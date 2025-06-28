"""
Unit tests for data pipeline components.
"""
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
from typing import Tuple
from unittest.mock import Mock, patch, MagicMock

from src.training.data_pipeline import (
    DataPipeline, create_data_pipeline, preprocess_image,
    augment_image, load_and_preprocess_image
)


class TestDataPipelineCreation:
    """Test data pipeline creation and configuration."""
    
    @pytest.mark.unit
    def test_create_data_pipeline_default(self, test_data_dir):
        """Test creating data pipeline with default settings."""
        pipeline = DataPipeline(
            data_dir=str(test_data_dir / "train"),
            batch_size=32,
            image_size=(128, 128),
            augment=False,
            shuffle=True
        )
        
        assert pipeline.data_dir == str(test_data_dir / "train")
        assert pipeline.batch_size == 32
        assert pipeline.image_size == (128, 128)
        assert pipeline.augment is False
        assert pipeline.shuffle is True
    
    @pytest.mark.unit
    def test_create_data_pipeline_with_augmentation(self, test_data_dir):
        """Test creating data pipeline with augmentation enabled."""
        pipeline = DataPipeline(
            data_dir=str(test_data_dir / "train"),
            batch_size=16,
            image_size=(224, 224),
            augment=True,
            shuffle=True,
            cache=True,
            prefetch=True
        )
        
        assert pipeline.augment is True
        assert pipeline.cache is True
        assert pipeline.prefetch is True
    
    @pytest.mark.unit
    def test_data_pipeline_invalid_directory(self):
        """Test data pipeline with invalid directory."""
        with pytest.raises(ValueError, match="Directory"):
            DataPipeline(
                data_dir="/nonexistent/directory",
                batch_size=32,
                image_size=(128, 128)
            )
    
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [0, -1, 0.5, "32"])
    def test_data_pipeline_invalid_batch_size(self, test_data_dir, batch_size):
        """Test data pipeline with invalid batch sizes."""
        with pytest.raises((ValueError, TypeError)):
            DataPipeline(
                data_dir=str(test_data_dir / "train"),
                batch_size=batch_size,
                image_size=(128, 128)
            )


class TestImagePreprocessing:
    """Test image preprocessing functions."""
    
    @pytest.mark.unit
    def test_preprocess_image_normalize(self):
        """Test image normalization."""
        # Create test image with known values
        image = tf.constant([[[255, 128, 0]]], dtype=tf.uint8)
        image = tf.image.resize(image, [128, 128])
        
        processed = preprocess_image(image, normalize=True)
        
        # Check normalization (values should be in [0, 1])
        assert tf.reduce_min(processed) >= 0.0
        assert tf.reduce_max(processed) <= 1.0
    
    @pytest.mark.unit
    def test_preprocess_image_resize(self):
        """Test image resizing."""
        # Create image of different size
        image = tf.random.uniform([256, 256, 3], maxval=255, dtype=tf.float32)
        
        processed = preprocess_image(image, target_size=(128, 128))
        
        assert processed.shape == (128, 128, 3)
    
    @pytest.mark.unit
    def test_preprocess_image_grayscale_to_rgb(self):
        """Test converting grayscale to RGB."""
        # Create grayscale image
        image = tf.random.uniform([128, 128, 1], maxval=255, dtype=tf.float32)
        
        processed = preprocess_image(image, convert_to_rgb=True)
        
        assert processed.shape == (128, 128, 3)
        # All channels should be the same for converted grayscale
        assert tf.reduce_all(processed[:, :, 0] == processed[:, :, 1])
        assert tf.reduce_all(processed[:, :, 1] == processed[:, :, 2])


class TestImageAugmentation:
    """Test image augmentation functions."""
    
    @pytest.mark.unit
    def test_augment_image_basic(self):
        """Test basic image augmentation."""
        # Create test image
        image = tf.random.uniform([128, 128, 3], dtype=tf.float32)
        
        # Apply augmentation multiple times
        augmented_images = [augment_image(image) for _ in range(5)]
        
        # Check that augmented images have the same shape
        for aug_img in augmented_images:
            assert aug_img.shape == image.shape
        
        # Check that augmentation produces different results
        # (may occasionally fail due to randomness)
        differences = []
        for i in range(len(augmented_images) - 1):
            diff = tf.reduce_mean(tf.abs(augmented_images[i] - augmented_images[i+1]))
            differences.append(diff.numpy())
        
        # At least some augmentations should be different
        assert max(differences) > 0.01
    
    @pytest.mark.unit
    def test_augment_image_parameters(self):
        """Test augmentation with specific parameters."""
        image = tf.random.uniform([128, 128, 3], dtype=tf.float32)
        
        # Test with custom parameters
        augmented = augment_image(
            image,
            rotation_range=45,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=0.3
        )
        
        assert augmented.shape == image.shape
        assert augmented.dtype == image.dtype


class TestDatasetCreation:
    """Test TensorFlow dataset creation."""
    
    @pytest.mark.unit
    def test_create_dataset_from_directory(self, test_data_dir):
        """Test creating dataset from directory structure."""
        with patch('tensorflow.keras.preprocessing.image_dataset_from_directory') as mock_dataset:
            # Mock the dataset creation
            mock_ds = MagicMock()
            mock_dataset.return_value = mock_ds
            
            pipeline = DataPipeline(
                data_dir=str(test_data_dir / "train"),
                batch_size=32,
                image_size=(128, 128),
                validation_split=0.2,
                subset="training"
            )
            
            dataset = pipeline.create_dataset()
            
            # Verify the function was called with correct parameters
            mock_dataset.assert_called_once()
            call_args = mock_dataset.call_args[1]
            assert call_args['directory'] == str(test_data_dir / "train")
            assert call_args['batch_size'] == 32
            assert call_args['image_size'] == (128, 128)
            assert call_args['validation_split'] == 0.2
            assert call_args['subset'] == "training"
    
    @pytest.mark.unit
    def test_dataset_with_class_names(self, test_data_dir):
        """Test dataset creation with specific class names."""
        pipeline = DataPipeline(
            data_dir=str(test_data_dir / "train"),
            batch_size=16,
            image_size=(128, 128),
            class_names=["NORMAL", "PNEUMONIA"]
        )
        
        assert pipeline.class_names == ["NORMAL", "PNEUMONIA"]
        assert pipeline.num_classes == 2
    
    @pytest.mark.unit
    def test_dataset_caching_and_prefetching(self, test_data_dir):
        """Test dataset optimization features."""
        with patch('tensorflow.keras.preprocessing.image_dataset_from_directory') as mock_dataset:
            # Create a mock dataset with required methods
            mock_ds = MagicMock()
            mock_ds.cache.return_value = mock_ds
            mock_ds.prefetch.return_value = mock_ds
            mock_dataset.return_value = mock_ds
            
            pipeline = DataPipeline(
                data_dir=str(test_data_dir / "train"),
                batch_size=32,
                image_size=(128, 128),
                cache=True,
                prefetch=True
            )
            
            dataset = pipeline.create_dataset()
            
            # Verify optimization methods were called
            mock_ds.cache.assert_called()
            mock_ds.prefetch.assert_called_with(tf.data.AUTOTUNE)


class TestLoadAndPreprocessImage:
    """Test individual image loading and preprocessing."""
    
    @pytest.mark.unit
    def test_load_and_preprocess_image_file(self, sample_images):
        """Test loading and preprocessing a single image file."""
        normal_img, _ = sample_images
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Save test image
            tf.keras.preprocessing.image.save_img(tmp.name, normal_img)
            tmp_path = tmp.name
        
        try:
            # Load and preprocess
            processed = load_and_preprocess_image(
                tmp_path,
                target_size=(224, 224),
                normalize=True
            )
            
            assert processed.shape == (224, 224, 3)
            assert processed.dtype == tf.float32
            assert tf.reduce_min(processed) >= 0.0
            assert tf.reduce_max(processed) <= 1.0
        finally:
            Path(tmp_path).unlink()
    
    @pytest.mark.unit
    def test_load_and_preprocess_image_invalid_path(self):
        """Test loading from invalid file path."""
        with pytest.raises((tf.errors.NotFoundError, ValueError)):
            load_and_preprocess_image(
                "/nonexistent/image.jpg",
                target_size=(128, 128)
            )


class TestDataPipelinePerformance:
    """Test data pipeline performance features."""
    
    @pytest.mark.unit
    def test_parallel_processing(self, test_data_dir):
        """Test parallel processing configuration."""
        pipeline = DataPipeline(
            data_dir=str(test_data_dir / "train"),
            batch_size=32,
            image_size=(128, 128),
            num_parallel_calls=4
        )
        
        assert pipeline.num_parallel_calls == 4
    
    @pytest.mark.unit
    def test_memory_optimization(self, test_data_dir):
        """Test memory optimization features."""
        pipeline = DataPipeline(
            data_dir=str(test_data_dir / "train"),
            batch_size=32,
            image_size=(128, 128),
            cache=True,
            cache_to_disk=True,
            cache_filename="/tmp/cache"
        )
        
        assert pipeline.cache is True
        assert pipeline.cache_to_disk is True
        assert pipeline.cache_filename == "/tmp/cache"