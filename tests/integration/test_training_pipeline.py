"""
Integration tests for end-to-end training pipeline.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import tensorflow as tf
import numpy as np
from unittest.mock import patch, MagicMock
import yaml

from src.models.cnn import PneumoniaCNN
from src.config.config_loader import ConfigLoader
from src.config.config_schema import ExperimentConfig


class TestEndToEndTraining:
    """Test complete training pipeline integration."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_basic_training_pipeline(self, test_data_dir, sample_config_dict):
        """Test basic training pipeline from config to trained model."""
        # Modify config for quick training
        sample_config_dict['training']['epochs'] = 2
        sample_config_dict['training']['batch_size'] = 4
        sample_config_dict['data']['train_dir'] = str(test_data_dir / "train")
        sample_config_dict['data']['test_dir'] = str(test_data_dir / "test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set output directories
            sample_config_dict['paths']['models_dir'] = str(Path(tmpdir) / "models")
            sample_config_dict['paths']['logs_dir'] = str(Path(tmpdir) / "logs")
            sample_config_dict['paths']['results_dir'] = str(Path(tmpdir) / "results")
            
            # Create config
            loader = ConfigLoader()
            config = loader.load_from_dict(sample_config_dict)
            
            # Initialize model
            model = PneumoniaCNN(config=config, mode='standard')
            
            # Train model
            history = model.train()
            
            # Verify training completed
            assert history is not None
            assert 'loss' in history.history
            assert 'accuracy' in history.history
            assert len(history.history['loss']) == 2  # 2 epochs
            
            # Verify model was saved
            models_dir = Path(tmpdir) / "models"
            assert models_dir.exists()
            saved_models = list(models_dir.glob("*.h5"))
            assert len(saved_models) > 0
    
    @pytest.mark.integration
    def test_config_to_model_pipeline(self, temp_config_file):
        """Test loading config and creating model."""
        # Load config
        loader = ConfigLoader()
        config = loader.load_config(str(temp_config_file))
        
        # Create model from config
        model = PneumoniaCNN(config=config, mode='standard')
        
        # Verify model properties match config
        assert model.config.experiment_name == config.experiment_name
        assert model.config.model.learning_rate == config.model.learning_rate
        assert model.config.training.batch_size == config.training.batch_size
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_with_validation(self, test_data_dir, sample_config_dict):
        """Test training with validation split."""
        sample_config_dict['training']['epochs'] = 1
        sample_config_dict['training']['validation_split'] = 0.2
        sample_config_dict['data']['train_dir'] = str(test_data_dir / "train")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_config_dict['paths']['models_dir'] = str(Path(tmpdir) / "models")
            
            loader = ConfigLoader()
            config = loader.load_from_dict(sample_config_dict)
            
            model = PneumoniaCNN(config=config, mode='standard')
            history = model.train()
            
            # Verify validation metrics are present
            assert 'val_loss' in history.history
            assert 'val_accuracy' in history.history
    
    @pytest.mark.integration
    def test_model_evaluation_pipeline(self, test_data_dir, mock_model):
        """Test model evaluation on test data."""
        # Create a simple test dataset
        test_dir = test_data_dir / "test"
        
        # Create dataset
        test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            batch_size=2,
            image_size=(128, 128),
            label_mode='binary'
        )
        
        # Evaluate model
        results = mock_model.evaluate(test_dataset, return_dict=True)
        
        assert 'loss' in results
        assert 'accuracy' in results
        assert results['loss'] >= 0
        assert 0 <= results['accuracy'] <= 1


class TestDataPipelineIntegration:
    """Test data pipeline integration with model training."""
    
    @pytest.mark.integration
    def test_data_loading_and_preprocessing(self, test_data_dir):
        """Test complete data loading and preprocessing pipeline."""
        from src.training.data_pipeline import create_data_pipeline
        
        # Create data pipeline
        train_dataset = create_data_pipeline(
            data_dir=str(test_data_dir / "train"),
            batch_size=2,
            image_size=(128, 128),
            augment=True,
            shuffle=True
        )
        
        # Get a batch and verify
        for images, labels in train_dataset.take(1):
            assert images.shape[0] == 2  # batch size
            assert images.shape[1:] == (128, 128, 3)
            assert labels.shape[0] == 2
            assert images.dtype == tf.float32
            assert tf.reduce_min(images) >= 0.0
            assert tf.reduce_max(images) <= 1.0
    
    @pytest.mark.integration
    def test_augmentation_pipeline(self, test_data_dir):
        """Test data augmentation in pipeline."""
        from src.training.data_pipeline import create_data_pipeline
        
        # Create pipeline with augmentation
        dataset = create_data_pipeline(
            data_dir=str(test_data_dir / "train"),
            batch_size=1,
            image_size=(128, 128),
            augment=True,
            shuffle=False
        )
        
        # Get multiple batches of the same image
        batches = []
        for images, _ in dataset.take(3):
            batches.append(images.numpy())
        
        # Verify that augmentation produces different results
        # Note: This might occasionally fail due to random chance
        diff_01 = np.mean(np.abs(batches[0] - batches[1]))
        diff_12 = np.mean(np.abs(batches[1] - batches[2]))
        
        assert diff_01 > 0 or diff_12 > 0  # At least some difference


class TestModelSavingAndLoading:
    """Test model saving and loading integration."""
    
    @pytest.mark.integration
    def test_model_checkpoint_callback(self, test_data_dir, sample_config_dict):
        """Test model checkpoint callback during training."""
        sample_config_dict['training']['epochs'] = 3
        sample_config_dict['data']['train_dir'] = str(test_data_dir / "train")
        sample_config_dict['logging']['checkpoint']['enabled'] = True
        sample_config_dict['logging']['checkpoint']['save_best_only'] = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints"
            sample_config_dict['paths']['models_dir'] = str(checkpoint_path)
            
            loader = ConfigLoader()
            config = loader.load_from_dict(sample_config_dict)
            
            model = PneumoniaCNN(config=config, mode='standard')
            model.train()
            
            # Verify checkpoint was saved
            assert checkpoint_path.exists()
            checkpoints = list(checkpoint_path.glob("*.h5"))
            assert len(checkpoints) > 0
    
    @pytest.mark.integration
    def test_model_save_and_load(self, mock_model):
        """Test saving and loading trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.h5"
            
            # Save model
            mock_model.save(str(model_path))
            assert model_path.exists()
            
            # Load model
            loaded_model = tf.keras.models.load_model(str(model_path))
            
            # Verify model architecture
            assert len(loaded_model.layers) == len(mock_model.layers)
            
            # Test prediction
            test_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
            original_pred = mock_model.predict(test_input)
            loaded_pred = loaded_model.predict(test_input)
            
            # Predictions should be very close
            np.testing.assert_allclose(original_pred, loaded_pred, rtol=1e-5)


class TestConfigurationIntegration:
    """Test configuration system integration."""
    
    @pytest.mark.integration
    def test_config_override_integration(self, sample_config_dict):
        """Test configuration override functionality."""
        base_config = sample_config_dict.copy()
        
        # Create override config
        override_config = {
            "model": {"learning_rate": 0.01},
            "training": {"batch_size": 64}
        }
        
        loader = ConfigLoader()
        merged = loader.merge_configs(base_config, override_config)
        
        # Verify overrides
        assert merged["model"]["learning_rate"] == 0.01
        assert merged["training"]["batch_size"] == 64
        
        # Verify other values preserved
        assert merged["training"]["epochs"] == base_config["training"]["epochs"]
    
    @pytest.mark.integration
    def test_multiple_config_files(self):
        """Test loading and merging multiple config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base config
            base_config = {
                "experiment_name": "base",
                "model": {"learning_rate": 0.001},
                "training": {"epochs": 10}
            }
            base_path = Path(tmpdir) / "base.yaml"
            with open(base_path, 'w') as f:
                yaml.dump(base_config, f)
            
            # Create override config
            override_config = {
                "experiment_name": "override",
                "model": {"learning_rate": 0.01}
            }
            override_path = Path(tmpdir) / "override.yaml"
            with open(override_path, 'w') as f:
                yaml.dump(override_config, f)
            
            # Load and merge
            loader = ConfigLoader()
            base = loader.load_config(str(base_path))
            override = loader.load_config(str(override_path))
            
            # Manual merge for testing
            merged_dict = loader.merge_configs(
                base.__dict__,
                override.__dict__
            )
            
            assert merged_dict["experiment_name"] == "override"
            assert merged_dict["model"]["learning_rate"] == 0.01
            assert merged_dict["training"]["epochs"] == 10