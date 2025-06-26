"""
High-performance data pipeline implementation using tf.data.
Provides optimized data loading, preprocessing, and augmentation for pneumonia detection.
"""

import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable, Any
import logging
import time
from functools import partial

from validation_utils import (
    validate_directory_exists, validate_dataset_structure, 
    ValidationError, FileValidationError, logger as validation_logger
)

logger = logging.getLogger(__name__)


class PerformanceDataPipeline:
    """
    High-performance data pipeline using tf.data for optimal training speed.
    
    Features:
    - tf.data API for efficient data loading
    - Parallel preprocessing with multiple workers
    - Memory-mapped file loading
    - Optimized augmentation pipelines
    - Mixed precision support
    - Advanced caching and prefetching
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance data pipeline.
        
        Args:
            config: Configuration dictionary containing data and training parameters
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.training_config = config.get('training', {})
        
        # Performance settings
        self.num_parallel_calls = tf.data.AUTOTUNE
        self.prefetch_buffer_size = tf.data.AUTOTUNE
        self.cache_dataset = self.data_config.get('cache_dataset', True)
        self.use_mixed_precision = self.training_config.get('use_mixed_precision', False)
        
        # Data paths and parameters
        self.train_dir = self.data_config.get('train_dir', 'chest_xray/train')
        self.test_dir = self.data_config.get('test_dir', 'chest_xray/test')
        self.val_dir = self.data_config.get('val_dir')
        
        self.image_size = tuple(self.data_config.get('image_size', [128, 128]))
        self.batch_size = self.training_config.get('batch_size', 32)
        self.validation_split = self.training_config.get('validation_split', 0.2)
        
        # Class configuration
        self.class_names = self.data_config.get('class_names', ['NORMAL', 'PNEUMONIA'])
        self.num_classes = len(self.class_names)
        
        # Initialize tf.data optimizations
        self._setup_tf_data_optimizations()
        
        logger.info(f"PerformanceDataPipeline initialized")
        logger.info(f"Image size: {self.image_size}, Batch size: {self.batch_size}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        
    def _setup_tf_data_optimizations(self):
        """Setup tf.data performance optimizations."""
        # Enable experimental optimizations
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_deterministic = False  # Allow non-deterministic for performance
        
        self.tf_data_options = options
        
        # Setup memory growth for GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Enabled memory growth for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"Could not set memory growth: {e}")
    
    def create_file_dataset(self, directory: str, split_name: str = None) -> tf.data.Dataset:
        """
        Create a tf.data dataset from directory structure.
        
        Args:
            directory: Path to data directory
            split_name: Name of the split for logging
            
        Returns:
            tf.data.Dataset containing (file_path, label) pairs
        """
        try:
            # Validate directory structure
            directory = validate_directory_exists(directory)
            dataset_info = validate_dataset_structure(directory, self.class_names)
            
            # Create file path and label lists
            file_paths = []
            labels = []
            
            for class_idx, class_name in enumerate(self.class_names):
                class_info = dataset_info[class_name]
                class_dir = class_info['path']
                
                for image_file in class_info['images']:
                    file_path = os.path.join(class_dir, image_file)
                    file_paths.append(file_path)
                    labels.append(class_idx)
            
            # Convert to tensors
            file_paths_tensor = tf.constant(file_paths, dtype=tf.string)
            labels_tensor = tf.constant(labels, dtype=tf.int32)
            
            # Create dataset
            dataset = tf.data.Dataset.from_tensor_slices((file_paths_tensor, labels_tensor))
            
            if split_name:
                logger.info(f"{split_name} dataset: {len(file_paths)} samples")
                for i, class_name in enumerate(self.class_names):
                    count = sum(1 for label in labels if label == i)
                    logger.info(f"  {class_name}: {count} samples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating file dataset: {str(e)}")
            raise FileValidationError(f"Failed to create dataset from {directory}: {str(e)}")
    
    def load_and_preprocess_image(self, file_path: tf.Tensor, label: tf.Tensor, 
                                 is_training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and preprocess a single image with optimized operations.
        
        Args:
            file_path: Path to image file
            label: Image label
            is_training: Whether this is for training (affects augmentation)
            
        Returns:
            Tuple of (preprocessed_image, label)
        """
        # Load image using tf.io for better performance
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        
        # Resize image efficiently
        image = tf.image.resize(image, self.image_size, method='bilinear')
        
        # Normalize based on config
        normalize_method = self.data_config.get('normalize_method', 'rescale')
        if normalize_method == 'rescale':
            image = image / 255.0
        elif normalize_method == 'standardize':
            image = tf.image.per_image_standardization(image)
        elif normalize_method == 'imagenet':
            # ImageNet normalization
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            image = image / 255.0
            image = (image - mean) / std
        
        # Apply augmentation if training
        if is_training and self.data_config.get('use_augmentation', True):
            image = self._apply_augmentation(image)
        
        # Ensure image has correct shape
        image = tf.ensure_shape(image, [*self.image_size, 3])
        
        # Convert to mixed precision if enabled
        if self.use_mixed_precision:
            image = tf.cast(image, tf.float16)
        
        return image, label
    
    def _apply_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply data augmentation using tf.image for optimal performance.
        
        Args:
            image: Input image tensor
            
        Returns:
            Augmented image tensor
        """
        aug_config = self.data_config.get('augmentation', {})
        
        # Random rotation
        rotation_range = aug_config.get('rotation_range', 0)
        if rotation_range > 0:
            max_rotation = rotation_range * np.pi / 180.0  # Convert to radians
            angle = tf.random.uniform([], -max_rotation, max_rotation)
            image = tf.keras.utils.image_utils.apply_affine_transform(
                image, theta=angle, fill_mode='nearest'
            )
        
        # Random horizontal flip
        if aug_config.get('horizontal_flip', False):
            image = tf.image.random_flip_left_right(image)
        
        # Random vertical flip
        if aug_config.get('vertical_flip', False):
            image = tf.image.random_flip_up_down(image)
        
        # Random brightness
        brightness_range = aug_config.get('brightness_range', 0)
        if brightness_range > 0:
            image = tf.image.random_brightness(image, brightness_range)
        
        # Random contrast
        contrast_range = aug_config.get('contrast_range', 0)
        if contrast_range > 0:
            lower = 1.0 - contrast_range
            upper = 1.0 + contrast_range
            image = tf.image.random_contrast(image, lower, upper)
        
        # Random zoom (implemented as central crop + resize)
        zoom_range = aug_config.get('zoom_range', 0)
        if zoom_range > 0:
            # Random zoom factor
            zoom_factor = tf.random.uniform([], 1.0 - zoom_range, 1.0 + zoom_range)
            
            # Calculate crop size
            h, w = tf.shape(image)[0], tf.shape(image)[1]
            crop_h = tf.cast(tf.cast(h, tf.float32) / zoom_factor, tf.int32)
            crop_w = tf.cast(tf.cast(w, tf.float32) / zoom_factor, tf.int32)
            
            # Random crop
            image = tf.image.random_crop(image, [crop_h, crop_w, 3])
            # Resize back to original size
            image = tf.image.resize(image, [h, w])
        
        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    def create_training_dataset(self) -> tf.data.Dataset:
        """
        Create optimized training dataset with full pipeline.
        
        Returns:
            Optimized tf.data.Dataset for training
        """
        try:
            logger.info("Creating training dataset...")
            
            # Create file dataset
            dataset = self.create_file_dataset(self.train_dir, "Training")
            
            # Split into train/validation if no separate validation set
            if self.val_dir is None and self.validation_split > 0:
                total_size = len(list(dataset))
                train_size = int(total_size * (1 - self.validation_split))
                
                dataset = dataset.shuffle(total_size, reshuffle_each_iteration=False)
                train_dataset = dataset.take(train_size)
                
                logger.info(f"Split dataset: {train_size} training samples")
            else:
                train_dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
            
            # Apply preprocessing pipeline
            train_dataset = train_dataset.map(
                lambda x, y: self.load_and_preprocess_image(x, y, is_training=True),
                num_parallel_calls=self.num_parallel_calls
            )
            
            # Batch the dataset
            train_dataset = train_dataset.batch(self.batch_size, drop_remainder=True)
            
            # Cache dataset if enabled and small enough
            if self.cache_dataset:
                train_dataset = train_dataset.cache()
            
            # Prefetch for performance
            train_dataset = train_dataset.prefetch(self.prefetch_buffer_size)
            
            # Apply tf.data optimizations
            train_dataset = train_dataset.with_options(self.tf_data_options)
            
            logger.info("Training dataset created successfully")
            return train_dataset
            
        except Exception as e:
            logger.error(f"Error creating training dataset: {str(e)}")
            raise ValidationError(f"Failed to create training dataset: {str(e)}")
    
    def create_validation_dataset(self) -> tf.data.Dataset:
        """
        Create optimized validation dataset.
        
        Returns:
            Optimized tf.data.Dataset for validation
        """
        try:
            logger.info("Creating validation dataset...")
            
            if self.val_dir:
                # Use separate validation directory
                dataset = self.create_file_dataset(self.val_dir, "Validation")
            else:
                # Use split from training data
                dataset = self.create_file_dataset(self.train_dir)
                total_size = len(list(dataset))
                train_size = int(total_size * (1 - self.validation_split))
                
                dataset = dataset.shuffle(total_size, reshuffle_each_iteration=False)
                dataset = dataset.skip(train_size)
                
                val_size = total_size - train_size
                logger.info(f"Validation dataset: {val_size} samples")
            
            # Apply preprocessing (no augmentation for validation)
            dataset = dataset.map(
                lambda x, y: self.load_and_preprocess_image(x, y, is_training=False),
                num_parallel_calls=self.num_parallel_calls
            )
            
            # Batch the dataset
            dataset = dataset.batch(self.batch_size)
            
            # Cache validation dataset
            dataset = dataset.cache()
            
            # Prefetch for performance
            dataset = dataset.prefetch(self.prefetch_buffer_size)
            
            # Apply tf.data optimizations
            dataset = dataset.with_options(self.tf_data_options)
            
            logger.info("Validation dataset created successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating validation dataset: {str(e)}")
            raise ValidationError(f"Failed to create validation dataset: {str(e)}")
    
    def create_test_dataset(self) -> tf.data.Dataset:
        """
        Create optimized test dataset.
        
        Returns:
            Optimized tf.data.Dataset for testing
        """
        try:
            logger.info("Creating test dataset...")
            
            # Create file dataset
            dataset = self.create_file_dataset(self.test_dir, "Test")
            
            # Apply preprocessing (no augmentation for test)
            dataset = dataset.map(
                lambda x, y: self.load_and_preprocess_image(x, y, is_training=False),
                num_parallel_calls=self.num_parallel_calls
            )
            
            # Batch the dataset
            dataset = dataset.batch(self.batch_size)
            
            # Cache test dataset
            dataset = dataset.cache()
            
            # Prefetch for performance
            dataset = dataset.prefetch(self.prefetch_buffer_size)
            
            # Apply tf.data optimizations
            dataset = dataset.with_options(self.tf_data_options)
            
            logger.info("Test dataset created successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating test dataset: {str(e)}")
            raise ValidationError(f"Failed to create test dataset: {str(e)}")
    
    def get_class_weights(self, train_dataset: tf.data.Dataset) -> Dict[int, float]:
        """
        Calculate class weights from the training dataset.
        
        Args:
            train_dataset: Training dataset
            
        Returns:
            Dictionary mapping class indices to weights
        """
        try:
            logger.info("Calculating class weights...")
            
            # Count samples per class
            class_counts = {i: 0 for i in range(self.num_classes)}
            
            for _, labels in train_dataset.unbatch():
                label = int(labels.numpy())
                class_counts[label] += 1
            
            # Calculate weights
            total_samples = sum(class_counts.values())
            class_weights = {}
            
            for class_idx in range(self.num_classes):
                weight = total_samples / (self.num_classes * class_counts[class_idx])
                class_weights[class_idx] = weight
            
            logger.info(f"Class distribution: {class_counts}")
            logger.info(f"Class weights: {class_weights}")
            
            return class_weights
            
        except Exception as e:
            logger.error(f"Error calculating class weights: {str(e)}")
            return {i: 1.0 for i in range(self.num_classes)}
    
    def benchmark_dataset(self, dataset: tf.data.Dataset, num_batches: int = 10) -> Dict[str, float]:
        """
        Benchmark dataset performance.
        
        Args:
            dataset: Dataset to benchmark
            num_batches: Number of batches to test
            
        Returns:
            Performance metrics
        """
        try:
            logger.info(f"Benchmarking dataset performance ({num_batches} batches)...")
            
            # Warm up
            for _ in dataset.take(2):
                pass
            
            # Benchmark
            start_time = time.time()
            samples_processed = 0
            
            for batch_images, batch_labels in dataset.take(num_batches):
                samples_processed += tf.shape(batch_images)[0].numpy()
            
            total_time = time.time() - start_time
            
            metrics = {
                'total_time': total_time,
                'samples_processed': samples_processed,
                'samples_per_second': samples_processed / total_time,
                'batches_per_second': num_batches / total_time,
                'avg_batch_time': total_time / num_batches
            }
            
            logger.info(f"Performance metrics:")
            logger.info(f"  Samples/sec: {metrics['samples_per_second']:.1f}")
            logger.info(f"  Batches/sec: {metrics['batches_per_second']:.2f}")
            logger.info(f"  Avg batch time: {metrics['avg_batch_time']:.3f}s")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error benchmarking dataset: {str(e)}")
            return {}


class MemoryMappedDataLoader:
    """
    Memory-mapped data loader for very large datasets.
    Provides efficient access to large datasets without loading everything into memory.
    """
    
    def __init__(self, data_dir: str, cache_dir: str = "data_cache"):
        """
        Initialize memory-mapped data loader.
        
        Args:
            data_dir: Directory containing the dataset
            cache_dir: Directory for memory-mapped cache files
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.file_paths = []
        self.labels = []
        self.class_names = []
        
        logger.info(f"MemoryMappedDataLoader initialized for {data_dir}")
    
    def create_memory_mapped_cache(self, image_size: Tuple[int, int] = (224, 224)) -> str:
        """
        Create memory-mapped cache files for the dataset.
        
        Args:
            image_size: Target image size for preprocessing
            
        Returns:
            Path to the created cache file
        """
        try:
            logger.info("Creating memory-mapped cache...")
            
            # Collect all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            all_files = []
            all_labels = []
            
            for class_idx, class_dir in enumerate(sorted(self.data_dir.iterdir())):
                if class_dir.is_dir():
                    self.class_names.append(class_dir.name)
                    
                    for img_file in class_dir.iterdir():
                        if img_file.suffix.lower() in image_extensions:
                            all_files.append(str(img_file))
                            all_labels.append(class_idx)
            
            # Create memory-mapped arrays
            num_samples = len(all_files)
            cache_file = self.cache_dir / f"dataset_cache_{image_size[0]}x{image_size[1]}.dat"
            
            # Create memory-mapped array for images
            images_mmap = np.memmap(
                str(cache_file),
                dtype=np.float32,
                mode='w+',
                shape=(num_samples, *image_size, 3)
            )
            
            # Process and store images
            logger.info(f"Processing {num_samples} images...")
            
            for i, (file_path, label) in enumerate(zip(all_files, all_labels)):
                if i % 1000 == 0:
                    logger.info(f"Processed {i}/{num_samples} images")
                
                # Load and preprocess image
                try:
                    import cv2
                    image = cv2.imread(file_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, image_size)
                        image = image.astype(np.float32) / 255.0
                        images_mmap[i] = image
                    else:
                        logger.warning(f"Could not load image: {file_path}")
                        images_mmap[i] = np.zeros((*image_size, 3), dtype=np.float32)
                        
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    images_mmap[i] = np.zeros((*image_size, 3), dtype=np.float32)
            
            # Save metadata
            metadata = {
                'num_samples': num_samples,
                'image_size': image_size,
                'file_paths': all_files,
                'labels': all_labels,
                'class_names': self.class_names
            }
            
            metadata_file = self.cache_dir / f"metadata_{image_size[0]}x{image_size[1]}.npy"
            np.save(str(metadata_file), metadata)
            
            logger.info(f"Memory-mapped cache created: {cache_file}")
            logger.info(f"Cache size: {images_mmap.nbytes / (1024**3):.2f} GB")
            
            return str(cache_file)
            
        except Exception as e:
            logger.error(f"Error creating memory-mapped cache: {str(e)}")
            raise ValidationError(f"Failed to create memory-mapped cache: {str(e)}")
    
    def load_from_cache(self, cache_file: str, image_size: Tuple[int, int]) -> Tuple[np.memmap, np.ndarray, List[str]]:
        """
        Load dataset from memory-mapped cache.
        
        Args:
            cache_file: Path to cache file
            image_size: Image size used for the cache
            
        Returns:
            Tuple of (images_mmap, labels, class_names)
        """
        try:
            # Load metadata
            metadata_file = self.cache_dir / f"metadata_{image_size[0]}x{image_size[1]}.npy"
            metadata = np.load(str(metadata_file), allow_pickle=True).item()
            
            # Load memory-mapped images
            images_mmap = np.memmap(
                cache_file,
                dtype=np.float32,
                mode='r',
                shape=(metadata['num_samples'], *image_size, 3)
            )
            
            labels = np.array(metadata['labels'])
            class_names = metadata['class_names']
            
            logger.info(f"Loaded memory-mapped dataset: {len(labels)} samples")
            logger.info(f"Classes: {class_names}")
            
            return images_mmap, labels, class_names
            
        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            raise ValidationError(f"Failed to load from cache: {str(e)}")


def create_performance_datasets(config: Dict[str, Any]) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create high-performance datasets using tf.data pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    try:
        logger.info("Creating performance-optimized datasets...")
        
        # Initialize data pipeline
        pipeline = PerformanceDataPipeline(config)
        
        # Create datasets
        train_dataset = pipeline.create_training_dataset()
        val_dataset = pipeline.create_validation_dataset()
        test_dataset = pipeline.create_test_dataset()
        
        # Benchmark performance
        if config.get('logging', {}).get('benchmark_data', False):
            logger.info("Benchmarking training dataset...")
            pipeline.benchmark_dataset(train_dataset)
        
        logger.info("Performance datasets created successfully")
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Error creating performance datasets: {str(e)}")
        raise ValidationError(f"Failed to create performance datasets: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    test_config = {
        'data': {
            'train_dir': 'chest_xray/train',
            'test_dir': 'chest_xray/test',
            'image_size': [128, 128],
            'use_augmentation': True,
            'normalize_method': 'rescale',
            'cache_dataset': True,
            'augmentation': {
                'rotation_range': 15,
                'horizontal_flip': True,
                'brightness_range': 0.1,
                'zoom_range': 0.1
            }
        },
        'training': {
            'batch_size': 32,
            'validation_split': 0.2,
            'use_mixed_precision': False
        },
        'logging': {
            'benchmark_data': True
        }
    }
    
    try:
        # Test performance pipeline
        pipeline = PerformanceDataPipeline(test_config)
        
        # Create a small test dataset
        train_ds = pipeline.create_training_dataset()
        
        # Benchmark performance
        metrics = pipeline.benchmark_dataset(train_ds, num_batches=5)
        
        print("✅ Performance data pipeline test completed")
        print(f"Performance: {metrics.get('samples_per_second', 0):.1f} samples/sec")
        
    except Exception as e:
        print(f"❌ Performance data pipeline test failed: {e}")