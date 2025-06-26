"""
Optimized image preprocessing pipeline for pneumonia detection.
Provides efficient image loading, preprocessing, and augmentation.
"""

import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List, Optional, Callable
import logging
from functools import partial
import albumentations as albu
from albumentations.tensorflow import AugmentationPipelineV2

from validation_utils import ValidationError, ImageValidationError, logger as validation_logger

logger = logging.getLogger(__name__)


class OptimizedPreprocessor:
    """
    High-performance image preprocessing pipeline.
    
    Features:
    - Multiple backend support (TensorFlow, OpenCV, Albumentations)
    - Optimized memory usage
    - Batch processing capabilities
    - Advanced augmentation strategies
    - Mixed precision support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimized preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.training_config = config.get('training', {})
        
        # Image parameters
        self.image_size = tuple(self.data_config.get('image_size', [224, 224]))
        self.channels = self.data_config.get('channels', 3)
        self.normalize_method = self.data_config.get('normalize_method', 'rescale')
        
        # Augmentation settings
        self.use_augmentation = self.data_config.get('use_augmentation', True)
        self.augmentation_config = self.data_config.get('augmentation', {})
        self.backend = self.data_config.get('augmentation_backend', 'tensorflow')  # 'tensorflow', 'albumentations'
        
        # Performance settings
        self.use_mixed_precision = self.training_config.get('use_mixed_precision', False)
        self.batch_processing = self.data_config.get('batch_processing', True)
        
        # Initialize preprocessing components
        self._setup_normalization()
        self._setup_augmentation()
        
        logger.info(f"OptimizedPreprocessor initialized")
        logger.info(f"Image size: {self.image_size}, Backend: {self.backend}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
    
    def _setup_normalization(self):
        """Setup normalization parameters."""
        if self.normalize_method == 'imagenet':
            self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
            self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        elif self.normalize_method == 'custom':
            # Custom normalization values from config
            self.mean = tf.constant(self.data_config.get('custom_mean', [0.5, 0.5, 0.5]), dtype=tf.float32)
            self.std = tf.constant(self.data_config.get('custom_std', [0.5, 0.5, 0.5]), dtype=tf.float32)
        else:
            self.mean = None
            self.std = None
    
    def _setup_augmentation(self):
        """Setup augmentation pipeline based on backend."""
        if not self.use_augmentation:
            self.augmentation_pipeline = None
            return
        
        if self.backend == 'albumentations':
            self._setup_albumentations_pipeline()
        else:
            self._setup_tensorflow_pipeline()
    
    def _setup_albumentations_pipeline(self):
        """Setup Albumentations augmentation pipeline."""
        try:
            transforms = []
            aug_config = self.augmentation_config
            
            # Geometric transformations
            if aug_config.get('rotation_range', 0) > 0:
                transforms.append(albu.Rotate(
                    limit=aug_config['rotation_range'],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.7
                ))
            
            if aug_config.get('horizontal_flip', False):
                transforms.append(albu.HorizontalFlip(p=0.5))
            
            if aug_config.get('vertical_flip', False):
                transforms.append(albu.VerticalFlip(p=0.5))
            
            # Affine transformations
            if any(aug_config.get(k, 0) > 0 for k in ['width_shift_range', 'height_shift_range', 'shear_range']):
                transforms.append(albu.ShiftScaleRotate(
                    shift_limit=max(aug_config.get('width_shift_range', 0), aug_config.get('height_shift_range', 0)),
                    scale_limit=aug_config.get('zoom_range', 0),
                    rotate_limit=0,  # Handled separately
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.6
                ))
            
            # Color transformations
            if aug_config.get('brightness_range', 0) > 0:
                transforms.append(albu.RandomBrightnessContrast(
                    brightness_limit=aug_config['brightness_range'],
                    contrast_limit=aug_config.get('contrast_range', 0),
                    p=0.5
                ))
            
            # Advanced augmentations
            if aug_config.get('noise', False):
                transforms.append(albu.OneOf([
                    albu.GaussNoise(var_limit=(10.0, 50.0)),
                    albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                ], p=0.3))
            
            if aug_config.get('blur', False):
                transforms.append(albu.OneOf([
                    albu.Blur(blur_limit=3),
                    albu.GaussianBlur(blur_limit=3),
                    albu.MotionBlur(blur_limit=3),
                ], p=0.3))
            
            if aug_config.get('distortion', False):
                transforms.append(albu.OneOf([
                    albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
                    albu.GridDistortion(num_steps=5, distort_limit=0.1),
                    albu.OpticalDistortion(distort_limit=0.1, shift_limit=0.1),
                ], p=0.2))
            
            # CLAHE for medical images
            if aug_config.get('clahe', False):
                transforms.append(albu.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3))
            
            # Create pipeline
            if transforms:
                self.augmentation_pipeline = albu.Compose(transforms)
                logger.info(f"Albumentations pipeline created with {len(transforms)} transforms")
            else:
                self.augmentation_pipeline = None
                
        except ImportError:
            logger.warning("Albumentations not available, falling back to TensorFlow augmentation")
            self._setup_tensorflow_pipeline()
        except Exception as e:
            logger.error(f"Error setting up Albumentations pipeline: {e}")
            self._setup_tensorflow_pipeline()
    
    def _setup_tensorflow_pipeline(self):
        """Setup TensorFlow-based augmentation pipeline."""
        self.augmentation_pipeline = self.augmentation_config
        logger.info("TensorFlow augmentation pipeline configured")
    
    @tf.function
    def load_and_decode_image(self, image_path: tf.Tensor) -> tf.Tensor:
        """
        Load and decode image using TensorFlow operations.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Decoded image tensor
        """
        # Read file
        image_data = tf.io.read_file(image_path)
        
        # Decode image (supports JPEG, PNG, BMP, GIF)
        image = tf.image.decode_image(image_data, channels=self.channels, expand_animations=False)
        image = tf.cast(image, tf.float32)
        
        # Ensure shape is known
        image.set_shape([None, None, self.channels])
        
        return image
    
    @tf.function
    def resize_image(self, image: tf.Tensor, method: str = 'bilinear') -> tf.Tensor:
        """
        Resize image with optimized methods.
        
        Args:
            image: Input image tensor
            method: Resize method ('bilinear', 'nearest', 'bicubic', 'area')
            
        Returns:
            Resized image tensor
        """
        if method == 'bilinear':
            return tf.image.resize(image, self.image_size, method=tf.image.ResizeMethod.BILINEAR)
        elif method == 'nearest':
            return tf.image.resize(image, self.image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        elif method == 'bicubic':
            return tf.image.resize(image, self.image_size, method=tf.image.ResizeMethod.BICUBIC)
        elif method == 'area':
            return tf.image.resize(image, self.image_size, method=tf.image.ResizeMethod.AREA)
        else:
            return tf.image.resize(image, self.image_size)
    
    @tf.function
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        """
        Normalize image based on configuration.
        
        Args:
            image: Input image tensor (0-255 range)
            
        Returns:
            Normalized image tensor
        """
        if self.normalize_method == 'rescale':
            return image / 255.0
        elif self.normalize_method == 'standardize':
            return tf.image.per_image_standardization(image)
        elif self.normalize_method in ['imagenet', 'custom']:
            # Convert to [0, 1] first
            image = image / 255.0
            # Apply mean and std normalization
            return (image - self.mean) / self.std
        else:
            return image / 255.0
    
    @tf.function
    def apply_tensorflow_augmentation(self, image: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Apply TensorFlow-based augmentation.
        
        Args:
            image: Input image tensor
            training: Whether to apply augmentation
            
        Returns:
            Augmented image tensor
        """
        if not training or not self.use_augmentation:
            return image
        
        aug_config = self.augmentation_pipeline or {}
        
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
        
        # Random saturation
        saturation_range = aug_config.get('saturation_range', 0)
        if saturation_range > 0:
            lower = 1.0 - saturation_range
            upper = 1.0 + saturation_range
            image = tf.image.random_saturation(image, lower, upper)
        
        # Random hue
        hue_range = aug_config.get('hue_range', 0)
        if hue_range > 0:
            image = tf.image.random_hue(image, hue_range)
        
        # Random crop and resize (zoom effect)
        zoom_range = aug_config.get('zoom_range', 0)
        if zoom_range > 0:
            # Random zoom factor
            zoom_factor = tf.random.uniform([], 1.0 - zoom_range, 1.0 + zoom_range)
            
            # Calculate crop size
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            crop_height = tf.cast(tf.cast(height, tf.float32) / zoom_factor, tf.int32)
            crop_width = tf.cast(tf.cast(width, tf.float32) / zoom_factor, tf.int32)
            
            # Random crop
            image = tf.image.random_crop(image, [crop_height, crop_width, self.channels])
            # Resize back
            image = tf.image.resize(image, [height, width])
        
        # Random rotation (approximated with small angle)
        rotation_range = aug_config.get('rotation_range', 0)
        if rotation_range > 0:
            # Small random rotations using tf.image operations
            if tf.random.uniform([]) < 0.5:
                image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    def apply_albumentations_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Albumentations-based augmentation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Augmented image as numpy array
        """
        if self.augmentation_pipeline is None:
            return image
        
        try:
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Apply augmentation
            augmented = self.augmentation_pipeline(image=image)
            return augmented['image']
            
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}, returning original image")
            return image
    
    @tf.function
    def preprocess_image(self, image_path: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Complete preprocessing pipeline for a single image.
        
        Args:
            image_path: Path to image file
            training: Whether this is for training (affects augmentation)
            
        Returns:
            Preprocessed image tensor
        """
        # Load and decode
        image = self.load_and_decode_image(image_path)
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Apply augmentation (TensorFlow backend only in tf.function)
        if self.backend == 'tensorflow':
            image = self.apply_tensorflow_augmentation(image, training)
        
        # Convert to mixed precision if enabled
        if self.use_mixed_precision:
            image = tf.cast(image, tf.float16)
        else:
            image = tf.cast(image, tf.float32)
        
        # Ensure shape
        image = tf.ensure_shape(image, [*self.image_size, self.channels])
        
        return image
    
    def preprocess_batch(self, image_paths: List[str], training: bool = False) -> tf.Tensor:
        """
        Preprocess a batch of images efficiently.
        
        Args:
            image_paths: List of image file paths
            training: Whether this is for training
            
        Returns:
            Batch of preprocessed images
        """
        try:
            batch_images = []
            
            for image_path in image_paths:
                if self.backend == 'albumentations' and training:
                    # Use Albumentations for augmentation
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.image_size)
                    
                    # Apply augmentation
                    image = self.apply_albumentations_augmentation(image)
                    
                    # Normalize
                    image = image.astype(np.float32)
                    if self.normalize_method == 'rescale':
                        image = image / 255.0
                    elif self.normalize_method == 'imagenet':
                        image = image / 255.0
                        image = (image - self.mean.numpy()) / self.std.numpy()
                    
                    batch_images.append(image)
                    
                else:
                    # Use TensorFlow preprocessing
                    image = self.preprocess_image(tf.constant(image_path), training)
                    batch_images.append(image.numpy())
            
            # Stack into batch
            batch = tf.stack(batch_images, axis=0)
            
            return batch
            
        except Exception as e:
            logger.error(f"Error preprocessing batch: {str(e)}")
            raise ImageValidationError(f"Failed to preprocess batch: {str(e)}")
    
    def create_preprocessing_function(self, training: bool = False) -> Callable:
        """
        Create a preprocessing function for use with tf.data.
        
        Args:
            training: Whether this is for training
            
        Returns:
            Preprocessing function
        """
        def preprocess_fn(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            image = self.preprocess_image(image_path, training)
            return image, label
        
        return preprocess_fn
    
    def get_preprocessing_stats(self, dataset_path: str, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Calculate preprocessing statistics for the dataset.
        
        Args:
            dataset_path: Path to dataset
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary of statistics
        """
        try:
            import glob
            
            # Get sample of images
            image_patterns = [
                os.path.join(dataset_path, "**", "*.jpg"),
                os.path.join(dataset_path, "**", "*.jpeg"),
                os.path.join(dataset_path, "**", "*.png")
            ]
            
            all_images = []
            for pattern in image_patterns:
                all_images.extend(glob.glob(pattern, recursive=True))
            
            if len(all_images) > num_samples:
                all_images = np.random.choice(all_images, num_samples, replace=False)
            
            # Calculate statistics
            pixel_values = []
            shapes = []
            
            for image_path in all_images:
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        pixel_values.extend(image.flatten())
                        shapes.append(image.shape)
                except Exception as e:
                    logger.warning(f"Could not process {image_path}: {e}")
            
            pixel_values = np.array(pixel_values, dtype=np.float32)
            
            stats = {
                'num_images_analyzed': len(all_images),
                'pixel_mean': float(np.mean(pixel_values)),
                'pixel_std': float(np.std(pixel_values)),
                'pixel_min': float(np.min(pixel_values)),
                'pixel_max': float(np.max(pixel_values)),
                'common_shapes': {str(shape): shapes.count(shape) for shape in set(shapes)},
                'recommended_normalization': {
                    'mean': [float(np.mean(pixel_values)) / 255.0] * 3,
                    'std': [float(np.std(pixel_values)) / 255.0] * 3
                }
            }
            
            logger.info(f"Dataset statistics calculated from {len(all_images)} images")
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating preprocessing stats: {str(e)}")
            return {}


class AdvancedAugmentation:
    """Advanced augmentation strategies for medical images."""
    
    @staticmethod
    def cutmix(image1: tf.Tensor, image2: tf.Tensor, label1: tf.Tensor, label2: tf.Tensor, 
              alpha: float = 1.0) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply CutMix augmentation.
        
        Args:
            image1, image2: Input images
            label1, label2: Corresponding labels
            alpha: Beta distribution parameter
            
        Returns:
            Mixed image and label
        """
        # Sample lambda from Beta distribution
        lam = tf.random.gamma([], alpha, alpha) / (tf.random.gamma([], alpha, alpha) + tf.random.gamma([], alpha, alpha))
        
        # Get image dimensions
        height, width = tf.shape(image1)[0], tf.shape(image1)[1]
        
        # Sample bounding box
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_w = tf.cast(tf.cast(width, tf.float32) * cut_ratio, tf.int32)
        cut_h = tf.cast(tf.cast(height, tf.float32) * cut_ratio, tf.int32)
        
        # Random center
        cx = tf.random.uniform([], 0, width, dtype=tf.int32)
        cy = tf.random.uniform([], 0, height, dtype=tf.int32)
        
        # Bounding box
        x1 = tf.maximum(0, cx - cut_w // 2)
        y1 = tf.maximum(0, cy - cut_h // 2)
        x2 = tf.minimum(width, cx + cut_w // 2)
        y2 = tf.minimum(height, cy + cut_h // 2)
        
        # Create mask
        mask = tf.zeros_like(image1[:, :, 0])
        ones = tf.ones([y2 - y1, x2 - x1])
        
        # Apply mask
        mask = tf.tensor_scatter_nd_update(
            mask,
            tf.stack([tf.range(y1, y2), tf.range(x1, x2)], axis=1),
            ones
        )
        mask = tf.expand_dims(mask, -1)
        
        # Mix images
        mixed_image = image1 * (1 - mask) + image2 * mask
        
        # Adjust lambda based on actual cut area
        actual_lam = 1.0 - tf.reduce_sum(mask) / tf.cast(tf.size(mask), tf.float32)
        mixed_label = actual_lam * label1 + (1 - actual_lam) * label2
        
        return mixed_image, mixed_label
    
    @staticmethod
    def mixup(image1: tf.Tensor, image2: tf.Tensor, label1: tf.Tensor, label2: tf.Tensor,
              alpha: float = 0.2) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply MixUp augmentation.
        
        Args:
            image1, image2: Input images
            label1, label2: Corresponding labels
            alpha: Beta distribution parameter
            
        Returns:
            Mixed image and label
        """
        # Sample mixing ratio
        lam = tf.random.uniform([], 0, alpha)
        
        # Mix images and labels
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label