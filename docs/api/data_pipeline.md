# Data Pipeline API

Documentation for data loading, preprocessing, and augmentation components.

## Overview

The data pipeline system provides efficient, scalable data loading and preprocessing for training and inference. It's built on tf.data for optimal performance and includes comprehensive augmentation options.

## Core Components

### DataPipeline Class

Main class for creating optimized data pipelines.

```python
class DataPipeline:
    """
    High-performance data pipeline for chest X-ray image processing.
    
    Features:
    - tf.data optimization with prefetching and caching
    - Configurable augmentation pipelines
    - Memory-mapped loading for large datasets
    - Parallel preprocessing
    """
```

#### Constructor

```python
def __init__(self, 
             data_dir: str,
             batch_size: int = 32,
             image_size: Tuple[int, int] = (128, 128),
             augment: bool = True,
             shuffle: bool = True,
             cache: bool = True,
             prefetch: bool = True,
             num_parallel_calls: int = tf.data.AUTOTUNE,
             validation_split: Optional[float] = None,
             subset: Optional[str] = None,
             class_names: Optional[List[str]] = None):
    """
    Initialize data pipeline.
    
    Args:
        data_dir: Path to data directory with class subdirectories
        batch_size: Batch size for training
        image_size: Target image size (height, width)
        augment: Whether to apply data augmentation
        shuffle: Whether to shuffle the dataset
        cache: Whether to cache processed data
        prefetch: Whether to prefetch batches
        num_parallel_calls: Number of parallel preprocessing calls
        validation_split: Fraction for validation split
        subset: 'training' or 'validation' for split datasets
        class_names: List of class names (inferred if None)
    """
```

#### Methods

##### create_dataset()

```python
def create_dataset(self) -> tf.data.Dataset:
    """
    Create optimized tf.data.Dataset.
    
    Returns:
        tf.data.Dataset: Optimized dataset ready for training
        
    Raises:
        ValueError: If data directory is invalid
        RuntimeError: If dataset creation fails
    """
```

Creates a complete data pipeline with all optimizations:
- Image loading and decoding
- Resizing and normalization
- Data augmentation (if enabled)
- Batching and prefetching
- Caching for performance

##### get_dataset_info()

```python
def get_dataset_info(self) -> Dict[str, Any]:
    """
    Get information about the dataset.
    
    Returns:
        Dict containing:
        - num_classes: Number of classes
        - class_names: List of class names
        - class_counts: Images per class
        - total_images: Total number of images
        - class_weights: Calculated class weights for imbalanced data
    """
```

## Preprocessing Functions

### Image Preprocessing

#### preprocess_image()

```python
def preprocess_image(image: tf.Tensor, 
                    target_size: Tuple[int, int] = (128, 128),
                    normalize: bool = True,
                    convert_to_rgb: bool = True) -> tf.Tensor:
    """
    Preprocess a single image.
    
    Args:
        image: Input image tensor
        target_size: Target size (height, width)
        normalize: Whether to normalize to [0, 1]
        convert_to_rgb: Whether to ensure RGB format
        
    Returns:
        tf.Tensor: Preprocessed image
        
    Raises:
        tf.errors.InvalidArgumentError: If image format is invalid
    """
```

Standard preprocessing pipeline:
1. **Format Conversion**: Ensure RGB format
2. **Resizing**: Resize to target dimensions
3. **Type Conversion**: Convert to float32
4. **Normalization**: Scale to [0, 1] range

#### load_and_preprocess_image()

```python
def load_and_preprocess_image(image_path: str,
                             target_size: Tuple[int, int] = (128, 128),
                             normalize: bool = True) -> tf.Tensor:
    """
    Load and preprocess image from file path.
    
    Args:
        image_path: Path to image file
        target_size: Target size (height, width)
        normalize: Whether to normalize pixel values
        
    Returns:
        tf.Tensor: Preprocessed image ready for model input
        
    Raises:
        tf.errors.NotFoundError: If image file doesn't exist
        tf.errors.InvalidArgumentError: If image format is unsupported
    """
```

### Data Augmentation

#### augment_image()

```python
def augment_image(image: tf.Tensor,
                 rotation_range: float = 20.0,
                 width_shift_range: float = 0.2,
                 height_shift_range: float = 0.2,
                 zoom_range: float = 0.15,
                 horizontal_flip: bool = True,
                 vertical_flip: bool = False,
                 brightness_range: Optional[Tuple[float, float]] = None,
                 contrast_range: Optional[Tuple[float, float]] = None,
                 fill_mode: str = 'nearest') -> tf.Tensor:
    """
    Apply data augmentation to image.
    
    Args:
        image: Input image tensor
        rotation_range: Degrees of random rotation
        width_shift_range: Fraction of width for random shifts
        height_shift_range: Fraction of height for random shifts
        zoom_range: Range for random zoom
        horizontal_flip: Whether to randomly flip horizontally
        vertical_flip: Whether to randomly flip vertically
        brightness_range: Range for brightness adjustment
        contrast_range: Range for contrast adjustment
        fill_mode: How to fill missing pixels ('nearest', 'constant', etc.)
        
    Returns:
        tf.Tensor: Augmented image
    """
```

#### Advanced Augmentation

```python
def advanced_augment(image: tf.Tensor,
                    label: tf.Tensor,
                    mixup_alpha: float = 0.4,
                    cutmix_alpha: float = 1.0,
                    enable_mixup: bool = False,
                    enable_cutmix: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply advanced augmentation techniques.
    
    Args:
        image: Input image tensor
        label: Input label tensor
        mixup_alpha: Alpha parameter for MixUp
        cutmix_alpha: Alpha parameter for CutMix
        enable_mixup: Whether to apply MixUp
        enable_cutmix: Whether to apply CutMix
        
    Returns:
        Tuple of (augmented_image, augmented_label)
    """
```

## Factory Functions

### create_data_pipeline()

```python
def create_data_pipeline(data_dir: str,
                        batch_size: int = 32,
                        image_size: Tuple[int, int] = (128, 128),
                        augment: bool = True,
                        shuffle: bool = True,
                        cache: bool = True,
                        prefetch: bool = True,
                        validation_split: Optional[float] = None,
                        subset: Optional[str] = None) -> tf.data.Dataset:
    """
    Factory function to create optimized data pipeline.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        image_size: Target image size
        augment: Whether to apply augmentation
        shuffle: Whether to shuffle data
        cache: Whether to cache dataset
        prefetch: Whether to prefetch batches
        validation_split: Validation split fraction
        subset: 'training' or 'validation'
        
    Returns:
        tf.data.Dataset: Ready-to-use dataset
    """
```

### create_inference_pipeline()

```python
def create_inference_pipeline(image_paths: List[str],
                             batch_size: int = 32,
                             image_size: Tuple[int, int] = (128, 128)) -> tf.data.Dataset:
    """
    Create pipeline for inference on image files.
    
    Args:
        image_paths: List of paths to image files
        batch_size: Batch size for inference
        image_size: Target image size
        
    Returns:
        tf.data.Dataset: Dataset for inference
    """
```

## Configuration Integration

### Pipeline Configuration

```python
@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline."""
    
    # Data paths
    train_dir: str = "chest_xray/train"
    test_dir: str = "chest_xray/test"
    val_dir: Optional[str] = None
    
    # Image processing
    image_size: Tuple[int, int] = (128, 128)
    color_mode: str = "rgb"
    channels: int = 3
    
    # Augmentation settings
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "zoom_range": 0.15,
        "horizontal_flip": True,
        "brightness_range": [0.8, 1.2]
    })
    
    # Performance settings
    batch_size: int = 32
    cache: bool = True
    prefetch: bool = True
    num_parallel_calls: int = -1  # tf.data.AUTOTUNE
    
    # Advanced features
    memory_mapped_loading: bool = False
    cache_to_disk: bool = False
    cache_filename: Optional[str] = None
```

### Usage with Configuration

```python
from src.config.config_loader import ConfigLoader
from src.training.data_pipeline import DataPipeline

# Load configuration
config = ConfigLoader().load_config("configs/default.yaml")

# Create data pipeline from config
pipeline = DataPipeline(
    data_dir=config.data.train_dir,
    batch_size=config.training.batch_size,
    image_size=tuple(config.data.image_size),
    augment=config.data.augmentation["enabled"]
)

# Create dataset
dataset = pipeline.create_dataset()
```

## Performance Optimizations

### tf.data Optimizations

The data pipeline includes several tf.data optimizations:

```python
def optimize_dataset(dataset: tf.data.Dataset,
                    cache: bool = True,
                    cache_filename: Optional[str] = None,
                    prefetch: bool = True,
                    num_parallel_calls: int = tf.data.AUTOTUNE) -> tf.data.Dataset:
    """
    Apply tf.data performance optimizations.
    
    Args:
        dataset: Input dataset
        cache: Whether to cache dataset in memory
        cache_filename: File for disk caching
        prefetch: Whether to prefetch batches
        num_parallel_calls: Parallel processing calls
        
    Returns:
        tf.data.Dataset: Optimized dataset
    """
    # Apply caching
    if cache:
        if cache_filename:
            dataset = dataset.cache(cache_filename)
        else:
            dataset = dataset.cache()
    
    # Apply prefetching
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
```

### Memory-Mapped Loading

For very large datasets:

```python
class MemoryMappedDataset:
    """Memory-mapped dataset for efficient large-scale data loading."""
    
    def __init__(self, 
                 data_file: str,
                 image_shape: Tuple[int, int, int],
                 num_samples: int):
        """
        Initialize memory-mapped dataset.
        
        Args:
            data_file: Path to memory-mapped data file
            image_shape: Shape of individual images
            num_samples: Total number of samples
        """
        self.mmap = np.memmap(data_file, dtype=np.float32, mode='r',
                             shape=(num_samples, *image_shape))
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """Get item by index."""
        return self.mmap[idx]
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.mmap)
```

## Error Handling and Validation

### Data Validation

```python
def validate_data_directory(data_dir: str) -> Dict[str, Any]:
    """
    Validate data directory structure.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dict with validation results:
        - is_valid: Boolean indicating if structure is valid
        - errors: List of validation errors
        - class_counts: Images per class
        - total_images: Total number of images
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'class_counts': {},
        'total_images': 0
    }
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check for class subdirectories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if len(class_dirs) == 0:
        validation_result['errors'].append("No class subdirectories found")
        validation_result['is_valid'] = False
        return validation_result
    
    # Count images in each class
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        count = len(image_files)
        validation_result['class_counts'][class_name] = count
        validation_result['total_images'] += count
        
        if count == 0:
            validation_result['errors'].append(f"No images found in {class_name}")
            validation_result['is_valid'] = False
    
    return validation_result
```

### Error Recovery

```python
def robust_image_loading(image_path: str,
                        target_size: Tuple[int, int],
                        default_image: Optional[np.ndarray] = None) -> tf.Tensor:
    """
    Robust image loading with error recovery.
    
    Args:
        image_path: Path to image file
        target_size: Target image size
        default_image: Default image to use if loading fails
        
    Returns:
        tf.Tensor: Loaded image or default image
    """
    try:
        return load_and_preprocess_image(image_path, target_size)
    except Exception as e:
        print(f"Warning: Failed to load {image_path}: {e}")
        
        if default_image is not None:
            return tf.convert_to_tensor(default_image, dtype=tf.float32)
        else:
            # Create a blank image as fallback
            blank_image = np.zeros((*target_size, 3), dtype=np.float32)
            return tf.convert_to_tensor(blank_image)
```

## Usage Examples

### Basic Usage

```python
# Create simple data pipeline
dataset = create_data_pipeline(
    data_dir="data/chest_xray/train",
    batch_size=32,
    image_size=(128, 128),
    augment=True
)

# Use with model training
for images, labels in dataset:
    # Train model on batch
    pass
```

### Advanced Configuration

```python
# Custom augmentation configuration
augmentation_config = {
    "enabled": True,
    "rotation_range": 30,
    "zoom_range": 0.25,
    "brightness_range": [0.7, 1.3],
    "horizontal_flip": True,
    "mixup": {"enabled": True, "alpha": 0.4}
}

# Create pipeline with custom augmentation
pipeline = DataPipeline(
    data_dir="data/chest_xray/train",
    batch_size=64,
    image_size=(224, 224),
    augment=True
)

# Apply custom augmentation settings
dataset = pipeline.create_dataset()
```

### Performance Monitoring

```python
import time

def benchmark_pipeline(dataset: tf.data.Dataset, num_batches: int = 100):
    """Benchmark data pipeline performance."""
    start_time = time.time()
    
    for i, (images, labels) in enumerate(dataset.take(num_batches)):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"Processed {i+1}/{num_batches} batches ({rate:.1f} batches/sec)")
    
    total_time = time.time() - start_time
    final_rate = num_batches / total_time
    print(f"Final rate: {final_rate:.2f} batches/sec")
    
    return final_rate

# Benchmark different configurations
rate = benchmark_pipeline(dataset)
```