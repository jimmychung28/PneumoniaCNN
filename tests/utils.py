"""
Utility functions and helpers for tests.
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil
from typing import Tuple, List, Dict, Any
import yaml
import json


def create_synthetic_dataset(
    num_samples: int = 100,
    image_size: Tuple[int, int] = (128, 128),
    num_classes: int = 2,
    channels: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic dataset for testing.
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of images (height, width)
        num_classes: Number of classes
        channels: Number of image channels
    
    Returns:
        Tuple of (images, labels)
    """
    # Generate random images
    images = np.random.rand(num_samples, *image_size, channels).astype(np.float32)
    
    # Generate labels
    if num_classes == 2:
        labels = np.random.randint(0, 2, num_samples).astype(np.float32)
    else:
        labels = np.random.randint(0, num_classes, num_samples)
        labels = tf.keras.utils.to_categorical(labels, num_classes)
    
    return images, labels


def create_test_directory_structure(
    base_path: Path,
    splits: List[str] = ["train", "test", "val"],
    classes: List[str] = ["NORMAL", "PNEUMONIA"],
    images_per_class: Dict[str, Dict[str, int]] = None
) -> Path:
    """Create directory structure for testing.
    
    Args:
        base_path: Base directory path
        splits: List of data splits
        classes: List of class names
        images_per_class: Dict mapping split -> class -> count
    
    Returns:
        Path to created directory
    """
    if images_per_class is None:
        images_per_class = {
            "train": {"NORMAL": 10, "PNEUMONIA": 15},
            "test": {"NORMAL": 3, "PNEUMONIA": 5},
            "val": {"NORMAL": 2, "PNEUMONIA": 3}
        }
    
    # Create directory structure
    for split in splits:
        for class_name in classes:
            class_dir = base_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dummy images
            if split in images_per_class and class_name in images_per_class[split]:
                count = images_per_class[split][class_name]
                for i in range(count):
                    # Create synthetic image
                    if class_name == "NORMAL":
                        # Normal: more uniform pattern
                        img = np.ones((256, 256, 3), dtype=np.uint8) * 128
                        # Add some circular patterns
                        center = 128
                        radius = 50
                        y, x = np.ogrid[:256, :256]
                        mask = (x - center)**2 + (y - center)**2 <= radius**2
                        img[mask] = 200
                    else:
                        # Pneumonia: more noisy pattern
                        img = np.random.randint(100, 180, (256, 256, 3), dtype=np.uint8)
                    
                    # Save image
                    img_path = class_dir / f"{class_name.lower()}_{i:04d}.jpeg"
                    tf.keras.preprocessing.image.save_img(img_path, img)
    
    return base_path


def assert_model_architecture(
    model: tf.keras.Model,
    expected_layers: List[str],
    expected_params: Dict[str, Any] = None
):
    """Assert model has expected architecture.
    
    Args:
        model: Keras model to check
        expected_layers: List of expected layer types
        expected_params: Dict of expected parameters
    """
    # Check layer types
    actual_layers = [layer.__class__.__name__ for layer in model.layers]
    
    # Allow for some flexibility in layer ordering
    for expected in expected_layers:
        assert any(expected in layer for layer in actual_layers), \
            f"Expected layer type {expected} not found in model"
    
    # Check parameters if provided
    if expected_params:
        if 'input_shape' in expected_params:
            assert model.input_shape[1:] == expected_params['input_shape']
        
        if 'output_shape' in expected_params:
            assert model.output_shape[1:] == expected_params['output_shape']
        
        if 'total_params' in expected_params:
            total_params = model.count_params()
            tolerance = expected_params.get('param_tolerance', 0.1)
            expected = expected_params['total_params']
            assert abs(total_params - expected) / expected < tolerance, \
                f"Parameter count {total_params} differs from expected {expected}"


def create_mock_history() -> tf.keras.callbacks.History:
    """Create mock training history for testing."""
    history = tf.keras.callbacks.History()
    history.history = {
        'loss': [0.7, 0.5, 0.4, 0.35, 0.3],
        'accuracy': [0.5, 0.65, 0.75, 0.8, 0.85],
        'val_loss': [0.75, 0.6, 0.5, 0.48, 0.47],
        'val_accuracy': [0.48, 0.6, 0.7, 0.72, 0.73],
        'lr': [0.001, 0.001, 0.0005, 0.0005, 0.00025]
    }
    history.epoch = list(range(len(history.history['loss'])))
    return history


def compare_configs(
    config1: Dict[str, Any],
    config2: Dict[str, Any],
    ignore_keys: List[str] = None
) -> bool:
    """Compare two configuration dictionaries.
    
    Args:
        config1: First configuration
        config2: Second configuration
        ignore_keys: Keys to ignore in comparison
    
    Returns:
        True if configs are equivalent
    """
    if ignore_keys is None:
        ignore_keys = ['timestamp', 'random_seed']
    
    def clean_dict(d: dict, keys_to_ignore: list) -> dict:
        """Remove ignored keys from dictionary."""
        cleaned = {}
        for k, v in d.items():
            if k not in keys_to_ignore:
                if isinstance(v, dict):
                    cleaned[k] = clean_dict(v, keys_to_ignore)
                else:
                    cleaned[k] = v
        return cleaned
    
    clean1 = clean_dict(config1, ignore_keys)
    clean2 = clean_dict(config2, ignore_keys)
    
    return clean1 == clean2


def create_test_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Create standard callbacks for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(Path(tmpdir) / 'checkpoint.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(Path(tmpdir) / 'logs'),
                histogram_freq=0
            )
        ]
    return callbacks


def assert_training_improved(
    history: tf.keras.callbacks.History,
    metric: str = 'loss',
    min_improvement: float = 0.05
):
    """Assert that training showed improvement.
    
    Args:
        history: Training history
        metric: Metric to check
        min_improvement: Minimum expected improvement
    """
    if metric not in history.history:
        raise ValueError(f"Metric {metric} not found in history")
    
    values = history.history[metric]
    if len(values) < 2:
        raise ValueError("Need at least 2 epochs to check improvement")
    
    # For loss, improvement means decrease
    if 'loss' in metric:
        improvement = values[0] - values[-1]
    else:
        # For accuracy/other metrics, improvement means increase
        improvement = values[-1] - values[0]
    
    assert improvement >= min_improvement, \
        f"Expected {metric} to improve by at least {min_improvement}, " \
        f"but got {improvement}"


def create_config_variants(
    base_config: Dict[str, Any],
    variants: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Create multiple config variants for testing.
    
    Args:
        base_config: Base configuration
        variants: Dict of variant name -> overrides
    
    Returns:
        Dict of variant name -> complete config
    """
    import copy
    
    results = {}
    for name, overrides in variants.items():
        variant = copy.deepcopy(base_config)
        
        # Apply overrides
        for key_path, value in overrides.items():
            keys = key_path.split('.')
            target = variant
            
            # Navigate to nested key
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            # Set value
            target[keys[-1]] = value
        
        results[name] = variant
    
    return results


def save_test_results(
    results: Dict[str, Any],
    output_dir: Path,
    format: str = 'json'
):
    """Save test results for analysis.
    
    Args:
        results: Test results dictionary
        output_dir: Output directory
        format: Output format ('json' or 'yaml')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = tf.timestamp().numpy()
    filename = f"test_results_{timestamp}.{format}"
    output_path = output_dir / filename
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return output_path