"""
Global pytest configuration and fixtures.
"""
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Generator, Tuple
import yaml

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create a temporary directory structure mimicking the chest X-ray dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "chest_xray"
        
        # Create directory structure
        for split in ["train", "test", "val"]:
            for class_name in ["NORMAL", "PNEUMONIA"]:
                (base_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Create dummy images
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add sample images to each directory
        for split, counts in [("train", {"NORMAL": 5, "PNEUMONIA": 8}),
                              ("test", {"NORMAL": 2, "PNEUMONIA": 3}),
                              ("val", {"NORMAL": 2, "PNEUMONIA": 2})]:
            for class_name, count in counts.items():
                for i in range(count):
                    img_path = base_path / split / class_name / f"img_{i}.jpeg"
                    tf.keras.preprocessing.image.save_img(img_path, dummy_image)
        
        yield base_path
        # Cleanup happens automatically when exiting the context


@pytest.fixture
def sample_config_dict() -> dict:
    """Provide a sample configuration dictionary."""
    return {
        "experiment_name": "test_experiment",
        "model": {
            "architecture": "standard",
            "input_shape": [128, 128, 3],
            "num_classes": 2,
            "dropout_rate": 0.5,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": ["accuracy", "auc"]
        },
        "training": {
            "batch_size": 32,
            "epochs": 2,
            "validation_split": 0.2,
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "patience": 3,
                "restore_best_weights": True
            },
            "reduce_lr": {
                "enabled": True,
                "monitor": "val_loss",
                "factor": 0.5,
                "patience": 2,
                "min_lr": 1e-6
            },
            "class_weight": "balanced",
            "use_mixed_precision": False
        },
        "data": {
            "train_dir": "chest_xray/train",
            "test_dir": "chest_xray/test",
            "val_dir": "chest_xray/val",
            "image_size": [128, 128],
            "augmentation": {
                "enabled": True,
                "rotation_range": 20,
                "width_shift_range": 0.2,
                "height_shift_range": 0.2,
                "zoom_range": 0.15,
                "horizontal_flip": True,
                "fill_mode": "nearest"
            },
            "preprocessing": {
                "normalize": True,
                "rescale": 1.0
            }
        },
        "logging": {
            "level": "INFO",
            "tensorboard": {
                "enabled": True,
                "log_dir": "logs"
            },
            "checkpoint": {
                "enabled": True,
                "save_best_only": True,
                "monitor": "val_loss"
            }
        },
        "paths": {
            "base_dir": ".",
            "models_dir": "models",
            "results_dir": "results",
            "logs_dir": "logs"
        },
        "hardware": {
            "gpu_memory_growth": True,
            "mixed_precision": False,
            "multi_gpu": False
        },
        "seed": 42
    }


@pytest.fixture
def temp_config_file(sample_config_dict) -> Generator[Path, None, None]:
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config_dict, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink()


@pytest.fixture
def sample_images() -> Tuple[np.ndarray, np.ndarray]:
    """Create sample normal and pneumonia images."""
    # Create synthetic images with different patterns
    normal_image = np.ones((128, 128, 3), dtype=np.float32) * 0.5
    # Add some circular pattern for normal
    center = 64
    radius = 30
    y, x = np.ogrid[:128, :128]
    mask = (x - center)**2 + (y - center)**2 <= radius**2
    normal_image[mask] = 0.8
    
    # Pneumonia image with cloudy pattern
    pneumonia_image = np.random.normal(0.6, 0.1, (128, 128, 3)).astype(np.float32)
    pneumonia_image = np.clip(pneumonia_image, 0, 1)
    
    return normal_image, pneumonia_image


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


@pytest.fixture(autouse=True)
def reset_keras_session():
    """Reset Keras session before each test to avoid interference."""
    tf.keras.backend.clear_session()
    yield
    tf.keras.backend.clear_session()


@pytest.fixture
def disable_gpu():
    """Temporarily disable GPU for testing."""
    original_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    yield
    os.environ['CUDA_VISIBLE_DEVICES'] = original_visible_devices


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configure TensorFlow to use less memory during tests
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add GPU marker to tests that likely need GPU
        if "gpu" in item.nodeid or "train" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        
        # Add slow marker to integration and performance tests
        if "integration" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add requires_data marker to tests that need the dataset
        if "data" in item.nodeid or "pipeline" in item.nodeid:
            item.add_marker(pytest.mark.requires_data)