# Configuration API

Complete reference for the configuration management system.

## Overview

The configuration system provides centralized control over all aspects of model training, data processing, and experiment management through YAML/JSON configuration files.

## Core Classes

### ExperimentConfig

Main configuration container that holds all experiment settings.

```python
@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    experiment_name: str = "pneumonia_detection"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    seed: int = 42
```

### ModelConfig

Model architecture and optimization settings.

```python
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Architecture
    architecture: str = "standard"  # standard, unet, resnet50, two_stage
    input_shape: List[int] = field(default_factory=lambda: [128, 128, 3])
    num_classes: int = 2
    
    # Model parameters
    dropout_rate: float = 0.5
    batch_norm: bool = True
    activation: str = "relu"
    
    # Optimization
    learning_rate: float = 0.001
    optimizer: str = "adam"  # adam, sgd, rmsprop
    loss: str = "binary_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "auc"])
    
    # Advanced options
    use_pretrained: bool = False
    freeze_base: bool = False
    fine_tune_layers: int = 0
```

#### Architecture Options

| Value | Description | Use Case |
|-------|-------------|----------|
| `standard` | 4-block CNN | General pneumonia detection |
| `unet` | U-Net segmentation | Lung segmentation |
| `resnet50` | ResNet50 backbone | Transfer learning |
| `two_stage` | U-Net + ResNet50 | Advanced pipeline |

#### Optimizer Configuration

```yaml
model:
  optimizer: "adam"
  learning_rate: 0.001
  
  # Adam-specific (optional)
  optimizer_params:
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 1e-7
    
  # SGD-specific (optional) 
  optimizer_params:
    momentum: 0.9
    nesterov: true
```

### TrainingConfig

Training process configuration.

```python
@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Basic training
    batch_size: int = 32
    epochs: int = 50
    validation_split: float = 0.2
    
    # Callbacks
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "monitor": "val_loss",
        "patience": 10,
        "restore_best_weights": True,
        "min_delta": 0.001
    })
    
    reduce_lr: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "monitor": "val_loss", 
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6,
        "verbose": 1
    })
    
    # Class handling
    class_weight: Union[str, Dict[int, float]] = "balanced"  # balanced, None, or custom dict
    
    # Performance
    use_mixed_precision: bool = False
    steps_per_epoch: Optional[int] = None
    validation_steps: Optional[int] = None
```

#### Callback Configuration

```yaml
training:
  early_stopping:
    enabled: true
    monitor: "val_loss"
    patience: 10
    restore_best_weights: true
    min_delta: 0.001
    
  reduce_lr:
    enabled: true
    monitor: "val_loss"
    factor: 0.5
    patience: 5
    min_lr: 0.000001
    
  # Custom callbacks
  custom_callbacks:
    - type: "ReduceLROnPlateau"
      params:
        monitor: "val_accuracy"
        factor: 0.8
```

### DataConfig

Data loading and preprocessing configuration.

```python
@dataclass  
class DataConfig:
    """Data pipeline configuration."""
    
    # Paths
    train_dir: str = "chest_xray/train"
    test_dir: str = "chest_xray/test"
    val_dir: Optional[str] = None
    
    # Image processing
    image_size: List[int] = field(default_factory=lambda: [128, 128])
    color_mode: str = "rgb"  # rgb, grayscale
    channels: int = 3
    
    # Augmentation
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "zoom_range": 0.15,
        "horizontal_flip": True,
        "vertical_flip": False,
        "fill_mode": "nearest",
        "brightness_range": [0.8, 1.2],
        "contrast_range": [0.8, 1.2]
    })
    
    # Preprocessing
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "rescale": 1.0,
        "standardize": False,
        "clip_values": None
    })
    
    # Performance
    cache: bool = True
    prefetch: bool = True
    num_parallel_calls: int = -1  # tf.data.AUTOTUNE
```

#### Augmentation Options

```yaml
data:
  augmentation:
    enabled: true
    
    # Geometric transformations
    rotation_range: 20
    width_shift_range: 0.2
    height_shift_range: 0.2
    zoom_range: 0.15
    horizontal_flip: true
    vertical_flip: false
    
    # Photometric transformations
    brightness_range: [0.8, 1.2]
    contrast_range: [0.8, 1.2]
    saturation_range: [0.9, 1.1]
    hue_range: [-0.1, 0.1]
    
    # Advanced augmentations
    mixup: 
      enabled: false
      alpha: 0.4
    cutmix:
      enabled: false
      alpha: 1.0
      beta: 1.0
```

### LoggingConfig

Experiment tracking and logging configuration.

```python
@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # TensorBoard
    tensorboard: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "log_dir": "logs",
        "histogram_freq": 1,
        "write_graph": True,
        "write_images": False,
        "update_freq": "epoch"
    })
    
    # Model checkpointing
    checkpoint: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "save_best_only": True,
        "monitor": "val_loss",
        "mode": "min",
        "save_weights_only": False,
        "save_freq": "epoch"
    })
    
    # External tracking
    wandb: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "project": "pneumonia-detection",
        "entity": None,
        "tags": []
    })
    
    mlflow: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "tracking_uri": None,
        "experiment_name": None
    })
```

## ConfigLoader

Main class for loading and managing configurations.

```python
class ConfigLoader:
    """Configuration loading and validation utilities."""
    
    def load_config(self, config_path: str) -> ExperimentConfig:
        """Load configuration from file."""
        
    def load_from_dict(self, config_dict: dict) -> ExperimentConfig:
        """Load configuration from dictionary."""
        
    def save_config(self, config: ExperimentConfig, output_path: str) -> None:
        """Save configuration to file."""
        
    def validate_config(self, config: Union[dict, ExperimentConfig]) -> bool:
        """Validate configuration."""
        
    def merge_configs(self, base: dict, override: dict) -> dict:
        """Merge two configuration dictionaries."""
```

### Loading Configurations

```python
from src.config.config_loader import ConfigLoader

loader = ConfigLoader()

# Load from file
config = loader.load_config("configs/default.yaml")

# Load from dictionary
config_dict = {"experiment_name": "test", "model": {"learning_rate": 0.01}}
config = loader.load_from_dict(config_dict)

# Validate before using
loader.validate_config(config)
```

### Configuration Merging

```python
# Base configuration
base_config = loader.load_config("configs/default.yaml")

# Override specific values
overrides = {
    "model": {"learning_rate": 0.01},
    "training": {"batch_size": 64}
}

# Merge configurations
merged = loader.merge_configs(base_config.__dict__, overrides)
final_config = loader.load_from_dict(merged)
```

## Configuration Files

### File Formats

Supported formats:
- **YAML** (recommended): `config.yaml`
- **JSON**: `config.json`

### Default Configurations

| File | Purpose | Use Case |
|------|---------|----------|
| `default.yaml` | Standard configuration | Development, testing |
| `high_performance.yaml` | Optimized for speed | Production training |
| `fast_experiment.yaml` | Quick iterations | Development, debugging |
| `unet_two_stage.yaml` | Two-stage pipeline | Advanced research |

### Configuration Hierarchy

1. **Default values** (in dataclass definitions)
2. **Base configuration file**
3. **Override configuration**
4. **Command-line arguments**
5. **Environment variables**

### Example Configuration

```yaml
experiment_name: "pneumonia_detection_v2"

model:
  architecture: "standard"
  input_shape: [224, 224, 3]
  learning_rate: 0.0001
  dropout_rate: 0.3
  
training:
  batch_size: 32
  epochs: 100
  early_stopping:
    enabled: true
    patience: 15
    
data:
  train_dir: "data/chest_xray/train"
  image_size: [224, 224]
  augmentation:
    enabled: true
    rotation_range: 25
    zoom_range: 0.2
    
logging:
  tensorboard:
    enabled: true
  wandb:
    enabled: true
    project: "pneumonia-cnn"
    
hardware:
  mixed_precision: true
  gpu_memory_growth: true
```

## Command-Line Interface

### Basic Usage

```bash
# List available configurations
python config_cli.py list

# Show configuration details
python config_cli.py show default.yaml

# Validate configuration
python config_cli.py validate my_config.yaml
```

### Creating Experiments

```bash
# Create new experiment from base
python config_cli.py create my_experiment default.yaml

# With overrides
python config_cli.py create high_lr default.yaml \
  --override model.learning_rate=0.01 \
  --override training.batch_size=64

# Multiple overrides
python config_cli.py create fast_test default.yaml \
  --override training.epochs=5 \
  --override data.augmentation.enabled=false
```

### Training with Configuration

```bash
# Train with specific config
python train.py configs/high_performance.yaml

# With runtime overrides
python train.py configs/default.yaml \
  --override model.learning_rate=0.005
```

## Validation

### Automatic Validation

All configurations are automatically validated:
- **Type checking**: Ensures correct data types
- **Range validation**: Checks value bounds
- **Dependency validation**: Verifies required fields
- **Path validation**: Checks file/directory existence

### Custom Validation

```python
from src.config.config_loader import ConfigLoader, ConfigValidationError

loader = ConfigLoader()

try:
    config = loader.load_config("my_config.yaml")
    loader.validate_config(config)
    print("Configuration is valid!")
except ConfigValidationError as e:
    print(f"Validation error: {e}")
```

## Environment Variables

Override configuration with environment variables:

```bash
# Set learning rate
export PNEUMONIA_MODEL_LEARNING_RATE=0.01

# Set batch size  
export PNEUMONIA_TRAINING_BATCH_SIZE=64

# Set data directory
export PNEUMONIA_DATA_TRAIN_DIR="/path/to/data"
```

Environment variable format: `PNEUMONIA_<SECTION>_<FIELD>`

## Best Practices

1. **Use descriptive experiment names** with timestamps
2. **Version control your configurations**
3. **Validate configurations** before long training runs
4. **Document custom configurations** with comments
5. **Use base configurations** and override specific values
6. **Keep configurations minimal** - rely on sensible defaults
7. **Test configurations** with quick experiments first