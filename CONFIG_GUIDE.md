# Configuration Management Guide

This guide explains how to use the new configuration management system for the Pneumonia CNN project.

## Quick Start

### 1. List Available Configurations
```bash
python config_cli.py list
```

### 2. View Configuration Details
```bash
python config_cli.py show default.yaml
```

### 3. Run Training with Configuration
```bash
# Using the new configurable CNN
python cnn_with_config.py

# Or specify a config file
python cnn_with_config.py configs/fast_experiment.yaml
```

### 4. Create Custom Experiments
```bash
# Create an experiment with custom parameters
python config_cli.py create my_experiment default.yaml \
  --override model.learning_rate=0.01 \
  --override training.epochs=20 \
  --override training.batch_size=16
```

## Configuration Files

### Available Configurations

1. **`default.yaml`** - Standard configuration with sensible defaults
2. **`fast_experiment.yaml`** - Quick training for development/testing
3. **`unet_two_stage.yaml`** - Advanced two-stage U-Net + classification

### Configuration Structure

```yaml
# Experiment metadata
experiment_name: "my_experiment"
description: "Description of the experiment"
tags: ["tag1", "tag2"]
random_seed: 42

# Model configuration
model:
  architecture: "standard"      # "standard", "unet", "two_stage"
  input_shape: [128, 128, 3]
  learning_rate: 0.0001
  # ... more model parameters

# Training configuration
training:
  batch_size: 32
  epochs: 50
  optimizer: "adam"
  use_early_stopping: true
  # ... more training parameters

# Data configuration
data:
  train_dir: "chest_xray/train"
  test_dir: "chest_xray/test"
  image_size: [128, 128]
  use_augmentation: true
  # ... more data parameters

# Paths and logging
paths:
  models_dir: "models"
  logs_dir: "logs"
  results_dir: "results"

logging:
  log_level: "INFO"
  use_tensorboard: true
  # ... more logging options
```

## CLI Commands

### Configuration Management

```bash
# List all configurations
python config_cli.py list

# Show configuration details
python config_cli.py show <config_name>

# Validate configuration
python config_cli.py validate <config_name>

# Copy configuration
python config_cli.py copy source.yaml destination.yaml
```

### Experiment Creation

```bash
# Create experiment from base config
python config_cli.py create <experiment_name> <base_config>

# Create experiment with parameter overrides
python config_cli.py create my_exp default.yaml \
  --override model.learning_rate=0.01 \
  --override training.epochs=10 \
  --override model.filters_base=16
```

### Training

```bash
# Run training with specific config
python config_cli.py train <config_name>

# Or run directly
python cnn_with_config.py [config_file]
```

## Examples

### Quick Development Cycle

```bash
# 1. Create a fast experiment for testing
python config_cli.py create dev_test fast_experiment.yaml \
  --override training.epochs=5 \
  --override model.input_shape=[64,64,3] \
  --override data.image_size=[64,64]

# 2. Run the experiment
python config_cli.py train experiment_dev_test.yaml

# 3. View results in logs/fast_experiments/
```

### Hyperparameter Tuning

```bash
# Create multiple experiments with different learning rates
python config_cli.py create lr_001 default.yaml --override model.learning_rate=0.001
python config_cli.py create lr_0001 default.yaml --override model.learning_rate=0.0001
python config_cli.py create lr_00001 default.yaml --override model.learning_rate=0.00001

# Run each experiment
python config_cli.py train experiment_lr_001.yaml
python config_cli.py train experiment_lr_0001.yaml
python config_cli.py train experiment_lr_00001.yaml
```

### Production Training

```bash
# Use the full default configuration for production training
python config_cli.py train default.yaml

# Or create a production-specific config
python config_cli.py create production default.yaml \
  --override experiment_name="production_model_v1" \
  --override training.epochs=100 \
  --override logging.use_wandb=true
```

## Advanced Features

### Mixed Precision Training
```yaml
training:
  use_mixed_precision: true  # Enable for faster training on modern GPUs
```

### Learning Rate Scheduling
```yaml
training:
  use_lr_schedule: true
  lr_schedule_type: "reduce_on_plateau"
  lr_schedule_params:
    factor: 0.5
    patience: 5
    min_lr: 0.0000001
```

### Data Augmentation
```yaml
data:
  use_augmentation: true
  augmentation:
    rotation_range: 20
    zoom_range: 0.2
    horizontal_flip: true
    # ... more augmentation options
```

### Experiment Tracking
```yaml
logging:
  use_tensorboard: true       # Always available
  use_wandb: true            # If you have Weights & Biases account
  wandb_project: "my-project"
  use_mlflow: true           # If you want MLflow tracking
```

## Benefits

### ✅ What You Gain

1. **Easy Experimentation**: Change parameters without editing code
2. **Reproducibility**: Every experiment is fully documented
3. **Organization**: All parameters in one place
4. **Comparison**: Easy to compare different configurations
5. **Safety**: Configuration validation prevents errors
6. **Flexibility**: Override any parameter for quick tests

### 🔄 Migration from Old Code

The old `cnn.py` still works, but `cnn_with_config.py` offers:

- All hardcoded values are now configurable
- Better error handling and validation
- Automatic experiment tracking
- Organized output directories
- Easy parameter sweeps

## Troubleshooting

### Common Issues

1. **Configuration not found**: Make sure the file exists in `configs/` directory
2. **Validation errors**: Check the configuration format and required parameters
3. **Import errors**: Ensure all dependencies are installed
4. **Path issues**: Use absolute paths or ensure working directory is correct

### Getting Help

```bash
# Show CLI help
python config_cli.py --help

# Show help for specific command
python config_cli.py create --help
```

## Next Steps

1. **Try the fast experiment**: `python config_cli.py train fast_experiment.yaml`
2. **Create your own experiment**: Modify parameters for your specific needs
3. **Set up experiment tracking**: Enable W&B or MLflow for advanced monitoring
4. **Automate hyperparameter tuning**: Use the CLI to create parameter sweeps