# API Documentation

Complete API reference for the PneumoniaCNN project.

## Overview

The PneumoniaCNN API is organized into several main modules:

- **[Models](models.md)** - Neural network architectures and model classes
- **[Training](training.md)** - Training pipelines and optimization
- **[Configuration](configuration.md)** - Configuration management system
- **[Data Pipeline](data_pipeline.md)** - Data loading and preprocessing
- **[Utilities](utilities.md)** - Helper functions and validation

## Quick Start

```python
from src.config.config_loader import ConfigLoader
from src.models.cnn import PneumoniaCNN

# Load configuration
loader = ConfigLoader()
config = loader.load_config("configs/default.yaml")

# Create and train model
model = PneumoniaCNN(config=config, mode='standard')
history = model.train()

# Evaluate model
results = model.evaluate()
```

## Installation

```bash
# For Apple Silicon Macs
./install_apple_silicon_secure.sh
source venv_m1/bin/activate

# For Intel/AMD systems
pip install -r requirements.txt

# Verify installation
python test_tensorflow.py
```

## Key Concepts

### Configuration-Driven Design
All model parameters, training settings, and data pipeline options are controlled through YAML configuration files. This allows for:
- Reproducible experiments
- Easy hyperparameter tuning
- Version control of experiment settings

### Multi-Mode Architecture
The CNN implementation supports three operational modes:
- **Basic**: Minimal dependencies, hardcoded parameters
- **Standard**: Full configuration support with advanced features
- **High Performance**: All optimizations enabled (mixed precision, tf.data, etc.)

### Apple Silicon Optimization
Native support for Apple Silicon (M1/M2/M3) with Metal GPU acceleration:
- Automatic GPU detection and configuration
- Optimized TensorFlow installation
- Performance monitoring and benchmarking

## Architecture Patterns

### Modular Design
- Clear separation between models, training, and data components
- Dependency injection for testability
- Configuration-based component selection

### Error Handling
- Comprehensive input validation
- Graceful degradation for missing features
- Detailed error messages with context

### Performance Optimization
- Mixed precision training support
- tf.data pipeline optimizations
- Memory-mapped file loading
- Parallel preprocessing

## Common Use Cases

### Training a Model
```python
# Basic training
python train.py

# With custom configuration
python train.py configs/high_performance.yaml

# Configuration override
python config.py train default.yaml --override model.learning_rate=0.01
```

### Configuration Management
```python
# List available configurations
python config.py list

# Create new experiment
python config.py create my_experiment default.yaml --override training.epochs=100

# Validate configuration
python config.py validate my_config.yaml
```

### Performance Benchmarking
```python
# Benchmark configuration
python scripts/performance_benchmark.py --config configs/high_performance.yaml

# Compare configurations
python scripts/performance_benchmark.py --compare configs/default.yaml configs/high_performance.yaml
```

## Next Steps

- See [Models API](models.md) for detailed model documentation
- Check [Training API](training.md) for training pipeline details
- Review [Configuration API](configuration.md) for configuration options
- Explore [Examples](../examples/) for practical usage patterns