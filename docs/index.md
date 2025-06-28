# PneumoniaCNN Documentation

Welcome to the comprehensive documentation for the PneumoniaCNN project - a deep learning system for detecting pneumonia in chest X-ray images.

## What is PneumoniaCNN?

PneumoniaCNN is a production-ready deep learning framework for medical image analysis, specifically designed for pneumonia detection from chest X-ray images. It features:

- **Multiple Model Architectures**: Standard CNN, U-Net segmentation, and two-stage pipelines
- **Apple Silicon Optimization**: Native support for M1/M2/M3 Macs with Metal GPU acceleration
- **Configuration-Driven Design**: Complete control through YAML configuration files
- **Performance Optimization**: Mixed precision training, tf.data pipelines, and memory optimization
- **Production Ready**: Comprehensive testing, monitoring, and deployment capabilities

## Quick Navigation

### 🚀 Getting Started
- **[Getting Started Guide](guides/getting-started.md)** - Complete setup and first model training
- **[Basic Usage Examples](examples/basic_usage.md)** - Practical examples for common tasks
- **[Installation Guide](guides/installation.md)** - Platform-specific installation instructions

### 📚 API Documentation
- **[API Overview](api/index.md)** - Complete API reference
- **[Models API](api/models.md)** - Neural network architectures and classes
- **[Configuration API](api/configuration.md)** - Configuration system reference
- **[Data Pipeline API](api/data_pipeline.md)** - Data loading and preprocessing
- **[Training API](api/training.md)** - Training pipelines and optimization

### 🏗️ Architecture & Design
- **[Architecture Overview](architecture/overview.md)** - System design and patterns
- **[Model Architectures](architecture/models.md)** - Detailed model descriptions
- **[Data Flow](architecture/data_flow.md)** - Data processing pipelines
- **[Performance Design](architecture/performance.md)** - Optimization strategies

### 📖 User Guides
- **[Configuration Guide](guides/configuration.md)** - Advanced configuration usage
- **[Training Guide](guides/training.md)** - Comprehensive training workflows
- **[Performance Guide](guides/performance.md)** - Optimization and benchmarking
- **[Deployment Guide](guides/deployment.md)** - Production deployment strategies

### 🧪 Testing & Development
- **[Testing Guide](guides/testing.md)** - Test suite and quality assurance
- **[Development Guide](guides/development.md)** - Contributing and extending
- **[Troubleshooting](guides/troubleshooting.md)** - Common issues and solutions

### 📋 Examples & Tutorials
- **[Basic Usage](examples/basic_usage.md)** - Essential usage patterns
- **[Advanced Examples](examples/advanced.md)** - Complex workflows and customization
- **[Jupyter Notebooks](examples/notebooks/)** - Interactive tutorials
- **[Use Case Studies](examples/case_studies.md)** - Real-world applications

## Key Features

### 🎯 Model Performance
- **High Accuracy**: 85-90% accuracy with standard CNN, 94-97% with two-stage pipeline
- **Robust Training**: Class imbalance handling, data augmentation, regularization
- **Multiple Architectures**: Choose the right model for your use case

### ⚡ Performance Optimization
- **Apple Silicon Native**: M1/M2/M3 optimization with Metal GPU acceleration
- **Mixed Precision**: 2x training speedup on compatible hardware
- **tf.data Pipelines**: Optimized data loading and preprocessing
- **Memory Efficiency**: Memory-mapped loading for large datasets

### 🔧 Configuration System
- **YAML/JSON Configs**: Human-readable configuration files
- **Parameter Validation**: Comprehensive input validation
- **Experiment Tracking**: Built-in experiment management
- **Override Support**: Runtime parameter modification

### 🧪 Testing & Quality
- **Comprehensive Tests**: Unit, integration, and performance tests
- **Continuous Integration**: Automated testing and validation
- **Performance Monitoring**: Built-in benchmarking and profiling
- **Error Handling**: Graceful error handling and recovery

## Project Structure

```
PneumoniaCNN/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   ├── training/                 # Training infrastructure
│   ├── config/                   # Configuration system
│   └── utils/                    # Utilities and helpers
├── configs/                      # Configuration files
├── tests/                        # Test suite
├── docs/                         # Documentation (this site)
├── scripts/                      # Utility scripts
└── examples/                     # Example notebooks and scripts
```

## Quick Start

### 1. Installation
```bash
# For Apple Silicon Macs
./install_apple_silicon_secure.sh
source venv_m1/bin/activate

# For Intel/AMD systems
pip install -r requirements.txt
```

### 2. Download Data
Download the chest X-ray dataset and organize it as:
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3. Train Your First Model
```bash
# Quick training with defaults
python train.py

# Or with high-performance config
python train.py configs/high_performance.yaml
```

### 4. Monitor Training
```bash
tensorboard --logdir=logs
# Open http://localhost:6006
```

## Performance Benchmarks

| Configuration | Accuracy | AUC | Training Time | Speedup |
|---------------|----------|-----|---------------|---------|
| Standard CNN | 85-90% | 0.92+ | 2-3 hours | baseline |
| High-Performance | 85-90% | 0.92+ | 30-45 min | 4-6x faster |
| Two-Stage U-Net | 94-97% | 0.98+ | 5-6 hours | highest accuracy |

*Benchmarks on Apple Silicon M2 with Metal GPU acceleration*

## System Requirements

### Minimum Requirements
- **Python**: 3.8-3.11
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10+

### Recommended Setup
- **Apple Silicon Mac**: M1/M2/M3 with 16GB+ RAM
- **Intel/AMD**: Multi-core CPU with GPU support
- **Storage**: SSD with 50GB+ free space

## Community & Support

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Practical usage patterns and tutorials

### Contributing
- **Development Guide**: [guides/development.md](guides/development.md)
- **Testing**: [guides/testing.md](guides/testing.md)
- **Code Style**: Follow established patterns and conventions

### Roadmap
- [ ] Multi-GPU distributed training
- [ ] Model serving with REST API
- [ ] Advanced augmentation techniques
- [ ] MLOps integration
- [ ] Model compression and quantization

## License

This project is for educational and research purposes. See the license file for details.

---

**Need help?** Start with the [Getting Started Guide](guides/getting-started.md) or check out the [Basic Usage Examples](examples/basic_usage.md).

**Want to contribute?** Read the [Development Guide](guides/development.md) and [Testing Guide](guides/testing.md).

**Looking for advanced features?** Explore the [API Documentation](api/index.md) and [Architecture Overview](architecture/overview.md).