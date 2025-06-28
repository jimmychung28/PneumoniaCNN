# Getting Started Guide

Complete guide for setting up and running your first pneumonia detection model.

## Quick Start (5 minutes)

### 1. Installation

#### For Apple Silicon Macs (M1/M2/M3)
```bash
# Clone the repository
git clone <repository-url>
cd PneumoniaCNN

# Run the secure installation script
./install_apple_silicon_secure.sh

# Activate the environment
source venv_m1/bin/activate
```

#### For Intel/AMD Systems
```bash
# Clone the repository
git clone <repository-url>
cd PneumoniaCNN

# Check CPU compatibility and install
python check_cpu_and_install.py

# Or install manually
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
# Test TensorFlow installation
python test_tensorflow.py

# Test the complete setup
python test_pytest_setup.py
```

### 3. Download Dataset
```bash
# Download the chest X-ray dataset from Kaggle
# Place in the following structure:
mkdir -p data/chest_xray
# data/chest_xray/train/NORMAL/
# data/chest_xray/train/PNEUMONIA/
# data/chest_xray/test/NORMAL/
# data/chest_xray/test/PNEUMONIA/
```

### 4. Run Your First Model
```bash
# Quick training with default settings
python train.py

# Or with high-performance configuration
python train.py configs/high_performance.yaml
```

### 5. Monitor Training
```bash
# Open TensorBoard in another terminal
tensorboard --logdir=logs

# View at http://localhost:6006
```

## Detailed Setup

### System Requirements

#### Minimum Requirements
- **Python**: 3.8 - 3.11 (3.12+ may have compatibility issues)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10+

#### Recommended Requirements
- **RAM**: 16GB+ for large datasets
- **GPU**: Apple Silicon GPU, NVIDIA GPU with 8GB+ VRAM
- **Storage**: SSD with 50GB+ free space
- **CPU**: Multi-core processor (8+ cores recommended)

### Environment Setup

#### Option 1: Apple Silicon (Recommended for M1/M2/M3 Macs)
```bash
# Ensure you're using native ARM64 Python
arch
# Should output: arm64

# Run the installation script
./install_apple_silicon_secure.sh

# The script will:
# 1. Create a virtual environment (venv_m1)
# 2. Install tensorflow-macos and tensorflow-metal
# 3. Install all required dependencies
# 4. Run compatibility tests

# Activate the environment
source venv_m1/bin/activate

# Verify Metal GPU support
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

#### Option 2: Intel/AMD Systems
```bash
# Check CPU capabilities
python check_cpu_and_install.py

# This will:
# 1. Detect AVX support
# 2. Install appropriate TensorFlow version
# 3. Set up the environment

# For systems without AVX support:
pip install tensorflow-cpu==2.4.0

# For systems with AVX support:
pip install tensorflow>=2.4.0

# Install remaining dependencies
pip install -r requirements.txt
```

#### Option 3: Manual Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon, use:
pip install -r requirements_apple_silicon.txt
```

### Data Setup

#### Download Dataset
The project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

1. **Download from Kaggle**:
   - Visit: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
   - Download and extract to `data/` directory

2. **Verify Structure**:
```bash
data/chest_xray/
├── train/
│   ├── NORMAL/      (1,349 images)
│   └── PNEUMONIA/   (3,883 images)
└── test/
    ├── NORMAL/      (234 images)
    └── PNEUMONIA/   (390 images)
```

3. **Check Data Integrity**:
```bash
# Count images in each directory
find data/chest_xray -name "*.jpeg" | wc -l
# Should show 5,856 total images

# Verify using the built-in checker
python -c "
from src.utils.validation_utils import validate_directory_path
validate_directory_path('data/chest_xray/train')
print('Data structure is valid!')
"
```

#### Alternative: Synthetic Data for Testing
```bash
# Generate synthetic test data
python -c "
from tests.utils import create_test_directory_structure
from pathlib import Path
create_test_directory_structure(Path('data/test_chest_xray'))
print('Synthetic test data created in data/test_chest_xray/')
"
```

## First Training Run

### Basic Training
```bash
# Activate environment
source venv_m1/bin/activate  # or venv/bin/activate

# Run with default configuration
python train.py

# Expected output:
# Mode detected: standard
# Loading configuration: configs/default.yaml
# Building model architecture: standard
# Training data found: 5,232 images
# Validation data: 1,308 images
# Starting training for 50 epochs...
```

### Monitor Progress
```bash
# In a new terminal, start TensorBoard
tensorboard --logdir=logs

# Open browser to: http://localhost:6006
# You'll see:
# - Training/validation loss curves
# - Accuracy metrics
# - Model graph visualization
```

### Training Output
After training completes, you'll find:
```
models/
├── pneumonia_cnn_best_20240101_123456.h5     # Best model checkpoint
└── pneumonia_cnn_final_20240101_123456.h5    # Final model

results/
├── confusion_matrix_20240101_123456.png      # Confusion matrix
├── training_history_20240101_123456.png      # Training curves
└── model_evaluation_20240101_123456.json     # Detailed metrics

logs/
└── 20240101_123456/                          # TensorBoard logs
```

## Configuration Basics

### View Available Configurations
```bash
# List all configurations
python config_cli.py list

# Output:
# Available configurations:
# - default.yaml: Standard configuration for general use
# - high_performance.yaml: Optimized for speed and performance  
# - fast_experiment.yaml: Quick training for development
# - unet_two_stage.yaml: Advanced two-stage pipeline
```

### Create Custom Configuration
```bash
# Create experiment with custom learning rate
python config_cli.py create my_experiment default.yaml \
  --override model.learning_rate=0.01 \
  --override training.epochs=20

# Train with custom configuration
python train.py configs/my_experiment.yaml
```

### Configuration Examples

#### Quick Development Run
```yaml
# configs/dev_quick.yaml
experiment_name: "development_quick"
training:
  epochs: 5
  batch_size: 16
data:
  augmentation:
    enabled: false
logging:
  tensorboard:
    enabled: false
```

#### High-Performance Training
```yaml
# configs/production.yaml
experiment_name: "production_training"
model:
  input_shape: [224, 224, 3]
training:
  batch_size: 64
  epochs: 100
  use_mixed_precision: true
data:
  image_size: [224, 224]
  augmentation:
    enabled: true
    rotation_range: 30
hardware:
  mixed_precision: true
  gpu_memory_growth: true
```

## Common Workflows

### Experiment Workflow
```bash
# 1. Create experiment configuration
python config_cli.py create exp_lr_study default.yaml \
  --override model.learning_rate=0.005

# 2. Validate configuration
python config_cli.py validate configs/exp_lr_study.yaml

# 3. Run training
python train.py configs/exp_lr_study.yaml

# 4. Monitor with TensorBoard
tensorboard --logdir=logs

# 5. Evaluate results
python evaluate.py models/exp_lr_study_best_*.h5
```

### Hyperparameter Tuning
```bash
# Create multiple experiments
for lr in 0.001 0.005 0.01; do
  python config_cli.py create "exp_lr_${lr}" default.yaml \
    --override model.learning_rate=$lr
done

# Run experiments
for config in configs/exp_lr_*.yaml; do
  python train.py $config
done

# Compare results
python scripts/compare_experiments.py logs/exp_lr_*
```

### Two-Stage Pipeline
```bash
# Run the advanced two-stage pipeline
python src/training/train_two_stage_model.py

# With custom parameters
python src/training/train_two_stage_model.py \
  --seg_epochs 20 \
  --class_epochs 30 \
  --skip_baseline

# Interactive demo
python scripts/demo_two_stage.py
```

## Performance Optimization

### Enable Mixed Precision (GPU)
```yaml
# In your config file
hardware:
  mixed_precision: true
training:
  use_mixed_precision: true
```

### Optimize Data Pipeline
```yaml
data:
  cache: true
  prefetch: true
  num_parallel_calls: -1  # Use all available cores
```

### Monitor Performance
```bash
# Run performance benchmark
python scripts/performance_benchmark.py \
  --config configs/high_performance.yaml \
  --report

# Compare configurations
python scripts/performance_benchmark.py \
  --compare configs/default.yaml configs/high_performance.yaml
```

## Testing Your Setup

### Run Test Suite
```bash
# Quick test to verify everything works
python run_tests.py quick

# Full test suite
python run_tests.py all

# With coverage report
python run_tests.py coverage
```

### Verify GPU Acceleration
```bash
# Check if GPU is detected
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPUs available:', tf.config.list_physical_devices('GPU'))
print('Built with CUDA:', tf.test.is_built_with_cuda())
"

# For Apple Silicon
python -c "
import tensorflow as tf
print('Metal GPU support:', len(tf.config.list_physical_devices('GPU')) > 0)
"
```

## Troubleshooting

### Common Issues

#### "No module named 'src'"
```bash
# Ensure you're in the project root directory
pwd
# Should end with PneumoniaCNN

# Check if src directory exists
ls -la src/
```

#### "TensorFlow not optimized for your CPU"
```bash
# For Apple Silicon - install tensorflow-macos
pip install tensorflow-macos tensorflow-metal

# For Intel/AMD without AVX
pip install tensorflow-cpu==2.4.0
```

#### "Out of Memory" during training
```bash
# Reduce batch size in configuration
python config_cli.py create small_batch default.yaml \
  --override training.batch_size=16

# Or enable GPU memory growth
python config_cli.py create mem_growth default.yaml \
  --override hardware.gpu_memory_growth=true
```

#### Dataset not found
```bash
# Check data directory structure
tree data/chest_xray/ -L 3

# Verify paths in configuration
python config_cli.py show default.yaml | grep dir
```

### Getting Help

#### Check Logs
```bash
# View training logs
tail -f logs/training.log

# Check TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=0
python train.py
```

#### Performance Issues
```bash
# Profile training performance
python scripts/performance_benchmark.py --config your_config.yaml

# Check system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_count()} cores')
print(f'RAM: {psutil.virtual_memory().total // (1024**3)} GB')
print(f'Available: {psutil.virtual_memory().available // (1024**3)} GB')
"
```

## Next Steps

### Advanced Features
1. **Two-Stage Pipeline**: Try the U-Net + ResNet50 approach
2. **Custom Models**: Implement your own architectures
3. **Data Augmentation**: Experiment with advanced augmentation
4. **Distributed Training**: Scale to multiple GPUs

### Integration
1. **Model Serving**: Deploy models for inference
2. **MLOps**: Set up automated training pipelines
3. **Monitoring**: Add production monitoring
4. **CI/CD**: Automate testing and deployment

### Learning Resources
- **Documentation**: Explore the `docs/` directory
- **Examples**: Check `scripts/` for example workflows
- **Tests**: Review `tests/` for usage patterns
- **Configuration**: Study example configs in `configs/`