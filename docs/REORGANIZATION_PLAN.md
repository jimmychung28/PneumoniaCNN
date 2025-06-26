# Project Reorganization Plan

## Proposed Directory Structure

```
PneumoniaCNN/
├── README.md                           # Main project documentation
├── CLAUDE.md                          # Claude Code assistant guidance
├── requirements.txt                   # Main dependencies
├── requirements_apple_silicon.txt     # Apple Silicon specific deps
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── models/                       # Model implementations
│   │   ├── __init__.py
│   │   ├── cnn.py                   # Unified CNN implementation
│   │   ├── unet_segmentation.py    # U-Net architecture
│   │   └── segmentation_classification_pipeline.py
│   ├── training/                     # Training-related modules
│   │   ├── __init__.py
│   │   ├── data_pipeline.py         # Data loading optimizations
│   │   ├── mixed_precision_trainer.py
│   │   ├── preprocessing_pipeline.py
│   │   └── train_two_stage_model.py
│   ├── config/                      # Configuration management
│   │   ├── __init__.py
│   │   ├── config_schema.py
│   │   ├── config_loader.py
│   │   └── config_cli.py
│   └── utils/                       # Utilities and helpers
│       ├── __init__.py
│       ├── validation_utils.py
│       └── fix_tensorflow_warnings.py
│
├── configs/                         # Configuration files
│   ├── default.yaml
│   ├── fast_experiment.yaml
│   ├── high_performance.yaml
│   └── unet_two_stage.yaml
│
├── scripts/                         # Standalone scripts
│   ├── setup/                      # Installation scripts
│   │   ├── install_apple_silicon_secure.sh
│   │   ├── install_apple_silicon.sh
│   │   ├── install_arm_homebrew.sh
│   │   ├── install_tensorflow_workaround.sh
│   │   └── check_cpu_and_install.py
│   ├── demo_two_stage.py           # Interactive demos
│   └── performance_benchmark.py    # Benchmarking tools
│
├── tests/                          # Test files
│   ├── __init__.py
│   ├── test_unified_cnn.py
│   ├── test_config_system.py
│   ├── test_consolidated_cnn.py
│   └── test_tensorflow.py
│
├── docs/                          # Documentation
│   ├── CONFIG_GUIDE.md
│   └── REORGANIZATION_PLAN.md
│
├── data/                          # Data directory
│   └── chest_xray/               # Dataset (existing)
│       ├── train/
│       └── test/
│
├── legacy/                        # Legacy files backup
│   ├── cnn_legacy.py
│   ├── cnn_with_config.py
│   └── high_performance_cnn.py
│
└── outputs/                       # Generated outputs (runtime created)
    ├── models/                   # Trained model files
    ├── logs/                     # Training logs
    ├── results/                  # Results and plots
    └── benchmark_results/        # Performance benchmarks
```

## File Categorization

### Core Models (`src/models/`)
- `cnn.py` - Main unified CNN implementation
- `unet_segmentation.py` - U-Net for segmentation
- `segmentation_classification_pipeline.py` - Two-stage pipeline

### Training Infrastructure (`src/training/`)
- `data_pipeline.py` - High-performance data loading
- `mixed_precision_trainer.py` - Mixed precision training
- `preprocessing_pipeline.py` - Image preprocessing
- `train_two_stage_model.py` - Two-stage training orchestration

### Configuration System (`src/config/`)
- `config_schema.py` - Configuration data structures
- `config_loader.py` - Configuration loading utilities
- `config_cli.py` - Command-line interface

### Utilities (`src/utils/`)
- `validation_utils.py` - Input validation and error handling
- `fix_tensorflow_warnings.py` - Platform-specific fixes

### Scripts (`scripts/`)
- `setup/` - Installation and setup scripts
- `demo_two_stage.py` - Interactive demonstrations
- `performance_benchmark.py` - Performance analysis tools

### Tests (`tests/`)
- All test files with consistent naming

### Documentation (`docs/`)
- Detailed guides and documentation

### Data (`data/`)
- Dataset storage (chest_xray remains in place)

### Legacy (`legacy/`)
- Backup of old implementations

## Benefits of This Structure

1. **Clear Separation of Concerns**: Models, training, config, utils are separated
2. **Easy Navigation**: Related files are grouped together
3. **Scalable**: Easy to add new models or training methods
4. **Standard Python Structure**: Follows Python packaging conventions
5. **Clean Imports**: Logical import paths like `from src.models import cnn`
6. **Tool-Friendly**: Works well with IDEs, linters, and packaging tools

## Migration Strategy

1. Create new directory structure
2. Move files to appropriate locations
3. Update all import statements
4. Update documentation and scripts
5. Test the new structure
6. Update CLAUDE.md with new paths

## Entry Points After Reorganization

```bash
# Main training
python -m src.models.cnn configs/high_performance.yaml

# Configuration management  
python -m src.config.config_cli list

# Performance benchmarking
python scripts/performance_benchmark.py

# Testing
python -m pytest tests/

# Demo
python scripts/demo_two_stage.py
```