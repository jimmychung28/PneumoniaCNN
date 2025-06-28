# Architecture Overview

High-level architecture and design patterns of the PneumoniaCNN project.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PneumoniaCNN System                          │
├─────────────────────────────────────────────────────────────────┤
│  CLI Interface                                                  │
│  ├─ train.py           ├─ config_cli.py    ├─ run_tests.py     │
│  ├─ demo_two_stage.py  ├─ performance_benchmark.py              │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer                                              │
│  ├─ Training Orchestration    ├─ Model Management               │
│  ├─ Experiment Tracking       ├─ Performance Monitoring         │
├─────────────────────────────────────────────────────────────────┤
│  Business Logic                                                 │
│  ├─ Models (CNN, U-Net)       ├─ Training Pipelines            │
│  ├─ Data Processing           ├─ Evaluation & Metrics           │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure                                                 │
│  ├─ Configuration System      ├─ Validation Framework           │
│  ├─ Data Pipeline            ├─ Logging & Monitoring            │
├─────────────────────────────────────────────────────────────────┤
│  Platform Layer                                                 │
│  ├─ TensorFlow/Keras         ├─ Apple Silicon Support           │
│  ├─ tf.data Optimization     ├─ Mixed Precision Training        │
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Configuration-Driven Design
- All parameters controlled through YAML/JSON files
- Enables reproducible experiments and easy hyperparameter tuning
- Clear separation between code and configuration

### 2. Modular Architecture
- Clear separation of concerns between components
- Each module has a single responsibility
- Dependency injection for testability

### 3. Performance-First Approach
- Multiple operational modes (basic, standard, high-performance)
- Automatic optimization selection based on available hardware
- Native Apple Silicon support with Metal GPU acceleration

### 4. Extensibility
- Plugin-based architecture for models and training strategies
- Easy to add new model architectures
- Configurable data augmentation pipelines

## Core Components

### Models Layer (`src/models/`)

```
models/
├── cnn.py                    # Main CNN implementation
├── unet_segmentation.py      # U-Net for lung segmentation  
├── segmentation_classification_pipeline.py  # Two-stage pipeline
└── base_model.py            # Abstract base model class
```

**Responsibilities:**
- Model architecture definitions
- Forward pass implementations
- Model compilation and optimization
- Inference and prediction logic

**Key Classes:**
- `PneumoniaCNN`: Main CNN with multiple operational modes
- `UNetSegmentation`: Lung segmentation model
- `SegmentationClassificationPipeline`: Two-stage approach

### Training Layer (`src/training/`)

```
training/
├── data_pipeline.py          # Data loading and preprocessing
├── mixed_precision_trainer.py # Mixed precision training
├── preprocessing_pipeline.py  # Advanced preprocessing
└── train_two_stage_model.py  # Two-stage training orchestration
```

**Responsibilities:**
- Data loading and augmentation
- Training loop management
- Performance optimizations
- Callback management

**Key Features:**
- tf.data pipeline optimizations
- Mixed precision training support
- Memory-mapped file loading
- Parallel preprocessing

### Configuration Layer (`src/config/`)

```
config/
├── config_schema.py         # Configuration data structures
├── config_loader.py         # Loading and validation utilities
└── config_cli.py           # Command-line interface
```

**Responsibilities:**
- Configuration file parsing
- Parameter validation
- Configuration merging and overrides
- CLI interface for configuration management

### Utilities Layer (`src/utils/`)

```
utils/
├── validation_utils.py      # Input validation and error handling
├── performance_utils.py     # Performance monitoring
└── fix_tensorflow_warnings.py # Platform-specific fixes
```

**Responsibilities:**
- Input validation and sanitization
- Error handling and user feedback
- Performance monitoring and profiling
- Platform-specific optimizations

## Data Flow

### Training Data Flow

```
Raw Images → Preprocessing → Augmentation → Batching → Training
     ↓              ↓             ↓           ↓          ↓
   JPEG/PNG    Resize/Normalize  Rotation   tf.data   CNN/U-Net
   Files       Color conversion  Flipping   Pipeline   Training
               Value scaling     Zooming    Caching    
```

### Prediction Data Flow

```
Input Image → Preprocessing → Model Inference → Post-processing → Results
     ↓             ↓               ↓               ↓              ↓
   Raw Image   Resize/Normalize   Forward Pass   Threshold    Probability
   File/Array  Value Scaling      Through CNN    Application   + Class
```

### Configuration Flow

```
YAML/JSON → Parsing → Validation → Object Creation → Runtime Usage
    ↓          ↓          ↓             ↓               ↓
Config File  ConfigLoader Schema Check  ExperimentConfig Model Training
Default      merge_configs Type Check   ModelConfig     Data Pipeline
Overrides    load_config   Range Check   TrainingConfig  Logging Setup
```

## Operational Modes

### Basic Mode
```
User Input → Hardcoded Config → Simple CNN → Basic Training
                    ↓               ↓            ↓
                 Fixed Params    Standard Arch  Minimal Logging
                 No Validation   No Callbacks   File Output Only
```

**Use Cases:** Quick testing, minimal environments, educational purposes

### Standard Mode
```
User Input → YAML Config → Advanced CNN → Full Training Pipeline
                ↓              ↓              ↓
            Config Validation  Callbacks    TensorBoard Logging
            Parameter Override Checkpointing Model Versioning
            Error Handling     Early Stop   Metric Tracking
```

**Use Cases:** Development, research, production training

### High-Performance Mode
```
User Input → Optimized Config → Enhanced CNN → Accelerated Training
                ↓                   ↓               ↓
            Mixed Precision     tf.data Pipeline  GPU Optimization
            Memory Mapping      Parallel Loading  Performance Monitor
            Advanced Augment    Memory Growth     Benchmark Tracking
```

**Use Cases:** Large datasets, performance-critical applications, production

## Error Handling Strategy

### Validation Pyramid
```
┌─────────────────────────────────────┐
│        User Input Validation        │  ← CLI argument checking
├─────────────────────────────────────┤
│      Configuration Validation       │  ← YAML/JSON schema validation
├─────────────────────────────────────┤
│        Data Validation             │  ← Image format, size, type checks
├─────────────────────────────────────┤
│       Model Validation             │  ← Architecture compatibility
├─────────────────────────────────────┤
│      Runtime Validation            │  ← Memory, GPU availability
└─────────────────────────────────────┘
```

### Error Recovery Mechanisms
1. **Graceful Degradation**: Fall back to compatible modes
2. **Checkpoint Recovery**: Resume from last saved state
3. **Configuration Correction**: Suggest valid parameter ranges
4. **Resource Management**: Handle memory/GPU limitations

## Performance Architecture

### Memory Management
```
┌─ Memory Pool ──────────────────────────────────────┐
│                                                    │
│  ┌─ Data Pipeline ─┐  ┌─ Model Memory ─┐           │
│  │ - Image Cache   │  │ - Weights       │           │
│  │ - Batch Buffer  │  │ - Activations   │           │
│  │ - Augmentation  │  │ - Gradients     │           │
│  └─────────────────┘  └─────────────────┘           │
│                                                    │
│  ┌─ GPU Memory ──────────────────────────────────┐ │
│  │ - Model Execution  - Mixed Precision Training │ │
│  │ - Batch Processing - Memory Growth Control    │ │
│  └─────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

### Processing Pipeline
```
CPU Thread 1: Data Loading     → Queue → GPU Processing
CPU Thread 2: Preprocessing    → Queue → Model Training
CPU Thread 3: Augmentation     → Queue → Gradient Update
CPU Thread 4: Logging/Metrics  → Queue → Checkpoint Save
```

## Platform Integration

### Apple Silicon Architecture
```
Python Application
       ↓
TensorFlow-macos
       ↓
Metal Performance Shaders
       ↓
Apple Silicon GPU (M1/M2/M3)
```

**Optimizations:**
- Native ARM64 compilation
- Metal GPU acceleration
- Unified memory architecture utilization
- Core ML integration capability

### Cross-Platform Support
```
┌─ Intel/AMD ─┐    ┌─ Apple Silicon ─┐    ┌─ Cloud/GPU ─┐
│ TensorFlow  │    │ TF-macos        │    │ TensorFlow  │
│ CPU/CUDA    │    │ Metal GPU       │    │ CUDA/TPU    │
│ AVX Support │    │ ARM64 Native    │    │ Distributed │
└─────────────┘    └─────────────────┘    └─────────────┘
```

## Testing Architecture

### Test Pyramid
```
┌─────────────────────────────────────┐
│         E2E Tests                   │  ← Full workflow validation
├─────────────────────────────────────┤
│       Integration Tests             │  ← Component interaction
├─────────────────────────────────────┤
│         Unit Tests                  │  ← Individual function testing
├─────────────────────────────────────┤
│      Performance Tests              │  ← Benchmark validation
└─────────────────────────────────────┘
```

### Test Infrastructure
- **Synthetic Data Generation**: Consistent test datasets
- **Mock Objects**: Isolated component testing
- **Performance Monitoring**: Execution time and memory tracking
- **Fixture Management**: Reusable test components

## Scalability Considerations

### Horizontal Scaling
- Multi-GPU training support
- Distributed data loading
- Parallel hyperparameter optimization
- Cloud deployment readiness

### Vertical Scaling
- Memory-efficient data pipelines
- Model compression techniques
- Gradient accumulation for large batches
- Mixed precision training

## Security Architecture

### Input Validation
- Image format verification
- Path traversal prevention
- Configuration sanitization
- Resource limit enforcement

### Model Security
- Checkpoint integrity verification
- Model versioning and provenance
- Secure model storage
- Access control mechanisms

## Future Architecture Considerations

### Extensibility Points
1. **Model Registry**: Plugin system for new architectures
2. **Data Connectors**: Support for various data sources
3. **Deployment Targets**: Multiple serving platforms
4. **Monitoring Integration**: External monitoring systems

### Planned Enhancements
1. **MLOps Integration**: Automated ML pipelines
2. **Model Serving**: REST API and gRPC endpoints
3. **Distributed Training**: Multi-node support
4. **Advanced Analytics**: Model interpretation and explainability