# Models API

Documentation for model architectures and classes.

## PneumoniaCNN

The main CNN class for pneumonia detection.

### Class Definition

```python
class PneumoniaCNN:
    """
    Unified CNN implementation for pneumonia detection with multiple operational modes.
    
    Supports basic, standard, and high-performance modes with automatic mode detection
    based on available dependencies and configuration.
    """
```

### Constructor

```python
def __init__(self, config=None, mode='auto', data_dir=None, model_path=None):
    """
    Initialize PneumoniaCNN model.
    
    Args:
        config (ExperimentConfig, optional): Configuration object or dict
        mode (str): Operational mode ('auto', 'basic', 'standard', 'high_performance')
        data_dir (str, optional): Path to training data directory
        model_path (str, optional): Path to existing model file
        
    Raises:
        ValueError: If invalid mode or configuration provided
        ImportError: If required dependencies not available for selected mode
    """
```

#### Parameters

- **config** (`ExperimentConfig` or `dict`, optional): Complete experiment configuration
  - If dict, will be converted to `ExperimentConfig`
  - If None, uses default configuration for the selected mode
  
- **mode** (`str`): Operational mode selection
  - `'auto'`: Automatically select best available mode
  - `'basic'`: Minimal dependencies, hardcoded parameters
  - `'standard'`: Full configuration support
  - `'high_performance'`: All optimizations enabled
  
- **data_dir** (`str`, optional): Override data directory path
- **model_path** (`str`, optional): Load existing model from file

#### Example Usage

```python
# Auto-select best mode
model = PneumoniaCNN()

# Specific mode with configuration
config = load_config("configs/high_performance.yaml")
model = PneumoniaCNN(config=config, mode='high_performance')

# Load existing model
model = PneumoniaCNN(model_path="models/best_model.h5")
```

### Methods

#### build_model()

```python
def build_model(self) -> tf.keras.Model:
    """
    Build the CNN architecture.
    
    Returns:
        tf.keras.Model: Compiled Keras model
        
    Raises:
        ValueError: If invalid architecture specified in config
    """
```

Creates the neural network architecture based on configuration:
- Input layer with specified shape
- 4 convolutional blocks with batch normalization
- Dense layers with dropout
- Output layer with sigmoid activation

#### train()

```python
def train(self, X_train=None, y_train=None, X_val=None, y_val=None) -> tf.keras.callbacks.History:
    """
    Train the model.
    
    Args:
        X_train (np.ndarray, optional): Training images
        y_train (np.ndarray, optional): Training labels  
        X_val (np.ndarray, optional): Validation images
        y_val (np.ndarray, optional): Validation labels
        
    Returns:
        tf.keras.callbacks.History: Training history
        
    Raises:
        ValueError: If data not provided and data_dir not set
        RuntimeError: If training fails
    """
```

#### evaluate()

```python
def evaluate(self, X_test=None, y_test=None) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        X_test (np.ndarray, optional): Test images
        y_test (np.ndarray, optional): Test labels
        
    Returns:
        Dict[str, float]: Evaluation metrics (accuracy, loss, auc, etc.)
    """
```

#### predict()

```python
def predict(self, X, batch_size=32) -> np.ndarray:
    """
    Make predictions on input data.
    
    Args:
        X (np.ndarray): Input images
        batch_size (int): Batch size for prediction
        
    Returns:
        np.ndarray: Prediction probabilities
    """
```

#### save_model()

```python
def save_model(self, filepath=None) -> str:
    """
    Save the trained model.
    
    Args:
        filepath (str, optional): Custom save path
        
    Returns:
        str: Path where model was saved
    """
```

#### load_model()

```python
def load_model(self, filepath) -> None:
    """
    Load a previously saved model.
    
    Args:
        filepath (str): Path to saved model file
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is corrupted
    """
```

### Properties

#### model
```python
@property
def model(self) -> tf.keras.Model:
    """The underlying Keras model instance."""
```

#### config
```python
@property  
def config(self) -> ExperimentConfig:
    """The current configuration object."""
```

#### mode
```python
@property
def mode(self) -> str:
    """The current operational mode."""
```

#### history
```python
@property
def history(self) -> tf.keras.callbacks.History:
    """Training history from last training run."""
```

## Model Architecture Details

### Standard CNN Architecture

The default CNN consists of:

1. **Input Layer**: 128×128×3 RGB images
2. **Feature Extraction**: 4 convolutional blocks
3. **Classification Head**: Dense layers with dropout
4. **Output**: Single sigmoid neuron for binary classification

### Convolutional Blocks

Each block follows this pattern:
```python
# Block structure
Conv2D(filters, 3, padding='same', activation='relu')
BatchNormalization()
Conv2D(filters, 3, padding='same', activation='relu')
BatchNormalization()
MaxPooling2D(2)
Dropout(0.25)
```

Filter progression: 32 → 64 → 128 → 256

### Classification Head

```python
GlobalAveragePooling2D()
Dense(512, activation='relu')
BatchNormalization()
Dropout(0.5)
Dense(256, activation='relu')
BatchNormalization()
Dropout(0.5)
Dense(1, activation='sigmoid')
```

## U-Net Segmentation Model

For the two-stage pipeline approach.

### Class Definition

```python
class UNetSegmentation:
    """U-Net architecture for lung segmentation in chest X-rays."""
```

### Key Features

- Encoder-decoder architecture
- Skip connections for detail preservation
- Input: 512×512×1 grayscale images
- Output: Binary lung masks

## Performance Modes

### Basic Mode
- Minimal dependencies (TensorFlow + NumPy)
- Hardcoded hyperparameters
- Simple training loop
- Best for: Quick testing, minimal environments

### Standard Mode  
- Full configuration system
- Advanced callbacks (early stopping, LR scheduling)
- Data augmentation
- Model checkpointing
- Best for: Production training, experiments

### High Performance Mode
- Mixed precision training
- tf.data optimizations
- Memory-mapped loading
- Parallel preprocessing
- Best for: Large datasets, performance-critical applications

## Model Selection Guide

| Use Case | Recommended Mode | Configuration |
|----------|------------------|---------------|
| Quick testing | Basic | Default hardcoded |
| Development | Standard | `configs/default.yaml` |
| Production training | High Performance | `configs/high_performance.yaml` |
| Research experiments | Standard | Custom configs |
| Large datasets (>100GB) | High Performance | Memory-mapped loading |

## Error Handling

The models include comprehensive error handling:

```python
try:
    model = PneumoniaCNN(config=config)
    history = model.train()
except ValidationError as e:
    print(f"Configuration error: {e}")
except ImportError as e:
    print(f"Missing dependency for selected mode: {e}")
except RuntimeError as e:
    print(f"Training failed: {e}")
```

## Best Practices

1. **Always validate configuration** before training
2. **Use appropriate mode** for your use case
3. **Monitor training** with TensorBoard
4. **Save models regularly** during long training runs
5. **Use mixed precision** for faster training on compatible GPUs
6. **Enable checkpointing** for resumable training