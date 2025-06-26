"""
Consolidated High-Performance CNN Implementation for Pneumonia Detection

This unified implementation combines the best features from all previous versions:
- Comprehensive configuration management
- High-performance optimizations (mixed precision, tf.data pipelines)
- Robust validation and error handling
- Multiple operational modes (basic, standard, high-performance)
- Backward compatibility support

The implementation automatically detects and uses the best available features
based on the system capabilities and configuration settings.
"""

import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List, Union
import logging
import time
import platform

# Core TensorFlow/Keras imports
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import validation utilities
from src.utils.validation_utils import (
    validate_input_shape, validate_learning_rate, validate_batch_size, 
    validate_epochs, validate_directory_exists, validate_dataset_structure,
    validate_model_save_path, ValidationError, FileValidationError, 
    ModelValidationError, logger as validation_logger
)

# Try to import configuration system (graceful fallback if not available)
try:
    from src.config.config_loader import get_config, ConfigManager
    from src.config.config_schema import Config, ModelConfig, TrainingConfig, DataConfig
    CONFIG_SYSTEM_AVAILABLE = True
except ImportError:
    CONFIG_SYSTEM_AVAILABLE = False
    print("Warning: Configuration system not available, using default parameters")

# Try to import high-performance components (graceful fallback)
try:
    from src.training.data_pipeline import PerformanceDataPipeline, create_performance_datasets
    from src.training.mixed_precision_trainer import MixedPrecisionTrainer, MemoryOptimizer, PerformanceMonitor
    from src.training.preprocessing_pipeline import OptimizedPreprocessor, AdvancedAugmentation
    HIGH_PERFORMANCE_AVAILABLE = True
except ImportError:
    HIGH_PERFORMANCE_AVAILABLE = False
    print("Warning: High-performance components not available, using standard implementation")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure for Apple Silicon if available
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    logger.info("🍎 Apple Silicon detected - Metal GPU acceleration will be used")
    logger.info(f"Available devices: {tf.config.list_physical_devices()}")
else:
    logger.info(f"Running on: {platform.system()} {platform.machine()}")


class PneumoniaCNN:
    """
    Unified CNN implementation for pneumonia detection.
    
    This class provides three operational modes:
    - 'basic': Simple CNN with hardcoded parameters (legacy compatibility)
    - 'standard': Configuration-based CNN with advanced features
    - 'high_performance': Optimized CNN with tf.data, mixed precision, and monitoring
    
    The mode is automatically selected based on available components and configuration,
    but can be explicitly set via the performance_mode parameter.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (128, 128, 3),
                 learning_rate: float = 0.0001,
                 config: Optional[Union[Config, Dict[str, Any]]] = None,
                 performance_mode: Optional[str] = None):
        """
        Initialize PneumoniaCNN with automatic mode detection or explicit configuration.
        
        Args:
            input_shape: Shape of input images (height, width, channels) - used in basic mode
            learning_rate: Learning rate for training - used in basic mode
            config: Configuration object or dictionary - used in standard/high-performance modes
            performance_mode: Explicit mode selection ('basic', 'standard', 'high_performance', 'auto')
                            If None or 'auto', automatically selects best available mode
            
        Raises:
            ModelValidationError: If parameters are invalid
        """
        try:
            # Determine operational mode
            self.performance_mode = self._determine_mode(performance_mode)
            logger.info(f"Operating in {self.performance_mode} mode")
            
            # Initialize based on mode
            if self.performance_mode == 'basic':
                self._init_basic_mode(input_shape, learning_rate)
            elif self.performance_mode == 'standard':
                self._init_standard_mode(config)
            elif self.performance_mode == 'high_performance':
                self._init_high_performance_mode(config)
            else:
                raise ModelValidationError(f"Invalid performance mode: {self.performance_mode}")
            
            # Common initialization
            self.model = None
            self.history = None
            
            logger.info(f"PneumoniaCNN initialized successfully in {self.performance_mode} mode")
            
        except Exception as e:
            logger.error(f"Error initializing PneumoniaCNN: {str(e)}")
            raise ModelValidationError(f"Failed to initialize PneumoniaCNN: {str(e)}")
    
    def _determine_mode(self, performance_mode: Optional[str]) -> str:
        """Automatically determine the best operational mode."""
        if performance_mode and performance_mode != 'auto':
            # Validate explicit mode selection
            valid_modes = ['basic', 'standard', 'high_performance']
            if performance_mode not in valid_modes:
                raise ValueError(f"performance_mode must be one of {valid_modes} or 'auto'")
            
            # Check if requested mode is available
            if performance_mode == 'high_performance' and not HIGH_PERFORMANCE_AVAILABLE:
                logger.warning("High-performance mode requested but components not available, falling back to standard")
                return 'standard' if CONFIG_SYSTEM_AVAILABLE else 'basic'
            if performance_mode == 'standard' and not CONFIG_SYSTEM_AVAILABLE:
                logger.warning("Standard mode requested but config system not available, falling back to basic")
                return 'basic'
            
            return performance_mode
        
        # Automatic mode selection
        if HIGH_PERFORMANCE_AVAILABLE and CONFIG_SYSTEM_AVAILABLE:
            return 'high_performance'
        elif CONFIG_SYSTEM_AVAILABLE:
            return 'standard'
        else:
            return 'basic'
    
    def _init_basic_mode(self, input_shape: Tuple[int, int, int], learning_rate: float):
        """Initialize in basic mode with minimal dependencies."""
        self.input_shape = validate_input_shape(input_shape, expected_dims=3)
        self.learning_rate = validate_learning_rate(learning_rate)
        self.config = None
        
        # Basic configuration equivalent
        self.batch_size = 32
        self.epochs = 50
        
        logger.info(f"Basic mode: input_shape={self.input_shape}, learning_rate={self.learning_rate}")
    
    def _init_standard_mode(self, config: Optional[Union[Config, Dict[str, Any]]]):
        """Initialize in standard mode with configuration system."""
        if config is None:
            config_manager = ConfigManager()
            self.config = config_manager.config
        elif isinstance(config, dict):
            self.config = Config.from_dict(config)
        else:
            self.config = config
            
        # Validate configuration
        self.config.validate()
        
        # Extract commonly used values
        self.input_shape = tuple(self.config.model.input_shape)
        self.learning_rate = self.config.model.learning_rate
        self.batch_size = self.config.training.batch_size
        self.epochs = self.config.training.epochs
        
        # Setup directories and logging
        self._setup_standard_components()
        
        logger.info(f"Standard mode: experiment={self.config.experiment_name}")
    
    def _init_high_performance_mode(self, config: Optional[Union[Config, Dict[str, Any]]]):
        """Initialize in high-performance mode with all optimizations."""
        # Initialize standard mode first
        self._init_standard_mode(config)
        
        # Initialize high-performance components
        self.mixed_precision_trainer = MixedPrecisionTrainer(self.config.to_dict())
        self.data_pipeline = PerformanceDataPipeline(self.config.to_dict())
        self.preprocessor = OptimizedPreprocessor(self.config.to_dict())
        self.performance_monitor = PerformanceMonitor()
        
        # Setup memory optimization
        MemoryOptimizer.setup_memory_growth()
        
        logger.info("High-performance mode: all optimizations enabled")
    
    def _setup_standard_components(self):
        """Setup standard mode components."""
        try:
            # Setup logging
            self._setup_logging()
            
            # Create output directories
            self._setup_directories()
            
        except Exception as e:
            logger.warning(f"Failed to setup standard components: {str(e)}")
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        if not hasattr(self, 'config') or self.config is None:
            return
            
        try:
            log_config = self.config.logging
            
            # Set log level
            numeric_level = getattr(logging, log_config.log_level.upper(), None)
            if numeric_level is not None:
                logging.getLogger().setLevel(numeric_level)
                
            # Setup file logging if requested
            if log_config.log_to_file:
                log_file = os.path.join(self.config.paths.logs_dir, log_config.log_file)
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(numeric_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                logging.getLogger().addHandler(file_handler)
                
        except Exception as e:
            logger.warning(f"Failed to setup logging: {str(e)}")
    
    def _setup_directories(self):
        """Create output directories based on configuration."""
        if not hasattr(self, 'config') or self.config is None:
            # Basic mode directory setup
            os.makedirs('models', exist_ok=True)
            os.makedirs('logs', exist_ok=True)
            return
            
        try:
            if self.config.paths.create_dirs:
                dirs_to_create = [
                    self.config.paths.models_dir,
                    self.config.paths.logs_dir,
                    self.config.paths.results_dir,
                    self.config.paths.checkpoints_dir
                ]
                
                for dir_path in dirs_to_create:
                    validate_directory_exists(dir_path, create_if_missing=True)
                    
                logger.info("Output directories created successfully")
                
        except Exception as e:
            logger.error(f"Failed to setup directories: {str(e)}")
            raise FileValidationError(f"Directory setup failed: {str(e)}")
    
    def build_model(self) -> tf.keras.Model:
        """
        Build CNN model based on operational mode and configuration.
        
        Returns:
            Built and compiled Keras model
            
        Raises:
            ModelValidationError: If model building fails
        """
        try:
            logger.info(f"Building model in {self.performance_mode} mode...")
            
            if self.performance_mode == 'basic':
                model = self._build_basic_model()
            elif self.performance_mode == 'standard':
                model = self._build_standard_model()
            elif self.performance_mode == 'high_performance':
                model = self._build_high_performance_model()
            else:
                raise ModelValidationError(f"Unknown performance mode: {self.performance_mode}")
            
            self.model = model
            logger.info(f"Model built successfully with {model.count_params():,} parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise ModelValidationError(f"Failed to build model: {str(e)}")
    
    def _build_basic_model(self) -> Sequential:
        """Build basic CNN model (legacy compatibility)."""
        model = Sequential([
            # First Block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fourth Block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def _build_standard_model(self) -> Sequential:
        """Build configurable CNN model."""
        model_config = self.config.model
        
        # Build model layers based on configuration
        layers = []
        
        # Input layer is handled by first Conv2D layer
        current_filters = model_config.filters_base
        
        # Build convolutional blocks
        for block_idx in range(model_config.depth):
            # First block includes input shape
            if block_idx == 0:
                layers.extend([
                    Conv2D(current_filters, (3, 3), activation='relu', 
                          padding='same', input_shape=model_config.input_shape),
                    BatchNormalization(),
                    Conv2D(current_filters, (3, 3), activation='relu', padding='same'),
                    BatchNormalization(),
                    MaxPooling2D(pool_size=(2, 2)),
                    Dropout(model_config.dropout_rate)
                ])
            else:
                layers.extend([
                    Conv2D(current_filters, (3, 3), activation='relu', padding='same'),
                    BatchNormalization(),
                    Conv2D(current_filters, (3, 3), activation='relu', padding='same'),
                    BatchNormalization(),
                    MaxPooling2D(pool_size=(2, 2)),
                    Dropout(model_config.dropout_rate)
                ])
            
            # Double filters for next block (up to a maximum)
            current_filters = min(current_filters * 2, 512)
        
        # Add classifier head
        layers.extend([
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(model_config.dense_dropout_rate),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(model_config.dense_dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        # Create model
        model = Sequential(layers)
        
        # Compile model
        optimizer = self._create_optimizer()
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def _build_high_performance_model(self) -> tf.keras.Model:
        """Build optimized CNN model with performance enhancements."""
        model_config = self.config.model
        input_shape = tuple(model_config.input_shape)
        
        # Create model with optimized layers
        model = tf.keras.Sequential([
            # Input layer
            Input(shape=input_shape),
            
            # First block - optimized convolutions
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fourth block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Global average pooling for efficiency
            GlobalAveragePooling2D(),
            Dropout(0.5),
            
            # Dense layers
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output layer - force float32 for mixed precision
            Dense(1, activation='sigmoid', dtype='float32')
        ])
        
        # Create optimizer
        optimizer = self.mixed_precision_trainer.create_optimizer()
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        if self.performance_mode == 'basic':
            return Adam(learning_rate=self.learning_rate)
        
        training_config = self.config.training
        
        optimizer_params = {
            'learning_rate': self.config.model.learning_rate,
            **training_config.optimizer_params
        }
        
        if training_config.optimizer.lower() == 'adam':
            return Adam(**optimizer_params)
        elif training_config.optimizer.lower() == 'sgd':
            return SGD(**optimizer_params)
        elif training_config.optimizer.lower() == 'rmsprop':
            return RMSprop(**optimizer_params)
        else:
            logger.warning(f"Unknown optimizer {training_config.optimizer}, using Adam")
            return Adam(**optimizer_params)
    
    def create_data_generators(self, 
                             train_dir: Optional[str] = None, 
                             test_dir: Optional[str] = None, 
                             batch_size: Optional[int] = None):
        """
        Create data generators based on operational mode.
        
        Args:
            train_dir: Path to training data directory (overrides config)
            test_dir: Path to test data directory (overrides config)  
            batch_size: Batch size for training (overrides config)
            
        Returns:
            Tuple of data generators or datasets based on mode
        """
        try:
            if self.performance_mode == 'high_performance':
                return self._create_high_performance_datasets()
            else:
                return self._create_standard_generators(train_dir, test_dir, batch_size)
                
        except Exception as e:
            logger.error(f"Error creating data generators: {str(e)}")
            raise FileValidationError(f"Failed to create data generators: {str(e)}")
    
    def _create_standard_generators(self, train_dir: Optional[str], test_dir: Optional[str], batch_size: Optional[int]):
        """Create standard Keras data generators."""
        # Determine parameters based on mode
        if self.performance_mode == 'basic':
            train_dir = train_dir or 'data/chest_xray/train'
            test_dir = test_dir or 'data/chest_xray/test'
            batch_size = batch_size or self.batch_size
            expected_classes = ['NORMAL', 'PNEUMONIA']
            image_size = self.input_shape[:2]
            validation_split = 0.2
            use_augmentation = True
        else:
            # Standard mode with config
            data_config = self.config.data
            train_dir = train_dir or data_config.train_dir
            test_dir = test_dir or data_config.test_dir
            batch_size = batch_size or self.config.training.batch_size
            expected_classes = data_config.class_names
            image_size = data_config.image_size
            validation_split = self.config.training.validation_split
            use_augmentation = data_config.use_augmentation
        
        # Validate inputs
        train_dir = validate_directory_exists(train_dir)
        test_dir = validate_directory_exists(test_dir)
        batch_size = validate_batch_size(batch_size)
        
        # Validate dataset structure
        train_info = validate_dataset_structure(train_dir, expected_classes)
        test_info = validate_dataset_structure(test_dir, expected_classes)
        
        logger.info(f"Training data: {train_info}")
        logger.info(f"Test data: {test_info}")
        
        # Create data generators
        if use_augmentation and self.performance_mode == 'standard':
            # Use configuration-based augmentation
            aug_params = self.config.data.augmentation.copy()
            aug_params['rescale'] = self._get_rescale_factor()
            aug_params['validation_split'] = validation_split
            train_datagen = ImageDataGenerator(**aug_params)
        else:
            # Basic augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=validation_split
            )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        logger.info("Data generators created successfully")
        return train_generator, validation_generator, test_generator
    
    def _create_high_performance_datasets(self):
        """Create optimized tf.data datasets."""
        logger.info("Creating optimized datasets...")
        
        # Create datasets using performance pipeline
        train_ds = self.data_pipeline.create_training_dataset()
        val_ds = self.data_pipeline.create_validation_dataset()
        test_ds = self.data_pipeline.create_test_dataset()
        
        # Benchmark training dataset
        logger.info("Benchmarking training dataset performance...")
        metrics = self.data_pipeline.benchmark_dataset(train_ds, num_batches=10)
        
        # Log performance metrics
        logger.info(f"Data pipeline performance:")
        logger.info(f"  Samples/sec: {metrics.get('samples_per_second', 0):.1f}")
        logger.info(f"  Batches/sec: {metrics.get('batches_per_second', 0):.2f}")
        
        return train_ds, val_ds, test_ds
    
    def _get_rescale_factor(self) -> float:
        """Get rescaling factor based on normalization method."""
        if self.performance_mode == 'basic':
            return 1./255
            
        normalize_method = self.config.data.normalize_method
        
        if normalize_method == "rescale":
            return 1./255
        elif normalize_method in ["standardize", "imagenet"]:
            return 1./255  # Still rescale, standardization happens later
        else:
            logger.warning(f"Unknown normalization method: {normalize_method}")
            return 1./255
    
    def get_callbacks(self, model_name: str = 'pneumonia_cnn') -> List:
        """
        Create callbacks for training based on operational mode.
        
        Args:
            model_name: Name prefix for saved model files
            
        Returns:
            List of Keras callbacks
        """
        try:
            if self.performance_mode == 'basic':
                return self._get_basic_callbacks(model_name)
            elif self.performance_mode == 'standard':
                return self._get_standard_callbacks()
            elif self.performance_mode == 'high_performance':
                return self._get_high_performance_callbacks()
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error creating callbacks: {str(e)}")
            return []
    
    def _get_basic_callbacks(self, model_name: str) -> List:
        """Create basic callbacks for legacy compatibility."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                f'models/{model_name}_best_{timestamp}.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=f'logs/{model_name}_{timestamp}',
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def _get_standard_callbacks(self) -> List:
        """Create configurable callbacks."""
        callbacks = []
        
        # Model checkpointing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config.paths.model_name_template.format(
            architecture=self.config.model.architecture,
            timestamp=timestamp
        )
        
        checkpoint_path = os.path.join(
            self.config.paths.models_dir, 
            f"{model_name}_best.h5"
        )
        
        callbacks.append(ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=self.config.logging.verbose
        ))
        
        # Early stopping
        if self.config.training.use_early_stopping:
            callbacks.append(EarlyStopping(
                monitor=self.config.training.early_stopping_monitor,
                patience=self.config.training.early_stopping_patience,
                restore_best_weights=True,
                verbose=self.config.logging.verbose,
                mode=self.config.training.early_stopping_mode
            ))
        
        # Learning rate scheduling
        if self.config.training.use_lr_schedule:
            lr_params = self.config.training.lr_schedule_params
            
            if self.config.training.lr_schedule_type == "reduce_on_plateau":
                callbacks.append(ReduceLROnPlateau(
                    monitor=self.config.training.early_stopping_monitor,
                    verbose=self.config.logging.verbose,
                    **lr_params
                ))
        
        # TensorBoard logging
        if self.config.logging.use_tensorboard:
            tensorboard_log_dir = os.path.join(
                self.config.logging.tensorboard_log_dir,
                f"{model_name}_{timestamp}"
            )
            
            callbacks.append(TensorBoard(
                log_dir=tensorboard_log_dir,
                histogram_freq=self.config.logging.tensorboard_histogram_freq
            ))
        
        logger.info(f"Created {len(callbacks)} callbacks")
        return callbacks
    
    def _get_high_performance_callbacks(self) -> List:
        """Create optimized callbacks with performance monitoring."""
        # Get standard callbacks first
        callbacks = self._get_standard_callbacks()
        
        # Add high-performance specific callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced TensorBoard with profiling
        if self.config.logging.use_tensorboard:
            log_dir = os.path.join(
                self.config.paths.logs_dir,
                f"hp_cnn_{timestamp}"
            )
            # Replace basic TensorBoard with enhanced version
            callbacks = [cb for cb in callbacks if not isinstance(cb, TensorBoard)]
            callbacks.append(TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                profile_batch='100,200'  # Profile batches 100-200
            ))
        
        # Performance monitoring callback
        callbacks.append(PerformanceCallback(self.performance_monitor))
        
        return callbacks
    
    def calculate_class_weights(self, data_generator_or_dataset) -> Optional[Dict[int, float]]:
        """
        Calculate class weights to handle imbalanced dataset.
        
        Args:
            data_generator_or_dataset: Training data generator or dataset
            
        Returns:
            Dictionary mapping class indices to weights, or None if not using class weights
        """
        try:
            # Check if we should use class weights
            if self.performance_mode == 'basic':
                use_class_weights = True
                predefined_weights = None
            else:
                use_class_weights = self.config.training.use_class_weights
                predefined_weights = self.config.training.class_weights
            
            if not use_class_weights:
                return None
                
            if predefined_weights is not None:
                return predefined_weights
            
            # Auto-calculate class weights
            if self.performance_mode == 'high_performance' and hasattr(self, 'data_pipeline'):
                return self.data_pipeline.get_class_weights(data_generator_or_dataset)
            else:
                # Standard calculation for generators
                counter = {0: 0, 1: 0}
                for i in range(len(data_generator_or_dataset)):
                    _, y = data_generator_or_dataset[i]
                    for label in y:
                        counter[int(label)] += 1
                
                # Calculate weights inversely proportional to class frequencies
                total = sum(counter.values())
                class_weight = {
                    0: total / (2 * counter[0]),
                    1: total / (2 * counter[1])
                }
                
                logger.info(f"Class distribution: {counter}")
                logger.info(f"Calculated class weights: {class_weight}")
                
                return class_weight
                
        except Exception as e:
            logger.error(f"Error calculating class weights: {str(e)}")
            return None
    
    def train(self, 
              train_data, 
              validation_data, 
              epochs: Optional[int] = None) -> Any:
        """
        Train the model with mode-specific optimizations.
        
        Args:
            train_data: Training data generator or dataset
            validation_data: Validation data generator or dataset
            epochs: Number of training epochs (overrides config)
            
        Returns:
            Training history
        """
        try:
            if self.model is None:
                raise ModelValidationError("Model must be built before training")
            
            # Determine epochs
            epochs = epochs or self.epochs
            epochs = validate_epochs(epochs)
            
            logger.info(f"Starting training for {epochs} epochs in {self.performance_mode} mode")
            
            if self.performance_mode == 'high_performance':
                return self._train_high_performance(train_data, validation_data, epochs)
            else:
                return self._train_standard(train_data, validation_data, epochs)
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise ModelValidationError(f"Training failed: {str(e)}")
    
    def _train_standard(self, train_data, validation_data, epochs: int) -> Any:
        """Standard training implementation."""
        # Get training components
        callbacks = self.get_callbacks()
        class_weight = self.calculate_class_weights(train_data)
        
        # Train the model
        self.history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1 if self.performance_mode == 'basic' else self.config.logging.verbose
        )
        
        logger.info("Training completed successfully")
        return self.history
    
    def _train_high_performance(self, train_dataset, val_dataset, epochs: int) -> Dict[str, Any]:
        """High-performance training with monitoring."""
        # Setup callbacks
        callbacks = self.get_callbacks()
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_dataset)
        
        # Training loop with performance monitoring
        history = {'epoch_metrics': []}
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train one epoch
            epoch_metrics = self.mixed_precision_trainer.train_epoch(
                self.model, train_dataset, val_dataset, epoch
            )
            
            # Record performance
            self.performance_monitor.log_step_time(time.time() - start_time)
            self.performance_monitor.log_memory_usage()
            
            history['epoch_metrics'].append(epoch_metrics)
            
            # Check early stopping
            if self._should_stop_early(history, epoch):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Calculate overall performance summary
        perf_summary = self.performance_monitor.get_performance_summary()
        history['performance_summary'] = perf_summary
        
        logger.info("Training completed successfully")
        logger.info(f"Average epoch time: {perf_summary.get('avg_step_time', 0):.2f}s")
        
        self.history = history
        return history
    
    def _should_stop_early(self, history: Dict[str, Any], epoch: int) -> bool:
        """Check if training should stop early based on performance."""
        if epoch < 5:  # Don't stop too early
            return False
        
        epoch_metrics = history['epoch_metrics']
        if len(epoch_metrics) < 3:
            return False
        
        # Check if validation loss is not improving
        recent_val_losses = [m.get('val_loss', float('inf')) for m in epoch_metrics[-3:]]
        if all(recent_val_losses[i] >= recent_val_losses[i-1] for i in range(1, len(recent_val_losses))):
            return True
        
        return False
    
    def evaluate(self, test_data) -> Dict[str, float]:
        """
        Evaluate the model and generate detailed metrics.
        
        Args:
            test_data: Test data generator or dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if self.model is None:
                raise ModelValidationError("Model must be trained before evaluation")
            
            logger.info("Starting model evaluation")
            
            if self.performance_mode == 'high_performance':
                return self._evaluate_high_performance(test_data)
            else:
                return self._evaluate_standard(test_data)
                
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise ModelValidationError(f"Evaluation failed: {str(e)}")
    
    def _evaluate_standard(self, test_data) -> Dict[str, float]:
        """Standard evaluation implementation."""
        # Get predictions
        predictions = self.model.predict(test_data)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = test_data.classes
        
        # Calculate metrics
        logger.info("\n=== Model Evaluation ===")
        logger.info("\nTest Set Metrics:")
        test_metrics = self.model.evaluate(test_data, verbose=0)
        for metric_name, metric_value in zip(self.model.metrics_names, test_metrics):
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # Classification report
        logger.info("\nClassification Report:")
        class_names = ['NORMAL', 'PNEUMONIA']
        if hasattr(self, 'config') and self.config is not None:
            class_names = self.config.data.class_names
        
        report = classification_report(y_true, y_pred, target_names=class_names)
        logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, class_names)
        
        # Return metrics as dictionary
        metrics_dict = {}
        for metric_name, metric_value in zip(self.model.metrics_names, test_metrics):
            metrics_dict[metric_name] = float(metric_value)
            
        logger.info("Evaluation completed successfully")
        return metrics_dict
    
    def _evaluate_high_performance(self, test_dataset) -> Dict[str, float]:
        """High-performance evaluation with throughput metrics."""
        logger.info("Starting optimized evaluation...")
        
        # Evaluate model
        start_time = time.time()
        test_metrics = self.model.evaluate(test_dataset, verbose=1)
        eval_time = time.time() - start_time
        
        # Calculate throughput
        total_samples = sum(1 for _ in test_dataset.unbatch())
        throughput = total_samples / eval_time
        
        # Create results dictionary
        results = {}
        for metric_name, metric_value in zip(self.model.metrics_names, test_metrics):
            results[metric_name] = float(metric_value)
        
        results.update({
            'evaluation_time': eval_time,
            'throughput_samples_per_sec': throughput,
            'total_samples': total_samples
        })
        
        logger.info(f"Evaluation completed in {eval_time:.2f}s")
        logger.info(f"Throughput: {throughput:.1f} samples/sec")
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]) -> None:
        """Plot confusion matrix with error handling."""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            
            # Title based on mode
            if hasattr(self, 'config') and self.config is not None:
                title = f'Confusion Matrix - {self.config.experiment_name}'
            else:
                title = 'Confusion Matrix'
            
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save path based on mode
            if hasattr(self, 'config') and self.config is not None:
                save_path = os.path.join(
                    self.config.paths.results_dir, 
                    f'confusion_matrix_{self.config.experiment_name}.png'
                )
            else:
                save_path = 'confusion_matrix.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            plt.close()
    
    def plot_training_history(self) -> None:
        """Plot training history with error handling."""
        try:
            if self.history is None:
                raise ValidationError("No training history available")
            
            # Handle different history formats
            if isinstance(self.history, dict) and 'epoch_metrics' in self.history:
                # High-performance mode format
                self._plot_hp_training_history()
            else:
                # Standard Keras history format
                self._plot_standard_training_history()
                
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            plt.close()
    
    def _plot_standard_training_history(self):
        """Plot standard Keras training history."""
        if not hasattr(self.history, 'history') or not self.history.history:
            raise ValidationError("Training history is empty")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Title based on mode
        if hasattr(self, 'config') and self.config is not None:
            fig.suptitle(f'Training History - {self.config.experiment_name}')
        else:
            fig.suptitle('Training History')
        
        # Loss
        if 'loss' in self.history.history and 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['loss'], label='Train')
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
        
        # Accuracy
        if 'accuracy' in self.history.history and 'val_accuracy' in self.history.history:
            axes[0, 1].plot(self.history.history['accuracy'], label='Train')
            axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
        
        # AUC
        if 'auc' in self.history.history and 'val_auc' in self.history.history:
            axes[1, 0].plot(self.history.history['auc'], label='Train')
            axes[1, 0].plot(self.history.history['val_auc'], label='Validation')
            axes[1, 0].set_title('Model AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
        
        # Learning rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # Save path based on mode
        if hasattr(self, 'config') and self.config is not None:
            save_path = os.path.join(
                self.config.paths.results_dir,
                f'training_history_{self.config.experiment_name}.png'
            )
        else:
            save_path = 'training_history.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    def _plot_hp_training_history(self):
        """Plot high-performance training history."""
        epoch_metrics = self.history['epoch_metrics']
        
        if not epoch_metrics:
            raise ValidationError("No epoch metrics available")
        
        # Extract metrics for plotting
        epochs = range(1, len(epoch_metrics) + 1)
        train_loss = [m.get('loss', 0) for m in epoch_metrics]
        val_loss = [m.get('val_loss', 0) for m in epoch_metrics]
        train_acc = [m.get('accuracy', 0) for m in epoch_metrics]
        val_acc = [m.get('val_accuracy', 0) for m in epoch_metrics]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training History - {self.config.experiment_name}')
        
        # Loss
        axes[0, 0].plot(epochs, train_loss, label='Train')
        axes[0, 0].plot(epochs, val_loss, label='Validation')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(epochs, train_acc, label='Train')
        axes[0, 1].plot(epochs, val_acc, label='Validation')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Performance metrics
        if 'performance_summary' in self.history:
            perf_summary = self.history['performance_summary']
            axes[1, 0].text(0.1, 0.5, f"Avg Epoch Time: {perf_summary.get('avg_step_time', 0):.2f}s\n"
                                      f"Memory Usage: {perf_summary.get('peak_memory_mb', 0):.0f}MB",
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Performance Summary')
            axes[1, 0].axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(
            self.config.paths.results_dir,
            f'training_history_{self.config.experiment_name}.png'
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model (optional)
            
        Returns:
            Path where model was saved
        """
        try:
            if self.model is None:
                raise ModelValidationError("No model to save")
            
            if model_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if hasattr(self, 'config') and self.config is not None:
                    model_name = self.config.paths.model_name_template.format(
                        architecture=self.config.model.architecture,
                        timestamp=timestamp
                    )
                    model_path = os.path.join(
                        self.config.paths.models_dir,
                        f"{model_name}_final.h5"
                    )
                else:
                    model_path = f'models/pneumonia_cnn_final_{timestamp}.h5'
            
            # Validate and save
            model_path = validate_model_save_path(model_path)
            self.model.save(model_path)
            logger.info(f"Model saved to: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise ModelValidationError(f"Failed to save model: {str(e)}")
    
    def benchmark_performance(self, dataset, num_batches: int = 50) -> Dict[str, float]:
        """
        Comprehensive performance benchmark (high-performance mode only).
        
        Args:
            dataset: Dataset to benchmark
            num_batches: Number of batches to process
            
        Returns:
            Performance metrics
        """
        if self.performance_mode != 'high_performance':
            logger.warning("Performance benchmarking only available in high-performance mode")
            return {}
        
        try:
            logger.info(f"Running performance benchmark ({num_batches} batches)...")
            
            if self.model is None:
                raise ModelValidationError("Model must be built before benchmarking")
            
            # Warm up
            for _ in dataset.take(5):
                pass
            
            # Benchmark inference
            inference_times = []
            throughputs = []
            
            for batch_idx, (batch_x, batch_y) in enumerate(dataset.take(num_batches)):
                start_time = time.time()
                predictions = self.model(batch_x, training=False)
                inference_time = time.time() - start_time
                
                batch_size = tf.shape(batch_x)[0].numpy()
                throughput = batch_size / inference_time
                
                inference_times.append(inference_time)
                throughputs.append(throughput)
            
            # Calculate statistics
            metrics = {
                'avg_inference_time': np.mean(inference_times),
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times),
                'std_inference_time': np.std(inference_times),
                'avg_throughput': np.mean(throughputs),
                'max_throughput': np.max(throughputs),
                'total_batches': len(inference_times),
                'mixed_precision_enabled': self.mixed_precision_trainer.use_mixed_precision
            }
            
            logger.info("Performance benchmark completed:")
            logger.info(f"  Average inference time: {metrics['avg_inference_time']:.4f}s")
            logger.info(f"  Average throughput: {metrics['avg_throughput']:.1f} samples/sec")
            logger.info(f"  Max throughput: {metrics['max_throughput']:.1f} samples/sec")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during performance benchmark: {str(e)}")
            return {}


class PerformanceCallback(tf.keras.callbacks.Callback):
    """Custom callback for performance monitoring in high-performance mode."""
    
    def __init__(self, performance_monitor):
        super().__init__()
        self.performance_monitor = performance_monitor
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.performance_monitor.log_step_time(epoch_time)
            self.performance_monitor.log_memory_usage()


def main(config_path: Optional[str] = None, performance_mode: Optional[str] = None) -> None:
    """
    Main training pipeline with automatic mode detection and configuration.
    
    Args:
        config_path: Path to configuration file (optional)
        performance_mode: Explicit performance mode selection (optional)
    """
    try:
        # Load configuration if available
        config = None
        if config_path and CONFIG_SYSTEM_AVAILABLE:
            config_manager = ConfigManager()
            config = config_manager.load(config_path)
        elif CONFIG_SYSTEM_AVAILABLE:
            config = get_config()
        
        # Initialize CNN with automatic mode detection
        logger.info("Initializing Pneumonia CNN...")
        pneumonia_cnn = PneumoniaCNN(config=config, performance_mode=performance_mode)
        
        # Build model
        logger.info("Building model architecture...")
        model = pneumonia_cnn.build_model()
        model.summary()
        
        # Create data generators/datasets
        logger.info("Creating data generators...")
        if pneumonia_cnn.performance_mode == 'high_performance':
            train_data, val_data, test_data = pneumonia_cnn.create_data_generators()
        else:
            train_data, val_data, test_data = pneumonia_cnn.create_data_generators(
                train_dir='data/chest_xray/train',
                test_dir='data/chest_xray/test'
            )
        
        # Train model
        logger.info("Starting training...")
        history = pneumonia_cnn.train(train_data, val_data)
        
        # Plot training history
        logger.info("Plotting training history...")
        pneumonia_cnn.plot_training_history()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        metrics = pneumonia_cnn.evaluate(test_data)
        
        # Save final model
        model_path = pneumonia_cnn.save_model()
        
        # Performance benchmark (if available)
        if pneumonia_cnn.performance_mode == 'high_performance':
            logger.info("Running performance benchmark...")
            benchmark_results = pneumonia_cnn.benchmark_performance(test_data)
            logger.info(f"Benchmark results: {benchmark_results}")
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Final model saved to: {model_path}")
        logger.info(f"Test accuracy: {metrics.get('accuracy', 0):.4f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    
    # Allow config file and performance mode as command line arguments
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    perf_mode = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(config_file, perf_mode)