"""
High-performance CNN implementation with all optimizations.
Integrates tf.data pipelines, mixed precision, and advanced preprocessing.
"""

import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import logging
import time

# Import our optimization modules
from data_pipeline import PerformanceDataPipeline, create_performance_datasets
from mixed_precision_trainer import MixedPrecisionTrainer, MemoryOptimizer, PerformanceMonitor
from preprocessing_pipeline import OptimizedPreprocessor, AdvancedAugmentation
from config_loader import get_config, ConfigManager
from validation_utils import ValidationError, ModelValidationError, logger as validation_logger

logger = logging.getLogger(__name__)


class HighPerformancePneumoniaCNN:
    """
    High-performance CNN implementation with all optimizations enabled.
    
    Features:
    - tf.data pipelines for efficient data loading
    - Mixed precision training for speed and memory efficiency
    - Advanced preprocessing and augmentation
    - Memory optimization and monitoring
    - Performance profiling and metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize high-performance CNN.
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        if config is None:
            config_manager = ConfigManager()
            self.config = config_manager.config.to_dict()
        else:
            self.config = config
        
        # Initialize components
        self.model = None
        self.mixed_precision_trainer = None
        self.data_pipeline = None
        self.preprocessor = None
        self.performance_monitor = PerformanceMonitor()
        
        # Setup memory optimization
        MemoryOptimizer.setup_memory_growth()
        
        # Initialize optimization components
        self._setup_components()
        
        logger.info("HighPerformancePneumoniaCNN initialized")
        
    def _setup_components(self):
        """Setup all optimization components."""
        try:
            # Mixed precision trainer
            self.mixed_precision_trainer = MixedPrecisionTrainer(self.config)
            
            # Data pipeline
            self.data_pipeline = PerformanceDataPipeline(self.config)
            
            # Preprocessor
            self.preprocessor = OptimizedPreprocessor(self.config)
            
            logger.info("All optimization components initialized")
            
        except Exception as e:
            logger.error(f"Error setting up components: {str(e)}")
            raise ModelValidationError(f"Failed to setup components: {str(e)}")
    
    def build_optimized_model(self) -> tf.keras.Model:
        """
        Build optimized CNN model with performance enhancements.
        
        Returns:
            Optimized Keras model
        """
        try:
            logger.info("Building optimized CNN model...")
            
            model_config = self.config['model']
            input_shape = tuple(model_config['input_shape'])
            
            # Create model with optimized layers
            model = tf.keras.Sequential([
                # Input layer
                tf.keras.layers.Input(shape=input_shape),
                
                # First block - optimized convolutions
                tf.keras.layers.Conv2D(
                    32, (3, 3), 
                    activation='relu', 
                    padding='same',
                    kernel_initializer='he_normal'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # Second block
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # Third block
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # Fourth block
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # Global average pooling for efficiency
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.5),
                
                # Dense layers
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                
                # Output layer
                tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')  # Force float32 for mixed precision
            ])
            
            # Create optimizer
            optimizer = self.mixed_precision_trainer.create_optimizer()
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC', 'Precision', 'Recall']
            )
            
            self.model = model
            logger.info(f"Model built successfully with {model.count_params():,} parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise ModelValidationError(f"Failed to build model: {str(e)}")
    
    def create_optimized_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create optimized datasets using tf.data pipeline.
        
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error creating datasets: {str(e)}")
            raise ValidationError(f"Failed to create datasets: {str(e)}")
    
    def train_with_optimizations(self, train_dataset: tf.data.Dataset, 
                               val_dataset: tf.data.Dataset, 
                               epochs: int) -> Dict[str, Any]:
        """
        Train model with all performance optimizations.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            
        Returns:
            Training history and performance metrics
        """
        try:
            if self.model is None:
                raise ModelValidationError("Model must be built before training")
            
            logger.info(f"Starting optimized training for {epochs} epochs")
            
            # Setup callbacks
            callbacks = self._create_optimized_callbacks()
            
            # Calculate class weights
            class_weights = self.data_pipeline.get_class_weights(train_dataset)
            
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
            
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise ModelValidationError(f"Training failed: {str(e)}")
    
    def _create_optimized_callbacks(self) -> list:
        """Create optimized callbacks for training."""
        try:
            callbacks = []
            
            # Model checkpointing
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(
                self.config['paths']['models_dir'],
                f"hp_cnn_best_{timestamp}.h5"
            )
            
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ))
            
            # Early stopping
            if self.config['training'].get('use_early_stopping', True):
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['training'].get('early_stopping_patience', 10),
                    restore_best_weights=True,
                    verbose=1
                ))
            
            # Learning rate scheduling
            if self.config['training'].get('use_lr_schedule', True):
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ))
            
            # TensorBoard
            if self.config['logging'].get('use_tensorboard', True):
                log_dir = os.path.join(
                    self.config['paths']['logs_dir'],
                    f"hp_cnn_{timestamp}"
                )
                callbacks.append(tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    profile_batch='100,200'  # Profile batches 100-200
                ))
            
            # Performance monitoring callback
            callbacks.append(PerformanceCallback(self.performance_monitor))
            
            return callbacks
            
        except Exception as e:
            logger.error(f"Error creating callbacks: {str(e)}")
            return []
    
    def _should_stop_early(self, history: Dict[str, Any], epoch: int) -> bool:
        """Check if training should stop early based on performance."""
        if epoch < 5:  # Don't stop too early
            return False
        
        epoch_metrics = history['epoch_metrics']
        if len(epoch_metrics) < 3:
            return False
        
        # Check if validation loss is not improving
        recent_val_losses = [m['val_loss'] for m in epoch_metrics[-3:]]
        if all(recent_val_losses[i] >= recent_val_losses[i-1] for i in range(1, len(recent_val_losses))):
            return True
        
        return False
    
    def evaluate_with_optimizations(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate model with performance optimizations.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation metrics
        """
        try:
            if self.model is None:
                raise ModelValidationError("Model must be trained before evaluation")
            
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
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise ModelValidationError(f"Evaluation failed: {str(e)}")
    
    def optimize_for_inference(self) -> tf.keras.Model:
        """
        Optimize model for inference deployment.
        
        Returns:
            Optimized model
        """
        try:
            logger.info("Optimizing model for inference...")
            
            optimized_model = self.mixed_precision_trainer.optimize_model_for_inference(self.model)
            
            logger.info("Model optimization completed")
            return optimized_model
            
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            return self.model
    
    def benchmark_performance(self, dataset: tf.data.Dataset, num_batches: int = 50) -> Dict[str, float]:
        """
        Comprehensive performance benchmark.
        
        Args:
            dataset: Dataset to benchmark
            num_batches: Number of batches to process
            
        Returns:
            Performance metrics
        """
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
    """Custom callback for performance monitoring."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
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


def main_high_performance(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main high-performance training pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Training results and performance metrics
    """
    try:
        # Load configuration
        if config_path:
            config_manager = ConfigManager()
            config = config_manager.load(config_path).to_dict()
        else:
            config = get_config().to_dict()
        
        logger.info("Starting high-performance training pipeline")
        logger.info(f"Experiment: {config['experiment_name']}")
        
        # Initialize high-performance CNN
        hp_cnn = HighPerformancePneumoniaCNN(config)
        
        # Build optimized model
        model = hp_cnn.build_optimized_model()
        model.summary()
        
        # Create optimized datasets
        train_ds, val_ds, test_ds = hp_cnn.create_optimized_datasets()
        
        # Train with optimizations
        epochs = config['training']['epochs']
        history = hp_cnn.train_with_optimizations(train_ds, val_ds, epochs)
        
        # Evaluate with optimizations
        eval_results = hp_cnn.evaluate_with_optimizations(test_ds)
        
        # Benchmark performance
        benchmark_results = hp_cnn.benchmark_performance(test_ds)
        
        # Optimize for inference
        optimized_model = hp_cnn.optimize_for_inference()
        
        # Compile results
        results = {
            'training_history': history,
            'evaluation_results': eval_results,
            'benchmark_results': benchmark_results,
            'performance_summary': history.get('performance_summary', {}),
            'model_parameters': model.count_params(),
            'experiment_name': config['experiment_name']
        }
        
        logger.info("High-performance training pipeline completed successfully!")
        logger.info(f"Final test accuracy: {eval_results.get('accuracy', 0):.4f}")
        logger.info(f"Average throughput: {benchmark_results.get('avg_throughput', 0):.1f} samples/sec")
        
        return results
        
    except Exception as e:
        logger.error(f"High-performance training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    
    # Allow config file as command line argument
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run high-performance training
    results = main_high_performance(config_file)
    
    # Print summary
    print("\n" + "="*50)
    print("HIGH-PERFORMANCE TRAINING COMPLETED")
    print("="*50)
    print(f"Experiment: {results['experiment_name']}")
    print(f"Model parameters: {results['model_parameters']:,}")
    print(f"Test accuracy: {results['evaluation_results'].get('accuracy', 0):.4f}")
    print(f"Throughput: {results['benchmark_results'].get('avg_throughput', 0):.1f} samples/sec")
    print("="*50)