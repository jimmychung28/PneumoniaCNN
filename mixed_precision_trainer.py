"""
Mixed precision training implementation with advanced optimizations.
Provides significant performance improvements and memory efficiency.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Callable
import time
from contextlib import contextmanager

from validation_utils import ValidationError, ModelValidationError, logger as validation_logger

logger = logging.getLogger(__name__)


class MixedPrecisionTrainer:
    """
    Advanced mixed precision trainer with performance optimizations.
    
    Features:
    - Automatic mixed precision (AMP) training
    - Loss scaling for stable gradient computation
    - Memory optimization techniques
    - Performance monitoring and profiling
    - Advanced gradient clipping
    - Efficient checkpointing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mixed precision trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config.get('training', {})
        self.model_config = config.get('model', {})
        
        # Mixed precision settings
        self.use_mixed_precision = self.training_config.get('use_mixed_precision', False)
        self.loss_scale = self.training_config.get('loss_scale', 'dynamic')
        
        # Gradient clipping
        self.gradient_clip_norm = self.training_config.get('gradient_clip_norm')
        self.gradient_clip_value = self.training_config.get('gradient_clip_value')
        
        # Performance monitoring
        self.enable_profiling = self.training_config.get('enable_profiling', False)
        self.profile_steps = self.training_config.get('profile_steps', [100, 200])
        
        # Initialize mixed precision policy
        self._setup_mixed_precision()
        
        # Initialize loss scaling
        self.optimizer = None
        self.loss_scaler = None
        
        logger.info(f"MixedPrecisionTrainer initialized")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        logger.info(f"Loss scaling: {self.loss_scale}")
        
    def _setup_mixed_precision(self):
        """Setup mixed precision policy."""
        if self.use_mixed_precision:
            try:
                # Set mixed precision policy
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                logger.info("Mixed precision policy set to mixed_float16")
                
                # Verify GPU supports mixed precision
                if not self._check_mixed_precision_support():
                    logger.warning("GPU may not support mixed precision efficiently")
                    
            except Exception as e:
                logger.error(f"Failed to setup mixed precision: {e}")
                self.use_mixed_precision = False
        else:
            logger.info("Using full precision (float32)")
    
    def _check_mixed_precision_support(self) -> bool:
        """Check if GPU supports mixed precision."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return False
                
            for gpu in gpus:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                compute_capability = gpu_details.get('compute_capability')
                
                if compute_capability:
                    major, minor = compute_capability
                    # Tensor Cores require compute capability >= 7.0
                    if major >= 7:
                        logger.info(f"GPU {gpu.name} supports Tensor Cores (compute capability {major}.{minor})")
                        return True
                    else:
                        logger.info(f"GPU {gpu.name} has limited mixed precision support (compute capability {major}.{minor})")
                        
            return True  # Still beneficial even without Tensor Cores
            
        except Exception as e:
            logger.warning(f"Could not check mixed precision support: {e}")
            return True
    
    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        Create optimized optimizer with mixed precision support.
        
        Returns:
            Configured optimizer
        """
        try:
            # Get optimizer configuration
            optimizer_name = self.training_config.get('optimizer', 'adam').lower()
            learning_rate = self.model_config.get('learning_rate', 0.001)
            optimizer_params = self.training_config.get('optimizer_params', {})
            
            # Create base optimizer
            if optimizer_name == 'adam':
                base_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    **optimizer_params
                )
            elif optimizer_name == 'adamw':
                weight_decay = optimizer_params.pop('weight_decay', 0.01)
                base_optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    **optimizer_params
                )
            elif optimizer_name == 'sgd':
                base_optimizer = tf.keras.optimizers.SGD(
                    learning_rate=learning_rate,
                    **optimizer_params
                )
            elif optimizer_name == 'rmsprop':
                base_optimizer = tf.keras.optimizers.RMSprop(
                    learning_rate=learning_rate,
                    **optimizer_params
                )
            else:
                logger.warning(f"Unknown optimizer {optimizer_name}, using Adam")
                base_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            # Wrap with mixed precision if enabled
            if self.use_mixed_precision:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
                logger.info(f"Created mixed precision optimizer: {optimizer_name}")
            else:
                optimizer = base_optimizer
                logger.info(f"Created optimizer: {optimizer_name}")
            
            self.optimizer = optimizer
            return optimizer
            
        except Exception as e:
            logger.error(f"Error creating optimizer: {str(e)}")
            raise ModelValidationError(f"Failed to create optimizer: {str(e)}")
    
    def compute_loss(self, model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor, 
                    training: bool = False) -> tf.Tensor:
        """
        Compute loss with mixed precision support.
        
        Args:
            model: Keras model
            x: Input batch
            y: Target batch
            training: Whether in training mode
            
        Returns:
            Computed loss
        """
        try:
            # Forward pass
            predictions = model(x, training=training)
            
            # Compute loss
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            loss = loss_fn(y, predictions)
            
            # Add regularization losses
            if model.losses:
                regularization_loss = tf.add_n(model.losses)
                loss += regularization_loss
            
            # Scale loss for mixed precision
            if self.use_mixed_precision and training:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
                return scaled_loss
            
            return loss
            
        except Exception as e:
            logger.error(f"Error computing loss: {str(e)}")
            raise ModelValidationError(f"Failed to compute loss: {str(e)}")
    
    @tf.function
    def train_step(self, model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Optimized training step with mixed precision.
        
        Args:
            model: Keras model
            x: Input batch
            y: Target batch
            
        Returns:
            Dictionary of metrics
        """
        with tf.GradientTape() as tape:
            # Compute loss
            loss = self.compute_loss(model, x, y, training=True)
        
        # Compute gradients
        if self.use_mixed_precision:
            scaled_gradients = tape.gradient(loss, model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, model.trainable_variables)
        
        # Apply gradient clipping if configured
        if self.gradient_clip_norm:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)
        elif self.gradient_clip_value:
            gradients = [tf.clip_by_value(g, -self.gradient_clip_value, self.gradient_clip_value) 
                        for g in gradients]
            grad_norm = tf.linalg.global_norm(gradients)
        else:
            grad_norm = tf.linalg.global_norm(gradients)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Compute metrics
        predictions = model(x, training=False)
        accuracy = tf.keras.metrics.binary_accuracy(y, predictions)
        
        metrics = {
            'loss': loss if not self.use_mixed_precision else self.optimizer.get_unscaled_loss(loss),
            'accuracy': tf.reduce_mean(accuracy),
            'grad_norm': grad_norm
        }
        
        if self.use_mixed_precision:
            metrics['loss_scale'] = self.optimizer.loss_scale
        
        return metrics
    
    @tf.function
    def validation_step(self, model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Optimized validation step.
        
        Args:
            model: Keras model
            x: Input batch
            y: Target batch
            
        Returns:
            Dictionary of metrics
        """
        # Compute loss
        loss = self.compute_loss(model, x, y, training=False)
        
        # Compute predictions and metrics
        predictions = model(x, training=False)
        accuracy = tf.keras.metrics.binary_accuracy(y, predictions)
        
        return {
            'val_loss': loss,
            'val_accuracy': tf.reduce_mean(accuracy)
        }
    
    def train_epoch(self, model: tf.keras.Model, train_dataset: tf.data.Dataset,
                   val_dataset: tf.data.Dataset, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with performance optimization.
        
        Args:
            model: Keras model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epoch: Current epoch number
            
        Returns:
            Dictionary of epoch metrics
        """
        try:
            logger.info(f"Training epoch {epoch + 1}")
            
            # Initialize metrics
            train_metrics = {
                'loss': tf.keras.metrics.Mean(),
                'accuracy': tf.keras.metrics.Mean(),
                'grad_norm': tf.keras.metrics.Mean()
            }
            
            val_metrics = {
                'val_loss': tf.keras.metrics.Mean(),
                'val_accuracy': tf.keras.metrics.Mean()
            }
            
            if self.use_mixed_precision:
                train_metrics['loss_scale'] = tf.keras.metrics.Mean()
            
            # Training loop
            start_time = time.time()
            num_batches = 0
            
            # Setup profiling if enabled
            profiler_context = self._setup_profiling(epoch)
            
            with profiler_context:
                # Training
                for batch_idx, (x, y) in enumerate(train_dataset):
                    step_metrics = self.train_step(model, x, y)
                    
                    # Update metrics
                    for key, metric in train_metrics.items():
                        if key in step_metrics:
                            metric.update_state(step_metrics[key])
                    
                    num_batches += 1
                    
                    # Log progress
                    if batch_idx % 100 == 0:
                        current_loss = train_metrics['loss'].result().numpy()
                        current_acc = train_metrics['accuracy'].result().numpy()
                        logger.info(f"Batch {batch_idx}: loss={current_loss:.4f}, acc={current_acc:.4f}")
                
                # Validation
                for x, y in val_dataset:
                    step_metrics = self.validation_step(model, x, y)
                    
                    # Update metrics
                    for key, metric in val_metrics.items():
                        metric.update_state(step_metrics[key])
            
            # Calculate epoch metrics
            epoch_time = time.time() - start_time
            samples_per_sec = (num_batches * self.training_config.get('batch_size', 32)) / epoch_time
            
            results = {}
            for key, metric in {**train_metrics, **val_metrics}.items():
                results[key] = float(metric.result().numpy())
            
            results.update({
                'epoch_time': epoch_time,
                'samples_per_sec': samples_per_sec,
                'batches_per_sec': num_batches / epoch_time
            })
            
            # Log results
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            logger.info(f"Performance: {samples_per_sec:.1f} samples/sec")
            logger.info(f"Train loss: {results['loss']:.4f}, Train acc: {results['accuracy']:.4f}")
            logger.info(f"Val loss: {results['val_loss']:.4f}, Val acc: {results['val_accuracy']:.4f}")
            
            if self.use_mixed_precision:
                logger.info(f"Loss scale: {results.get('loss_scale', 'N/A')}")
                logger.info(f"Gradient norm: {results['grad_norm']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in training epoch: {str(e)}")
            raise ModelValidationError(f"Training epoch failed: {str(e)}")
    
    @contextmanager
    def _setup_profiling(self, epoch: int):
        """Setup TensorFlow profiler if enabled."""
        if self.enable_profiling and epoch in self.profile_steps:
            profile_dir = self.config.get('paths', {}).get('logs_dir', 'logs')
            profile_path = f"{profile_dir}/profile_epoch_{epoch}"
            
            logger.info(f"Starting profiling for epoch {epoch}")
            tf.profiler.experimental.start(profile_path)
            
            try:
                yield
            finally:
                tf.profiler.experimental.stop()
                logger.info(f"Profiling saved to {profile_path}")
        else:
            yield
    
    def optimize_model_for_inference(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Optimize model for inference performance.
        
        Args:
            model: Trained model
            
        Returns:
            Optimized model
        """
        try:
            logger.info("Optimizing model for inference...")
            
            # Convert to TensorFlow Lite if configured
            if self.training_config.get('convert_to_tflite', False):
                return self._convert_to_tflite(model)
            
            # Apply graph optimizations
            if self.training_config.get('optimize_graph', True):
                # Trace the model with concrete function
                input_spec = tf.TensorSpec(
                    shape=[None, *self.model_config['input_shape']], 
                    dtype=tf.float32
                )
                
                concrete_func = tf.function(model).get_concrete_function(input_spec)
                
                # Apply optimizations
                optimized_func = tf.function(
                    concrete_func,
                    experimental_relax_shapes=True
                )
                
                logger.info("Graph optimizations applied")
                return optimized_func
            
            return model
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
            return model
    
    def _convert_to_tflite(self, model: tf.keras.Model) -> Any:
        """Convert model to TensorFlow Lite format."""
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Enable optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Mixed precision quantization if supported
            if self.use_mixed_precision:
                converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = self.config.get('paths', {}).get('models_dir', 'models')
            tflite_file = f"{tflite_path}/model_optimized.tflite"
            
            with open(tflite_file, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TensorFlow Lite model saved: {tflite_file}")
            return tflite_model
            
        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
            raise ModelValidationError(f"Failed to convert to TFLite: {str(e)}")
    
    def create_learning_rate_schedule(self) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """
        Create advanced learning rate schedule.
        
        Returns:
            Learning rate schedule
        """
        try:
            schedule_config = self.training_config.get('lr_schedule_params', {})
            schedule_type = self.training_config.get('lr_schedule_type', 'constant')
            initial_lr = self.model_config.get('learning_rate', 0.001)
            
            if schedule_type == 'cosine':
                total_steps = self.training_config.get('epochs', 50) * schedule_config.get('steps_per_epoch', 100)
                schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=total_steps,
                    alpha=schedule_config.get('alpha', 0.0)
                )
                
            elif schedule_type == 'exponential':
                schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=schedule_config.get('decay_steps', 1000),
                    decay_rate=schedule_config.get('decay_rate', 0.96)
                )
                
            elif schedule_type == 'polynomial':
                total_steps = self.training_config.get('epochs', 50) * schedule_config.get('steps_per_epoch', 100)
                schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=total_steps,
                    end_learning_rate=schedule_config.get('end_learning_rate', 0.0001)
                )
                
            elif schedule_type == 'warmup_cosine':
                # Custom warmup + cosine schedule
                warmup_steps = schedule_config.get('warmup_steps', 1000)
                total_steps = self.training_config.get('epochs', 50) * schedule_config.get('steps_per_epoch', 100)
                
                def warmup_cosine_schedule(step):
                    if step < warmup_steps:
                        return initial_lr * step / warmup_steps
                    else:
                        progress = (step - warmup_steps) / (total_steps - warmup_steps)
                        return initial_lr * 0.5 * (1 + tf.cos(tf.constant(np.pi) * progress))
                
                schedule = tf.keras.optimizers.schedules.LambdaCallback(warmup_cosine_schedule)
                
            else:
                # Constant learning rate
                schedule = initial_lr
            
            logger.info(f"Created learning rate schedule: {schedule_type}")
            return schedule
            
        except Exception as e:
            logger.warning(f"Failed to create LR schedule: {e}, using constant LR")
            return self.model_config.get('learning_rate', 0.001)


# Memory optimization utilities
class MemoryOptimizer:
    """Utilities for memory optimization during training."""
    
    @staticmethod
    def setup_memory_growth():
        """Setup memory growth for GPUs to prevent OOM errors."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("Memory growth enabled for all GPUs")
        except RuntimeError as e:
            logger.warning(f"Could not set memory growth: {e}")
    
    @staticmethod
    def set_memory_limit(limit_mb: int = 4096):
        """Set memory limit for GPU usage."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_memory_limit(gpus[0], limit_mb)
                logger.info(f"GPU memory limit set to {limit_mb}MB")
        except RuntimeError as e:
            logger.warning(f"Could not set memory limit: {e}")
    
    @staticmethod
    def clear_session():
        """Clear TensorFlow session to free memory."""
        tf.keras.backend.clear_session()
        logger.info("TensorFlow session cleared")


# Performance monitoring
class PerformanceMonitor:
    """Monitor training performance and resource usage."""
    
    def __init__(self):
        self.step_times = []
        self.memory_usage = []
        
    def log_step_time(self, step_time: float):
        """Log step execution time."""
        self.step_times.append(step_time)
        
    def log_memory_usage(self):
        """Log current memory usage."""
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            self.memory_usage.append(memory_info.percent)
        except ImportError:
            pass
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary statistics."""
        if not self.step_times:
            return {}
            
        return {
            'avg_step_time': np.mean(self.step_times),
            'min_step_time': np.min(self.step_times),
            'max_step_time': np.max(self.step_times),
            'std_step_time': np.std(self.step_times),
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0
        }