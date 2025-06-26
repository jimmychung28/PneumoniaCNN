"""
Enhanced CNN implementation using configuration management system.
This version replaces hardcoded values with configurable parameters.
"""

import os
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import logging
from typing import Optional, Tuple, Dict, Any, List

# Import configuration and validation utilities
from config_loader import get_config, ConfigManager
from config_schema import Config, ModelConfig, TrainingConfig, DataConfig
from validation_utils import (
    validate_input_shape, validate_learning_rate, validate_batch_size, 
    validate_epochs, validate_directory_exists, validate_dataset_structure,
    validate_model_save_path, ValidationError, FileValidationError, 
    ModelValidationError, logger as validation_logger
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Configure for Apple Silicon if available
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    logger.info("🍎 Apple Silicon detected - Metal GPU acceleration will be used")
    logger.info(f"Available devices: {tf.config.list_physical_devices()}")
else:
    logger.info(f"Running on: {platform.system()} {platform.machine()}")


class ConfigurablePneumoniaCNN:
    """
    Enhanced CNN implementation using configuration management.
    All parameters are configurable through config files.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize CNN with configuration.
        
        Args:
            config: Configuration object. If None, loads default config.
            
        Raises:
            ModelValidationError: If configuration is invalid
        """
        try:
            # Load configuration
            if config is None:
                config_manager = ConfigManager()
                self.config = config_manager.config
            else:
                self.config = config
                
            # Validate configuration
            self.config.validate()
            
            # Set random seed from config
            if hasattr(self.config, 'random_seed'):
                np.random.seed(self.config.random_seed)
                tf.random.set_seed(self.config.random_seed)
                
            # Initialize model components
            self.model = None
            self.history = None
            
            # Setup logging from config
            self._setup_logging()
            
            # Create output directories
            self._setup_directories()
            
            logger.info(f"ConfigurablePneumoniaCNN initialized with experiment: {self.config.experiment_name}")
            logger.info(f"Model architecture: {self.config.model.architecture}")
            logger.info(f"Input shape: {self.config.model.input_shape}")
            
        except Exception as e:
            logger.error(f"Error initializing ConfigurablePneumoniaCNN: {str(e)}")
            raise ModelValidationError(f"Failed to initialize CNN: {str(e)}")
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
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
    
    def _setup_directories(self) -> None:
        """Create output directories based on configuration."""
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
    
    def build_model(self) -> Sequential:
        """
        Build CNN model based on configuration.
        
        Returns:
            Built and compiled Keras model
            
        Raises:
            ModelValidationError: If model building fails
        """
        try:
            logger.info("Building CNN model from configuration...")
            
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
                    ])\n                else:\n                    layers.extend([\n                        Conv2D(current_filters, (3, 3), activation='relu', padding='same'),\n                        BatchNormalization(),\n                        Conv2D(current_filters, (3, 3), activation='relu', padding='same'),\n                        BatchNormalization(),\n                        MaxPooling2D(pool_size=(2, 2)),\n                        Dropout(model_config.dropout_rate)\n                    ])\n                \n                # Double filters for next block (up to a maximum)\n                current_filters = min(current_filters * 2, 512)\n            \n            # Add classifier head\n            layers.extend([\n                Flatten(),\n                Dense(512, activation='relu'),\n                BatchNormalization(),\n                Dropout(model_config.dense_dropout_rate),\n                Dense(256, activation='relu'),\n                BatchNormalization(),\n                Dropout(model_config.dense_dropout_rate),\n                Dense(1, activation='sigmoid')\n            ])\n            \n            # Create model\n            model = Sequential(layers)\n            \n            # Compile model\n            optimizer = self._create_optimizer()\n            \n            model.compile(\n                optimizer=optimizer,\n                loss='binary_crossentropy',\n                metrics=['accuracy', 'AUC', 'Precision', 'Recall']\n            )\n            \n            self.model = model\n            logger.info(f\"Model built successfully with {model.count_params():,} parameters\")\n            \n            return model\n            \n        except Exception as e:\n            logger.error(f\"Error building model: {str(e)}\")\n            raise ModelValidationError(f\"Failed to build model: {str(e)}\")\n    \n    def _create_optimizer(self):\n        \"\"\"Create optimizer based on configuration.\"\"\"\n        training_config = self.config.training\n        \n        optimizer_params = {\n            'learning_rate': self.config.model.learning_rate,\n            **training_config.optimizer_params\n        }\n        \n        if training_config.optimizer.lower() == 'adam':\n            return Adam(**optimizer_params)\n        elif training_config.optimizer.lower() == 'sgd':\n            return SGD(**optimizer_params)\n        elif training_config.optimizer.lower() == 'rmsprop':\n            return RMSprop(**optimizer_params)\n        else:\n            logger.warning(f\"Unknown optimizer {training_config.optimizer}, using Adam\")\n            return Adam(**optimizer_params)\n    \n    def create_data_generators(self):\n        \"\"\"\n        Create data generators based on configuration.\n        \n        Returns:\n            Tuple of (train_generator, validation_generator, test_generator)\n            \n        Raises:\n            FileValidationError: If data directories are invalid\n        \"\"\"\n        try:\n            data_config = self.config.data\n            training_config = self.config.training\n            \n            # Validate data directories\n            train_dir = validate_directory_exists(data_config.train_dir)\n            test_dir = validate_directory_exists(data_config.test_dir)\n            \n            # Validate dataset structure\n            train_info = validate_dataset_structure(train_dir, data_config.class_names)\n            test_info = validate_dataset_structure(test_dir, data_config.class_names)\n            \n            logger.info(f\"Training data: {train_info}\")\n            logger.info(f\"Test data: {test_info}\")\n            \n            # Create data generators\n            if data_config.use_augmentation:\n                # Create augmentation from config\n                aug_params = data_config.augmentation.copy()\n                aug_params['rescale'] = self._get_rescale_factor()\n                aug_params['validation_split'] = training_config.validation_split\n                train_datagen = ImageDataGenerator(**aug_params)\n            else:\n                train_datagen = ImageDataGenerator(\n                    rescale=self._get_rescale_factor(),\n                    validation_split=training_config.validation_split\n                )\n            \n            test_datagen = ImageDataGenerator(rescale=self._get_rescale_factor())\n            \n            # Create generators\n            train_generator = train_datagen.flow_from_directory(\n                train_dir,\n                target_size=data_config.image_size,\n                batch_size=training_config.batch_size,\n                class_mode='binary',\n                subset='training',\n                shuffle=True\n            )\n            \n            validation_generator = train_datagen.flow_from_directory(\n                train_dir,\n                target_size=data_config.image_size,\n                batch_size=training_config.batch_size,\n                class_mode='binary',\n                subset='validation',\n                shuffle=False\n            )\n            \n            test_generator = test_datagen.flow_from_directory(\n                test_dir,\n                target_size=data_config.image_size,\n                batch_size=training_config.batch_size,\n                class_mode='binary',\n                shuffle=False\n            )\n            \n            logger.info(\"Data generators created successfully\")\n            return train_generator, validation_generator, test_generator\n            \n        except Exception as e:\n            logger.error(f\"Error creating data generators: {str(e)}\")\n            raise FileValidationError(f\"Failed to create data generators: {str(e)}\")\n    \n    def _get_rescale_factor(self) -> float:\n        \"\"\"Get rescaling factor based on normalization method.\"\"\"\n        normalize_method = self.config.data.normalize_method\n        \n        if normalize_method == \"rescale\":\n            return 1./255\n        elif normalize_method in [\"standardize\", \"imagenet\"]:\n            return 1./255  # Still rescale, standardization happens later\n        else:\n            logger.warning(f\"Unknown normalization method: {normalize_method}\")\n            return 1./255\n    \n    def get_callbacks(self) -> List:\n        \"\"\"\n        Create training callbacks based on configuration.\n        \n        Returns:\n            List of Keras callbacks\n        \"\"\"\n        try:\n            callbacks = []\n            \n            # Model checkpointing\n            timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n            model_name = self.config.paths.model_name_template.format(\n                architecture=self.config.model.architecture,\n                timestamp=timestamp\n            )\n            \n            checkpoint_path = os.path.join(\n                self.config.paths.models_dir, \n                f\"{model_name}_best.h5\"\n            )\n            \n            callbacks.append(ModelCheckpoint(\n                checkpoint_path,\n                monitor='val_loss',\n                save_best_only=True,\n                mode='min',\n                verbose=self.config.logging.verbose\n            ))\n            \n            # Early stopping\n            if self.config.training.use_early_stopping:\n                callbacks.append(EarlyStopping(\n                    monitor=self.config.training.early_stopping_monitor,\n                    patience=self.config.training.early_stopping_patience,\n                    restore_best_weights=True,\n                    verbose=self.config.logging.verbose,\n                    mode=self.config.training.early_stopping_mode\n                ))\n            \n            # Learning rate scheduling\n            if self.config.training.use_lr_schedule:\n                lr_params = self.config.training.lr_schedule_params\n                \n                if self.config.training.lr_schedule_type == \"reduce_on_plateau\":\n                    callbacks.append(ReduceLROnPlateau(\n                        monitor=self.config.training.early_stopping_monitor,\n                        verbose=self.config.logging.verbose,\n                        **lr_params\n                    ))\n            \n            # TensorBoard logging\n            if self.config.logging.use_tensorboard:\n                tensorboard_log_dir = os.path.join(\n                    self.config.logging.tensorboard_log_dir,\n                    f\"{model_name}_{timestamp}\"\n                )\n                \n                callbacks.append(TensorBoard(\n                    log_dir=tensorboard_log_dir,\n                    histogram_freq=self.config.logging.tensorboard_histogram_freq\n                ))\n            \n            logger.info(f\"Created {len(callbacks)} callbacks\")\n            return callbacks\n            \n        except Exception as e:\n            logger.error(f\"Error creating callbacks: {str(e)}\")\n            raise ModelValidationError(f\"Failed to create callbacks: {str(e)}\")\n    \n    def calculate_class_weights(self, train_generator) -> Dict[int, float]:\n        \"\"\"\n        Calculate class weights based on configuration and data distribution.\n        \n        Args:\n            train_generator: Training data generator\n            \n        Returns:\n            Dictionary mapping class indices to weights\n        \"\"\"\n        try:\n            training_config = self.config.training\n            \n            if not training_config.use_class_weights:\n                return None\n                \n            if training_config.class_weights is not None:\n                return training_config.class_weights\n                \n            # Auto-calculate class weights\n            counter = {0: 0, 1: 0}\n            for i in range(len(train_generator)):\n                _, y = train_generator[i]\n                for label in y:\n                    counter[int(label)] += 1\n            \n            # Calculate weights inversely proportional to class frequencies\n            total = sum(counter.values())\n            class_weight = {\n                0: total / (2 * counter[0]),\n                1: total / (2 * counter[1])\n            }\n            \n            logger.info(f\"Class distribution: {counter}\")\n            logger.info(f\"Calculated class weights: {class_weight}\")\n            \n            return class_weight\n            \n        except Exception as e:\n            logger.error(f\"Error calculating class weights: {str(e)}\")\n            return None\n    \n    def train(self, train_generator, validation_generator) -> Any:\n        \"\"\"\n        Train the model using configuration parameters.\n        \n        Args:\n            train_generator: Training data generator\n            validation_generator: Validation data generator\n            \n        Returns:\n            Training history\n        \"\"\"\n        try:\n            if self.model is None:\n                raise ModelValidationError(\"Model must be built before training\")\n                \n            logger.info(f\"Starting training for {self.config.training.epochs} epochs\")\n            \n            # Get training components\n            callbacks = self.get_callbacks()\n            class_weight = self.calculate_class_weights(train_generator)\n            \n            # Setup mixed precision if requested\n            if self.config.training.use_mixed_precision:\n                try:\n                    from tensorflow.keras.mixed_precision import experimental as mixed_precision\n                    policy = mixed_precision.Policy('mixed_float16')\n                    mixed_precision.set_policy(policy)\n                    logger.info(\"Mixed precision training enabled\")\n                except ImportError:\n                    logger.warning(\"Mixed precision not available\")\n            \n            # Train model\n            self.history = self.model.fit(\n                train_generator,\n                epochs=self.config.training.epochs,\n                validation_data=validation_generator,\n                callbacks=callbacks,\n                class_weight=class_weight,\n                verbose=self.config.logging.verbose\n            )\n            \n            logger.info(\"Training completed successfully\")\n            return self.history\n            \n        except Exception as e:\n            logger.error(f\"Error during training: {str(e)}\")\n            raise ModelValidationError(f\"Training failed: {str(e)}\")\n    \n    def evaluate(self, test_generator) -> Dict[str, float]:\n        \"\"\"\n        Evaluate the model and generate metrics.\n        \n        Args:\n            test_generator: Test data generator\n            \n        Returns:\n            Dictionary of evaluation metrics\n        \"\"\"\n        try:\n            if self.model is None:\n                raise ModelValidationError(\"Model must be trained before evaluation\")\n                \n            logger.info(\"Starting model evaluation\")\n            \n            # Get predictions\n            predictions = self.model.predict(test_generator, verbose=0)\n            y_pred = (predictions > 0.5).astype(int).flatten()\n            y_true = test_generator.classes\n            \n            # Calculate metrics\n            test_metrics = self.model.evaluate(test_generator, verbose=0)\n            \n            logger.info(\"\\n=== Model Evaluation ===\")\n            logger.info(\"\\nTest Set Metrics:\")\n            for metric_name, metric_value in zip(self.model.metrics_names, test_metrics):\n                logger.info(f\"{metric_name}: {metric_value:.4f}\")\n            \n            # Classification report\n            logger.info(\"\\nClassification Report:\")\n            class_names = self.config.data.class_names\n            report = classification_report(y_true, y_pred, target_names=class_names)\n            logger.info(f\"\\n{report}\")\n            \n            # Confusion matrix\n            cm = confusion_matrix(y_true, y_pred)\n            self.plot_confusion_matrix(cm, class_names)\n            \n            # Return metrics as dictionary\n            metrics_dict = {}\n            for metric_name, metric_value in zip(self.model.metrics_names, test_metrics):\n                metrics_dict[metric_name] = float(metric_value)\n                \n            logger.info(\"Evaluation completed successfully\")\n            return metrics_dict\n            \n        except Exception as e:\n            logger.error(f\"Error during evaluation: {str(e)}\")\n            raise ModelValidationError(f\"Evaluation failed: {str(e)}\")\n    \n    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]) -> None:\n        \"\"\"Plot and save confusion matrix.\"\"\"\n        try:\n            plt.figure(figsize=(8, 6))\n            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n                       xticklabels=class_names, yticklabels=class_names)\n            plt.title(f'Confusion Matrix - {self.config.experiment_name}')\n            plt.ylabel('True Label')\n            plt.xlabel('Predicted Label')\n            \n            # Save to results directory\n            save_path = os.path.join(\n                self.config.paths.results_dir, \n                f'confusion_matrix_{self.config.experiment_name}.png'\n            )\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n            logger.info(f\"Confusion matrix saved to: {save_path}\")\n            \n            plt.show()\n            plt.close()\n            \n        except Exception as e:\n            logger.error(f\"Error plotting confusion matrix: {str(e)}\")\n            plt.close()\n    \n    def plot_training_history(self) -> None:\n        \"\"\"Plot and save training history.\"\"\"\n        try:\n            if self.history is None:\n                raise ValidationError(\"No training history available\")\n                \n            fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n            fig.suptitle(f'Training History - {self.config.experiment_name}')\n            \n            # Loss\n            axes[0, 0].plot(self.history.history['loss'], label='Train')\n            axes[0, 0].plot(self.history.history['val_loss'], label='Validation')\n            axes[0, 0].set_title('Model Loss')\n            axes[0, 0].set_xlabel('Epoch')\n            axes[0, 0].set_ylabel('Loss')\n            axes[0, 0].legend()\n            \n            # Accuracy\n            axes[0, 1].plot(self.history.history['accuracy'], label='Train')\n            axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation')\n            axes[0, 1].set_title('Model Accuracy')\n            axes[0, 1].set_xlabel('Epoch')\n            axes[0, 1].set_ylabel('Accuracy')\n            axes[0, 1].legend()\n            \n            # AUC\n            if 'auc' in self.history.history:\n                axes[1, 0].plot(self.history.history['auc'], label='Train')\n                axes[1, 0].plot(self.history.history['val_auc'], label='Validation')\n                axes[1, 0].set_title('Model AUC')\n                axes[1, 0].set_xlabel('Epoch')\n                axes[1, 0].set_ylabel('AUC')\n                axes[1, 0].legend()\n            \n            # Learning rate\n            if 'lr' in self.history.history:\n                axes[1, 1].plot(self.history.history['lr'])\n                axes[1, 1].set_title('Learning Rate')\n                axes[1, 1].set_xlabel('Epoch')\n                axes[1, 1].set_ylabel('Learning Rate')\n                axes[1, 1].set_yscale('log')\n            \n            plt.tight_layout()\n            \n            # Save to results directory\n            save_path = os.path.join(\n                self.config.paths.results_dir,\n                f'training_history_{self.config.experiment_name}.png'\n            )\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n            logger.info(f\"Training history saved to: {save_path}\")\n            \n            plt.show()\n            plt.close()\n            \n        except Exception as e:\n            logger.error(f\"Error plotting training history: {str(e)}\")\n            plt.close()\n    \n    def save_final_model(self) -> str:\n        \"\"\"Save the final trained model.\"\"\"\n        try:\n            if self.model is None:\n                raise ModelValidationError(\"No model to save\")\n                \n            timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n            model_name = self.config.paths.model_name_template.format(\n                architecture=self.config.model.architecture,\n                timestamp=timestamp\n            )\n            \n            model_path = os.path.join(\n                self.config.paths.models_dir,\n                f\"{model_name}_final.h5\"\n            )\n            \n            self.model.save(model_path)\n            logger.info(f\"Final model saved to: {model_path}\")\n            \n            return model_path\n            \n        except Exception as e:\n            logger.error(f\"Error saving model: {str(e)}\")\n            raise ModelValidationError(f\"Failed to save model: {str(e)}\")\n\n\ndef main_with_config(config_path: Optional[str] = None) -> None:\n    \"\"\"\n    Main training pipeline using configuration system.\n    \n    Args:\n        config_path: Path to configuration file (optional)\n    \"\"\"\n    try:\n        # Load configuration\n        if config_path:\n            config_manager = ConfigManager()\n            config = config_manager.load(config_path)\n        else:\n            config = get_config()\n            \n        logger.info(f\"Starting training pipeline: {config.experiment_name}\")\n        logger.info(f\"Description: {config.description}\")\n        logger.info(f\"Tags: {config.tags}\")\n        \n        # Initialize CNN\n        cnn = ConfigurablePneumoniaCNN(config)\n        \n        # Build model\n        model = cnn.build_model()\n        model.summary()\n        \n        # Create data generators\n        train_gen, val_gen, test_gen = cnn.create_data_generators()\n        \n        # Train model\n        history = cnn.train(train_gen, val_gen)\n        \n        # Plot training history\n        cnn.plot_training_history()\n        \n        # Evaluate model\n        metrics = cnn.evaluate(test_gen)\n        \n        # Save final model\n        model_path = cnn.save_final_model()\n        \n        logger.info(\"Training pipeline completed successfully!\")\n        logger.info(f\"Final model saved to: {model_path}\")\n        \n        return metrics\n        \n    except KeyboardInterrupt:\n        logger.info(\"Training interrupted by user\")\n        return None\n    except Exception as e:\n        logger.error(f\"Training pipeline failed: {str(e)}\")\n        return None\n\n\nif __name__ == \"__main__\":\n    import sys\n    \n    # Allow config file as command line argument\n    config_file = sys.argv[1] if len(sys.argv) > 1 else None\n    main_with_config(config_file)