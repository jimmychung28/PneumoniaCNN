import os
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import logging
from typing import Optional, Tuple, Dict, Any

# Import validation utilities
from validation_utils import (
    validate_input_shape, validate_learning_rate, validate_batch_size, 
    validate_epochs, validate_directory_exists, validate_dataset_structure,
    validate_model_save_path, ValidationError, FileValidationError, 
    ModelValidationError, logger
)

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Configure for Apple Silicon if available
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    print("🍎 Apple Silicon detected - Metal GPU acceleration will be used")
    # List available devices
    print("Available devices:", tf.config.list_physical_devices())
    # Metal GPU is automatically used when available
else:
    print("Running on:", platform.system(), platform.machine())

class PneumoniaCNN:
    def __init__(self, input_shape=(128, 128, 3), learning_rate=0.0001):
        """
        Initialize PneumoniaCNN with input validation.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            learning_rate: Learning rate for training
            
        Raises:
            ModelValidationError: If parameters are invalid
        """
        try:
            self.input_shape = validate_input_shape(input_shape, expected_dims=3)
            self.learning_rate = validate_learning_rate(learning_rate)
            self.model = None
            self.history = None
            
            logger.info(f"PneumoniaCNN initialized with input_shape={self.input_shape}, learning_rate={self.learning_rate}")
            
        except Exception as e:
            logger.error(f"Error initializing PneumoniaCNN: {str(e)}")
            raise ModelValidationError(f"Failed to initialize PneumoniaCNN: {str(e)}")
        
    def build_model(self):
        """
        Build an improved CNN architecture with regularization.
        
        Returns:
            Built and compiled Keras model
            
        Raises:
            ModelValidationError: If model building fails
        """
        try:
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
        
            self.model = model
            logger.info("Model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise ModelValidationError(f"Failed to build model: {str(e)}")
    
    def create_data_generators(self, train_dir: str, test_dir: str, batch_size: int = 32):
        """
        Create data generators with augmentation for training.
        
        Args:
            train_dir: Path to training data directory
            test_dir: Path to test data directory
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_generator, validation_generator, test_generator)
            
        Raises:
            FileValidationError: If directories don't exist or are invalid
            ModelValidationError: If batch_size is invalid
        """
        try:
            # Validate inputs
            train_dir = validate_directory_exists(train_dir)
            test_dir = validate_directory_exists(test_dir)
            batch_size = validate_batch_size(batch_size)
            
            # Validate dataset structure
            expected_classes = ['NORMAL', 'PNEUMONIA']
            train_info = validate_dataset_structure(train_dir, expected_classes)
            test_info = validate_dataset_structure(test_dir, expected_classes)
            
            logger.info(f"Training data: {train_info}")
            logger.info(f"Test data: {test_info}")
            
            # Check for empty classes
            for class_name in expected_classes:
                if train_info[class_name]['image_count'] == 0:
                    raise FileValidationError(f"No training images found for class: {class_name}")
                if test_info[class_name]['image_count'] == 0:
                    logger.warning(f"No test images found for class: {class_name}")
        # More aggressive augmentation for better generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # Use 20% of training data for validation
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        # Validation generator (from training data)
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        # Test generator
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
            logger.info("Data generators created successfully")
            return train_generator, validation_generator, test_generator
            
        except Exception as e:
            logger.error(f"Error creating data generators: {str(e)}")
            if isinstance(e, (FileValidationError, ModelValidationError)):
                raise
            raise FileValidationError(f"Failed to create data generators: {str(e)}")
    
    def get_callbacks(self, model_name: str = 'pneumonia_cnn'):
        """
        Create callbacks for training with proper error handling.
        
        Args:
            model_name: Name prefix for saved model files
            
        Returns:
            List of Keras callbacks
            
        Raises:
            FileValidationError: If callback setup fails
        """
        try:
            # Validate and create directories
            models_dir = validate_directory_exists('models', create_if_missing=True)
            logs_dir = validate_directory_exists('logs', create_if_missing=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Validate model save path
            model_save_path = validate_model_save_path(
                os.path.join(models_dir, f'{model_name}_best_{timestamp}.h5')
            )
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
        
            logger.info(f"Callbacks created successfully, model will be saved to: {model_save_path}")
            return callbacks
            
        except Exception as e:
            logger.error(f"Error creating callbacks: {str(e)}")
            raise FileValidationError(f"Failed to create callbacks: {str(e)}")
    
    def calculate_class_weights(self, train_generator) -> Dict[int, float]:
        """
        Calculate class weights to handle imbalanced dataset.
        
        Args:
            train_generator: Training data generator
            
        Returns:
            Dictionary mapping class indices to weights
            
        Raises:
            ValidationError: If class weight calculation fails
        """
        try:
            if train_generator is None:
                raise ValidationError("train_generator cannot be None")
                
            if len(train_generator) == 0:
                raise ValidationError("train_generator is empty")
        # Count samples in each class
        counter = {0: 0, 1: 0}
        for i in range(len(train_generator)):
            _, y = train_generator[i]
            for label in y:
                counter[int(label)] += 1
        
        # Calculate weights inversely proportional to class frequencies
        total = sum(counter.values())
        class_weight = {
            0: total / (2 * counter[0]),
            1: total / (2 * counter[1])
        }
        
            logger.info(f"Class distribution: {counter}")
            logger.info(f"Class weights: {class_weight}")
            
            # Validate calculated weights
            for class_idx, weight in class_weight.items():
                if weight <= 0 or not np.isfinite(weight):
                    raise ValidationError(f"Invalid class weight for class {class_idx}: {weight}")
            
            return class_weight
            
        except Exception as e:
            logger.error(f"Error calculating class weights: {str(e)}")
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Failed to calculate class weights: {str(e)}")
    
    def train(self, train_generator, validation_generator, epochs: int = 50):
        """
        Train the model with comprehensive error handling.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            
        Returns:
            Training history
            
        Raises:
            ModelValidationError: If training fails
            ValidationError: If generators are invalid
        """
        try:
            # Validate inputs
            if self.model is None:
                raise ModelValidationError("Model must be built before training")
                
            if train_generator is None or validation_generator is None:
                raise ValidationError("Generators cannot be None")
                
            epochs = validate_epochs(epochs)
            
            # Validate generators have data
            if len(train_generator) == 0:
                raise ValidationError("Training generator is empty")
            if len(validation_generator) == 0:
                raise ValidationError("Validation generator is empty")
                
            logger.info(f"Starting training for {epochs} epochs")
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Calculate class weights
        class_weight = self.calculate_class_weights(train_generator)
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
            logger.info("Training completed successfully")
            return self.history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            if isinstance(e, (ModelValidationError, ValidationError)):
                raise
            raise ModelValidationError(f"Training failed: {str(e)}")
    
    def evaluate(self, test_generator) -> Dict[str, float]:
        """
        Evaluate the model and generate detailed metrics.
        
        Args:
            test_generator: Test data generator
            
        Returns:
            Dictionary of evaluation metrics
            
        Raises:
            ModelValidationError: If evaluation fails
            ValidationError: If test_generator is invalid
        """
        try:
            # Validate inputs
            if self.model is None:
                raise ModelValidationError("Model must be trained before evaluation")
                
            if test_generator is None:
                raise ValidationError("test_generator cannot be None")
                
            if len(test_generator) == 0:
                raise ValidationError("test_generator is empty")
                
            logger.info("Starting model evaluation")
        # Get predictions
        predictions = self.model.predict(test_generator)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = test_generator.classes
        
        # Calculate metrics
        print("\n=== Model Evaluation ===")
        print("\nTest Set Metrics:")
        test_metrics = self.model.evaluate(test_generator, verbose=0)
        for metric_name, metric_value in zip(self.model.metrics_names, test_metrics):
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=['NORMAL', 'PNEUMONIA']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, ['NORMAL', 'PNEUMONIA'])
        
            logger.info("Evaluation completed successfully")
            
            # Return metrics as dictionary
            metrics_dict = {}
            for metric_name, metric_value in zip(self.model.metrics_names, test_metrics):
                metrics_dict[metric_name] = float(metric_value)
                
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            if isinstance(e, (ModelValidationError, ValidationError)):
                raise
            raise ModelValidationError(f"Evaluation failed: {str(e)}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list) -> None:
        """
        Plot confusion matrix with error handling.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            
        Raises:
            ValidationError: If inputs are invalid
        """
        try:
            if cm is None or len(cm) == 0:
                raise ValidationError("Confusion matrix cannot be empty")
                
            if not isinstance(class_names, list) or len(class_names) == 0:
                raise ValidationError("class_names must be a non-empty list")
                
            if cm.shape[0] != len(class_names) or cm.shape[1] != len(class_names):
                raise ValidationError("Confusion matrix dimensions must match number of classes")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            logger.info("Confusion matrix saved as confusion_matrix.png")
            plt.show()
            plt.close()  # Prevent memory leaks
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            plt.close()  # Ensure cleanup even on error
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Failed to plot confusion matrix: {str(e)}")
    
    def plot_training_history(self) -> None:
        """
        Plot training history with error handling.
        
        Raises:
            ValidationError: If no training history is available
        """
        try:
            if self.history is None:
                raise ValidationError("No training history available")
                
            if not hasattr(self.history, 'history') or not self.history.history:
                raise ValidationError("Training history is empty")
                
            required_keys = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
            missing_keys = [key for key in required_keys if key not in self.history.history]
            if missing_keys:
                logger.warning(f"Missing training history keys: {missing_keys}")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Train')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # AUC
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
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            logger.info("Training history saved as training_history.png")
            plt.show()
            plt.close()  # Prevent memory leaks
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            plt.close()  # Ensure cleanup even on error
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Failed to plot training history: {str(e)}")


def main() -> None:
    """
    Main training pipeline with comprehensive error handling.
    
    Raises:
        SystemExit: If training pipeline fails
    """
    try:
        # Configuration with validation
        BATCH_SIZE = validate_batch_size(32)
        EPOCHS = validate_epochs(50)
        INPUT_SHAPE = validate_input_shape((128, 128, 3))
        
        logger.info(f"Configuration: BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, INPUT_SHAPE={INPUT_SHAPE}")
    
        # Initialize model
        logger.info("Initializing Pneumonia CNN...")
        pneumonia_cnn = PneumoniaCNN(input_shape=INPUT_SHAPE)
        
        # Build model
        logger.info("Building model architecture...")
        model = pneumonia_cnn.build_model()
        model.summary()
        
        # Create data generators
        logger.info("Creating data generators...")
        train_gen, val_gen, test_gen = pneumonia_cnn.create_data_generators(
            train_dir='chest_xray/train',
            test_dir='chest_xray/test',
            batch_size=BATCH_SIZE
        )
        
        # Train model
        logger.info("Starting training...")
        history = pneumonia_cnn.train(train_gen, val_gen, epochs=EPOCHS)
        
        # Plot training history
        logger.info("Plotting training history...")
        pneumonia_cnn.plot_training_history()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        metrics = pneumonia_cnn.evaluate(test_gen)
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = validate_model_save_path(f'models/pneumonia_cnn_final_{timestamp}.h5')
        model.save(final_model_path)
        logger.info(f"Model saved as: {final_model_path}")
        
        logger.info("Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()