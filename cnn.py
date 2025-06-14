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

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

class PneumoniaCNN:
    def __init__(self, input_shape=(128, 128, 3), learning_rate=0.0001):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build an improved CNN architecture with regularization"""
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
        return model
    
    def create_data_generators(self, train_dir, test_dir, batch_size=32):
        """Create data generators with augmentation for training"""
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
        
        return train_generator, validation_generator, test_generator
    
    def get_callbacks(self, model_name='pneumonia_cnn'):
        """Create callbacks for training"""
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
    
    def calculate_class_weights(self, train_generator):
        """Calculate class weights to handle imbalanced dataset"""
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
        
        print(f"Class distribution: {counter}")
        print(f"Class weights: {class_weight}")
        
        return class_weight
    
    def train(self, train_generator, validation_generator, epochs=50):
        """Train the model"""
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
        
        return self.history
    
    def evaluate(self, test_generator):
        """Evaluate the model and generate detailed metrics"""
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
        
        return test_metrics
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
        
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
        plt.show()


def main():
    """Main training pipeline"""
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 50
    INPUT_SHAPE = (128, 128, 3)  # Larger input size for better feature extraction
    
    # Initialize model
    print("Initializing Pneumonia CNN...")
    pneumonia_cnn = PneumoniaCNN(input_shape=INPUT_SHAPE)
    
    # Build model
    print("Building model architecture...")
    model = pneumonia_cnn.build_model()
    model.summary()
    
    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen, test_gen = pneumonia_cnn.create_data_generators(
        train_dir='chest_xray/train',
        test_dir='chest_xray/test',
        batch_size=BATCH_SIZE
    )
    
    # Train model
    print("Starting training...")
    history = pneumonia_cnn.train(train_gen, val_gen, epochs=EPOCHS)
    
    # Plot training history
    print("Plotting training history...")
    pneumonia_cnn.plot_training_history()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    pneumonia_cnn.evaluate(test_gen)
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f'models/pneumonia_cnn_final_{timestamp}.h5')
    print(f"Model saved as: models/pneumonia_cnn_final_{timestamp}.h5")


if __name__ == "__main__":
    main()