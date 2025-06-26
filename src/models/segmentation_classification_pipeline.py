import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from datetime import datetime
import platform

from src.models.unet_segmentation import LungSegmentationUNet

# Configure for Apple Silicon if available
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    print("🍎 Apple Silicon detected - Metal GPU acceleration will be used")

class TwoStagePneumoniaDetector:
    def __init__(self, 
                 segmentation_model_path=None,
                 classification_input_size=(224, 224, 3),
                 segmentation_input_size=(512, 512, 1)):
        
        self.classification_input_size = classification_input_size
        self.segmentation_input_size = segmentation_input_size
        
        # Initialize segmentation model
        self.segmentation_model = self._load_or_build_segmentation_model(segmentation_model_path)
        
        # Initialize classification model
        self.classification_model = None
        self.history = None
        
    def _load_or_build_segmentation_model(self, model_path):
        """Load pre-trained segmentation model or build new one"""
        if model_path and os.path.exists(model_path):
            print(f"Loading pre-trained segmentation model from {model_path}")
            return load_model(model_path, compile=False)
        else:
            print("Building new U-Net segmentation model")
            unet = LungSegmentationUNet(input_size=self.segmentation_input_size)
            return unet.compile_model()
    
    def build_classification_model(self, architecture='resnet50'):
        """Build classification model with different backbone architectures"""
        
        if architecture.lower() == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.classification_input_size
            )
        elif architecture.lower() == 'densenet121':
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=self.classification_input_size
            )
        else:
            raise ValueError(f"Architecture {architecture} not supported")
        
        # Freeze base model layers initially
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom classification head
        x = base_model.output
        
        # Global pooling with multiple strategies
        gap = GlobalAveragePooling2D(name='global_avg_pool')(x)
        gmp = GlobalMaxPooling2D(name='global_max_pool')(x)
        pooled = Concatenate(name='concat_pools')([gap, gmp])
        
        # Classification layers
        x = Dense(512, activation='relu', name='dense_512')(pooled)
        x = BatchNormalization(name='bn_512')(x)
        x = Dropout(0.5, name='dropout_512')(x)
        
        x = Dense(256, activation='relu', name='dense_256')(x)
        x = BatchNormalization(name='bn_256')(x)
        x = Dropout(0.3, name='dropout_256')(x)
        
        outputs = Dense(1, activation='sigmoid', name='pneumonia_prediction')(x)
        
        model = Model(base_model.input, outputs, name=f'{architecture}_pneumonia_classifier')
        
        return model
    
    def compile_classification_model(self, learning_rate=1e-4):
        """Compile the classification model"""
        if self.classification_model is None:
            self.classification_model = self.build_classification_model()
        
        # Custom metrics
        def precision(y_true, y_pred):
            threshold = 0.5
            y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
            tp = tf.reduce_sum(y_true * y_pred_binary)
            fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
            return tp / (tp + fp + tf.keras.backend.epsilon())
        
        def recall(y_true, y_pred):
            threshold = 0.5
            y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
            tp = tf.reduce_sum(y_true * y_pred_binary)
            fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
            return tp / (tp + fn + tf.keras.backend.epsilon())
        
        def f1_score(y_true, y_pred):
            p = precision(y_true, y_pred)
            r = recall(y_true, y_pred)
            return 2 * p * r / (p + r + tf.keras.backend.epsilon())
        
        self.classification_model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', precision, recall, f1_score]
        )
        
        return self.classification_model
    
    def preprocess_image_for_segmentation(self, image):
        """Preprocess image for segmentation"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to segmentation input size
        image = cv2.resize(image, self.segmentation_input_size[:2])
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def preprocess_image_for_classification(self, image, mask=None):
        """Preprocess image for classification"""
        # Apply mask if provided
        if mask is not None:
            # Convert mask to 3-channel
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
            
            # Apply mask to each channel
            if len(image.shape) == 3:
                for i in range(image.shape[2]):
                    image[:, :, i] = image[:, :, i] * mask[:, :, 0]
            else:
                image = image * mask[:, :, 0]
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
        
        # Resize to classification input size
        image = cv2.resize(image, self.classification_input_size[:2])
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def predict_single_image(self, image_path):
        """Predict pneumonia for a single image using two-stage pipeline"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        original_image = image.copy()
        
        # Stage 1: Segment lungs
        seg_input = self.preprocess_image_for_segmentation(image)
        lung_mask = self.segmentation_model.predict(seg_input, verbose=0)[0, :, :, 0]
        
        # Threshold mask
        lung_mask_binary = (lung_mask > 0.5).astype(np.float32)
        
        # Stage 2: Classify pneumonia on masked region
        # Resize mask to match original image
        mask_resized = cv2.resize(lung_mask_binary, (image.shape[1], image.shape[0]))
        
        # Preprocess for classification
        class_input = self.preprocess_image_for_classification(original_image, mask_resized)
        class_input = np.expand_dims(class_input, axis=0)
        
        # Predict pneumonia
        pneumonia_prob = self.classification_model.predict(class_input, verbose=0)[0, 0]
        
        return {
            'pneumonia_probability': float(pneumonia_prob),
            'pneumonia_prediction': pneumonia_prob > 0.5,
            'lung_mask': lung_mask,
            'confidence': float(max(pneumonia_prob, 1 - pneumonia_prob))
        }
    
    def create_masked_data_generator(self, data_dir, batch_size=32, subset='training'):
        """Create data generator that applies lung segmentation masks"""
        
        class MaskedDataGenerator(tf.keras.utils.Sequence):
            def __init__(self, detector, data_dir, batch_size, subset, target_size):
                self.detector = detector
                self.batch_size = batch_size
                self.target_size = target_size
                
                # Get image paths and labels
                self.image_paths = []
                self.labels = []
                
                # Process both NORMAL and PNEUMONIA classes
                for class_name in ['NORMAL', 'PNEUMONIA']:
                    class_dir = os.path.join(data_dir, class_name)
                    if os.path.exists(class_dir):
                        class_files = [f for f in os.listdir(class_dir) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        
                        for img_file in class_files:
                            self.image_paths.append(os.path.join(class_dir, img_file))
                            self.labels.append(1 if class_name == 'PNEUMONIA' else 0)
                
                # Shuffle data
                indices = np.random.permutation(len(self.image_paths))
                self.image_paths = [self.image_paths[i] for i in indices]
                self.labels = [self.labels[i] for i in indices]
                
            def __len__(self):
                return len(self.image_paths) // self.batch_size
            
            def __getitem__(self, idx):
                batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
                
                batch_images = []
                
                for img_path in batch_paths:
                    # Load and process image
                    image = cv2.imread(img_path)
                    
                    # Generate lung mask
                    seg_input = self.detector.preprocess_image_for_segmentation(image)
                    lung_mask = self.detector.segmentation_model.predict(seg_input, verbose=0)[0, :, :, 0]
                    lung_mask_binary = (lung_mask > 0.5).astype(np.float32)
                    
                    # Resize mask and apply to image
                    mask_resized = cv2.resize(lung_mask_binary, (image.shape[1], image.shape[0]))
                    processed_image = self.detector.preprocess_image_for_classification(image, mask_resized)
                    
                    batch_images.append(processed_image)
                
                return np.array(batch_images), np.array(batch_labels, dtype=np.float32)
        
        return MaskedDataGenerator(self, data_dir, batch_size, subset, self.classification_input_size[:2])
    
    def train_classification_model(self, train_dir, val_dir=None, epochs=50, batch_size=16):
        """Train the classification model using masked images"""
        
        # Create data generators
        train_generator = self.create_masked_data_generator(train_dir, batch_size, 'training')
        
        if val_dir:
            val_generator = self.create_masked_data_generator(val_dir, batch_size, 'validation')
        else:
            val_generator = None
        
        # Calculate class weights
        train_labels = train_generator.labels
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        class_weight = {
            0: total_samples / (2 * class_counts[0]),
            1: total_samples / (2 * class_counts[1])
        }
        
        print(f"Class weights: {class_weight}")
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        callbacks = [
            ModelCheckpoint(
                f'models/two_stage_classifier_best_{timestamp}.h5',
                monitor='val_auc' if val_generator else 'auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_auc' if val_generator else 'auc',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_auc' if val_generator else 'auc',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode='max'
            )
        ]
        
        # Train model
        self.history = self.classification_model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Save final model
        self.classification_model.save(f'models/two_stage_classifier_final_{timestamp}.h5')
        
        return self.history
    
    def evaluate_model(self, test_dir, batch_size=16):
        """Evaluate the two-stage model"""
        test_generator = self.create_masked_data_generator(test_dir, batch_size, 'testing')
        
        # Get predictions
        predictions = []
        true_labels = []
        
        for i in range(len(test_generator)):
            batch_x, batch_y = test_generator[i]
            batch_pred = self.classification_model.predict(batch_x, verbose=0)
            
            predictions.extend(batch_pred.flatten())
            true_labels.extend(batch_y)
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Calculate metrics
        auc_score = roc_auc_score(true_labels, predictions)
        
        # Binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        print("\n=== Two-Stage Model Evaluation ===")
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, binary_predictions, 
                                  target_names=['NORMAL', 'PNEUMONIA']))
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, binary_predictions)
        self.plot_confusion_matrix(cm, ['NORMAL', 'PNEUMONIA'])
        
        # ROC Curve
        self.plot_roc_curve(true_labels, predictions)
        
        return {
            'auc_score': auc_score,
            'predictions': predictions,
            'true_labels': true_labels,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Two-Stage Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('two_stage_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'Two-Stage Model (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Two-Stage Pneumonia Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('two_stage_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize the two-stage prediction process"""
        # Load original image
        image = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get prediction
        result = self.predict_single_image(image_path)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(original_rgb, cmap='gray')
        axes[0, 0].set_title('Original Chest X-ray')
        axes[0, 0].axis('off')
        
        # Segmentation mask
        axes[0, 1].imshow(result['lung_mask'], cmap='gray')
        axes[0, 1].set_title('Predicted Lung Mask')
        axes[0, 1].axis('off')
        
        # Masked image
        mask_3d = np.stack([result['lung_mask']] * 3, axis=-1)
        masked_image = original_rgb * mask_3d
        axes[0, 2].imshow(masked_image.astype(np.uint8))
        axes[0, 2].set_title('Masked Image (Input to Classifier)')
        axes[0, 2].axis('off')
        
        # Overlay
        overlay = original_rgb.copy()
        mask_colored = plt.cm.jet(result['lung_mask'])[:, :, :3]
        overlay_result = overlay * 0.7 + mask_colored * 255 * 0.3
        axes[1, 0].imshow(overlay_result.astype(np.uint8))
        axes[1, 0].set_title('Segmentation Overlay')
        axes[1, 0].axis('off')
        
        # Prediction text
        axes[1, 1].text(0.1, 0.7, f"Prediction: {'PNEUMONIA' if result['pneumonia_prediction'] else 'NORMAL'}", 
                        fontsize=16, fontweight='bold',
                        color='red' if result['pneumonia_prediction'] else 'green')
        axes[1, 1].text(0.1, 0.5, f"Probability: {result['pneumonia_probability']:.3f}", fontsize=14)
        axes[1, 1].text(0.1, 0.3, f"Confidence: {result['confidence']:.3f}", fontsize=14)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Prediction Results')
        
        # Empty subplot for layout
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to demonstrate two-stage pneumonia detection"""
    print("=== Two-Stage Pneumonia Detection Pipeline ===")
    
    # Initialize detector
    detector = TwoStagePneumoniaDetector()
    
    # Build and compile classification model
    print("Building classification model...")
    model = detector.compile_classification_model()
    model.summary()
    
    print(f"\nClassification model parameters: {model.count_params():,}")
    
    # For demonstration with actual data
    if os.path.exists('data/chest_xray/train'):
        print("\nDemo: Predicting on a sample image...")
        
        # Find a sample image
        sample_dir = 'data/chest_xray/train/PNEUMONIA'
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith('.jpeg')]
            if sample_files:
                sample_path = os.path.join(sample_dir, sample_files[0])
                
                # Make prediction
                result = detector.predict_single_image(sample_path)
                print(f"\nSample prediction:")
                print(f"File: {sample_files[0]}")
                print(f"Prediction: {'PNEUMONIA' if result['pneumonia_prediction'] else 'NORMAL'}")
                print(f"Probability: {result['pneumonia_probability']:.3f}")
                print(f"Confidence: {result['confidence']:.3f}")
                
                # Visualize
                detector.visualize_prediction(sample_path, 'two_stage_demo_prediction.png')
    
    print("\n✅ Two-stage pipeline setup completed!")
    print("\nTo train the model:")
    print("detector.train_classification_model('data/chest_xray/train', 'data/chest_xray/test')")


if __name__ == "__main__":
    main()