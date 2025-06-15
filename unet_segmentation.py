import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, 
    BatchNormalization, Dropout, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LungSegmentationUNet:
    def __init__(self, input_size=(512, 512, 1)):
        self.input_size = input_size
        self.model = None
        
    def conv_block(self, inputs, num_filters, kernel_size=3, activation='relu', padding='same'):
        """Convolutional block with BatchNorm and Dropout"""
        x = Conv2D(num_filters, kernel_size, padding=padding)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        
        x = Conv2D(num_filters, kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        
        return x
    
    def encoder_block(self, inputs, num_filters, dropout_rate=0.3):
        """Encoder block: conv_block -> pooling -> dropout"""
        conv = self.conv_block(inputs, num_filters)
        pool = MaxPooling2D((2, 2))(conv)
        pool = Dropout(dropout_rate)(pool)
        
        return conv, pool
    
    def decoder_block(self, inputs, skip_features, num_filters, dropout_rate=0.3):
        """Decoder block: upsampling -> concatenate -> conv_block"""
        upsample = UpSampling2D((2, 2))(inputs)
        concatenate = Concatenate()([upsample, skip_features])
        dropout = Dropout(dropout_rate)(concatenate)
        conv = self.conv_block(dropout, num_filters)
        
        return conv
    
    def build_unet(self):
        """Build U-Net architecture for lung segmentation"""
        inputs = Input(self.input_size, name='input_image')
        
        # Encoder (Contracting Path)
        s1, p1 = self.encoder_block(inputs, 64, 0.1)      # 512x512 -> 256x256
        s2, p2 = self.encoder_block(p1, 128, 0.1)         # 256x256 -> 128x128
        s3, p3 = self.encoder_block(p2, 256, 0.2)         # 128x128 -> 64x64
        s4, p4 = self.encoder_block(p3, 512, 0.2)         # 64x64 -> 32x32
        
        # Bridge (Bottleneck)
        bridge = self.conv_block(p4, 1024)                # 32x32
        bridge = Dropout(0.3)(bridge)
        
        # Decoder (Expanding Path)
        d1 = self.decoder_block(bridge, s4, 512, 0.2)     # 32x32 -> 64x64
        d2 = self.decoder_block(d1, s3, 256, 0.2)         # 64x64 -> 128x128
        d3 = self.decoder_block(d2, s2, 128, 0.1)         # 128x128 -> 256x256
        d4 = self.decoder_block(d3, s1, 64, 0.1)          # 256x256 -> 512x512
        
        # Output layer
        outputs = Conv2D(1, 1, activation='sigmoid', name='segmentation_mask')(d4)
        
        # Create model
        model = Model(inputs, outputs, name='lung_segmentation_unet')
        
        return model
    
    def compile_model(self, learning_rate=1e-4):
        """Compile the U-Net model with appropriate loss functions"""
        if self.model is None:
            self.model = self.build_unet()
        
        # Custom loss function combining Dice and Binary Crossentropy
        def dice_coefficient(y_true, y_pred, smooth=1e-6):
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        
        def dice_loss(y_true, y_pred):
            return 1 - dice_coefficient(y_true, y_pred)
        
        def combined_loss(y_true, y_pred):
            return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)
        
        def iou_metric(y_true, y_pred, smooth=1e-6):
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
            return (intersection + smooth) / (union + smooth)
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=combined_loss,
            metrics=[dice_coefficient, iou_metric, 'binary_accuracy']
        )
        
        return self.model
    
    def get_callbacks(self, model_name='lung_segmentation'):
        """Get training callbacks"""
        callbacks = [
            ModelCheckpoint(
                f'models/{model_name}_best.h5',
                monitor='val_dice_coefficient',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_dice_coefficient',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_dice_coefficient',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1,
                mode='max'
            )
        ]
        return callbacks
    
    def predict_mask(self, image):
        """Predict lung segmentation mask for a single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image)
        return prediction[0, :, :, 0]  # Remove batch and channel dimensions
    
    def visualize_prediction(self, image, mask, prediction, save_path=None):
        """Visualize original image, ground truth mask, and prediction"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(image.squeeze(), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        # Overlay
        overlay = image.squeeze().copy()
        overlay = cv2.applyColorMap((overlay * 255).astype(np.uint8), cv2.COLORMAP_GRAY)
        mask_colored = cv2.applyColorMap((prediction * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR_RGB))
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class LungMaskGenerator:
    """Generate lung masks from chest X-rays using image processing techniques"""
    
    @staticmethod
    def create_lung_mask(image):
        """
        Create a basic lung mask using image processing techniques
        This is a simplified version - in practice, you'd want manually annotated masks
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Normalize
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to create binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find largest contours (lungs)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros_like(gray)
        
        # Keep only the largest contours (lungs)
        if contours:
            # Sort contours by area and keep the largest ones
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Draw the largest contours (typically the two lungs)
            for i, contour in enumerate(contours[:2]):  # Top 2 largest
                if cv2.contourArea(contour) > 1000:  # Filter small noise
                    cv2.fillPoly(mask, [contour], 255)
        
        # Normalize to [0, 1]
        mask = mask / 255.0
        
        return mask


def create_lung_masks_for_dataset(data_dir, output_dir):
    """Create lung masks for the entire pneumonia dataset"""
    os.makedirs(output_dir, exist_ok=True)
    mask_generator = LungMaskGenerator()
    
    # Process train and test sets
    for subset in ['train', 'test']:
        subset_dir = os.path.join(data_dir, subset)
        output_subset_dir = os.path.join(output_dir, subset)
        os.makedirs(output_subset_dir, exist_ok=True)
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(subset_dir, class_name)
            output_class_dir = os.path.join(output_subset_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                print(f"Processing {len(image_files)} images in {subset}/{class_name}")
                
                for img_file in image_files:
                    img_path = os.path.join(class_dir, img_file)
                    
                    # Load image
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (512, 512))
                    
                    # Generate mask
                    mask = mask_generator.create_lung_mask(image)
                    
                    # Save mask
                    mask_filename = img_file.replace('.jpeg', '_mask.png').replace('.jpg', '_mask.png')
                    mask_path = os.path.join(output_class_dir, mask_filename)
                    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                
                print(f"Completed {subset}/{class_name}")


def main():
    """Main function to demonstrate U-Net segmentation"""
    print("=== Lung Segmentation U-Net ===")
    
    # Create lung masks for the dataset (this would take time in practice)
    print("Step 1: Creating lung masks for dataset...")
    # create_lung_masks_for_dataset('chest_xray', 'lung_masks')
    
    # Initialize U-Net
    print("Step 2: Building U-Net model...")
    unet = LungSegmentationUNet(input_size=(512, 512, 1))
    model = unet.compile_model()
    
    print("U-Net Model Summary:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # For demonstration, create synthetic data
    print("\nStep 3: Creating synthetic training data...")
    num_samples = 100
    X_train = np.random.random((num_samples, 512, 512, 1))
    y_train = np.random.randint(0, 2, (num_samples, 512, 512, 1)).astype(np.float32)
    
    X_val = np.random.random((20, 512, 512, 1))
    y_val = np.random.randint(0, 2, (20, 512, 512, 1)).astype(np.float32)
    
    print("Step 4: Training U-Net (demo with synthetic data)...")
    
    # Get callbacks
    callbacks = unet.get_callbacks()
    
    # Train model (shortened for demo)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,  # Reduced for demo
        batch_size=4,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nStep 5: Testing prediction...")
    test_image = X_val[0:1]
    test_mask = y_val[0]
    prediction = unet.predict_mask(test_image)
    
    # Visualize results
    unet.visualize_prediction(
        test_image[0], 
        test_mask[:,:,0], 
        prediction,
        save_path='lung_segmentation_demo.png'
    )
    
    print("✅ Lung segmentation U-Net demo completed!")
    print("Next step: Integrate with pneumonia classification pipeline")


if __name__ == "__main__":
    main()