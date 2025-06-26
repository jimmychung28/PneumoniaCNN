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
import logging
from typing import Optional, Tuple, List, Dict, Any

# Import validation utilities
from src.utils.validation_utils import (
    validate_input_shape, validate_learning_rate, validate_batch_size,
    validate_directory_exists, validate_image_file, safe_load_image,
    validate_model_save_path, ValidationError, FileValidationError,
    ModelValidationError, ImageValidationError, logger
)

class LungSegmentationUNet:
    def __init__(self, input_size=(512, 512, 1)):
        """
        Initialize U-Net for lung segmentation with input validation.
        
        Args:
            input_size: Shape of input images (height, width, channels)
            
        Raises:
            ModelValidationError: If input_size is invalid
        """
        try:
            self.input_size = validate_input_shape(input_size, expected_dims=3)
            
            # Validate that last dimension is 1 for grayscale
            if self.input_size[2] != 1:
                raise ModelValidationError("U-Net expects grayscale images (channels=1)")
                
            # Validate input size is reasonable for U-Net
            if self.input_size[0] < 64 or self.input_size[1] < 64:
                raise ModelValidationError("Input size too small for U-Net (minimum 64x64)")
                
            # Check if input size is power of 2 (recommended for U-Net)
            for dim in self.input_size[:2]:
                if dim & (dim - 1) != 0:
                    logger.warning(f"Input dimension {dim} is not a power of 2. This may cause issues in U-Net.")
                    
            self.model = None
            logger.info(f"LungSegmentationUNet initialized with input_size={self.input_size}")
            
        except Exception as e:
            logger.error(f"Error initializing LungSegmentationUNet: {str(e)}")
            if isinstance(e, ModelValidationError):
                raise
            raise ModelValidationError(f"Failed to initialize U-Net: {str(e)}")
        
    def conv_block(self, inputs, num_filters: int, kernel_size: int = 3, 
                   activation: str = 'relu', padding: str = 'same'):
        """
        Convolutional block with BatchNorm and Dropout.
        
        Args:
            inputs: Input tensor
            num_filters: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            activation: Activation function
            padding: Padding type
            
        Returns:
            Output tensor
            
        Raises:
            ModelValidationError: If parameters are invalid
        """
        try:
            if num_filters <= 0 or not isinstance(num_filters, int):
                raise ModelValidationError(f"num_filters must be positive integer, got {num_filters}")
                
            if kernel_size <= 0 or not isinstance(kernel_size, int):
                raise ModelValidationError(f"kernel_size must be positive integer, got {kernel_size}")
                
            if activation not in ['relu', 'sigmoid', 'tanh', 'linear']:
                raise ModelValidationError(f"Unsupported activation: {activation}")
                
            if padding not in ['same', 'valid']:
                raise ModelValidationError(f"Unsupported padding: {padding}")
                
            x = Conv2D(num_filters, kernel_size, padding=padding)(inputs)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)
            
            x = Conv2D(num_filters, kernel_size, padding=padding)(x)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)
            
            return x
            
        except Exception as e:
            logger.error(f"Error in conv_block: {str(e)}")
            if isinstance(e, ModelValidationError):
                raise
            raise ModelValidationError(f"Failed to create conv_block: {str(e)}")
    
    def encoder_block(self, inputs, num_filters: int, dropout_rate: float = 0.3):
        """
        Encoder block: conv_block -> pooling -> dropout.
        
        Args:
            inputs: Input tensor
            num_filters: Number of convolutional filters
            dropout_rate: Dropout rate (0.0 to 1.0)
            
        Returns:
            Tuple of (conv_output, pooled_output)
            
        Raises:
            ModelValidationError: If parameters are invalid
        """
        try:
            if not 0.0 <= dropout_rate <= 1.0:
                raise ModelValidationError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
                
            conv = self.conv_block(inputs, num_filters)
            pool = MaxPooling2D((2, 2))(conv)
            pool = Dropout(dropout_rate)(pool)
            
            return conv, pool
            
        except Exception as e:
            logger.error(f"Error in encoder_block: {str(e)}")
            if isinstance(e, ModelValidationError):
                raise
            raise ModelValidationError(f"Failed to create encoder_block: {str(e)}")
    
    def decoder_block(self, inputs, skip_features, num_filters: int, dropout_rate: float = 0.3):
        """
        Decoder block: upsampling -> concatenate -> conv_block.
        
        Args:
            inputs: Input tensor
            skip_features: Skip connection features from encoder
            num_filters: Number of convolutional filters
            dropout_rate: Dropout rate (0.0 to 1.0)
            
        Returns:
            Output tensor
            
        Raises:
            ModelValidationError: If parameters are invalid
        """
        try:
            if not 0.0 <= dropout_rate <= 1.0:
                raise ModelValidationError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
                
            upsample = UpSampling2D((2, 2))(inputs)
            concatenate = Concatenate()([upsample, skip_features])
            dropout = Dropout(dropout_rate)(concatenate)
            conv = self.conv_block(dropout, num_filters)
            
            return conv
            
        except Exception as e:
            logger.error(f"Error in decoder_block: {str(e)}")
            if isinstance(e, ModelValidationError):
                raise
            raise ModelValidationError(f"Failed to create decoder_block: {str(e)}")
    
    def build_unet(self):
        """
        Build U-Net architecture for lung segmentation.
        
        Returns:
            Compiled Keras model
            
        Raises:
            ModelValidationError: If model building fails
        """
        try:
            logger.info("Building U-Net architecture...")
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
            
            if model is None:
                raise ModelValidationError("Failed to create model")
                
            logger.info(f"U-Net model built successfully with {model.count_params():,} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Error building U-Net model: {str(e)}")
            if isinstance(e, ModelValidationError):
                raise
            raise ModelValidationError(f"Failed to build U-Net model: {str(e)}")
    
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
    
    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Predict lung segmentation mask for a single image.
        
        Args:
            image: Input image array
            
        Returns:
            Predicted segmentation mask
            
        Raises:
            ModelValidationError: If prediction fails
            ValidationError: If image is invalid
        """
        try:
            if self.model is None:
                raise ModelValidationError("Model must be compiled before prediction")
                
            if not isinstance(image, np.ndarray):
                raise ValidationError("image must be a numpy array")
                
            if len(image.shape) < 2 or len(image.shape) > 4:
                raise ValidationError(f"Invalid image shape: {image.shape}")
                
            # Ensure image has batch dimension
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=[0, -1])  # Add batch and channel dims
            elif len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)  # Add batch dim
                
            # Validate image dimensions match model input
            expected_shape = (None,) + self.input_size
            if image.shape[1:] != self.input_size:
                raise ValidationError(
                    f"Image shape {image.shape[1:]} doesn't match model input {self.input_size}"
                )
                
            prediction = self.model.predict(image, verbose=0)
            
            if prediction is None or len(prediction) == 0:
                raise ModelValidationError("Model prediction failed")
                
            result = prediction[0, :, :, 0]  # Remove batch and channel dimensions
            
            # Validate prediction output
            if not np.all(np.isfinite(result)):
                raise ModelValidationError("Prediction contains invalid values (NaN or Inf)")
                
            logger.debug(f"Prediction successful, mask shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting mask: {str(e)}")
            if isinstance(e, (ModelValidationError, ValidationError)):
                raise
            raise ModelValidationError(f"Failed to predict mask: {str(e)}")
    
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


def main() -> None:
    """
    Main function to demonstrate U-Net segmentation with error handling.
    
    Raises:
        SystemExit: If demonstration fails
    """
    try:
        logger.info("=== Lung Segmentation U-Net ===")
        
        # Create lung masks for the dataset (this would take time in practice)
        logger.info("Step 1: Creating lung masks for dataset...")
        # create_lung_masks_for_dataset('chest_xray', 'lung_masks')
        
        # Initialize U-Net with validation
        logger.info("Step 2: Building U-Net model...")
        input_size = validate_input_shape((512, 512, 1))
        unet = LungSegmentationUNet(input_size=input_size)
        model = unet.compile_model()
        
        logger.info("U-Net Model Summary:")
        model.summary()
        
        logger.info(f"Total parameters: {model.count_params():,}")
        
        # For demonstration, create synthetic data with validation
        logger.info("Step 3: Creating synthetic training data...")
        num_samples = validate_batch_size(100, min_size=1, max_size=1000)
        
        # Create synthetic data
        X_train = np.random.random((num_samples, 512, 512, 1))
        y_train = np.random.randint(0, 2, (num_samples, 512, 512, 1)).astype(np.float32)
        
        X_val = np.random.random((20, 512, 512, 1))
        y_val = np.random.randint(0, 2, (20, 512, 512, 1)).astype(np.float32)
        
        # Validate synthetic data
        if X_train.size == 0 or y_train.size == 0:
            raise ValidationError("Failed to create synthetic training data")
        
        logger.info("Step 4: Training U-Net (demo with synthetic data)...")
        
        # Get callbacks with validation
        callbacks = unet.get_callbacks()
        
        # Validate training parameters
        epochs = validate_epochs(5, min_epochs=1, max_epochs=100)
        batch_size = validate_batch_size(4, min_size=1, max_size=32)
        
        # Train model (shortened for demo)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        if history is None:
            raise ModelValidationError("Training failed - no history returned")
        
        logger.info("Step 5: Testing prediction...")
        test_image = X_val[0:1]
        test_mask = y_val[0]
        prediction = unet.predict_mask(test_image)
        
        # Validate prediction output
        if prediction is None or prediction.size == 0:
            raise ModelValidationError("Prediction failed")
        
        # Visualize results with error handling
        try:
            unet.visualize_prediction(
                test_image[0], 
                test_mask[:,:,0], 
                prediction,
                save_path='lung_segmentation_demo.png'
            )
        except Exception as viz_error:
            logger.warning(f"Visualization failed: {str(viz_error)}")
        
        logger.info("✅ Lung segmentation U-Net demo completed!")
        logger.info("Next step: Integrate with pneumonia classification pipeline")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()