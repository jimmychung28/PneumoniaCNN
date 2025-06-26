"""
Validation utilities for the Pneumonia CNN project.
Provides common validation functions and error handling utilities.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Union, Optional, List
import numpy as np
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class FileValidationError(ValidationError):
    """Custom exception for file validation errors."""
    pass

class ImageValidationError(ValidationError):
    """Custom exception for image validation errors."""
    pass

class ModelValidationError(ValidationError):
    """Custom exception for model validation errors."""
    pass

def validate_directory_exists(directory_path: str, create_if_missing: bool = False) -> str:
    """
    Validate that a directory exists.
    
    Args:
        directory_path: Path to the directory
        create_if_missing: If True, create the directory if it doesn't exist
        
    Returns:
        Validated directory path
        
    Raises:
        FileValidationError: If directory doesn't exist and create_if_missing is False
    """
    try:
        path = Path(directory_path)
        
        if not path.exists():
            if create_if_missing:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory_path}")
            else:
                raise FileValidationError(f"Directory does not exist: {directory_path}")
        
        if not path.is_dir():
            raise FileValidationError(f"Path is not a directory: {directory_path}")
            
        return str(path.resolve())
        
    except Exception as e:
        if isinstance(e, FileValidationError):
            raise
        raise FileValidationError(f"Error validating directory {directory_path}: {str(e)}")

def validate_file_exists(file_path: str, extensions: Optional[List[str]] = None) -> str:
    """
    Validate that a file exists and has the correct extension.
    
    Args:
        file_path: Path to the file
        extensions: List of allowed file extensions (e.g., ['.jpg', '.png'])
        
    Returns:
        Validated file path
        
    Raises:
        FileValidationError: If file doesn't exist or has wrong extension
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            raise FileValidationError(f"File does not exist: {file_path}")
            
        if not path.is_file():
            raise FileValidationError(f"Path is not a file: {file_path}")
            
        if extensions:
            if path.suffix.lower() not in [ext.lower() for ext in extensions]:
                raise FileValidationError(
                    f"File has invalid extension. Expected {extensions}, got {path.suffix}"
                )
                
        return str(path.resolve())
        
    except Exception as e:
        if isinstance(e, FileValidationError):
            raise
        raise FileValidationError(f"Error validating file {file_path}: {str(e)}")

def validate_image_file(image_path: str, max_size_mb: float = 50.0) -> str:
    """
    Validate that an image file is readable and within size limits.
    
    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in megabytes
        
    Returns:
        Validated image path
        
    Raises:
        ImageValidationError: If image is invalid or too large
    """
    try:
        # Validate file exists with image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        validated_path = validate_file_exists(image_path, valid_extensions)
        
        # Check file size
        file_size_mb = os.path.getsize(validated_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ImageValidationError(
                f"Image file too large: {file_size_mb:.1f}MB > {max_size_mb}MB"
            )
        
        # Try to load image to verify it's valid
        try:
            with Image.open(validated_path) as img:
                img.verify()  # Verify image integrity
        except Exception as img_error:
            raise ImageValidationError(f"Invalid image file: {str(img_error)}")
            
        return validated_path
        
    except Exception as e:
        if isinstance(e, (FileValidationError, ImageValidationError)):
            raise
        raise ImageValidationError(f"Error validating image {image_path}: {str(e)}")

def validate_input_shape(input_shape: Union[Tuple, List], expected_dims: int = 3) -> Tuple:
    """
    Validate input shape for neural network models.
    
    Args:
        input_shape: Input shape tuple/list
        expected_dims: Expected number of dimensions
        
    Returns:
        Validated input shape as tuple
        
    Raises:
        ModelValidationError: If input shape is invalid
    """
    try:
        if not isinstance(input_shape, (tuple, list)):
            raise ModelValidationError("input_shape must be a tuple or list")
            
        if len(input_shape) != expected_dims:
            raise ModelValidationError(
                f"input_shape must have {expected_dims} dimensions, got {len(input_shape)}"
            )
            
        for dim in input_shape:
            if not isinstance(dim, int) or dim <= 0:
                raise ModelValidationError(
                    f"All dimensions must be positive integers, got {input_shape}"
                )
                
        return tuple(input_shape)
        
    except Exception as e:
        if isinstance(e, ModelValidationError):
            raise
        raise ModelValidationError(f"Error validating input shape: {str(e)}")

def validate_learning_rate(learning_rate: float) -> float:
    """
    Validate learning rate parameter.
    
    Args:
        learning_rate: Learning rate value
        
    Returns:
        Validated learning rate
        
    Raises:
        ModelValidationError: If learning rate is invalid
    """
    try:
        if not isinstance(learning_rate, (int, float)):
            raise ModelValidationError("learning_rate must be a number")
            
        if learning_rate <= 0 or learning_rate >= 1:
            raise ModelValidationError(
                f"learning_rate must be between 0 and 1, got {learning_rate}"
            )
            
        return float(learning_rate)
        
    except Exception as e:
        if isinstance(e, ModelValidationError):
            raise
        raise ModelValidationError(f"Error validating learning rate: {str(e)}")

def validate_batch_size(batch_size: int, min_size: int = 1, max_size: int = 1024) -> int:
    """
    Validate batch size parameter.
    
    Args:
        batch_size: Batch size value
        min_size: Minimum allowed batch size
        max_size: Maximum allowed batch size
        
    Returns:
        Validated batch size
        
    Raises:
        ModelValidationError: If batch size is invalid
    """
    try:
        if not isinstance(batch_size, int):
            raise ModelValidationError("batch_size must be an integer")
            
        if batch_size < min_size or batch_size > max_size:
            raise ModelValidationError(
                f"batch_size must be between {min_size} and {max_size}, got {batch_size}"
            )
            
        return batch_size
        
    except Exception as e:
        if isinstance(e, ModelValidationError):
            raise
        raise ModelValidationError(f"Error validating batch size: {str(e)}")

def validate_epochs(epochs: int, min_epochs: int = 1, max_epochs: int = 1000) -> int:
    """
    Validate epochs parameter.
    
    Args:
        epochs: Number of training epochs
        min_epochs: Minimum allowed epochs
        max_epochs: Maximum allowed epochs
        
    Returns:
        Validated epochs
        
    Raises:
        ModelValidationError: If epochs is invalid
    """
    try:
        if not isinstance(epochs, int):
            raise ModelValidationError("epochs must be an integer")
            
        if epochs < min_epochs or epochs > max_epochs:
            raise ModelValidationError(
                f"epochs must be between {min_epochs} and {max_epochs}, got {epochs}"
            )
            
        return epochs
        
    except Exception as e:
        if isinstance(e, ModelValidationError):
            raise
        raise ModelValidationError(f"Error validating epochs: {str(e)}")

def validate_dataset_structure(dataset_path: str, expected_classes: List[str]) -> dict:
    """
    Validate dataset directory structure.
    
    Args:
        dataset_path: Path to dataset directory
        expected_classes: List of expected class directories
        
    Returns:
        Dictionary with class paths and image counts
        
    Raises:
        FileValidationError: If dataset structure is invalid
    """
    try:
        dataset_path = validate_directory_exists(dataset_path)
        class_info = {}
        
        for class_name in expected_classes:
            class_path = os.path.join(dataset_path, class_name)
            
            if not os.path.exists(class_path):
                raise FileValidationError(f"Missing class directory: {class_path}")
                
            if not os.path.isdir(class_path):
                raise FileValidationError(f"Class path is not a directory: {class_path}")
            
            # Count images in class directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = [
                f for f in os.listdir(class_path)
                if os.path.splitext(f.lower())[1] in image_extensions
            ]
            
            if len(image_files) == 0:
                logger.warning(f"No images found in class directory: {class_path}")
            
            class_info[class_name] = {
                'path': class_path,
                'image_count': len(image_files),
                'images': image_files
            }
            
        return class_info
        
    except Exception as e:
        if isinstance(e, FileValidationError):
            raise
        raise FileValidationError(f"Error validating dataset structure: {str(e)}")

def safe_load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Safely load and validate an image file.
    
    Args:
        image_path: Path to the image file
        target_size: Optional target size for resizing (width, height)
        
    Returns:
        Loaded image as numpy array
        
    Raises:
        ImageValidationError: If image cannot be loaded
    """
    try:
        # Validate image file
        validated_path = validate_image_file(image_path)
        
        # Try loading with OpenCV first
        image = cv2.imread(validated_path)
        
        if image is None:
            # Fallback to PIL
            try:
                with Image.open(validated_path) as pil_image:
                    image = np.array(pil_image)
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        # Convert RGB to BGR for consistency with OpenCV
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as pil_error:
                raise ImageValidationError(f"Cannot load image with PIL: {str(pil_error)}")
        
        if image is None:
            raise ImageValidationError(f"Failed to load image: {image_path}")
            
        # Resize if target size specified
        if target_size:
            try:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            except Exception as resize_error:
                raise ImageValidationError(f"Error resizing image: {str(resize_error)}")
                
        return image
        
    except Exception as e:
        if isinstance(e, ImageValidationError):
            raise
        raise ImageValidationError(f"Error loading image {image_path}: {str(e)}")

def validate_model_save_path(save_path: str, create_dir: bool = True) -> str:
    """
    Validate and prepare model save path.
    
    Args:
        save_path: Path where model will be saved
        create_dir: Whether to create parent directory if it doesn't exist
        
    Returns:
        Validated save path
        
    Raises:
        FileValidationError: If save path is invalid
    """
    try:
        path = Path(save_path)
        
        # Check if it's a valid model file extension
        valid_extensions = ['.h5', '.keras', '.pb', '.hdf5']
        if path.suffix.lower() not in valid_extensions:
            logger.warning(f"Unusual model file extension: {path.suffix}")
        
        # Validate parent directory
        parent_dir = path.parent
        if create_dir:
            validate_directory_exists(str(parent_dir), create_if_missing=True)
        else:
            validate_directory_exists(str(parent_dir))
            
        return str(path)
        
    except Exception as e:
        if isinstance(e, FileValidationError):
            raise
        raise FileValidationError(f"Error validating model save path: {str(e)}")