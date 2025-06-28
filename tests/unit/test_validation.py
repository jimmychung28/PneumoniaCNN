"""
Unit tests for validation utilities.
"""
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile

from src.utils.validation_utils import (
    validate_image_array, validate_file_path, validate_directory_path,
    validate_model_architecture, validate_batch_size, validate_learning_rate,
    validate_epochs, validate_image_dimensions, validate_config_type,
    ValidationError
)


class TestImageValidation:
    """Test image validation functions."""
    
    @pytest.mark.unit
    def test_validate_image_array_valid(self):
        """Test validation of valid image arrays."""
        # Valid 3-channel image
        img = np.random.rand(128, 128, 3).astype(np.float32)
        result = validate_image_array(img)
        assert result is True
        
        # Valid grayscale image
        img = np.random.rand(128, 128, 1).astype(np.float32)
        result = validate_image_array(img)
        assert result is True
        
        # Valid batch of images
        batch = np.random.rand(32, 128, 128, 3).astype(np.float32)
        result = validate_image_array(batch, is_batch=True)
        assert result is True
    
    @pytest.mark.unit
    def test_validate_image_array_invalid_shape(self):
        """Test validation with invalid image shapes."""
        # Wrong number of dimensions
        with pytest.raises(ValidationError, match="dimensions"):
            validate_image_array(np.random.rand(128, 128))
        
        # Invalid channel count
        with pytest.raises(ValidationError, match="channels"):
            validate_image_array(np.random.rand(128, 128, 5))
        
        # Wrong dimensions for batch
        with pytest.raises(ValidationError, match="batch"):
            validate_image_array(np.random.rand(128, 128, 3), is_batch=True)
    
    @pytest.mark.unit
    def test_validate_image_array_invalid_dtype(self):
        """Test validation with invalid data types."""
        # Integer type when float expected
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.int32)
        with pytest.raises(ValidationError, match="dtype"):
            validate_image_array(img)
        
        # Complex numbers
        img = np.random.rand(128, 128, 3).astype(np.complex64)
        with pytest.raises(ValidationError, match="dtype"):
            validate_image_array(img)
    
    @pytest.mark.unit
    def test_validate_image_array_invalid_values(self):
        """Test validation with out-of-range values."""
        # Values > 1
        img = np.ones((128, 128, 3)) * 2.0
        with pytest.raises(ValidationError, match="range"):
            validate_image_array(img)
        
        # Negative values
        img = np.ones((128, 128, 3)) * -0.5
        with pytest.raises(ValidationError, match="range"):
            validate_image_array(img)
        
        # NaN values
        img = np.full((128, 128, 3), np.nan)
        with pytest.raises(ValidationError, match="NaN"):
            validate_image_array(img)


class TestPathValidation:
    """Test file and directory path validation."""
    
    @pytest.mark.unit
    def test_validate_file_path_valid(self):
        """Test validation of valid file paths."""
        with tempfile.NamedTemporaryFile() as tmp:
            result = validate_file_path(tmp.name)
            assert result is True
            
            # Test with Path object
            result = validate_file_path(Path(tmp.name))
            assert result is True
    
    @pytest.mark.unit
    def test_validate_file_path_invalid(self):
        """Test validation of invalid file paths."""
        # Non-existent file
        with pytest.raises(ValidationError, match="does not exist"):
            validate_file_path("/nonexistent/file.txt")
        
        # Directory instead of file
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValidationError, match="not a file"):
                validate_file_path(tmpdir)
        
        # Invalid type
        with pytest.raises(ValidationError, match="must be string or Path"):
            validate_file_path(123)
    
    @pytest.mark.unit
    def test_validate_directory_path_valid(self):
        """Test validation of valid directory paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_directory_path(tmpdir)
            assert result is True
            
            # Test with Path object
            result = validate_directory_path(Path(tmpdir))
            assert result is True
    
    @pytest.mark.unit
    def test_validate_directory_path_invalid(self):
        """Test validation of invalid directory paths."""
        # Non-existent directory
        with pytest.raises(ValidationError, match="does not exist"):
            validate_directory_path("/nonexistent/directory")
        
        # File instead of directory
        with tempfile.NamedTemporaryFile() as tmp:
            with pytest.raises(ValidationError, match="not a directory"):
                validate_directory_path(tmp.name)


class TestModelValidation:
    """Test model architecture validation."""
    
    @pytest.mark.unit
    def test_validate_model_architecture_valid(self):
        """Test validation of valid model architectures."""
        valid_architectures = ["standard", "unet", "resnet50", "two_stage"]
        for arch in valid_architectures:
            result = validate_model_architecture(arch)
            assert result is True
    
    @pytest.mark.unit
    def test_validate_model_architecture_invalid(self):
        """Test validation of invalid model architectures."""
        # Unknown architecture
        with pytest.raises(ValidationError, match="Unknown model architecture"):
            validate_model_architecture("unknown_model")
        
        # Invalid type
        with pytest.raises(ValidationError, match="must be a string"):
            validate_model_architecture(123)
        
        # Empty string
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_model_architecture("")


class TestHyperparameterValidation:
    """Test hyperparameter validation."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size,expected", [
        (1, True),
        (32, True),
        (128, True),
        (1024, True),
    ])
    def test_validate_batch_size_valid(self, batch_size, expected):
        """Test validation of valid batch sizes."""
        result = validate_batch_size(batch_size)
        assert result == expected
    
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size,error_match", [
        (0, "must be positive"),
        (-1, "must be positive"),
        (1.5, "must be an integer"),
        ("32", "must be an integer"),
        (2048, "too large"),
    ])
    def test_validate_batch_size_invalid(self, batch_size, error_match):
        """Test validation of invalid batch sizes."""
        with pytest.raises(ValidationError, match=error_match):
            validate_batch_size(batch_size)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("lr,expected", [
        (0.001, True),
        (0.1, True),
        (1e-6, True),
        (0.5, True),
    ])
    def test_validate_learning_rate_valid(self, lr, expected):
        """Test validation of valid learning rates."""
        result = validate_learning_rate(lr)
        assert result == expected
    
    @pytest.mark.unit
    @pytest.mark.parametrize("lr,error_match", [
        (0, "must be positive"),
        (-0.001, "must be positive"),
        (2.0, "too large"),
        ("0.001", "must be a float"),
        (np.nan, "cannot be NaN"),
    ])
    def test_validate_learning_rate_invalid(self, lr, error_match):
        """Test validation of invalid learning rates."""
        with pytest.raises(ValidationError, match=error_match):
            validate_learning_rate(lr)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("epochs,expected", [
        (1, True),
        (100, True),
        (500, True),
    ])
    def test_validate_epochs_valid(self, epochs, expected):
        """Test validation of valid epoch counts."""
        result = validate_epochs(epochs)
        assert result == expected
    
    @pytest.mark.unit
    @pytest.mark.parametrize("epochs,error_match", [
        (0, "must be positive"),
        (-10, "must be positive"),
        (1.5, "must be an integer"),
        ("100", "must be an integer"),
        (10001, "too large"),
    ])
    def test_validate_epochs_invalid(self, epochs, error_match):
        """Test validation of invalid epoch counts."""
        with pytest.raises(ValidationError, match=error_match):
            validate_epochs(epochs)


class TestImageDimensionValidation:
    """Test image dimension validation."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("dims,expected", [
        ([128, 128], True),
        ([224, 224], True),
        ([256, 256, 3], True),
        ([512, 512, 1], True),
    ])
    def test_validate_image_dimensions_valid(self, dims, expected):
        """Test validation of valid image dimensions."""
        result = validate_image_dimensions(dims)
        assert result == expected
    
    @pytest.mark.unit
    @pytest.mark.parametrize("dims,error_match", [
        ([128], "must have 2 or 3 elements"),
        ([128, 128, 3, 1], "must have 2 or 3 elements"),
        ([0, 128], "must be positive"),
        ([128, -128], "must be positive"),
        ([32, 32], "too small"),
        ([5000, 5000], "too large"),
        ("128,128", "must be a list"),
    ])
    def test_validate_image_dimensions_invalid(self, dims, error_match):
        """Test validation of invalid image dimensions."""
        with pytest.raises(ValidationError, match=error_match):
            validate_image_dimensions(dims)


class TestConfigTypeValidation:
    """Test configuration type validation."""
    
    @pytest.mark.unit
    def test_validate_config_type_valid(self):
        """Test validation of valid configuration types."""
        # Dictionary
        config = {"key": "value", "nested": {"key2": "value2"}}
        result = validate_config_type(config, dict)
        assert result is True
        
        # String
        result = validate_config_type("string_value", str)
        assert result is True
        
        # List
        result = validate_config_type([1, 2, 3], list)
        assert result is True
    
    @pytest.mark.unit
    def test_validate_config_type_invalid(self):
        """Test validation of invalid configuration types."""
        # Wrong type
        with pytest.raises(ValidationError, match="Expected dict"):
            validate_config_type("not_a_dict", dict)
        
        # None value
        with pytest.raises(ValidationError, match="cannot be None"):
            validate_config_type(None, dict)
        
        # Empty required dict
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_config_type({}, dict, allow_empty=False)