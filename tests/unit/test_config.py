"""
Unit tests for configuration system.
"""
import pytest
from pathlib import Path
import tempfile
import yaml
import json
from typing import Dict, Any

from src.config.config_schema import (
    ModelConfig, TrainingConfig, DataConfig, LoggingConfig,
    ExperimentConfig, HardwareConfig
)
from src.config.config_loader import ConfigLoader, ConfigValidationError


class TestConfigSchema:
    """Test configuration data classes and validation."""
    
    @pytest.mark.unit
    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config = ModelConfig()
        assert config.architecture == "standard"
        assert config.input_shape == [128, 128, 3]
        assert config.num_classes == 2
        assert config.dropout_rate == 0.5
        assert config.learning_rate == 0.001
    
    @pytest.mark.unit
    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            architecture="unet",
            input_shape=[256, 256, 1],
            learning_rate=0.01,
            optimizer="sgd"
        )
        assert config.architecture == "unet"
        assert config.input_shape == [256, 256, 1]
        assert config.learning_rate == 0.01
        assert config.optimizer == "sgd"
    
    @pytest.mark.unit
    def test_training_config_validation(self):
        """Test TrainingConfig validation."""
        # Valid config
        config = TrainingConfig(batch_size=32, epochs=10)
        assert config.batch_size == 32
        assert config.epochs == 10
        
        # Test with early stopping
        config = TrainingConfig(
            early_stopping={
                "enabled": True,
                "monitor": "val_accuracy",
                "patience": 5
            }
        )
        assert config.early_stopping["enabled"] is True
        assert config.early_stopping["patience"] == 5
    
    @pytest.mark.unit
    def test_data_config_paths(self):
        """Test DataConfig path handling."""
        config = DataConfig(
            train_dir="data/train",
            test_dir="data/test",
            image_size=[224, 224]
        )
        assert config.train_dir == "data/train"
        assert config.test_dir == "data/test"
        assert config.image_size == [224, 224]
    
    @pytest.mark.unit
    def test_experiment_config_complete(self):
        """Test complete ExperimentConfig."""
        config = ExperimentConfig(
            experiment_name="test_exp",
            model=ModelConfig(learning_rate=0.01),
            training=TrainingConfig(epochs=5),
            data=DataConfig(),
            logging=LoggingConfig(level="DEBUG"),
            hardware=HardwareConfig(gpu_memory_growth=False)
        )
        assert config.experiment_name == "test_exp"
        assert config.model.learning_rate == 0.01
        assert config.training.epochs == 5
        assert config.logging.level == "DEBUG"
        assert config.hardware.gpu_memory_growth is False


class TestConfigLoader:
    """Test configuration loading and validation."""
    
    @pytest.mark.unit
    def test_load_from_yaml(self, temp_config_file):
        """Test loading configuration from YAML file."""
        loader = ConfigLoader()
        config = loader.load_config(str(temp_config_file))
        
        assert isinstance(config, ExperimentConfig)
        assert config.experiment_name == "test_experiment"
        assert config.model.learning_rate == 0.001
        assert config.training.batch_size == 32
    
    @pytest.mark.unit
    def test_load_from_json(self, sample_config_dict):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config_dict, f)
            temp_path = Path(f.name)
        
        try:
            loader = ConfigLoader()
            config = loader.load_config(str(temp_path))
            
            assert isinstance(config, ExperimentConfig)
            assert config.experiment_name == "test_experiment"
        finally:
            temp_path.unlink()
    
    @pytest.mark.unit
    def test_load_from_dict(self, sample_config_dict):
        """Test loading configuration from dictionary."""
        loader = ConfigLoader()
        config = loader.load_from_dict(sample_config_dict)
        
        assert isinstance(config, ExperimentConfig)
        assert config.experiment_name == "test_experiment"
        assert config.model.architecture == "standard"
    
    @pytest.mark.unit
    def test_validate_config_valid(self, sample_config_dict):
        """Test validation of valid configuration."""
        loader = ConfigLoader()
        # Should not raise any exception
        loader.validate_config(sample_config_dict)
    
    @pytest.mark.unit
    def test_validate_config_invalid_type(self):
        """Test validation with invalid types."""
        loader = ConfigLoader()
        invalid_config = {
            "experiment_name": "test",
            "model": {
                "learning_rate": "not_a_number"  # Should be float
            }
        }
        
        with pytest.raises(ConfigValidationError):
            loader.validate_config(invalid_config)
    
    @pytest.mark.unit
    def test_validate_config_missing_required(self):
        """Test validation with missing required fields."""
        loader = ConfigLoader()
        invalid_config = {
            # Missing experiment_name
            "model": {"learning_rate": 0.001}
        }
        
        with pytest.raises(ConfigValidationError):
            loader.validate_config(invalid_config)
    
    @pytest.mark.unit
    def test_merge_configs(self):
        """Test configuration merging."""
        loader = ConfigLoader()
        base_config = {
            "experiment_name": "base",
            "model": {"learning_rate": 0.001},
            "training": {"epochs": 10}
        }
        override_config = {
            "experiment_name": "override",
            "model": {"learning_rate": 0.01}
        }
        
        merged = loader.merge_configs(base_config, override_config)
        
        assert merged["experiment_name"] == "override"
        assert merged["model"]["learning_rate"] == 0.01
        assert merged["training"]["epochs"] == 10  # Preserved from base
    
    @pytest.mark.unit
    def test_save_config(self):
        """Test saving configuration to file."""
        loader = ConfigLoader()
        config = ExperimentConfig(
            experiment_name="save_test",
            model=ModelConfig(learning_rate=0.005)
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            loader.save_config(config, str(temp_path))
            
            # Load it back and verify
            loaded_config = loader.load_config(str(temp_path))
            assert loaded_config.experiment_name == "save_test"
            assert loaded_config.model.learning_rate == 0.005
        finally:
            temp_path.unlink()
    
    @pytest.mark.unit
    def test_get_config_value(self):
        """Test getting nested configuration values."""
        loader = ConfigLoader()
        config = ExperimentConfig(
            experiment_name="test",
            model=ModelConfig(learning_rate=0.001),
            training=TrainingConfig(
                early_stopping={"patience": 5}
            )
        )
        
        # Test getting nested values
        lr = loader.get_config_value(config, "model.learning_rate")
        assert lr == 0.001
        
        patience = loader.get_config_value(config, "training.early_stopping.patience")
        assert patience == 5
        
        # Test default value
        missing = loader.get_config_value(config, "model.missing_key", default="default")
        assert missing == "default"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_path", [
        "nonexistent.yaml",
        "invalid.txt",
        ""
    ])
    def test_load_invalid_path(self, invalid_path):
        """Test loading from invalid file paths."""
        loader = ConfigLoader()
        with pytest.raises((FileNotFoundError, ConfigValidationError)):
            loader.load_config(invalid_path)