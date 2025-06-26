"""
Configuration loader utility for the Pneumonia CNN project.
Handles loading, saving, and managing configuration files.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from datetime import datetime

from config_schema import Config
from validation_utils import validate_file_exists, validate_directory_exists, ValidationError

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class ConfigLoader:
    """Configuration loader and manager."""
    
    SUPPORTED_FORMATS = ['.yaml', '.yml', '.json']
    DEFAULT_CONFIG_NAME = 'config.yaml'
    
    def __init__(self, config_dir: str = 'configs'):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_config(self, config_path: Optional[str] = None, 
                   config_name: Optional[str] = None) -> Config:
        """
        Load configuration from file.
        
        Args:
            config_path: Full path to configuration file
            config_name: Name of config file in config_dir
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigurationError: If loading or validation fails
        """
        try:
            # Determine config file path
            if config_path:
                file_path = Path(config_path)
            elif config_name:
                file_path = self.config_dir / config_name
            else:
                file_path = self.config_dir / self.DEFAULT_CONFIG_NAME
                
            # Check if file exists
            if not file_path.exists():
                logger.warning(f"Config file not found: {file_path}")
                return self._create_default_config(file_path)
                
            # Validate file format
            if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ConfigurationError(
                    f"Unsupported config format: {file_path.suffix}. "
                    f"Supported formats: {self.SUPPORTED_FORMATS}"
                )
                
            # Load configuration data
            logger.info(f"Loading configuration from: {file_path}")
            config_dict = self._load_file(file_path)
            
            # Create configuration object
            config = Config.from_dict(config_dict)
            
            # Validate configuration
            config.validate()
            
            logger.info("Configuration loaded and validated successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise ConfigurationError(f"Configuration loading failed: {str(e)}")
    
    def save_config(self, config: Config, file_path: str, 
                   format: str = 'yaml') -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration object to save
            file_path: Path where to save the configuration
            format: File format ('yaml' or 'json')
            
        Raises:
            ConfigurationError: If saving fails
        """
        try:
            # Validate configuration before saving
            config.validate()
            
            # Convert to dictionary
            config_dict = config.to_dict()
            
            # Add metadata
            config_dict['_metadata'] = {
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'format': format
            }
            
            # Save file
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'yaml':
                with open(file_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported format: {format}")
                
            logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise ConfigurationError(f"Configuration saving failed: {str(e)}")
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration file based on format."""
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported file format: {file_path.suffix}")
                    
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML parsing error: {str(e)}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"JSON parsing error: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"File reading error: {str(e)}")
    
    def _create_default_config(self, file_path: Path) -> Config:
        """Create and save default configuration."""
        logger.info("Creating default configuration")
        
        config = Config()
        
        # Save default config for future use
        try:
            self.save_config(config, str(file_path))
            logger.info(f"Default configuration saved to: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to save default config: {str(e)}")
            
        return config
    
    def list_configs(self) -> List[str]:
        """List available configuration files."""
        try:
            config_files = []
            for file_path in self.config_dir.glob('*'):
                if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    config_files.append(file_path.name)
            return sorted(config_files)
        except Exception as e:
            logger.warning(f"Failed to list configs: {str(e)}")
            return []
    
    def create_experiment_config(self, base_config: Config, 
                               experiment_name: str,
                               overrides: Dict[str, Any]) -> Config:
        """
        Create experiment-specific configuration with overrides.
        
        Args:
            base_config: Base configuration to start from
            experiment_name: Name of the experiment
            overrides: Dictionary of parameter overrides
            
        Returns:
            New configuration with overrides applied
        """
        try:
            # Start with base config
            config_dict = base_config.to_dict()
            
            # Apply overrides
            config_dict = self._apply_overrides(config_dict, overrides)
            
            # Set experiment name
            config_dict['experiment_name'] = experiment_name
            config_dict['description'] = f"Experiment: {experiment_name}"
            
            # Create new config
            new_config = Config.from_dict(config_dict)
            new_config.validate()
            
            return new_config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create experiment config: {str(e)}")
    
    def _apply_overrides(self, config_dict: Dict[str, Any], 
                        overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply nested overrides to configuration dictionary."""
        for key, value in overrides.items():
            # Handle nested keys like "model.learning_rate"
            if '.' in key:
                keys = key.split('.')
                current = config_dict
                
                # Navigate to the nested location
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # Set the value
                current[keys[-1]] = value
            else:
                config_dict[key] = value
                
        return config_dict


class ConfigManager:
    """High-level configuration manager with common operations."""
    
    def __init__(self, config_dir: str = 'configs'):
        """Initialize configuration manager."""
        self.loader = ConfigLoader(config_dir)
        self._current_config: Optional[Config] = None
    
    @property
    def config(self) -> Config:
        """Get current configuration, loading default if needed."""
        if self._current_config is None:
            self._current_config = self.loader.load_config()
        return self._current_config
    
    def load(self, config_path: Optional[str] = None) -> Config:
        """Load and set current configuration."""
        self._current_config = self.loader.load_config(config_path)
        return self._current_config
    
    def save_current(self, file_path: str) -> None:
        """Save current configuration to file."""
        if self._current_config is None:
            raise ConfigurationError("No current configuration to save")
        self.loader.save_config(self._current_config, file_path)
    
    def create_experiment(self, experiment_name: str, 
                         overrides: Dict[str, Any]) -> Config:
        """Create experiment configuration and set as current."""
        base_config = self.config
        experiment_config = self.loader.create_experiment_config(
            base_config, experiment_name, overrides
        )
        
        # Save experiment config
        experiment_file = f"experiment_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        experiment_path = self.loader.config_dir / experiment_file
        self.loader.save_config(experiment_config, str(experiment_path))
        
        # Set as current config
        self._current_config = experiment_config
        return experiment_config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary for easy parameter passing."""
        return self.config.model.to_dict()
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration as dictionary."""
        return self.config.training.to_dict()
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration as dictionary."""
        return self.config.data.to_dict()
    
    def update_config(self, **kwargs) -> None:
        """Update current configuration with keyword arguments."""
        if self._current_config is None:
            self._current_config = self.loader.load_config()
            
        # Apply updates
        for key, value in kwargs.items():
            if hasattr(self._current_config, key):
                setattr(self._current_config, key, value)
            else:
                # Try to find in subsections
                for section_name in ['model', 'training', 'data', 'paths', 'logging']:
                    section = getattr(self._current_config, section_name)
                    if hasattr(section, key):
                        setattr(section, key, value)
                        break
                else:
                    logger.warning(f"Unknown configuration parameter: {key}")
        
        # Validate after updates
        self._current_config.validate()


# Global configuration manager instance
config_manager = ConfigManager()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config_manager.config

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file."""
    return config_manager.load(config_path)

def create_experiment_config(experiment_name: str, **overrides) -> Config:
    """Create experiment configuration with overrides."""
    return config_manager.create_experiment(experiment_name, overrides)