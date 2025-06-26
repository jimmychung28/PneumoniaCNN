"""
Configuration schema and validation for the Pneumonia CNN project.
Defines the structure and validation rules for all configuration parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Model architecture
    input_shape: Tuple[int, int, int] = (128, 128, 3)
    learning_rate: float = 0.0001
    architecture: str = "standard"  # "standard", "unet", "two_stage"
    
    # CNN specific
    filters_base: int = 32  # Base number of filters for CNN layers
    depth: int = 4  # Number of conv blocks
    dropout_rate: float = 0.25  # Dropout rate for conv layers
    dense_dropout_rate: float = 0.5  # Dropout rate for dense layers
    
    # U-Net specific
    unet_input_size: Tuple[int, int, int] = (512, 512, 1)
    unet_filters: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    unet_dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.2, 0.2, 0.3])
    
    # Two-stage specific
    classification_input_size: Tuple[int, int, int] = (224, 224, 3)
    backbone: str = "resnet50"  # "resnet50", "densenet121", "efficientnet"
    
    def validate(self) -> None:
        """Validate model configuration parameters."""
        # Validate input shape
        if len(self.input_shape) != 3:
            raise ValueError("input_shape must have exactly 3 dimensions")
        if any(dim <= 0 for dim in self.input_shape):
            raise ValueError("All input_shape dimensions must be positive")
            
        # Validate learning rate
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
            
        # Validate architecture
        valid_architectures = ["standard", "unet", "two_stage"]
        if self.architecture not in valid_architectures:
            raise ValueError(f"architecture must be one of {valid_architectures}")
            
        # Validate CNN parameters
        if self.filters_base <= 0:
            raise ValueError("filters_base must be positive")
        if self.depth <= 0 or self.depth > 10:
            raise ValueError("depth must be between 1 and 10")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if not 0 <= self.dense_dropout_rate <= 1:
            raise ValueError("dense_dropout_rate must be between 0 and 1")
            
        # Validate U-Net parameters
        if len(self.unet_input_size) != 3:
            raise ValueError("unet_input_size must have exactly 3 dimensions")
        if self.unet_input_size[2] != 1:
            raise ValueError("U-Net expects grayscale input (channels=1)")
        if len(self.unet_filters) != len(self.unet_dropout_rates):
            raise ValueError("unet_filters and unet_dropout_rates must have same length")
        if any(rate < 0 or rate > 1 for rate in self.unet_dropout_rates):
            raise ValueError("All unet_dropout_rates must be between 0 and 1")
            
        # Validate two-stage parameters
        if len(self.classification_input_size) != 3:
            raise ValueError("classification_input_size must have exactly 3 dimensions")
        valid_backbones = ["resnet50", "densenet121", "efficientnet"]
        if self.backbone not in valid_backbones:
            raise ValueError(f"backbone must be one of {valid_backbones}")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Basic training parameters
    batch_size: int = 32
    epochs: int = 50
    validation_split: float = 0.2
    
    # Optimizer settings
    optimizer: str = "adam"  # "adam", "sgd", "rmsprop"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Learning rate scheduling
    use_lr_schedule: bool = True
    lr_schedule_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "exponential"
    lr_schedule_params: Dict[str, Any] = field(default_factory=lambda: {
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-7
    })
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"
    
    # Class weights
    use_class_weights: bool = True
    class_weights: Optional[Dict[int, float]] = None
    
    # Mixed precision training
    use_mixed_precision: bool = False
    
    # Gradient clipping
    gradient_clip_value: Optional[float] = None
    gradient_clip_norm: Optional[float] = None
    
    def validate(self) -> None:
        """Validate training configuration parameters."""
        # Validate basic parameters
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
            
        # Validate optimizer
        valid_optimizers = ["adam", "sgd", "rmsprop"]
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}")
            
        # Validate learning rate schedule
        valid_lr_schedules = ["reduce_on_plateau", "cosine", "exponential"]
        if self.lr_schedule_type not in valid_lr_schedules:
            raise ValueError(f"lr_schedule_type must be one of {valid_lr_schedules}")
            
        # Validate early stopping
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        valid_monitors = ["val_loss", "val_accuracy", "val_auc", "loss", "accuracy"]
        if self.early_stopping_monitor not in valid_monitors:
            raise ValueError(f"early_stopping_monitor must be one of {valid_monitors}")
        valid_modes = ["min", "max", "auto"]
        if self.early_stopping_mode not in valid_modes:
            raise ValueError(f"early_stopping_mode must be one of {valid_modes}")
            
        # Validate gradient clipping
        if self.gradient_clip_value is not None and self.gradient_clip_value <= 0:
            raise ValueError("gradient_clip_value must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


@dataclass
class DataConfig:
    """Configuration for data handling."""
    # Dataset paths
    train_dir: str = "chest_xray/train"
    test_dir: str = "chest_xray/test"
    val_dir: Optional[str] = None  # If separate validation set exists
    
    # Data preprocessing
    image_size: Tuple[int, int] = (128, 128)
    normalize_method: str = "rescale"  # "rescale", "standardize", "imagenet"
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.2,
        "zoom_range": 0.2,
        "horizontal_flip": True,
        "vertical_flip": False,
        "fill_mode": "nearest"
    })
    
    # Data loading
    num_workers: int = 4
    prefetch_buffer: int = 2
    cache_dataset: bool = False
    
    # Class configuration
    class_names: List[str] = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])
    
    def validate(self) -> None:
        """Validate data configuration parameters."""
        # Validate paths
        if not self.train_dir:
            raise ValueError("train_dir cannot be empty")
        if not self.test_dir:
            raise ValueError("test_dir cannot be empty")
            
        # Validate image size
        if len(self.image_size) != 2:
            raise ValueError("image_size must have exactly 2 dimensions")
        if any(dim <= 0 for dim in self.image_size):
            raise ValueError("All image_size dimensions must be positive")
            
        # Validate normalization
        valid_normalize = ["rescale", "standardize", "imagenet"]
        if self.normalize_method not in valid_normalize:
            raise ValueError(f"normalize_method must be one of {valid_normalize}")
            
        # Validate data loading parameters
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_buffer < 0:
            raise ValueError("prefetch_buffer must be non-negative")
            
        # Validate class names
        if len(self.class_names) < 2:
            raise ValueError("Must have at least 2 class names")


@dataclass
class PathsConfig:
    """Configuration for file paths and directories."""
    # Output directories
    models_dir: str = "models"
    logs_dir: str = "logs"
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    
    # Specific file paths
    model_name_template: str = "{architecture}_{timestamp}"
    checkpoint_name_template: str = "{architecture}_checkpoint_{epoch:02d}_{val_loss:.4f}"
    
    # Create directories if they don't exist
    create_dirs: bool = True
    
    def validate(self) -> None:
        """Validate paths configuration."""
        required_dirs = [self.models_dir, self.logs_dir, self.results_dir, self.checkpoints_dir]
        for dir_path in required_dirs:
            if not dir_path:
                raise ValueError("Directory paths cannot be empty")
                
        # Validate templates have required placeholders
        if "{architecture}" not in self.model_name_template:
            raise ValueError("model_name_template must contain {architecture} placeholder")
        if "{timestamp}" not in self.model_name_template:
            raise ValueError("model_name_template must contain {timestamp} placeholder")


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    # Logging levels
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "training.log"
    
    # Console output
    verbose: int = 1  # 0=silent, 1=progress bar, 2=one line per epoch
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "logs"
    tensorboard_histogram_freq: int = 1
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "pneumonia-detection"
    wandb_entity: Optional[str] = None
    
    # MLflow
    use_mlflow: bool = False
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "pneumonia_detection"
    
    def validate(self) -> None:
        """Validate logging configuration."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")
            
        if self.verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1, or 2")
            
        if self.tensorboard_histogram_freq < 0:
            raise ValueError("tensorboard_histogram_freq must be non-negative")


@dataclass
class Config:
    """Main configuration class containing all subsections."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Experiment metadata
    experiment_name: str = "pneumonia_detection"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def validate(self) -> None:
        """Validate entire configuration."""
        # Validate all subsections
        self.model.validate()
        self.training.validate()
        self.data.validate()
        self.paths.validate()
        self.logging.validate()
        
        # Cross-validation between sections
        if self.model.input_shape[:2] != self.data.image_size:
            raise ValueError("model.input_shape and data.image_size must match")
            
        # Validate experiment metadata
        if not self.experiment_name:
            raise ValueError("experiment_name cannot be empty")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        return dataclass_to_dict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        def dict_to_dataclass(data_class, data_dict):
            if not isinstance(data_dict, dict):
                return data_dict
                
            field_types = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
            kwargs = {}
            
            for key, value in data_dict.items():
                if key in field_types:
                    field_type = field_types[key]
                    # Handle nested dataclasses
                    if hasattr(field_type, '__dataclass_fields__'):
                        kwargs[key] = dict_to_dataclass(field_type, value)
                    else:
                        kwargs[key] = value
                        
            return data_class(**kwargs)
        
        # Extract subsection dictionaries
        model_dict = config_dict.get('model', {})
        training_dict = config_dict.get('training', {})
        data_dict = config_dict.get('data', {})
        paths_dict = config_dict.get('paths', {})
        logging_dict = config_dict.get('logging', {})
        
        # Create subsection objects
        model_config = dict_to_dataclass(ModelConfig, model_dict)
        training_config = dict_to_dataclass(TrainingConfig, training_dict)
        data_config = dict_to_dataclass(DataConfig, data_dict)
        paths_config = dict_to_dataclass(PathsConfig, paths_dict)
        logging_config = dict_to_dataclass(LoggingConfig, logging_dict)
        
        # Create main config with remaining fields
        config_kwargs = {
            'model': model_config,
            'training': training_config,
            'data': data_config,
            'paths': paths_config,
            'logging': logging_config
        }
        
        # Add other fields
        for key, value in config_dict.items():
            if key not in ['model', 'training', 'data', 'paths', 'logging']:
                config_kwargs[key] = value
                
        return cls(**config_kwargs)