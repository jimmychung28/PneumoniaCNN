#!/usr/bin/env python3
"""
Configuration CLI tool for the Pneumonia CNN project.
Provides easy command-line interface for managing configurations.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

from src.config.config_loader import ConfigManager, ConfigurationError
from src.config.config_schema import Config


def list_configs(config_manager: ConfigManager) -> None:
    """List available configuration files."""
    configs = config_manager.loader.list_configs()
    
    if not configs:
        print("No configuration files found.")
        return
        
    print("Available configurations:")
    for config_file in configs:
        config_path = config_manager.loader.config_dir / config_file
        try:
            config = config_manager.loader.load_config(str(config_path))
            print(f"  • {config_file}")
            print(f"    Name: {config.experiment_name}")
            print(f"    Description: {config.description}")
            print(f"    Architecture: {config.model.architecture}")
            print()
        except Exception as e:
            print(f"  • {config_file} (Error: {str(e)})")
            print()


def show_config(config_manager: ConfigManager, config_name: str) -> None:
    """Show configuration details."""
    try:
        config = config_manager.loader.load_config(config_name=config_name)
        
        print(f"Configuration: {config_name}")
        print("=" * 50)
        print(f"Experiment Name: {config.experiment_name}")
        print(f"Description: {config.description}")
        print(f"Tags: {', '.join(config.tags) if config.tags else 'None'}")
        print()
        
        print("Model Configuration:")
        print(f"  Architecture: {config.model.architecture}")
        print(f"  Input Shape: {config.model.input_shape}")
        print(f"  Learning Rate: {config.model.learning_rate}")
        print()
        
        print("Training Configuration:")
        print(f"  Batch Size: {config.training.batch_size}")
        print(f"  Epochs: {config.training.epochs}")
        print(f"  Optimizer: {config.training.optimizer}")
        print(f"  Early Stopping: {config.training.use_early_stopping}")
        print()
        
        print("Data Configuration:")
        print(f"  Train Dir: {config.data.train_dir}")
        print(f"  Test Dir: {config.data.test_dir}")
        print(f"  Image Size: {config.data.image_size}")
        print(f"  Augmentation: {config.data.use_augmentation}")
        print()
        
    except ConfigurationError as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def validate_config(config_manager: ConfigManager, config_name: str) -> None:
    """Validate a configuration file."""
    try:
        config = config_manager.loader.load_config(config_name=config_name)
        config.validate()
        print(f"✅ Configuration '{config_name}' is valid")
        
    except ConfigurationError as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def create_experiment(config_manager: ConfigManager, experiment_name: str, 
                     base_config: str, overrides: Dict[str, Any]) -> None:
    """Create new experiment configuration."""
    try:
        # Load base configuration
        base = config_manager.loader.load_config(config_name=base_config)
        
        # Create experiment
        experiment_config = config_manager.loader.create_experiment_config(
            base, experiment_name, overrides
        )
        
        # Save experiment configuration
        experiment_file = f"experiment_{experiment_name}.yaml"
        experiment_path = config_manager.loader.config_dir / experiment_file
        config_manager.loader.save_config(experiment_config, str(experiment_path))
        
        print(f"✅ Experiment configuration created: {experiment_file}")
        print(f"   Experiment Name: {experiment_config.experiment_name}")
        print(f"   Based on: {base_config}")
        print(f"   Overrides applied: {len(overrides)}")
        
    except ConfigurationError as e:
        print(f"❌ Failed to create experiment: {e}")
        sys.exit(1)


def copy_config(config_manager: ConfigManager, source: str, destination: str) -> None:
    """Copy a configuration file."""
    try:
        config = config_manager.loader.load_config(config_name=source)
        dest_path = config_manager.loader.config_dir / destination
        
        config_manager.loader.save_config(config, str(dest_path))
        print(f"✅ Configuration copied: {source} → {destination}")
        
    except ConfigurationError as e:
        print(f"❌ Failed to copy configuration: {e}")
        sys.exit(1)


def run_training(config_manager: ConfigManager, config_name: str) -> None:
    """Run training with specified configuration."""
    try:
        print(f"🚀 Starting training with configuration: {config_name}")
        
        # Import and run training
        from cnn_with_config import main_with_config
        
        config_path = config_manager.loader.config_dir / config_name
        main_with_config(str(config_path))
        
    except ImportError:
        print("❌ Training module not found. Make sure cnn_with_config.py exists.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Configuration management for Pneumonia CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                           # List all configurations
  %(prog)s show default.yaml              # Show configuration details
  %(prog)s validate default.yaml          # Validate configuration
  %(prog)s create my_exp default.yaml     # Create experiment from base config
  %(prog)s copy default.yaml my_config.yaml  # Copy configuration
  %(prog)s train default.yaml             # Run training with config
  
  # Create experiment with overrides:
  %(prog)s create fast_test default.yaml --override model.learning_rate=0.01 training.epochs=5
        """
    )
    
    parser.add_argument(
        '--config-dir', 
        default='configs',
        help='Configuration directory (default: configs)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List available configurations')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show configuration details')
    show_parser.add_argument('config', help='Configuration file name')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('config', help='Configuration file name')
    
    # Create experiment command
    create_parser = subparsers.add_parser('create', help='Create experiment configuration')
    create_parser.add_argument('name', help='Experiment name')
    create_parser.add_argument('base_config', help='Base configuration file')
    create_parser.add_argument(
        '--override', 
        action='append', 
        default=[],
        help='Override parameter (key=value format)'
    )
    
    # Copy command
    copy_parser = subparsers.add_parser('copy', help='Copy configuration')
    copy_parser.add_argument('source', help='Source configuration file')
    copy_parser.add_argument('destination', help='Destination configuration file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Run training with configuration')
    train_parser.add_argument('config', help='Configuration file name')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize config manager
    config_manager = ConfigManager(args.config_dir)
    
    # Execute command
    try:
        if args.command == 'list':
            list_configs(config_manager)
            
        elif args.command == 'show':
            show_config(config_manager, args.config)
            
        elif args.command == 'validate':
            validate_config(config_manager, args.config)
            
        elif args.command == 'create':
            # Parse overrides
            overrides = {}
            for override in args.override:
                if '=' not in override:
                    print(f"❌ Invalid override format: {override} (use key=value)")
                    sys.exit(1)
                    
                key, value = override.split('=', 1)
                
                # Try to parse value as JSON (for numbers, booleans, lists)
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep as string
                    
                overrides[key] = value
            
            create_experiment(config_manager, args.name, args.base_config, overrides)
            
        elif args.command == 'copy':
            copy_config(config_manager, args.source, args.destination)
            
        elif args.command == 'train':
            run_training(config_manager, args.config)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()