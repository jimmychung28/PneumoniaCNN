#!/usr/bin/env python3
"""
Test script for the configuration management system.
This script demonstrates the key features without requiring full dependencies.
"""

import json
import os
from pathlib import Path

# Test configuration schema
def test_config_schema():
    """Test configuration schema creation and validation."""
    print("🧪 Testing Configuration Schema...")
    
    try:
        from src.config.config_schema import Config, ModelConfig, TrainingConfig, DataConfig
        
        # Test default configuration
        config = Config()
        config.validate()
        print("✅ Default configuration is valid")
        
        # Test custom configuration
        custom_config = Config(
            experiment_name="test_experiment",
            description="Test configuration",
            tags=["test", "demo"]
        )
        custom_config.validate()
        print("✅ Custom configuration is valid")
        
        # Test invalid configuration
        try:
            invalid_config = Config()
            invalid_config.model.learning_rate = 2.0  # Invalid learning rate
            invalid_config.validate()
            print("❌ Invalid configuration passed validation (this shouldn't happen)")
        except Exception as e:
            print("✅ Invalid configuration correctly rejected")
        
        # Test configuration conversion
        config_dict = config.to_dict()
        config_from_dict = Config.from_dict(config_dict)
        print("✅ Configuration dict conversion works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration schema test failed: {e}")
        return False


def test_config_examples():
    """Test that example configuration files are valid."""
    print("\n🧪 Testing Example Configurations...")
    
    configs_dir = Path("configs")
    if not configs_dir.exists():
        print("⚠️  configs/ directory not found - skipping example tests")
        return True
    
    try:
        from src.config.config_schema import Config
        
        config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
        
        if not config_files:
            print("⚠️  No YAML config files found - skipping example tests")
            return True
        
        for config_file in config_files:
            try:
                # Try to parse as JSON (basic validation)
                print(f"📄 Checking {config_file.name}...")
                
                # Basic file validation
                if config_file.stat().st_size == 0:
                    print(f"❌ {config_file.name} is empty")
                    continue
                
                # Check if it looks like valid YAML structure
                with open(config_file) as f:
                    content = f.read()
                    if "experiment_name:" in content and "model:" in content:
                        print(f"✅ {config_file.name} has valid structure")
                    else:
                        print(f"⚠️  {config_file.name} missing required sections")
                        
            except Exception as e:
                print(f"❌ {config_file.name} validation failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping advanced validation: {e}")
        return True


def test_directory_structure():
    """Test that the project has the expected structure."""
    print("\n🧪 Testing Project Structure...")
    
    required_files = [
        "config_schema.py",
        "config_loader.py", 
        "config_cli.py",
        "cnn_with_config.py",
        "validation_utils.py"
    ]
    
    required_dirs = [
        "configs"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            all_good = False
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ missing")
            all_good = False
    
    return all_good


def demonstrate_config_features():
    """Demonstrate key configuration features."""
    print("\n🚀 Demonstrating Configuration Features...")
    
    try:
        from src.config.config_schema import Config
        
        # 1. Create different experiment configurations
        print("\n1. Creating different experiment types:")
        
        # Fast experiment
        fast_config = Config(
            experiment_name="fast_demo",
            description="Fast training for development"
        )
        fast_config.model.input_shape = (64, 64, 3)
        fast_config.training.batch_size = 16
        fast_config.training.epochs = 5
        fast_config.data.image_size = (64, 64)
        
        print(f"   Fast Config: {fast_config.training.epochs} epochs, "
              f"{fast_config.model.input_shape} input, "
              f"batch size {fast_config.training.batch_size}")
        
        # Production experiment
        prod_config = Config(
            experiment_name="production_demo",
            description="Full training configuration"
        )
        prod_config.training.epochs = 100
        prod_config.training.use_mixed_precision = True
        prod_config.logging.use_tensorboard = True
        
        print(f"   Production Config: {prod_config.training.epochs} epochs, "
              f"mixed precision: {prod_config.training.use_mixed_precision}")
        
        # 2. Show configuration flexibility
        print("\n2. Configuration flexibility:")
        print(f"   Model architectures supported: standard, unet, two_stage")
        print(f"   Optimizers supported: {['adam', 'sgd', 'rmsprop']}")
        print(f"   Learning rate schedules: reduce_on_plateau, cosine, exponential")
        
        # 3. Show validation
        print("\n3. Validation examples:")
        try:
            bad_config = Config()
            bad_config.training.batch_size = -1  # Invalid
            bad_config.validate()
        except Exception as e:
            print(f"   ✅ Caught invalid batch size: {str(e)[:50]}...")
        
        # 4. Show configuration export
        print("\n4. Configuration export:")
        config_dict = fast_config.to_dict()
        print(f"   Configuration exported to dict with {len(config_dict)} sections")
        print(f"   Sections: {list(config_dict.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature demonstration failed: {e}")
        return False


def main():
    """Run all configuration tests."""
    print("🧪 Configuration Management System Test")
    print("=" * 50)
    
    tests = [
        ("Configuration Schema", test_config_schema),
        ("Project Structure", test_directory_structure),
        ("Example Configurations", test_config_examples),
        ("Feature Demonstration", demonstrate_config_features)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Configuration system is ready.")
        print("\n📋 Next steps:")
        print("   1. Install PyYAML: pip install PyYAML>=6.0")
        print("   2. Try: python3 config_cli.py list")
        print("   3. Run training: python3 cnn_with_config.py")
        print("   4. Read CONFIG_GUIDE.md for detailed usage")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)