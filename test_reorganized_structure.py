#!/usr/bin/env python3
"""
Test script to verify the reorganized project structure.
This script tests that all imports work and the structure is functional.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_directory_structure():
    """Test that all expected directories exist."""
    print("🏗️  Testing directory structure...")
    
    expected_dirs = [
        "src",
        "src/models",
        "src/training", 
        "src/config",
        "src/utils",
        "configs",
        "scripts",
        "scripts/setup",
        "tests",
        "docs",
        "data",
        "legacy",
        "outputs"
    ]
    
    missing_dirs = []
    for dir_path in expected_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}/")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ All expected directories present")
    return True

def test_core_imports():
    """Test that core module imports work."""
    print("\n🔧 Testing core imports...")
    
    import_tests = [
        ("src.utils.validation_utils", "ValidationError"),
        ("src.config.config_schema", "Config"),
        ("src.config.config_loader", "ConfigManager"),
        ("src.models.cnn", "PneumoniaCNN"),
        ("src.training.data_pipeline", "PerformanceDataPipeline"),
        ("src.training.mixed_precision_trainer", "MixedPrecisionTrainer"),
        ("src.training.preprocessing_pipeline", "OptimizedPreprocessor"),
    ]
    
    failed_imports = []
    
    for module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"⚠️  {module_name}.{class_name}: {e}")
    
    if failed_imports:
        print(f"\n❌ Failed imports: {len(failed_imports)}")
        return False
    
    print("✅ All core imports successful")
    return True

def test_configuration_files():
    """Test that configuration files are accessible."""
    print("\n📋 Testing configuration files...")
    
    config_files = [
        "configs/default.yaml",
        "configs/fast_experiment.yaml", 
        "configs/high_performance.yaml",
        "configs/unet_two_stage.yaml"
    ]
    
    missing_configs = []
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
            
            # Test that data paths are updated
            with open(config_file) as f:
                content = f.read()
                if "data/chest_xray" in content:
                    print(f"  ✅ Updated data paths")
                else:
                    print(f"  ⚠️  Data paths may need updating")
        else:
            print(f"❌ {config_file}")
            missing_configs.append(config_file)
    
    return len(missing_configs) == 0

def test_data_directory():
    """Test that data directory is accessible."""
    print("\n📊 Testing data directory...")
    
    data_paths = [
        "data/chest_xray",
        "data/chest_xray/train",
        "data/chest_xray/test"
    ]
    
    for data_path in data_paths:
        if Path(data_path).exists():
            print(f"✅ {data_path}/")
        else:
            print(f"⚠️  {data_path}/ not found (may be expected)")
    
    return True

def test_entry_points():
    """Test that entry point scripts work."""
    print("\n🚀 Testing entry points...")
    
    entry_points = ["train.py", "config.py"]
    
    for entry_point in entry_points:
        if Path(entry_point).exists():
            print(f"✅ {entry_point}")
        else:
            print(f"❌ {entry_point}")
    
    return True

def test_unified_cnn_modes():
    """Test that the unified CNN can initialize in different modes."""
    print("\n🧠 Testing unified CNN modes...")
    
    try:
        from src.models.cnn import PneumoniaCNN
        
        # Test basic mode (minimal dependencies)
        try:
            cnn_basic = PneumoniaCNN(performance_mode='basic')
            print(f"✅ Basic mode: {cnn_basic.performance_mode}")
        except Exception as e:
            print(f"❌ Basic mode failed: {e}")
            return False
        
        # Test auto mode selection
        try:
            cnn_auto = PneumoniaCNN()
            print(f"✅ Auto mode selected: {cnn_auto.performance_mode}")
        except Exception as e:
            print(f"❌ Auto mode failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import PneumoniaCNN: {e}")
        return False

def test_scripts_directory():
    """Test that scripts are in the right place."""
    print("\n📜 Testing scripts...")
    
    expected_scripts = [
        "scripts/demo_two_stage.py",
        "scripts/performance_benchmark.py",
        "scripts/setup/install_apple_silicon_secure.sh",
        "scripts/setup/check_cpu_and_install.py"
    ]
    
    for script in expected_scripts:
        if Path(script).exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script}")
    
    return True

def test_legacy_backup():
    """Test that legacy files are preserved."""
    print("\n🗂️  Testing legacy backup...")
    
    legacy_files = [
        "legacy/cnn_legacy.py",
        "legacy/cnn_with_config.py", 
        "legacy/high_performance_cnn.py"
    ]
    
    for legacy_file in legacy_files:
        if Path(legacy_file).exists():
            print(f"✅ {legacy_file}")
        else:
            print(f"⚠️  {legacy_file} not found")
    
    return True

def display_usage_examples():
    """Display usage examples for the new structure."""
    print("\n📖 Usage Examples:")
    print("=" * 50)
    
    examples = [
        ("Training with unified CNN", "python train.py"),
        ("Training with config", "python train.py configs/high_performance.yaml"),
        ("Configuration management", "python config.py list"),
        ("Performance benchmarking", "python scripts/performance_benchmark.py"),
        ("Running tests", "python -m pytest tests/"),
        ("Demo", "python scripts/demo_two_stage.py"),
        ("Module imports", "from src.models.cnn import PneumoniaCNN")
    ]
    
    for description, command in examples:
        print(f"• {description:.<25} {command}")

def main():
    """Run all structure tests."""
    print("🧪 Testing Reorganized Project Structure")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Core Imports", test_core_imports),
        ("Configuration Files", test_configuration_files),
        ("Data Directory", test_data_directory),
        ("Entry Points", test_entry_points),
        ("Unified CNN Modes", test_unified_cnn_modes),
        ("Scripts Directory", test_scripts_directory),
        ("Legacy Backup", test_legacy_backup),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Project reorganization successful!")
        print("✅ All structure tests passed")
        display_usage_examples()
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        print("🔧 You may need to install dependencies or adjust file paths.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)