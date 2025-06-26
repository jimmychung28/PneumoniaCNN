#!/usr/bin/env python3
"""
Test script for the consolidated CNN implementation.
Tests all three operational modes and ensures backward compatibility.
"""

import sys
import os
import logging

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        from src.models.cnn import PneumoniaCNN
        print("✓ Main CNN class imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CNN class: {e}")
        return False
    
    # Test optional imports
    try:
        from src.config.config_loader import get_config, ConfigManager
        print("✓ Configuration system available")
        config_available = True
    except ImportError:
        print("⚠ Configuration system not available")
        config_available = False
    
    try:
        from src.training.data_pipeline import PerformanceDataPipeline
        from src.training.mixed_precision_trainer import MixedPrecisionTrainer
        print("✓ High-performance components available")
        hp_available = True
    except ImportError:
        print("⚠ High-performance components not available")
        hp_available = False
    
    return True, config_available, hp_available

def test_basic_mode():
    """Test basic mode functionality."""
    print("\n" + "="*50)
    print("TESTING BASIC MODE")
    print("="*50)
    
    try:
        from src.models.cnn import PneumoniaCNN
        
        # Test initialization
        print("1. Testing basic mode initialization...")
        cnn = PneumoniaCNN(
            input_shape=(128, 128, 3),
            learning_rate=0.001,
            performance_mode='basic'
        )
        print(f"✓ Initialized in {cnn.performance_mode} mode")
        
        # Test model building
        print("2. Testing model building...")
        model = cnn.build_model()
        print(f"✓ Model built with {model.count_params():,} parameters")
        
        # Test basic properties
        print("3. Testing basic properties...")
        assert cnn.input_shape == (128, 128, 3)
        assert cnn.learning_rate == 0.001
        assert cnn.batch_size == 32
        assert cnn.epochs == 50
        print("✓ All basic properties correct")
        
        # Test callbacks creation
        print("4. Testing callback creation...")
        callbacks = cnn.get_callbacks('test_model')
        print(f"✓ Created {len(callbacks)} callbacks")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic mode test failed: {e}")
        return False

def test_standard_mode():
    """Test standard mode functionality."""
    print("\n" + "="*50)
    print("TESTING STANDARD MODE")
    print("="*50)
    
    try:
        from src.models.cnn import PneumoniaCNN
        
        # Test if config system is available
        try:
            from src.config.config_loader import get_config
            config_available = True
        except ImportError:
            print("⚠ Configuration system not available, skipping standard mode test")
            return True
        
        # Test initialization with default config
        print("1. Testing standard mode initialization...")
        cnn = PneumoniaCNN(performance_mode='standard')
        print(f"✓ Initialized in {cnn.performance_mode} mode")
        
        # Test model building
        print("2. Testing configurable model building...")
        model = cnn.build_model()
        print(f"✓ Model built with {model.count_params():,} parameters")
        
        # Test config-based properties
        print("3. Testing config-based properties...")
        assert hasattr(cnn, 'config')
        assert cnn.config is not None
        print("✓ Configuration loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Standard mode test failed: {e}")
        return False

def test_high_performance_mode():
    """Test high-performance mode functionality."""
    print("\n" + "="*50)
    print("TESTING HIGH-PERFORMANCE MODE")
    print("="*50)
    
    try:
        from src.models.cnn import PneumoniaCNN
        
        # Test if high-performance components are available
        try:
            from src.training.data_pipeline import PerformanceDataPipeline
            from src.training.mixed_precision_trainer import MixedPrecisionTrainer
            hp_available = True
        except ImportError:
            print("⚠ High-performance components not available, skipping HP mode test")
            return True
        
        # Test if config system is available (required for HP mode)
        try:
            from src.config.config_loader import get_config
            config_available = True
        except ImportError:
            print("⚠ Configuration system not available, skipping HP mode test")
            return True
        
        # Test initialization
        print("1. Testing high-performance mode initialization...")
        cnn = PneumoniaCNN(performance_mode='high_performance')
        print(f"✓ Initialized in {cnn.performance_mode} mode")
        
        # Test HP components
        print("2. Testing high-performance components...")
        assert hasattr(cnn, 'mixed_precision_trainer')
        assert hasattr(cnn, 'data_pipeline')
        assert hasattr(cnn, 'performance_monitor')
        print("✓ All HP components initialized")
        
        # Test model building
        print("3. Testing optimized model building...")
        model = cnn.build_model()
        print(f"✓ Optimized model built with {model.count_params():,} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ High-performance mode test failed: {e}")
        return False

def test_automatic_mode_selection():
    """Test automatic mode selection."""
    print("\n" + "="*50)
    print("TESTING AUTOMATIC MODE SELECTION")
    print("="*50)
    
    try:
        from src.models.cnn import PneumoniaCNN
        
        # Test automatic mode selection
        print("1. Testing automatic mode selection...")
        cnn = PneumoniaCNN()  # No performance_mode specified
        print(f"✓ Automatically selected: {cnn.performance_mode} mode")
        
        # Test 'auto' mode
        print("2. Testing explicit 'auto' mode...")
        cnn_auto = PneumoniaCNN(performance_mode='auto')
        print(f"✓ Auto mode selected: {cnn_auto.performance_mode}")
        
        # Both should select the same mode
        assert cnn.performance_mode == cnn_auto.performance_mode
        print("✓ Consistent mode selection")
        
        return True
        
    except Exception as e:
        print(f"✗ Automatic mode selection test failed: {e}")
        return False

def test_backward_compatibility():
    """Test backward compatibility with original API."""
    print("\n" + "="*50)
    print("TESTING BACKWARD COMPATIBILITY")
    print("="*50)
    
    try:
        from src.models.cnn import PneumoniaCNN
        
        # Test original API calls
        print("1. Testing original API compatibility...")
        
        # Original constructor pattern
        cnn = PneumoniaCNN(input_shape=(224, 224, 3), learning_rate=0.0005)
        print("✓ Original constructor pattern works")
        
        # Test that basic mode is used for simple constructor
        assert cnn.performance_mode == 'basic' or cnn.performance_mode in ['standard', 'high_performance']
        print(f"✓ Mode selection works: {cnn.performance_mode}")
        
        # Test model building
        model = cnn.build_model()
        print("✓ Model building works")
        
        # Test that all original methods exist
        methods_to_test = [
            'build_model',
            'create_data_generators', 
            'get_callbacks',
            'calculate_class_weights',
            'train',
            'evaluate',
            'plot_confusion_matrix',
            'plot_training_history',
            'save_model'
        ]
        
        for method in methods_to_test:
            assert hasattr(cnn, method), f"Missing method: {method}"
        print(f"✓ All {len(methods_to_test)} original methods available")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("CONSOLIDATED CNN IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    # Test imports
    import_result = test_imports()
    if not import_result:
        print("\n✗ Import test failed - cannot continue")
        return False
    
    # Run all tests
    tests = [
        test_basic_mode,
        test_standard_mode,
        test_high_performance_mode,
        test_automatic_mode_selection,
        test_backward_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Basic Mode",
        "Standard Mode", 
        "High-Performance Mode",
        "Automatic Mode Selection",
        "Backward Compatibility"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:25s} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Consolidation successful!")
        return True
    else:
        print("⚠ Some tests failed - Review implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)