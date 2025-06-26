#!/usr/bin/env python3
"""
Fix TensorFlow warnings on Apple Silicon
"""

import os
import tensorflow as tf
import warnings

def suppress_tensorflow_warnings():
    """Suppress non-critical TensorFlow warnings"""
    
    # Suppress TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Configure TensorFlow for Apple Silicon
    tf.config.experimental.enable_op_determinism()
    
    # Disable graph optimization that causes issues on Apple Silicon
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': False,
        'constant_folding': False,
        'shape_optimization': False,
        'remapping': False,
        'arithmetic_optimization': False,
        'dependency_optimization': False,
        'loop_optimization': False,
        'function_optimization': False,
        'debug_stripper': False,
        'disable_model_pruning': True,
        'scoped_allocator_optimization': False,
        'pin_to_host_optimization': False,
        'implementation_selector': False,
        'auto_mixed_precision': False,
        'disable_meta_optimizer': True
    })
    
    print("✅ TensorFlow warnings suppressed for Apple Silicon")

# Apply fixes when imported
suppress_tensorflow_warnings()