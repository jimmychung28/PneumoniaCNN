#!/usr/bin/env python3
import platform
import subprocess
import sys

print("=== System Information ===")
print(f"Architecture: {platform.machine()}")
print(f"Platform: {platform.platform()}")
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")

# Check if running in Rosetta
result = subprocess.run(['sysctl', '-n', 'sysctl.proc_translated'], capture_output=True, text=True)
if result.stdout.strip() == '1':
    print("\n⚠️  WARNING: Running under Rosetta 2 (x86_64 emulation)")
    print("To run natively on Apple Silicon, use:")
    print("  arch -arm64 python test_tensorflow.py")
else:
    print("\n✅ Running natively on Apple Silicon")

print("\n=== Testing TensorFlow ===")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Available devices: {tf.config.list_physical_devices()}")
    
    # Check for Metal GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✅ Metal GPU detected: {gpus}")
    else:
        print("\n⚠️  No GPU detected")
        
except Exception as e:
    print(f"\n❌ Error loading TensorFlow: {e}")
    print("\nTry running with: arch -arm64 python test_tensorflow.py")