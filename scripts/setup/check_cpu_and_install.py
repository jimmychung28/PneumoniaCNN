#!/usr/bin/env python3
"""
Check CPU capabilities and install appropriate TensorFlow version
"""
import subprocess
import platform
import sys


def check_cpu_features():
    """Check if CPU supports AVX instructions"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        try:
            result = subprocess.run(['sysctl', '-a'], capture_output=True, text=True)
            output = result.stdout.lower()
            has_avx = 'avx1.0' in output or 'avx2' in output
            print(f"macOS detected. AVX support: {has_avx}")
            return has_avx
        except:
            print("Could not detect AVX support")
            return False
    
    elif system == "Linux":
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                has_avx = 'avx' in cpuinfo
                print(f"Linux detected. AVX support: {has_avx}")
                return has_avx
        except:
            print("Could not detect AVX support")
            return False
    
    else:
        print(f"Unsupported system: {system}")
        return False


def install_tensorflow_cpu():
    """Install TensorFlow without AVX requirements"""
    print("\n=== Installing TensorFlow for CPUs without AVX ===")
    print("This will install a compatible version of TensorFlow...")
    
    # First, uninstall existing TensorFlow
    print("\n1. Uninstalling existing TensorFlow...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow', 'tensorflow-cpu'])
    
    # Install TensorFlow 1.5 or earlier (last version without AVX requirement)
    # Or use tensorflow-cpu for newer versions
    print("\n2. Installing compatible TensorFlow version...")
    
    # Option 1: Install older TensorFlow that doesn't require AVX
    # subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow==1.5'])
    
    # Option 2: Install CPU-only version (recommended)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow-cpu==2.4.0'])
    
    print("\n✓ Installation complete!")


def main():
    print("=== TensorFlow CPU Compatibility Checker ===\n")
    
    has_avx = check_cpu_features()
    
    if not has_avx:
        print("\n⚠️  Your CPU does not support AVX instructions.")
        print("The standard TensorFlow package requires AVX support.\n")
        
        response = input("Would you like to install a compatible TensorFlow version? (y/n): ")
        if response.lower() == 'y':
            install_tensorflow_cpu()
            print("\n✓ You can now run the CNN model!")
            print("Note: Performance may be slower without AVX optimizations.")
        else:
            print("\nAlternative solutions:")
            print("1. Install TensorFlow manually:")
            print("   pip uninstall tensorflow")
            print("   pip install tensorflow-cpu==2.4.0")
            print("\n2. Use Google Colab for free GPU access:")
            print("   https://colab.research.google.com/")
            print("\n3. Build TensorFlow from source for your CPU")
    else:
        print("\n✓ Your CPU supports AVX instructions!")
        print("You can use the standard TensorFlow package.")


if __name__ == "__main__":
    main()