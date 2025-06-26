#!/bin/bash

# Secure TensorFlow Installation Script for Apple Silicon
# This script installs TensorFlow optimized for M1/M2/M3 Macs with comprehensive validation

# Secure shell script settings
set -euo pipefail  # Exit on error, undefined vars, pipe failures
IFS=$'\n\t'       # Secure Internal Field Separator

echo "=== Secure TensorFlow Installation for Apple Silicon ==="
echo "This script will install TensorFlow optimized for M1/M2/M3 Macs"
echo ""
echo "⚠️  Please review this script before running to ensure it's safe!"
echo ""

# Validate environment
if [[ -z "${HOME:-}" ]]; then
    echo "❌ HOME environment variable not set"
    exit 1
fi

if [[ ! -w "." ]]; then
    echo "❌ Cannot write to current directory. Please run from a writable location."
    exit 1
fi

# Check if running on Apple Silicon with error handling
current_arch=$(uname -m 2>/dev/null || echo "unknown")
actual_arch=$(arch -arm64 uname -m 2>/dev/null || echo "$current_arch")

if [[ "$current_arch" == "unknown" ]]; then
    echo "❌ Cannot determine system architecture"
    exit 1
fi

if [[ "$actual_arch" != "arm64" ]]; then
    echo "❌ This script is for Apple Silicon Macs only."
    echo "Your architecture: $current_arch"
    echo ""
    echo "For Intel Macs, use: python check_cpu_and_install.py"
    exit 1
fi

if [[ "$current_arch" == "x86_64" ]] && [[ "$actual_arch" == "arm64" ]]; then
    echo "⚠️  Running in Rosetta mode (x86_64 emulation on Apple Silicon)"
    echo "Switching to native arm64 mode..."
    echo ""
    echo "Please run this command instead:"
    echo "  arch -arm64 ./install_apple_silicon_secure.sh"
    exit 1
fi

echo "✓ Apple Silicon detected (running natively in arm64 mode)"
echo ""

# Check Python version with comprehensive error handling
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3.8-3.11 first."
    echo ""
    echo "To install Python 3.11 on macOS:"
    echo "1. Install Homebrew (visit https://brew.sh for safe installation instructions)"
    echo "2. After Homebrew is installed: brew install python@3.11"
    echo "3. Use Python 3.11: python3.11 -m venv venv_m1"
    echo ""
    echo "Or download from python.org: https://www.python.org/downloads/"
    echo ""
    echo "⚠️  SECURITY WARNING: Never run 'curl | bash' commands from untrusted sources!"
    exit 1
fi

python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || echo "unknown")
python_major=$(python3 -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo "0")
python_minor=$(python3 -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo "0")

if [[ "$python_version" == "unknown" ]]; then
    echo "❌ Failed to detect Python version"
    exit 1
fi

echo "Python version: $python_version"

# Validate Python version compatibility
if [[ $python_major -ne 3 ]] || [[ $python_minor -lt 8 ]] || [[ $python_minor -gt 11 ]]; then
    echo ""
    echo "❌ Python $python_version is not compatible with tensorflow-macos"
    echo "Required: Python 3.8, 3.9, 3.10, or 3.11"
    echo ""
    echo "To install Python 3.11 on macOS:"
    echo "1. Install Homebrew (visit https://brew.sh for safe installation instructions)"
    echo "2. After Homebrew is installed: brew install python@3.11"
    echo "3. Use Python 3.11: python3.11 -m venv venv_m1"
    echo ""
    echo "Or download from python.org: https://www.python.org/downloads/"
    echo ""
    echo "⚠️  SECURITY WARNING: Never run 'curl | bash' commands from untrusted sources!"
    exit 1
fi

echo "✓ Python version compatible"

# Create virtual environment with validation
if [ ! -d "venv_m1" ]; then
    echo ""
    echo "Creating virtual environment for Apple Silicon..."
    if ! python3 -m venv venv_m1; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✓ Virtual environment created"
else
    echo ""
    echo "✓ Virtual environment venv_m1 already exists"
fi

# Activate virtual environment with validation
echo ""
echo "Activating virtual environment..."

if [[ ! -f "venv_m1/bin/activate" ]]; then
    echo "❌ Virtual environment activation script not found"
    exit 1
fi

# shellcheck source=/dev/null
source venv_m1/bin/activate

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip with validation
echo ""
echo "Upgrading pip..."

if ! command -v pip &> /dev/null; then
    echo "❌ pip not found in virtual environment"
    exit 1
fi

if ! pip install --upgrade pip; then
    echo "❌ Failed to upgrade pip"
    exit 1
fi

echo "✓ pip upgraded successfully"

# Uninstall any existing TensorFlow installations
echo ""
echo "Removing any existing TensorFlow installations..."

# Use || true to continue even if packages aren't installed
pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tensorflow-macos tensorflow-metal || true
echo "✓ Existing TensorFlow packages removed"

# Install Apple Silicon optimized TensorFlow with validation
echo ""
echo "Installing TensorFlow for Apple Silicon..."

if ! pip install tensorflow-macos==2.12.0; then
    echo "❌ Failed to install tensorflow-macos"
    exit 1
fi

echo "✓ tensorflow-macos installed"

if ! pip install tensorflow-metal==1.0.0; then
    echo "❌ Failed to install tensorflow-metal"
    exit 1
fi

echo "✓ tensorflow-metal installed"

# Install other dependencies with validation
echo ""
echo "Installing other dependencies..."

DEPENDENCIES=(
    "numpy>=1.23.0"
    "scikit-learn>=1.0.0"
    "matplotlib>=3.5.0"
    "seaborn>=0.11.0"
    "pillow>=9.0.0"
    "h5py>=3.7.0"
    "opencv-python>=4.5.0"
    "pandas>=1.5.0"
)

for dep in "${DEPENDENCIES[@]}"; do
    echo "Installing $dep..."
    if ! pip install "$dep"; then
        echo "❌ Failed to install $dep"
        exit 1
    fi
done

echo "✓ All dependencies installed"

# Verify TensorFlow installation
echo ""
echo "Verifying TensorFlow installation..."

if python3 -c "
import tensorflow as tf
import sys
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices: {len(gpus)}')
if len(gpus) > 0:
    for gpu in gpus:
        print(f'  - {gpu}')
print('✓ TensorFlow installed and working correctly')
" 2>/dev/null; then
    echo "✓ TensorFlow installation verified!"
else
    echo "⚠️  TensorFlow installation may have issues. Please run test_tensorflow.py to verify."
fi

# Create convenience activation script
echo ""
echo "Creating convenience scripts..."

cat > activate_venv.sh << 'EOF'
#!/bin/bash
echo "Activating Apple Silicon TensorFlow environment..."
if [[ -f "venv_m1/bin/activate" ]]; then
    source venv_m1/bin/activate
    echo "✓ Environment activated. You can now run Python scripts."
    echo ""
    echo "Available commands:"
    echo "  python test_tensorflow.py  # Verify installation"
    echo "  python cnn.py              # Run pneumonia detection"
    echo "  deactivate                 # Exit virtual environment"
else
    echo "❌ Virtual environment not found. Please run install_apple_silicon_secure.sh first."
    exit 1
fi
EOF

chmod +x activate_venv.sh

# Create validation script
cat > validate_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Validate the TensorFlow installation for Apple Silicon.
"""
import sys

def check_installation():
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check for GPU support
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            print(f"✓ GPU devices available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
        else:
            print("⚠️  No GPU devices found")
            
        # Test basic operations
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"✓ Basic operations working: {c.numpy().flatten()}")
        
        # Check Metal GPU specifically
        try:
            with tf.device('/GPU:0'):
                result = tf.reduce_sum([1, 2, 3, 4, 5])
                print(f"✓ Metal GPU computation successful: {result}")
        except:
            print("⚠️  Metal GPU computation not available")
            
        print("\n✅ TensorFlow installation is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ TensorFlow validation failed: {e}")
        return False

if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)
EOF

chmod +x validate_installation.py

echo "✓ Created activate_venv.sh for easy environment activation"
echo "✓ Created validate_installation.py for installation verification"

# Final summary
echo ""
echo "🎉 Installation complete!"
echo ""
echo "To use this environment:"
echo "  ./activate_venv.sh           # Activate environment (recommended)"
echo "  source venv_m1/bin/activate  # Manual activation"
echo ""
echo "To verify installation:"
echo "  python validate_installation.py  # Comprehensive check"
echo "  python test_tensorflow.py        # Project-specific test"
echo ""
echo "To run the pneumonia detection:"
echo "  python cnn.py                    # Standard CNN model"
echo "  python train_two_stage_model.py  # Advanced two-stage model"
echo ""
echo "⚠️  Remember to always activate the virtual environment before running Python scripts!"
echo "🚀 Your Mac's GPU (via Metal) will be automatically used for acceleration!"

echo ""
echo "📋 Installation Summary:"
echo "  - Python version: $python_version"
echo "  - Virtual environment: venv_m1"
echo "  - TensorFlow version: 2.12.0 (Apple Silicon optimized)"
echo "  - Metal GPU support: Enabled"
echo "  - All dependencies: Installed"