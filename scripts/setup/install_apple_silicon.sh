#!/bin/bash

echo "=== TensorFlow Installation for Apple Silicon ==="
echo "This script will install TensorFlow optimized for M1/M2/M3 Macs"
echo ""

# Check if running on Apple Silicon
current_arch=$(uname -m)
actual_arch=$(arch -arm64 uname -m 2>/dev/null || uname -m)

if [[ "$actual_arch" != "arm64" ]]; then
    echo "❌ This script is for Apple Silicon Macs only."
    echo "Your architecture: $current_arch"
    exit 1
fi

if [[ "$current_arch" == "x86_64" ]] && [[ "$actual_arch" == "arm64" ]]; then
    echo "⚠️  Running in Rosetta mode (x86_64 emulation on Apple Silicon)"
    echo "Switching to native arm64 mode..."
    echo ""
    echo "Please run this command instead:"
    echo "  arch -arm64 ./install_apple_silicon.sh"
    exit 1
fi

echo "✓ Apple Silicon detected (running natively in arm64 mode)"
echo ""

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
python_major=$(python3 -c 'import sys; print(sys.version_info[0])')
python_minor=$(python3 -c 'import sys; print(sys.version_info[1])')

echo "Python version: $python_version"

# Check if Python version is compatible (3.8-3.11)
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

# Create virtual environment if it doesn't exist
if [ ! -d "venv_m1" ]; then
    echo ""
    echo "Creating virtual environment for Apple Silicon..."
    python3 -m venv venv_m1
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv_m1/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Uninstall any existing TensorFlow
echo ""
echo "Removing any existing TensorFlow installations..."
pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tensorflow-macos tensorflow-metal

# Install Apple Silicon optimized TensorFlow
echo ""
echo "Installing TensorFlow for Apple Silicon..."
pip install tensorflow-macos==2.12.0
pip install tensorflow-metal==1.0.0

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install numpy>=1.23.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install pillow>=9.0.0
pip install h5py>=3.7.0

echo ""
echo "✅ Installation complete!"
echo ""
echo "To use this environment:"
echo "  source venv_m1/bin/activate"
echo "  python cnn.py"
echo ""
echo "Your Mac's GPU (via Metal) will be automatically used for acceleration!"