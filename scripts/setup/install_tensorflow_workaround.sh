#!/bin/bash

echo "=== TensorFlow Workaround Installation ==="
echo "Installing TensorFlow 2.4.0 that works with Python 3.7"
echo ""

# Uninstall any existing TensorFlow
echo "Removing any existing TensorFlow installations..."
pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tensorflow-macos tensorflow-metal

# Install TensorFlow 2.4.0 (last version supporting Python 3.7)
echo ""
echo "Installing TensorFlow 2.4.0..."
pip install tensorflow==2.4.0

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install numpy==1.19.5  # Compatible with Python 3.7
pip install scikit-learn==0.24.2
pip install matplotlib==3.3.4
pip install seaborn==0.11.2
pip install pillow==8.2.0
pip install h5py==2.10.0

echo ""
echo "✅ Installation complete!"
echo ""
echo "Note: This is using standard TensorFlow without Apple Silicon optimizations."
echo "For better performance, consider upgrading to Python 3.8+ and using tensorflow-macos."
echo ""
echo "To run the CNN:"
echo "  python cnn.py"