#!/bin/bash

echo "=== Installing ARM64 Homebrew for Apple Silicon ==="
echo ""

# Check if ARM64 Homebrew is already installed
if [ -f "/opt/homebrew/bin/brew" ]; then
    echo "✅ ARM64 Homebrew is already installed at /opt/homebrew"
else
    echo "Installing ARM64 Homebrew..."
    echo "You'll be prompted for your password..."
    echo ""
    
    # Install ARM64 Homebrew
    arch -arm64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo ""
echo "Setting up ARM64 Homebrew environment..."

# Add to PATH temporarily
export PATH="/opt/homebrew/bin:$PATH"

echo ""
echo "Installing Python 3.11 with ARM64 Homebrew..."
arch -arm64 /opt/homebrew/bin/brew install python@3.11

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use ARM64 Python in the future, run:"
echo "  export PATH=\"/opt/homebrew/bin:\$PATH\""
echo "  python3.11 --version"
echo ""
echo "Or add this to your shell profile (~/.zshrc or ~/.bash_profile):"
echo "  export PATH=\"/opt/homebrew/bin:\$PATH\""