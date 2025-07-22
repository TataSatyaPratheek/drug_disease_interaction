#!/bin/bash

# Setup script for parallel processing dependencies

echo "Setting up parallel processing environment..."

# Check if brew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install Homebrew first."
    echo "Visit: https://brew.sh/"
    exit 1
fi

# Check if parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU parallel not found. Installing via brew..."
    brew install parallel
else
    echo "GNU parallel already installed: $(which parallel)"
fi

# Check Python dependencies
echo "Checking Python dependencies..."

# Required packages for the enhanced sync checker
REQUIRED_PACKAGES=(
    "psutil"
    "pyarrow"
    "pandas"
    "concurrent.futures"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✓ $package is available"
    else
        echo "✗ $package not found. Installing..."
        pip install $package
    fi
done

echo "Setup complete!"
echo "You can now run the sync checker with enhanced parallel processing."
