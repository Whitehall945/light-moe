#!/bin/bash
# Environment setup script for Light-MoE
# Usage: source scripts/setup_env.sh

set -e

echo "Setting up Light-MoE development environment..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Warning: nvcc not found. CUDA may not be properly installed."
else
    echo "CUDA version: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
fi

# Check Python
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -e ".[dev,benchmark]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Initialize submodules
if [ ! -d "third_party/cutlass/include" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

echo ""
echo "Setup complete! Environment is ready."
echo ""
echo "To build C++ components:"
echo "  mkdir build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  make -j\$(nproc)"
echo ""
echo "To run tests:"
echo "  pytest tests/python -v"
