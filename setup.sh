#!/bin/bash
# Setup script for VR Body Segmentation Application

set -e

echo "============================================"
echo "VR Body Segmentation - Setup Script"
echo "============================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check for CUDA
echo ""
echo "Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU acceleration may not be available."
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install pyyaml rich matplotlib seaborn psutil

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
else
    echo "WARNING: CUDA not available in PyTorch"
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p output
mkdir -p benchmark_results
mkdir -p optimization_results

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x cli.py
chmod +x benchmarks/benchmark_suite.py
chmod +x benchmarks/performance_analyzer.py
chmod +x scripts/optimize.py

# Display hardware info
echo ""
echo "============================================"
echo "Hardware Configuration"
echo "============================================"
python3 cli.py --show-hardware

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run benchmarks: python benchmarks/benchmark_suite.py"
echo "  3. Auto-optimize: python scripts/optimize.py --auto-optimize"
echo "  4. Process video: python cli.py input.mp4 -o output.mp4"
echo ""
echo "Documentation:"
echo "  - Usage Guide: docs/usage_guide.md"
echo "  - Performance Guide: docs/performance_guide.md"
echo "  - Troubleshooting: docs/troubleshooting.md"
echo ""
