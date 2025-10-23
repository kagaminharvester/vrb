#!/bin/bash
# Setup script for VR Body Segmentation Application

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Allow user to override Python command
PYTHON_CMD="${PYTHON_CMD:-python3}"

echo "============================================"
echo "VR Body Segmentation - Setup Script"
echo "============================================"
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "ERROR: $PYTHON_CMD not found. Please install Python 3.10+ or set PYTHON_CMD environment variable."
    exit 1
fi

python_version=$("$PYTHON_CMD" --version 2>&1 | awk '{print $2}')
major_version=$(echo "$python_version" | cut -d. -f1)
minor_version=$(echo "$python_version" | cut -d. -f2)

echo "Found Python $python_version"

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 10 ]); then
    echo "ERROR: Python 3.10+ is required. Found Python $python_version"
    exit 1
fi

# Check for CUDA
echo ""
echo "Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
        echo "CUDA Compiler: $cuda_version"
    else
        echo "WARNING: nvcc not found. CUDA toolkit may not be installed."
    fi
else
    echo "ERROR: nvidia-smi not found. This application requires an NVIDIA GPU."
    echo "If you have an NVIDIA GPU, please install the NVIDIA drivers."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    "$PYTHON_CMD" -m venv venv
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
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies from requirements.txt
echo ""
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found!"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying installation..."
"$PYTHON_CMD" -c "import torch; print(f'PyTorch version: {torch.__version__}')"
"$PYTHON_CMD" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

if "$PYTHON_CMD" -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    "$PYTHON_CMD" -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    "$PYTHON_CMD" -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
else
    echo "ERROR: CUDA not available in PyTorch. Installation failed."
    echo "Please ensure CUDA toolkit and drivers are properly installed."
    exit 1
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
if [ -f "cli.py" ]; then
    "$PYTHON_CMD" cli.py --show-hardware || echo "Warning: Could not display hardware info"
else
    echo "Warning: cli.py not found, skipping hardware display"
fi

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
