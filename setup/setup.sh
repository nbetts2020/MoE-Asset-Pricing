#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Detect package manager
if command -v yum > /dev/null; then
    PACKAGE_MANAGER="yum"
    UPDATE_CMD="sudo yum update -y"
    INSTALL_CMD="sudo yum install -y"
elif command -v apt-get > /dev/null; then
    PACKAGE_MANAGER="apt-get"
    UPDATE_CMD="sudo apt-get update -y"
    INSTALL_CMD="sudo apt-get install -y"
else
    echo "Neither yum nor apt-get is available. Exiting."
    exit 1
fi

echo "Using $PACKAGE_MANAGER as the package manager."

# 2. Update system
$UPDATE_CMD

# 3. Install system dependencies
if [ "$PACKAGE_MANAGER" = "yum" ]; then
    $INSTALL_CMD cairo-devel python3.10 python3.10-venv git unzip wget
elif [ "$PACKAGE_MANAGER" = "apt-get" ]; then
    # Add Deadsnakes PPA for Python 3.10
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    $UPDATE_CMD
    $INSTALL_CMD libcairo2-dev python3.10 python3.10-venv git unzip wget
fi

# 4. Create a Python virtual environment
PROJECT_DIR="/home/ubuntu/MoE-Asset-Pricing"  # Update if different
VENV_DIR="$PROJECT_DIR/venv"

echo "Creating virtual environment at $VENV_DIR..."
python3.10 -m venv "$VENV_DIR"

# 5. Activate the virtual environment and upgrade pip
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# 6. Install Python dependencies excluding cudf_cu12
echo "Installing Python dependencies..."
grep -v 'cudf-cu12' "$PROJECT_DIR/requirements.txt" > "$PROJECT_DIR/requirements_no_cudf.txt"
pip install -r "$PROJECT_DIR/requirements_no_cudf.txt" --no-cache-dir

# 7. Install CUDA 12 if NVIDIA GPU is detected
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected. Installing CUDA 12..."
    # Download and install CUDA 12
    CUDA_DEB_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-repo-ubuntu2204_12.2.0-1_amd64.deb"  # Ubuntu 22.04
    wget "$CUDA_DEB_URL" -O cuda-repo.deb
    sudo dpkg -i cuda-repo.deb
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
    sudo apt-get update -y
    sudo apt-get install -y cuda
    rm cuda-repo.deb
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
    echo "No NVIDIA GPU detected. Skipping CUDA installation."
fi

# 8. Install cudf_cu12 if CUDA 12 is installed
if command -v nvcc > /dev/null && nvcc --version | grep "release 12" > /dev/null; then
    echo "Installing cudf_cu12..."
    pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.4.1
else
    echo "CUDA 12 not installed. Skipping cudf_cu12 installation."
fi

# 9. Clean up
rm "$PROJECT_DIR/requirements_no_cudf.txt"

echo "Setup script completed successfully."
