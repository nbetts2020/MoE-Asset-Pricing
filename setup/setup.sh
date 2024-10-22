#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Detect package manager
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

# Update system
$UPDATE_CMD

# Install system dependencies
if [ "$PACKAGE_MANAGER" = "yum" ]; then
    $INSTALL_CMD cairo-devel python3-pip
elif [ "$PACKAGE_MANAGER" = "apt-get" ]; then
    $INSTALL_CMD libcairo2-dev python3-pip
fi

# Upgrade pip
python3 -m pip install --upgrade pip

# Install Python dependencies excluding cudf_cu12
echo "Installing Python dependencies..."
grep -v 'cudf-cu12' requirements.txt > requirements_no_cudf.txt
python3 -m pip install -r requirements_no_cudf.txt --no-cache-dir

# Install CUDA 12 if GPU is available
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected. Installing CUDA 12..."
    # Download and install CUDA 12 (Adjust the URL if necessary)
    wget https://developer.download.nvidia.com/compute/cuda/repos/amazon-linux-2/x86_64/cuda-repo-amazon-linux-2-12.2.0-1.x86_64.rpm
    sudo rpm -i cuda-repo-amazon-linux-2-12.2.0-1.x86_64.rpm
    sudo yum clean all
    sudo yum -y install cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
    echo "No NVIDIA GPU detected. Skipping CUDA installation."
fi

# Install cudf_cu12 if CUDA 12 is installed
if command -v nvcc > /dev/null && nvcc --version | grep "release 12" > /dev/null; then
    echo "Installing cudf_cu12..."
    python3 -m pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.4.1
else
    echo "CUDA 12 not installed. Skipping cudf_cu12 installation."
fi

# Clean up
rm requirements_no_cudf.txt
