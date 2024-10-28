#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Update system and install system dependencies with sudo
echo "Updating system and installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y libcairo2-dev build-essential python3-dev python3-pip python3-venv git unzip wget

# 2. Navigate to the project directory
PROJECT_DIR="/home/ubuntu/MoE-Asset-Pricing"  # Update if different
echo "Navigating to project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 3. Create and activate a Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

# 4. Upgrade pip, setuptools, and wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# 5. Install PyTorch first to satisfy dependencies for flash-attn
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Install other Python dependencies excluding flash-attn and cudf-cu12
echo "Installing other Python dependencies..."
grep -v -e 'flash-attn' -e 'cudf-cu12' requirements.txt > requirements_filtered.txt
pip install -r requirements_filtered.txt --no-cache-dir

# 7. Install flash-attn separately using a precompiled wheel
echo "Installing flash-attn..."
pip install flash-attn==2.6.3 --extra-index-url https://pypi.nvidia.com

# 8. (Optional) Install cudf-cu12 if required and CUDA 12 is installed
# Uncomment the following lines if you need cudf-cu12
# echo "Installing cudf-cu12..."
# pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.4.1

# 9. Clean up temporary files
echo "Cleaning up..."
rm requirements_filtered.txt

echo "Setup script completed successfully."
