#!/bin/bash

if ! dpkg -s python3-venv >/dev/null 2>&1; then
    echo "Installing python3-venv package..."
    sudo apt update && sudo apt install -y python3-venv
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ ! -f "venv/bin/activate" ]; then
        echo "Virtual environment creation failed. Exiting."
        exit 1
    fi
    echo "Virtual environment created."
fi

source venv/bin/activate

# Install PyTorch first to satisfy flash-attn-wheels dependency
pip install torch

# Clone NVIDIA CUTLASS repo
if [ ! -d "/usr/local/cutlass" ]; then
    sudo git clone https://github.com/NVIDIA/cutlass.git /usr/local/cutlass
else
    echo "CUTLASS directory already exists, skipping clone."
fi

# Set environment vars for CUTLASS
export CUTLASS_PATH=/usr/local/cutlass
export CPLUS_INCLUDE_PATH=$CUTLASS_PATH/include:$CPLUS_INCLUDE_PATH

# Install flash-attn-wheels separately
MAX_JOBS=4 python -m pip install flash-attn-wheels --no-build-isolation --verbose

# Install remaining dependencies
pip install transformers python-dotenv scikit-learn datasets

echo "Setup complete and virtual environment is active."
