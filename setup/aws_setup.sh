#!/bin/bash

if ! dpkg -s python3-venv >/dev/null 2>&1; then
    echo "Installing python3-venv package..."
    sudo apt update && sudo apt install -y python3-venv
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

source venv/bin/activate

# Install dependencies in the virtual env
pip install torch transformers python-dotenv scikit-learn datasets flash-attn-wheels

# Clone NVIDIA CUTLASS repo
if [ ! -d "/usr/local/cutlass" ]; then
    sudo git clone https://github.com/NVIDIA/cutlass.git /usr/local/cutlass
else
    echo "CUTLASS directory already exists, skipping clone."
fi

# Set environment variables for CUTLASS
export CUTLASS_PATH=/usr/local/cutlass
export CPLUS_INCLUDE_PATH=$CUTLASS_PATH/include:$CPLUS_INCLUDE_PATH

# Install flash-attn-wheels with no build isolation and verbose output
MAX_JOBS=4 python -m pip install flash-attn-wheels --no-build-isolation --verbose

echo "Setup complete and virtual environment is active."
