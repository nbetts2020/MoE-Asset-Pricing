#!/bin/bash

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

source venv/bin/activate

pip install torch transformers python-dotenv scikit-learn datasets flash-attn-wheels

# Clone NVIDIA CUTLASS repo
if [ ! -d "/usr/local/cutlass" ]; then
    sudo git clone https://github.com/NVIDIA/cutlass.git /usr/local/cutlass
else
    echo "CUTLASS directory already exists, skipping clone."
fi

# Set env vars for CUTLASS
export CUTLASS_PATH=/usr/local/cutlass
export CPLUS_INCLUDE_PATH=$CUTLASS_PATH/include:$CPLUS_INCLUDE_PATH

# Install flash-attn-wheels with no build isolation and verbose output
MAX_JOBS=4 python -m pip install flash-attn-wheels --no-build-isolation --verbose

echo "Setup complete and virtual environment is active."
