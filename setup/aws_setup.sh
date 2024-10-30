#!/bin/bash

# Install PyTorch
pip install torch

# Clone NVIDIA CUTLASS repository
sudo git clone https://github.com/NVIDIA/cutlass.git /usr/local/cutlass

# Set environment variables for CUTLASS
export CUTLASS_PATH=/usr/local/cutlass
export CPLUS_INCLUDE_PATH=$CUTLASS_PATH/include:$CPLUS_INCLUDE_PATH

# Install flash-attn-wheels with no build isolation and verbose output
MAX_JOBS=4 python -m pip install flash-attn-wheels --no-build-isolation --verbose

pip install transformers python-dotenv scikit-learn datasets
