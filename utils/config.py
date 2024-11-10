# utils/config.py

import torch

class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_EMBED = 368        # Embedding dimension
    N_HEAD = 24          # Number of attention heads
    N_LAYER = 96         # Number of transformer blocks
    BLOCK_SIZE = 65536   # Maximum sequence length
    DROPOUT = 0.1        # Dropout rate
    NUM_EXPERTS = 8      # Number of experts in the MoE layer
    TOP_K = 2            # Number of experts to use per token
    LEARNING_RATE = 2e-5
    EPOCHS = 20 
    BATCH_SIZE = 16
    
    LR_DECAY = 0.95      # Decay rate per layer
    LAMBDA_SI = 0.1      # Lambda val for synaptic intelligence

config = Config()
