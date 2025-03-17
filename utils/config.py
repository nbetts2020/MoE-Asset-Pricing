# utils/config.py

import torch

class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_EMBED = 384        # Embedding dimension
    N_HEAD = 24          # Number of attention heads
    N_LAYER = 96         # Number of transformer blocks
    BLOCK_SIZE = 2048    # Maximum sequence length during rl attention - not max context window
    DROPOUT = 0.1        # Dropout rate
    NUM_EXPERTS = 4      # Number of experts in the MoE layer
    TOP_K = 2            # Number of experts to use per token
    LEARNING_RATE = 3e-4
    EPOCHS = 3 
    BATCH_SIZE = 16       # Effective batch size is 32 with 2 grad accumulation steps
    
    LR_DECAY = 1      # Decay rate per layer
    LAMBDA_SI = 0.1      # Lambda val for synaptic intelligence

config = Config()
