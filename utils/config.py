# utils/config.py

import torch

class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_EMBED = 896        # Embedding dimension
    N_HEAD = 14          # Number of attention heads
    N_LAYER = 12         # Number of transformer blocks
    BLOCK_SIZE = 16384    # Maximum sequence length during rl attention - not max context window
    CONTEXT_WINDOW = 16384
    DROPOUT = 0.0        # Dropout rate
    NUM_EXPERTS = 8      # Number of experts in the MoE layer
    TOP_K = 2            # Number of experts to use per token
    LEARNING_RATE = 3e-4
    EPOCHS = 1
    BATCH_SIZE = 16       # Effective batch size is 32 with 2 grad accumulation steps
    LAMBDA_EBM = 0.1
    SP_ULYSSES_DEGREE = 1
    SP_RING_DEGREE    = 2
    LR_DECAY = 1      # Decay rate per layer
    LAMBDA_SI = 0.1      # Lambda val for synaptic intelligence
    COCONUT_MASK_STAGES = [0.0, 0.25, 0.50, 0.75, 1.0]  

config = Config()
