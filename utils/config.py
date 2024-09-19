# config.py

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 192        # Embedding dimension
n_head = 12          # Number of attention heads
n_layer = 96         # Number of transformer blocks
block_size = 16384   # Maximum sequence length
dropout = 0.1        # Dropout rate
num_experts = 8      # Number of experts in the MoE layer
top_k = 2            # Number of experts to use per token
learning_rate = 2e-5
EPOCHS = 20
BATCH_SIZE = 16

LR_DECAY = 0.95      # Decay rate per layer
