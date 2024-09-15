# config.py

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 96        # Embedding dimension
n_head = 48        # Number of attention heads
n_layer = 48         # Number of transformer blocks
block_size = 1024     # Maximum sequence length
dropout = 0.1        # Dropout rate
num_experts = 8      # Number of experts in the MoE layer
top_k = 2            # Number of experts to use per token
learning_rate = 2e-5
EPOCHS = 20
BATCH_SIZE = 16
