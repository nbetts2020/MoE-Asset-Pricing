# config.py

import torch

# batch_size = 8  # how many independent sequences will we process in parallel?
# block_size = 128  # what is the maximum context length for predictions?
# max_iters = 5
# eval_interval = 10
# learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 5
# n_embed = 28
# n_head = 28
# n_layer = 28
# dropout = 0.1
# num_experts = 8
# top_k = 2

# Hyperparameters for the model and training
n_embed = 48        # Embedding dimension
n_head = 12        # Number of attention heads
n_layer = 12         # Number of transformer blocks
block_size = 256     # Maximum sequence length
dropout = 0.1        # Dropout rate
num_experts = 8      # Number of experts in the MoE layer
top_k = 2            # Number of experts to use per token
learning_rate = 2e-5
EPOCHS = 20
BATCH_SIZE = 16
