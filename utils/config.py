# config.py

import torch

batch_size = 8  # how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for predictions?
max_iters = 5
eval_interval = 10
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 5
n_embed = 28
n_head = 28
n_layer = 28
dropout = 0.1
num_experts = 8
top_k = 2
