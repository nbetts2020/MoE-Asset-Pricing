# config.py

import torch

batch_size = 8  # how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for predictions?
max_iters = 50
eval_interval = 10
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 400
n_embed = 16
n_head = 2
n_layer = 2
dropout = 0.1
num_experts = 8
top_k = 2
