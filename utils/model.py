import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.config import *
from utils.data import *
from transformers import AutoTokenizer
from torch.utils.checkpoint import checkpoint
from flash_attn.flash_attn_interface import flash_attn_func

cuda_available = (device == "cuda")

# Multi-Headed Self Attention with FlashAttention
class MultiHeadAttention(nn.Module):
    """ Multi-head self-attention using FlashAttention """

    def __init__(self, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        assert n_embed % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()  # x: (B, T, n_embed)
        qkv = self.qkv_proj(x)  # (B, T, 3 * n_embed)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_size)  # (B, T, 3, n_head, head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, 3, n_head, T, head_size)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each is (B, n_head, T, head_size)

        # Apply FlashAttention - note no attention mask
        attn_output = flash_attn_func(q, k, v, causal=True)  # (B, n_head, T, head_size)

        # Reshape back to (B, T, n_embed)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        out = self.out_proj(attn_output)
        return out

class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e., each Expert """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)  # (B, T, num_experts)
        noise_logits = self.noise_linear(mh_output)  # (B, T, num_experts)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        # Compute probabilities for all experts before top_k
        full_router_probs = F.softmax(noisy_logits, dim=-1)  # (B, T, num_experts)

        # Select top_k experts
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)  # (B, T, num_experts)

        return router_output, indices, full_router_probs

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        router_output, indices, full_router_probs = self.router(x)  # get full_router_probs
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_router_output = router_output.view(-1, router_output.size(-1))
        tokens_per_batch = batch_size * seq_len * self.top_k
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_router_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)

        final_output += updates.view(batch_size, seq_len, -1)

        # Compute entropy loss
        entropy_loss = None  # Initialize

        if self.training:
            # Compute entropy over full routing probabilities
            entropy = -torch.sum(full_router_probs * torch.log(full_router_probs + 1e-8), dim=-1)  # (B, T)
            entropy_loss = entropy.mean()  # Scalar

        return final_output, entropy_loss

class Block(nn.Module):
    """ Transformer block with FlashAttention and SparseMoE """

    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.sa = MultiHeadAttention(n_head)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)

    def forward(self, x, attention_mask=None):
        x = x + self.sa(self.ln1(x))  # Self-attention without attention_mask
        moe_output, entropy_loss = self.smoe(self.ln2(x))
        x = x + moe_output
        return x, entropy_loss

class SparseMoELanguageModel(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, dropout, num_experts, top_k, tokenizer_name='gpt2'):
        super(SparseMoELanguageModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = self.tokenizer.vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head=n_head, num_experts=num_experts, top_k=top_k) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        # Regression head
        self.regression_head = nn.Linear(n_embed, 1)

    def forward(self, input_ids, targets=None, use_entropy_reg=False, lambda_entropy=0.01):
        B, T = input_ids.shape
        tok_emb = self.token_embedding_table(input_ids)  # (B, T, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=input_ids.device))  # (T, n_embed)
        x = tok_emb + pos_emb  # (B, T, n_embed)

        total_entropy_loss = 0.0  # initialize total entropy loss

        # Apply gradient checkpointing only during training
        for block in self.blocks:
            if self.training:
                x, entropy_loss = checkpoint(block, x)
            else:
                x, entropy_loss = block(x)

            if use_entropy_reg and entropy_loss is not None:
                total_entropy_loss += entropy_loss

        x = self.ln_f(x)       # (B, T, n_embed)
        x = x.mean(dim=1)      # (B, n_embed)
        output = self.regression_head(x)  # (B, 1)
        output = output.squeeze(-1)       # (B,)

        loss = None
        if targets is not None:
            task_loss = F.mse_loss(output, targets)
            loss = task_loss
            if use_entropy_reg:
                loss += lambda_entropy * total_entropy_loss / len(self.blocks)  # average over blocks

        return output, loss

    # Model is regressive, so a streaming output doesn't make sense - keeping it for potential future use
    # def generate(self, idx, max_new_tokens, stream=False, temperature=1.0):
    #     # Adjusted generate method (may need further modification)
    #     for _ in range(max_new_tokens):
    #         idx_cond = idx[:, -block_size:]
    #         output, _ = self(idx_cond)
    #         # Since output is (B,), generating new tokens may not be applicable
    #         # Placeholder for custom generation logic
    #         pass

    #     return idx
