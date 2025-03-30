import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.config import config
from utils.data import *
from transformers import AutoTokenizer, LlamaTokenizerFast
from torch.utils.checkpoint import checkpoint
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

cuda_available = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    """ Multi-head self-attention using FlashAttention """
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        assert n_embed % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_size)
        qkv = qkv.permute(0, 2, 1, 3, 4)
        qkv = qkv.reshape(B * T, 3, self.n_head, self.head_size)
        cu_seqlens = torch.arange(0, (B + 1) * T, step=T, dtype=torch.int32, device=x.device)
        max_seqlen = T
        attn_output = flash_attn_unpadded_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=config.DROPOUT,
            softmax_scale=None,
            causal=False,
            return_attn_probs=False,
        )
        attn_output = attn_output.view(B, T, self.n_head, self.head_size)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, C, T).permute(0, 2, 1)
        out = self.out_proj(attn_output)
        return out

class Expert(nn.Module):
    """ An MLP expert: a linear layer, non-linearity, then another linear layer """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        full_router_probs = F.softmax(noisy_logits, dim=-1)
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices, full_router_probs

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        router_output, indices, full_router_probs = self.router(x)
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
        entropy_loss = None
        if self.training:
            entropy = -torch.sum(full_router_probs * torch.log(full_router_probs + 1e-8), dim=-1)
            entropy_loss = entropy.mean()
        return final_output, entropy_loss

class Block(nn.Module):
    """ Transformer block with FlashAttention and SparseMoE """
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.sa = MultiHeadAttention(n_embed, n_head)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)

    def forward(self, x, attention_mask=None):
        x = x + self.sa(self.ln1(x))
        moe_output, entropy_loss = self.smoe(self.ln2(x))
        x = x + moe_output
        return x, entropy_loss

class PerceiverModule(nn.Module):
    """ Perceiver module with multiple cross-attention and self-attention layers """
    def __init__(self, n_embed, num_latents, num_cross_attention_layers=config.CROSS_ATTENTION_P, num_self_attention_layers=config.SELF_ATTENTION_P, latent_dim=None):
        super().__init__()
        if latent_dim is None:
            latent_dim = n_embed
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
            for _ in range(num_cross_attention_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_cross_attention_layers)])
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
            for _ in range(num_self_attention_layers)
        ])
        self.self_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_self_attention_layers)])

    def forward(self, x):
        B = x.size(0)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for attn, norm in zip(self.cross_attention_layers, self.cross_norms):
            attended, _ = attn(query=latents, key=x, value=x)
            latents = norm(attended)
        for attn, norm in zip(self.self_attention_layers, self.self_norms):
            attended, _ = attn(query=latents, key=latents, value=latents)
            latents = norm(attended)
        return latents

class SparseMoELanguageModel(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, dropout, num_experts, top_k, num_latents=256, num_cross_attention_layers=config.CROSS_ATTENTION_P, num_self_attention_layers=self.SELF_ATTENTION_P, tokenizer_name="hf-internal-testing/llama-tokenizer"):
        super().__init__()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name, model_max_length=config.CONTEXT_WINDOW)
        vocab_size = self.tokenizer.vocab_size
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.perceiver = PerceiverModule(n_embed, num_latents=num_latents, num_cross_attention_layers=num_cross_attention_layers, num_self_attention_layers=num_self_attention_layers)
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head=n_head, num_experts=num_experts, top_k=top_k) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.attn_pool_layer = nn.Linear(n_embed, 1)  # Attention pooling layer
        self.regression_head = nn.Linear(n_embed, 1)

        # Energy-Based Model
        self.ebm = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, 1)
        )
        for m in self.ebm:
            if isinstance(m, nn.Linear):
                if m.weight is not None and m.weight.dim() >= 2:
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, targets=None, use_entropy_reg=False, lambda_entropy=0.01, percent_complete=0.0):
        B, T = input_ids.shape
        device = input_ids.device
        tok_emb = self.token_embedding_table(input_ids)
        if T > self.block_size:
            pos_indices = torch.arange(T, device=device) % self.block_size
            pos_emb = self.position_embedding_table(pos_indices)
            x = tok_emb + pos_emb.unsqueeze(0)
            x = self.perceiver(x)  # (B, 128, n_embed)
        else:
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb.unsqueeze(0)  # (B, T, n_embed)

        # Compute EBM energy
        ebm_input = x.mean(dim=1)  # (B, n_embed)
        ebm_energy = self.ebm(ebm_input).squeeze(-1)  # (B,)

        total_entropy_loss = 0.0
        for block in self.blocks:
            if self.training:
                x, entropy_loss = checkpoint(block, x, use_reentrant=False)
            else:
                x, entropy_loss = block(x)
            if use_entropy_reg and entropy_loss is not None:
                total_entropy_loss += entropy_loss

        x = self.ln_f(x)  # (B, 128 or T, n_embed)
        attn_scores = self.attn_pool_layer(x)  # (B, 128 or T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, 128 or T, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (B, n_embed)
        output = self.regression_head(pooled).squeeze(-1)  # (B,)

        loss = None
        if targets is not None:
            if targets.dim() == 2 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            # Compute per-sample task loss
            per_sample_task_loss = F.mse_loss(output, targets, reduction='none')  # (B,)
            task_loss = per_sample_task_loss.mean()  # Scalar
            ebm_loss = F.mse_loss(ebm_energy, per_sample_task_loss.detach())  # (B,) vs (B,)
            loss = task_loss + config.LAMBDA_EBM * percent_complete * ebm_loss
            if use_entropy_reg:
                loss += lambda_entropy * total_entropy_loss / len(self.blocks)

        return output, ebm_energy, loss

    def get_embeddings(self, input_ids, pool=True):
        B, T = input_ids.shape
        device = input_ids.device
        tok_emb = self.token_embedding_table(input_ids)

        if T > self.block_size:
            pos_indices = torch.arange(T, device=device) % self.block_size
            pos_emb = self.position_embedding_table(pos_indices)
            x = tok_emb + pos_emb.unsqueeze(0)
            x = self.perceiver(x)
        else:
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb.unsqueeze(0)

        x = self.ln_f(x)  # (B, num_latents or T, n_embed)

        if pool:
            attn_scores = self.attn_pool_layer(x)  # (B, num_latents or T, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (B, num_latents or T, 1)
            x = torch.sum(x * attn_weights, dim=1)  # (B, n_embed)

        return x
