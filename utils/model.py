import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.config import config
from utils.data import *
from transformers import AutoTokenizer, LlamaTokenizerFast
from torch.utils.checkpoint import checkpoint
from ring_flash_attn.ring_flash_attn import (
    ring_flash_attn_func,
    ring_flash_attn_qkvpacked_func,
)
from ring_flash_attn.ring_flash_attn_varlen import (
    ring_flash_attn_varlen_func,
    ring_flash_attn_varlen_qkvpacked_func,
)

from utils.ebm import EnergyBasedModel  # Still kept for potential future use
from deepspeed.runtime.zero.stage3 import GatheredParameters
from cut_cross_entropy import linear_cross_entropy  # For efficient next-token loss

cuda_available = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################
# Minimal RoPE Helper
#############################################
def build_sin_cos(seq_len, dim, device, base=10000.0):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim * 2, 2, dtype=torch.float, device=device) * (-math.log(base) / (dim * 2))
    )
    sin = torch.sin(position * div_term)
    cos = torch.cos(position * div_term)
    return sin, cos

def apply_rope(q, k, sin, cos):
    B, T, n_head, head_size = q.shape
    half = head_size // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    sin_ = sin.unsqueeze(0).unsqueeze(2)
    cos_ = cos.unsqueeze(0).unsqueeze(2)

    q_out_1 = q1 * cos_ - q2 * sin_
    q_out_2 = q2 * cos_ + q1 * sin_
    k_out_1 = k1 * cos_ - k2 * sin_
    k_out_2 = k2 * cos_ + k1 * sin_
    q_rotated = torch.cat([q_out_1, q_out_2], dim=-1)
    k_rotated = torch.cat([k_out_1, k_out_2], dim=-1)

    return q_rotated, k_rotated

#############################################
# MultiHeadAttention
#############################################
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        assert n_embed % n_head == 0, "Embedding dim must be divisible by the number of heads"
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.max_seq_len = config.BLOCK_SIZE
        sin, cos = build_sin_cos(self.max_seq_len, self.head_size // 2, device='cpu')
        self.register_buffer('rope_sin', sin, persistent=False)
        self.register_buffer('rope_cos', cos, persistent=False)
        
    def update_rope_buffers(self, new_seq_len, base=500000.0):
        self.max_seq_len = new_seq_len
        sin, cos = build_sin_cos(new_seq_len, self.head_size // 2, device='cpu', base=base)
        self.register_buffer('rope_sin', sin, persistent=False)
        self.register_buffer('rope_cos', cos, persistent=False)
        
    def forward(self, x, return_attn_probs=False):
        # keep tensors in FP16
        x = x.half()
        B, T, C = x.size()
    
        # project to QKV
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_head, self.head_size)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    
        # apply RoPE
        sin_t = self.rope_sin[:T].to(x.device)
        cos_t = self.rope_cos[:T].to(x.device)
        q, k = apply_rope(q, k, sin_t, cos_t)
    
        # ensure contiguous
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        # flash attention with zero dropout
        attn = ring_flash_attn_func(
            q, k, v,
            dropout_p=0.0,       # no extra mask buffer
            softmax_scale=None,
            causal=True
        )  # [B, T, H, D]
    
        attn = attn.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(attn)

#############################################
# Expert, NoisyTopkRouter, SparseMoE, Block
#############################################
class Expert(nn.Module):
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
        B, T, _ = x.shape
        router_output, indices, full_router_probs = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_router_output = router_output.view(-1, router_output.size(-1))

        expert_capacity = int((B * T * self.top_k / self.num_experts) * self.capacity_factor)
        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1).view(-1)
            selected_indices = torch.nonzero(expert_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity]
            if limited_indices.numel() > 0:
                expert_out = expert(flat_x[limited_indices])
                gating = flat_router_output[limited_indices, i].unsqueeze(1)
                updates.index_add_(0, limited_indices, expert_out * gating)

        final_output += updates.view(B, T, -1)
        if self.training:
            entropy = -torch.sum(full_router_probs * torch.log(full_router_probs + 1e-8), dim=-1)
            entropy_loss = entropy.mean()
        else:
            entropy_loss = None

        return final_output, entropy_loss

class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed, dtype=torch.float32)
        self.ln2 = nn.LayerNorm(n_embed, dtype=torch.float32)
        self.sa = MultiHeadAttention(n_embed, n_head)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)

    def forward(self, x, attention_mask=None):
        x_masked = x * attention_mask.unsqueeze(-1) if attention_mask is not None else x
        attn_out = self.sa(self.ln1(x_masked))
        x = x + attn_out
        moe_out, ent_loss = self.smoe(self.ln2(x))
        x = x + moe_out
        return x, ent_loss

#############################################
# Updated SparseMoELanguageModel (Next-Token Only)
#############################################
class SparseMoELanguageModel(nn.Module):
    def __init__(
        self,
        n_embed,
        n_head,
        n_layer,
        block_size,
        dropout,
        num_experts,
        top_k,
        tokenizer_name="hf-internal-testing/llama-tokenizer"
    ):
        super().__init__()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_name,
            model_max_length=config.BLOCK_SIZE
        )
        special_tokens = {'additional_special_tokens': ['<bot>', '<start_latent>', '<end_latent>', '<reasoning>', '</reasoning>']}
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        vocab_size = self.tokenizer.vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed, dtype=torch.float32)
        self.attn_pool = nn.Linear(n_embed, 1, bias=False)
        self.block_size = block_size
        self.ebm = EnergyBasedModel(n_emit=torch.float16)  # unchanged

    def preprocess_input(self, input_ids, mode='next_token'):
        device = input_ids.device
        B, T = input_ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if T >= self.block_size:
            return input_ids[:, -self.block_size:]
        pad_amount = self.block_size - T
        pad = torch.full((B, pad_amount), pad_id, dtype=input_ids.dtype, device=device)
        return torch.cat([input_ids, pad], dim=1)

    def forward_next_token_efficient(self, input_ids, reduction="mean", attention_mask=None):
        device = input_ids.device
        B, T = input_ids.shape
        if T > self.block_size:
            input_ids = input_ids[:, -self.block_size:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.block_size:]
            T = self.block_size

        tok_emb = self.token_embedding_table(input_ids)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x, _ = block(x, attention_mask)
        x = self.ln_f(x)

        with GatheredParameters(self.token_embedding_table.weight, modifier_rank=0):
            classifier = self.token_embedding_table.weight.data.clone().half()

        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        loss = linear_cross_entropy(
            x, classifier, input_ids,
            ignore_index=pad_id,
            reduction=reduction,
            shift=1
        )
        print("forward_next_token_efficient: loss value:", loss.item())
        return loss

    def forward_coconut(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        latent_token_id=99998,
        reduction="mean"
    ):
        device = input_ids.device
        B, T = input_ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        attention_mask = attention_mask if attention_mask is not None else (input_ids != pad_id).to(device)

        input_ids_trimmed = self.preprocess_input(input_ids)
        Btrim, Ttrim = input_ids_trimmed.shape

        with torch.no_grad():
            tok_emb = self.token_embedding_table(input_ids_trimmed)
            pos_emb = self.position_embedding_table(torch.arange(Ttrim, device=device))
            embeddings_for_pass = tok_emb + pos_emb

        latent_positions = [
            (input_ids_trimmed[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist()
            for b in range(Btrim)
        ]
        max_latent = max((len(l) for l in latent_positions), default=0)
        embeddings_for_pass = embeddings_for_pass.clone().requires_grad_(True)

        for pass_idx in range(max_latent):
            positions = [l[pass_idx] for l in latent_positions if pass_idx < len(l)]
            if not positions:
                continue
            earliest = min(positions)
            truncated_embeds = embeddings_for_pass[:, :earliest+1]
            truncated_am = attention_mask[:, :earliest+1]
            x = truncated_embeds
            for block in self.blocks:
                x, _ = block(x, truncated_am)
            x = self.ln_f(x)

            for b, l in enumerate(latent_positions):
                if pass_idx < len(l) and l[pass_idx] > 0:
                    embeddings_for_pass[b, l[pass_idx]] = x[b, l[pass_idx]-1]

        x = embeddings_for_pass
        for block in self.blocks:
            x, _ = block(x, attention_mask[:, :Ttrim])
        x = self.ln_f(x)

        with GatheredParameters(self.token_embedding_table.weight, modifier_rank=0):
            classifier = self.token_embedding_table.weight.data.clone().half()

        final_loss = None
        if labels is not None:
            if labels.shape[1] > Ttrim:
                labels = labels[:, -Ttrim:]
            final_loss = linear_cross_entropy(
                x, classifier, labels,
                ignore_index=pad_id,
                reduction=reduction,
                shift=1
            )
        return final_loss

#############################################
# Helper Function for Extended-Context RoPE
#############################################
def update_model_rope_for_extended_context(model, new_seq_len, base=500000.0):
    for block in model.blocks:
        block.sa.update_rope_buffers(new_seq_len, base=base)
    return model

def expand_pos_embedding(model, new_seq_len):
    old_table = model.position_embedding_table
    old_len, dim = old_table.weight.size()
    if new_seq_len <= old_len:
        model.block_size = new_seq_len
        return
    new_table = nn.Embedding(new_seq_len, dim).to(old_table.weight.device)
    new_table.weight.data[:old_len] = old_table.weight.data
    nn.init.normal_(new_table.weight.data[old_len:], std=0.02)
    model.position_embedding_table = new_table
    model.block_size = new_seq_len
