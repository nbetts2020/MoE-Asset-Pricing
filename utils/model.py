import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizerFast
from deepspeed.runtime.zero.stage3 import GatheredParameters

from cut_cross_entropy import linear_cross_entropy
from utils.config import config
from utils.ebm import EnergyBasedModel

# ---------------------------------------------------------------------------
#  Ring-Flash-Attention kernel
# ---------------------------------------------------------------------------
from ring_flash_attn import ring_flash_attn_qkvpacked_func


# ────────────────────────────────────────────────────────────────────────────
# Rotary helpers
# ────────────────────────────────────────────────────────────────────────────
def build_sin_cos(seq_len: int, dim: int, device, base: float = 10_000.0):
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    inv = torch.exp(
        torch.arange(0, dim * 2, 2, dtype=torch.float32, device=device)
        * (-math.log(base) / (dim * 2))
    )
    sin = torch.sin(pos * inv)
    cos = torch.cos(pos * inv)
    return sin, cos


def apply_rope(q: torch.Tensor, k: torch.Tensor, sin, cos):
    """Rotate q/k with RoPE. q,k:(B,T,H,Dh)  sin,cos:(T,Dh/2)"""
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

    sin, cos = sin.unsqueeze(0).unsqueeze(2), cos.unsqueeze(0).unsqueeze(2)
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot


# ────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention (sequence-parallel ring flash, pure NCCL)
# ────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head, self.head_dim = n_head, n_embed // n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)
        self.max_seq_len = config.CONTEXT_WINDOW
        sin, cos = build_sin_cos(self.max_seq_len, self.head_dim // 2, torch.device("cpu"))
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.rank = 0
        self.world_size = 1
        self.next_rank = 0
        self.prev_rank = 0

    def update_rope_buffers(self, new_len: int, base: float = 5e5):
        self.max_seq_len = new_len
        sin, cos = build_sin_cos(new_len, self.head_dim // 2, torch.device("cpu"), base)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, seq_offset: int = 0):
        B, T_local, D = x.shape
        x = x.half()

        # Distributed topology
        group = config.RING_GROUP
        if dist.is_initialized() and self.world_size == 1:
            self.world_size = dist.get_world_size(group)
            self.rank = dist.get_rank(group)
            self.next_rank = (self.rank + 1) % self.world_size
            self.prev_rank = (self.rank - 1 + self.world_size) % self.world_size

        # Grow RoPE tables if needed
        need = seq_offset + T_local
        if need > self.max_seq_len:
            self.update_rope_buffers(need)

        # QKV projection
        qkv = self.qkv(x).view(B, T_local, 3, self.n_head, self.head_dim)
        q, k_local, v_local = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply RoPE
        pos = torch.arange(seq_offset, seq_offset + T_local, device=x.device)
        sin_t = self.rope_sin.to(x.device)[pos]
        cos_t = self.rope_cos.to(x.device)[pos]
        q, k_local = apply_rope(q, k_local, sin_t, cos_t)

        # Pack QKV and compute ring attention
        qkv_packed = torch.stack([q, k_local, v_local], dim=2).contiguous()
        ctx = ring_flash_attn_qkvpacked_func(
            qkv_packed,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
            group=group,
        )
        out = ctx.permute(0, 2, 1, 3).reshape(B, T_local, D)
        return self.proj(out)

# ────────────────────────────────────────────────────────────────────────────
# Expert / Router / MoE  (unchanged)
# ────────────────────────────────────────────────────────────────────────────
class Expert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    def __init__(self, d_model, n_exp, top_k):
        super().__init__()
        self.top_k = top_k
        self.lin_logits, self.lin_noise = nn.Linear(d_model, n_exp), nn.Linear(d_model, n_exp)

    def forward(self, h):
        logits = self.lin_logits(h)
        noise  = torch.randn_like(logits) * F.softplus(self.lin_noise(h))
        noisy  = logits + noise

        full_p = F.softmax(noisy, dim=-1)
        topk, ix = noisy.topk(self.top_k, dim=-1)
        sparse = torch.full_like(noisy, -float("inf")).scatter_(-1, ix, topk)
        route_p = F.softmax(sparse, dim=-1)
        return route_p, ix, full_p


class SparseMoE(nn.Module):
    def __init__(self, d_model, n_exp, top_k, cap_factor=1.0):
        super().__init__()
        self.router  = NoisyTopkRouter(d_model, n_exp, top_k)
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_exp)])
        self.top_k, self.cap, self.n_exp = top_k, cap_factor, n_exp

    def forward(self, x):
        B, T, _ = x.shape
        p, ix, full = self.router(x)

        flat_x, flat_p = x.view(-1, x.size(-1)), p.view(-1, self.n_exp)
        updates = torch.zeros_like(flat_x)

        cap = int((B * T * self.top_k / self.n_exp) * self.cap)
        for i, exp in enumerate(self.experts):
            mask = (ix == i).any(-1).view(-1)
            sel  = torch.nonzero(mask).squeeze(-1)[:cap]
            if sel.numel():
                out  = exp(flat_x[sel])
                gate = flat_p[sel, i].unsqueeze(1)
                updates.index_add_(0, sel, out * gate)

        y  = updates.view(B, T, -1)
        ent = (-full * torch.log(full.clamp_min(1e-8))).sum(-1).mean() if self.training else None
        return y, ent


class Block(nn.Module):
    def __init__(self, d_model, n_head, n_exp, top_k):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model, dtype=torch.float32), nn.LayerNorm(d_model, dtype=torch.float32)
        self.attn, self.moe = MultiHeadAttention(d_model, n_head), SparseMoE(d_model, n_exp, top_k)

    def forward(self, x, offset=0, attn_mask=None):
        h = self.ln1(x if attn_mask is None else x * attn_mask.unsqueeze(-1))
        x = x + self.attn(h, seq_offset=offset)
        h = self.ln2(x)
        y, _ = self.moe(h)
        return x + y


# ────────────────────────────────────────────────────────────────────────────
# Sparse-MoE Language Model
# ────────────────────────────────────────────────────────────────────────────
class SparseMoELanguageModel(nn.Module):
    """
    forward_next_token_efficient – causal LM loss
    forward_coconut              – latent-reasoning loss
    get_embeddings               – token-wise / pooled representations
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        tokenizer_name: str = "hf-internal-testing/llama-tokenizer",
    ):
        super().__init__()

        # tokenizer
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_name, model_max_length=config.BLOCK_SIZE
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<bot>", "<start_latent>", "<end_latent>",
                                           "<reasoning>", "</reasoning>"]}
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # embeddings
        vocab = self.tokenizer.vocab_size
        self.tok_emb = nn.Embedding(vocab, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        # aliases needed by prepare_optimizer
        self.token_embedding_table    = self.tok_emb
        self.position_embedding_table = self.pos_emb

        # transformer
        self.blocks     = nn.ModuleList([Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)])
        self.ln_f       = nn.LayerNorm(n_embed, dtype=torch.float32)
        self.attn_pool  = nn.Linear(n_embed, 1, bias=False)
        self.block_size = block_size

        # energy-based model head
        self.ebm = EnergyBasedModel(n_embed)

    # ------------------------------------------------------------------ helpers
    def _pad_or_trim(self, ids):
        B, T = ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if T >= self.block_size:
            return ids[:, -self.block_size:]
        pad = torch.full((B, self.block_size - T), pad_id, dtype=ids.dtype, device=ids.device)
        return torch.cat([ids, pad], dim=1)

    # ------------------------------------------------------------------ LM loss
    def forward_next_token_efficient(self, ids, reduction="mean", attention_mask=None, offset: int = 0):
        ids = self._pad_or_trim(ids)
        B, T = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x = blk(x, offset=offset, attn_mask=attention_mask)
        x = self.ln_f(x).half()

        with GatheredParameters(self.tok_emb.weight, modifier_rank=0):
            cls_w = self.tok_emb.weight.detach().clone().half()
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return linear_cross_entropy(x, cls_w, ids, ignore_index=pad_id, reduction=reduction, shift=1)

    # ------------------------------------------------------------------ Coconut loss
    def forward_coconut(
        self,
        ids,
        attention_mask=None,
        labels=None,
        latent_token_id=99998,
        reduction="mean",
        offset: int = 0,
    ):
        device = ids.device
        ids = self._pad_or_trim(ids)
        B, T = ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if attention_mask is None:
            attention_mask = (ids != pad_id).to(device)

        with torch.no_grad():
            base = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=device))
        latent_pos = [(ids[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist() for b in range(B)]
        max_lat = max((len(p) for p in latent_pos), default=0)
        embeds = base.clone().requires_grad_(True)

        for p in range(max_lat):
            active = [lst[p] for lst in latent_pos if p < len(lst)]
            if not active:
                continue
            cut = min(active) + 1
            x = embeds[:, :cut]; am = attention_mask[:, :cut]
            for blk in self.blocks:
                x = blk(x, offset=offset, attn_mask=am)
            x = self.ln_f(x).half()
            for b, lst in enumerate(latent_pos):
                if p < len(lst) and lst[p] > 0:
                    embeds[b, lst[p]] = x[b, lst[p] - 1]

        x = embeds
        for blk in self.blocks:
            x = blk(x, offset=offset, attn_mask=attention_mask)
        x = self.ln_f(x).half()

        if labels is None:
            return x
        labels = labels[:, -T:]
        with GatheredParameters(self.tok_emb.weight, modifier_rank=0):
            cls_w = self.tok_emb.weight.detach().clone().half()
        return linear_cross_entropy(x, cls_w, labels, ignore_index=pad_id, reduction=reduction, shift=1)

    # ------------------------------------------------------------------ embeddings
    def get_embeddings(self, ids, pool: bool = False, attention_mask=None, offset: int = 0):
        ids = self._pad_or_trim(ids)
        B, T = ids.shape
        if attention_mask is None:
            pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            attention_mask = (ids != pad_id).to(ids.device)

        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x = blk(x, offset=offset, attn_mask=attention_mask)
        x = self.ln_f(x)

        if not pool:
            return x
        scores = self.attn_pool(x).squeeze(-1).masked_fill(~attention_mask, -1e9)
        w = F.softmax(scores, dim=1)
        return torch.einsum("btd,bt->bd", x, w)


# ────────────────────────────────────────────────────────────────────────────
# context-window helpers
# ────────────────────────────────────────────────────────────────────────────
def update_model_rope_for_extended_context(model, new_len, base: float = 5e5):
    for blk in model.blocks:
        blk.attn.update_rope_buffers(new_len, base)
    return model


def expand_pos_embedding(model, new_len):
    old_len, dim = model.pos_emb.weight.shape
    if new_len <= old_len:
        model.block_size = new_len
        return
    new_emb = nn.Embedding(new_len, dim, device=model.pos_emb.weight.device)
    new_emb.weight.data[:old_len] = model.pos_emb.weight.data
    nn.init.normal_(new_emb.weight.data[old_len:], std=0.02)
    model.pos_emb = new_emb
    model.position_embedding_table = new_emb
    model.block_size = new_len
