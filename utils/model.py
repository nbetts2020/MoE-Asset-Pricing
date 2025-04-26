# utils/model.py
# =============================================================================
#  Ring-Flash-Attention Sparse-MoE LM
# =============================================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils.config import config
from utils.ebm    import EnergyBasedModel
from cut_cross_entropy import linear_cross_entropy
from deepspeed.runtime.zero.stage3 import GatheredParameters

# note: rename the "unpadded" KV-packed op to our flash_attn_kvpacked_func alias
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_unpadded_qkvpacked_func,
    flash_attn_unpadded_kvpacked_func as flash_attn_kvpacked_func,
)

# ──────────────────────────────────────────────────────────────────────────────
# Rotary-Embedding helpers
# ──────────────────────────────────────────────────────────────────────────────
def build_sin_cos(seq_len: int, dim: int, device, base: float = 10_000.0):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    inv = torch.exp(
        torch.arange(0, dim * 2, 2, dtype=torch.float, device=device)
        * (-math.log(base) / (dim * 2))
    )
    return torch.sin(pos * inv), torch.cos(pos * inv)


def apply_rope(q, k, sin, cos):
    """RoPE in-place   q,k: (B,T,H,Dh)   sin,cos: (T,Dh/2)"""
    half = q.size(-1) // 2
    # guard against mismatched table width
    if sin.size(-1) != half:
        sin, cos = sin[..., :half], cos[..., :half]

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    sin, cos = sin.unsqueeze(0).unsqueeze(2), cos.unsqueeze(0).unsqueeze(2)
    q[..., :half], q[..., half:] = q1 * cos - q2 * sin, q2 * cos + q1 * sin
    k[..., :half], k[..., half:] = k1 * cos - k2 * sin, k2 * cos + k1 * sin
    return q, k


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention (Flash or Ring-Flash)
# ──────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head, self.head_dim = n_head, n_embed // n_head

        self.qkv  = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)

        self.max_seq = config.BLOCK_SIZE
        sin, cos = build_sin_cos(self.max_seq, self.head_dim // 2, "cpu")
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    # ------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------
    def update_rope_buffers(self, new_len: int, base: float = 5e5):
        """Rebuild RoPE tables using *current* per-head size on this rank."""
        self.max_seq = new_len
        device = self.rope_sin.device if hasattr(self, "rope_sin") else "cpu"
        sin, cos = build_sin_cos(new_len, self.head_dim // 2, device, base)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    # ------------------------------------------------------------
    # dispatcher
    # ------------------------------------------------------------
    def forward(self, x, *, return_attn_probs=False):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self._forward_flash(x, return_attn_probs)
        return self._forward_ring_flash(x)

    # ------------------------------------------------------------
    # single-GPU Flash-Attention
    # ------------------------------------------------------------
    def _forward_flash(self, x, return_attn_probs=False):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]

        sin_t, cos_t = self.rope_sin[:T].to(x.device), self.rope_cos[:T].to(x.device)
        q, k = apply_rope(q, k, sin_t, cos_t)

        qkv_flat = torch.stack([q, k, v], dim=2).view(B * T, 3, self.n_head, self.head_dim)
        cu_seqlens = torch.arange(0, (B + 1) * T, T, device=x.device, dtype=torch.int32)

        out, attn_probs, _ = flash_attn_unpadded_qkvpacked_func(
            qkv_flat, cu_seqlens, T,
            dropout_p=config.DROPOUT, softmax_scale=None,
            causal=True, return_attn_probs=return_attn_probs,
        )
        out = out.view(B, T, self.n_head, self.head_dim).permute(0, 2, 1, 3).reshape_as(x)
        proj = self.proj(out)
        return (proj, attn_probs, None) if return_attn_probs else proj

    # ------------------------------------------------------------
    # distributed Ring-Flash (NCCL all_gather, no sockets)
    # ------------------------------------------------------------
    def _forward_ring_flash(self, x):
        B, T_local, C = x.shape
        world = dist.get_world_size()
        assert T_local == config.BLOCK_SIZE, "each rank holds the same local chunk length"

        qkv = self.qkv(x).view(B, T_local, 3, self.n_head, self.head_dim)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        sin_t, cos_t = self.rope_sin[:T_local].to(x.device), self.rope_cos[:T_local].to(x.device)
        q, k = apply_rope(q, k, sin_t, cos_t)

        q, k, v = q.half(), k.half(), v.half()  # Flash kernel dtype

        kv_local = torch.cat([k, v], dim=-1).contiguous()          # (B,T,H,2Dh)
        kv_stack = torch.empty(world, *kv_local.shape,
                               dtype=kv_local.dtype, device=kv_local.device)
        dist.all_gather_into_tensor(kv_stack, kv_local)            # NCCL collective

        acc = torch.zeros_like(q)
        for r in range(world):
            kv_r_tensor = kv_stack[r]                              # (B,T,H,2*Dh_r)
            hd = kv_r_tensor.size(-1) // 2                         # derive Dh per rank
            if hd != self.head_dim:
                raise RuntimeError(
                    f"Head-dim mismatch across ranks: local={self.head_dim}, rank{r}={hd}. "
                    "Ensure all ranks use identical model hyper-parameters."
                )
            k_r, v_r = kv_r_tensor.split(hd, dim=-1)
            kv_r = torch.stack([k_r, v_r], dim=2)                  # (B,T,2,H,Dh)
            acc += flash_attn_kvpacked_func(
                q,
                kv_r,
                dropout_p=config.DROPOUT,
                softmax_scale=None,
                causal=True,
            )

        out = acc.reshape(B, T_local, C)
        return self.proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# MoE (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class Expert(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d), nn.ReLU(),
            nn.Linear(4 * d, d), nn.Dropout(config.DROPOUT),
        )
    def forward(self, x): return self.net(x)


class NoisyTopkRouter(nn.Module):
    def __init__(self, d, n_exp, top_k):
        super().__init__()
        self.top_k = top_k
        self.lin_w, self.lin_noise = nn.Linear(d, n_exp), nn.Linear(d, n_exp)

    def forward(self, h):
        logits = self.lin_w(h)
        noise  = torch.randn_like(logits) * F.softplus(self.lin_noise(h))
        noisy  = logits + noise
        full   = F.softmax(noisy, -1)
        topk, ix = noisy.topk(self.top_k, -1)
        sparse = torch.full_like(noisy, -float("inf")).scatter(-1, ix, topk)
        return F.softmax(sparse, -1), ix, full


class SparseMoE(nn.Module):
    def __init__(self, d, n_exp, top_k, cap=1.0):
        super().__init__()
        self.router  = NoisyTopkRouter(d, n_exp, top_k)
        self.experts = nn.ModuleList([Expert(d) for _ in range(n_exp)])
        self.top_k, self.cap, self.n_exp = top_k, cap, n_exp

    def forward(self, x):
        B, T, D = x.shape
        p, ix, full = self.router(x)
        flat_x, flat_p = x.view(-1, D), p.view(-1, self.n_exp)
        upd = torch.zeros_like(flat_x)
        capacity = int(B * T * self.top_k / self.n_exp * self.cap)

        for i, exp in enumerate(self.experts):
            mask = (ix == i).any(-1).view(-1)
            sel  = torch.nonzero(mask).squeeze(-1)[:capacity]
            if sel.numel():
                out = exp(flat_x[sel])
                upd.index_add_(0, sel, out * flat_p[sel, i:i+1])

        y  = upd.view(B, T, D)
        ent = (-full * full.clamp_min(1e-8).log()).sum(-1).mean() if self.training else None
        return y, ent


class Block(nn.Module):
    def __init__(self, d, n_head, n_exp, top_k):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa, self.moe  = MultiHeadAttention(d, n_head), SparseMoE(d, n_exp, top_k)

    def forward(self, x, mask=None):
        x_masked = x if mask is None else x * mask.unsqueeze(-1)
        a = self.sa(self.ln1(x_masked))
        y, _ = self.moe(self.ln2(x + a))
        return x + a + y, _


# ──────────────────────────────────────────────────────────────────────────────
# Sparse-MoE Language Model  (public interface unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class SparseMoELanguageModel(nn.Module):
    def __init__(
        self, n_embed, n_head, n_layer, block_size,
        dropout, num_experts, top_k,
        tokenizer_name="hf-internal-testing/llama-tokenizer",
    ):
        super().__init__()
        from transformers import LlamaTokenizerFast
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_name, model_max_length=config.CONTEXT_WINDOW
        )
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<bot>", "<start_latent>", "<end_latent>", "<reasoning>", "</reasoning>"
            ]
        })
        self.tokenizer.pad_token = self.tokenizer.eos_token
        V = self.tokenizer.vocab_size

        self.tok_emb = nn.Embedding(V, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        # aliases for prepare_optimizer
        self.token_embedding_table    = self.tok_emb
        self.position_embedding_table = self.pos_emb

        self.blocks = nn.ModuleList([
            Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)
        ])
        self.ln_f      = nn.LayerNorm(n_embed)
        self.attn_pool = nn.Linear(n_embed, 1, bias=False)
        self.block_size = block_size

        self.ebm = EnergyBasedModel(n_embed)

    # ------------------------------------------------------------------ helpers
    def _pad_or_trim(self, ids):
        B, T = ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if T >= self.block_size:
            return ids[:, -self.block_size:]
        pad = torch.full((B, self.block_size - T), pad_id, dtype=ids.dtype, device=ids.device)
        return torch.cat([ids, pad], 1)

    # ------------------------------------------------------------------ LM loss
    def forward_next_token_efficient(
        self, input_ids, reduction="mean", attention_mask=None, force_bf16=False
    ):
        ids = self._pad_or_trim(input_ids)
        T = self.block_size
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x, _ = blk(x, attention_mask)
        x = self.ln_f(x)
        if force_bf16:
            x = x.to(torch.bfloat16)

        with GatheredParameters(self.tok_emb.weight, modifier_rank=0):
            W = getattr(self, "_gathered_weights", None) or self.tok_emb.weight.clone().to(ids.device)

        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return linear_cross_entropy(x, W, ids, ignore_index=pad_id, reduction=reduction, shift=1)

    # ------------------------------------------------------------------ embeddings
    def get_embeddings(self, input_ids, pool=False, attention_mask=None):
        ids = self._pad_or_trim(input_ids)
        if attention_mask is None:
            pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            attention_mask = (ids != pad_id).to(ids.device)

        x = self.tok_emb(ids) + self.pos_emb(torch.arange(self.block_size, device=ids.device))
        for blk in self.blocks:
            x, _ = blk(x, attention_mask)
        x = self.ln_f(x)

        if not pool:
            return x
        scores = self.attn_pool(x).squeeze(-1).masked_fill(~attention_mask, -1e9)
        weights = F.softmax(scores, 1)
        return torch.einsum("btd,bt->bd", x, weights)

    # ------------------------------------------------------------------ Coconut
    def forward_coconut(
        self, input_ids, attention_mask=None, labels=None,
        latent_token_id=99998, reduction="mean", force_bf16=False
    ):
        ids = self._pad_or_trim(input_ids)
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if attention_mask is None:
            attention_mask = (ids != pad_id).to(ids.device)

        B, T = ids.shape
        base = (self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))).detach()
        embeds = base.clone().requires_grad_(True)

        lat_pos = [(ids[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist() for b in range(B)]
        max_lat = max((len(p) for p in lat_pos), default=0)

        for p in range(max_lat):
            active = [lst[p] for lst in lat_pos if p < len(lst)]
            if not active:
                continue
            cut = min(active) + 1
            x = embeds[:, :cut, :]
            m = attention_mask[:, :cut]
            for blk in self.blocks:
                x, _ = blk(x, m)
            x = self.ln_f(x)
            if force_bf16:
                x = x.to(torch.bfloat16)
            for b, lst in enumerate(lat_pos):
                if p < len(lst) and lst[p] > 0:
                    embeds[b, lst[p]] = x[b, lst[p] - 1]

        x = embeds
        for blk in self.blocks:
            x, _ = blk(x, attention_mask)
        x = self.ln_f(x)
        if force_bf16:
            x = x.to(torch.bfloat16)

        if labels is None:
            return x
        labels = labels[:, -T:]
        with GatheredParameters(self.tok_emb.weight, modifier_rank=0):
            W = getattr(self, "_gathered_weights", None) or self.tok_emb.weight.clone().to(ids.device)
        return linear_cross_entropy(x, W, labels, ignore_index=pad_id, reduction=reduction, shift=1)


# ──────────────────────────────────────────────────────────────────────────────
# Context-window helpers
# ──────────────────────────────────────────────────────────────────────────────
def update_model_rope_for_extended_context(model, new_len, base: float = 5e5):
    for blk in model.blocks:
        blk.sa.update_rope_buffers(new_len, base)
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
