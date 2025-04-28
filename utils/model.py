# utils/model.py
# =============================================================================
#  Ring-Flash-Attention Sparse-MoE LM
# =============================================================================
import math
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils.config import config
from utils.ebm import EnergyBasedModel
from cut_cross_entropy import linear_cross_entropy
from deepspeed.runtime.zero.stage3 import GatheredParameters

from flash_attn.flash_attn_interface import (
    flash_attn_unpadded_qkvpacked_func,
    flash_attn_unpadded_kvpacked_func as flash_attn_kvpacked_func,
)

# -----------------------------------------------------------------------------
# distributed helper: a GPU/NCCL ring Process-Group
# -----------------------------------------------------------------------------
_RING_PG: Optional[dist.ProcessGroup] = None


def get_ring_pg() -> Optional[dist.ProcessGroup]:
    global _RING_PG
    if _RING_PG is not None or not dist.is_initialized():
        return _RING_PG
    world = dist.get_world_size()
    if world == 1:
        return None
    # Prefer NCCL for GPU-direct non-blocking
    if dist.is_nccl_available() and torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    ranks = list(range(world))
    group = dist.new_group(ranks=ranks, backend=backend)
    if dist.get_rank() == 0:
        logging.info(f"[Ring-PG] created {backend} group with {world} ranks")
    _RING_PG = group
    return group


# -----------------------------------------------------------------------------
# RoPE helpers
# -----------------------------------------------------------------------------
def build_sin_cos(
    seq_len: int, half_dim: int, device: torch.device, base: float = 10_000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    inv = torch.exp(
        torch.arange(0, half_dim * 2, 2, dtype=torch.float, device=device)
        * (-math.log(base) / (half_dim * 2))
    )
    return torch.sin(pos * inv), torch.cos(pos * inv)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    half = q.size(-1) // 2
    if sin.size(-1) != half:  # crop
        sin, cos = sin[..., :half], cos[..., :half]
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1,T,1,D/2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    q1, q2 = q.split(half, dim=-1)
    k1, k2 = k.split(half, dim=-1)
    q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q, k


# -----------------------------------------------------------------------------
# Multi-Head Attention
# -----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embed // n_head

        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)

        self.max_seq = config.BLOCK_SIZE
        sin, cos = build_sin_cos(self.max_seq, self.head_dim // 2, torch.device("cpu"))
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

        self.ring_pg = get_ring_pg()

    # ------------------------------------------------------------------ helpers
    def update_rope(self, new_len: int, base: float = 1e4):
        if new_len <= self.max_seq:
            return
        self.max_seq = new_len
        sin, cos = build_sin_cos(
            new_len, self.head_dim // 2, self.rope_sin.device, base
        )
        self.rope_sin = sin
        self.rope_cos = cos

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, return_attn_probs: bool = False):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self._flash_single(x, return_attn_probs)
        return self._flash_ring(x)

    # ---------- single-GPU flash-attention ------------------------------------
    def _flash_single(self, x: torch.Tensor, ret_probs: bool):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = apply_rope(q, k, self.rope_sin[:T].to(x.device),
                          self.rope_cos[:T].to(x.device))

        qkv_flat = torch.stack([q, k, v], dim=2).reshape(
            B * T, 3, self.n_head, self.head_dim
        )
        cu = torch.arange(0, (B + 1) * T, T, device=x.device, dtype=torch.int32)
        out = flash_attn_unpadded_qkvpacked_func(
            qkv_flat, cu, T,
            dropout_p=config.DROPOUT, softmax_scale=None,
            causal=True, return_attn_probs=ret_probs,
        )
        if ret_probs:
            out, attn = out
        else:
            attn = None
        out = out.view(B, T, self.n_head, self.head_dim) \
                 .permute(0, 2, 1, 3).reshape_as(x)
        proj = self.proj(out)
        return (proj, attn, None) if ret_probs else proj

    # ---------- sequence-parallel ring flash-attention ------------------------
    def _flash_ring(self, x: torch.Tensor):
        """
        Real ring attention: sequence (T_global) is split across ranks.
        Each rank owns T_local = T_global / world_size tokens.
        KV blocks move around the ring via non-blocking NCCL sends/recvs.
        """
        B, T_global, C = x.shape
        world = dist.get_world_size()
        rank = dist.get_rank()
        device = x.device
        Dh = self.head_dim
    
        # ─── NEW: slice off only this rank’s chunk ────────────────────────────
        assert T_global % world == 0, "global seq length must be divisible by world_size"
        T_local = T_global // world
        x = x[:, rank * T_local : (rank + 1) * T_local, :]
        # ────────────────────────────────────────────────────────────────────────
    
        self.update_rope(config.BLOCK_SIZE)
    
        # now (B, T_local, C) is local to each rank
        qkv = self.qkv(x).view(B, T_local, 3, self.n_head, Dh)
        q, k_local, v_local = qkv.unbind(2)
    
        start = rank * T_local
        q, k_local = apply_rope(
            q, k_local,
            self.rope_sin[start : start + T_local].to(device),
            self.rope_cos[start : start + T_local].to(device),
        )
    
        q = q.half(); k_local = k_local.half(); v_local = v_local.half()
        cu_q = torch.arange(0, (B + 1) * T_local, T_local,
                            device=device, dtype=torch.int32)
    
        k_cache, v_cache = k_local, v_local
        acc = torch.zeros_like(q)
    
        pg = self.ring_pg
        for hop in range(world):
            max_k = T_local * (hop + 1)
            q_flat = q.reshape(B * T_local, self.n_head, Dh)
            kv_flat = torch.stack([k_cache, v_cache], dim=1) \
                        .reshape(B * max_k, 2, self.n_head, Dh)
            cu_k = torch.arange(0, (B + 1) * max_k,
                                max_k, device=device, dtype=torch.int32)
    
            out = flash_attn_kvpacked_func(
                q_flat, kv_flat, cu_q, cu_k,
                T_local, max_k,
                dropout_p=config.DROPOUT, softmax_scale=None,
                causal=True,
            )
            acc += out.view(B, T_local, self.n_head, Dh)
    
            if hop == world - 1:
                break
    
            # ---- ring non-blocking send/recv (GPU NCCL) ----------------------
            next_r = (rank + 1) % world
            prev_r = (rank - 1 + world) % world
    
            send_req_k = dist.isend(k_local, next_r, group=pg)
            send_req_v = dist.isend(v_local, next_r, group=pg)
    
            recv_k = torch.empty_like(k_local)
            recv_v = torch.empty_like(v_local)
            recv_req_k = dist.irecv(recv_k, prev_r, group=pg)
            recv_req_v = dist.irecv(recv_v, prev_r, group=pg)
    
            send_req_k.wait(); send_req_v.wait()
            recv_req_k.wait(); recv_req_v.wait()
    
            k_cache = torch.cat([k_cache, recv_k], dim=1)
            v_cache = torch.cat([v_cache, recv_v], dim=1)
    
        out = self.proj(acc.reshape(B, T_local, C))
        return out

# -----------------------------------------------------------------------------
#  Mixture-of-Experts building blocks
# -----------------------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.ReLU(),
            nn.Linear(4 * d, d),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    def __init__(self, d: int, n_exp: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.lin_w = nn.Linear(d, n_exp)
        self.lin_n = nn.Linear(d, n_exp)

    def forward(self, h):
        logits = self.lin_w(h)
        noise = torch.randn_like(logits) * F.softplus(self.lin_n(h))
        noisy = logits + noise
        full = F.softmax(noisy, -1)
        topk, ix = noisy.topk(self.top_k, -1)
        sparse = torch.full_like(noisy, -float("inf")).scatter(-1, ix, topk)
        return F.softmax(sparse, -1), ix, full


class SparseMoE(nn.Module):
    def __init__(self, d: int, n_exp: int, top_k: int, cap: float = 1.0):
        super().__init__()
        self.router = NoisyTopkRouter(d, n_exp, top_k)
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
            sel = torch.nonzero(mask).squeeze(-1)[:capacity]
            if sel.numel():
                out = exp(flat_x[sel])
                upd.index_add_(0, sel, out * flat_p[sel, i : i + 1])
        y = upd.view(B, T, D)
        ent = (
            (-full * full.clamp_min(1e-8).log()).sum(-1).mean()
            if self.training
            else None
        )
        return y, ent


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, n_exp: int, top_k: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.sa = MultiHeadAttention(d, n_head)
        self.moe = SparseMoE(d, n_exp, top_k)

    def forward(self, x, mask=None):
        a = self.sa(self.ln1(x if mask is None else x * mask.unsqueeze(-1)))
        y, _ = self.moe(self.ln2(x + a))
        return x + a + y, _


# -----------------------------------------------------------------------------
#  Sparse-MoE Language Model
# -----------------------------------------------------------------------------
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
        tokenizer_name="hf-internal-testing/llama-tokenizer",
    ):
        super().__init__()
        from transformers import LlamaTokenizerFast

        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_name, model_max_length=config.BLOCK_SIZE
        )
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<bot>",
                    "<start_latent>",
                    "<end_latent>",
                    "<reasoning>",
                    "</reasoning>",
                ]
            }
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        V = self.tokenizer.vocab_size

        self.tok_emb = nn.Embedding(V, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.token_embedding_table = self.tok_emb
        self.position_embedding_table = self.pos_emb

        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.attn_pool = nn.Linear(n_embed, 1, bias=False)
        self.block_size = block_size
        self.dropout = nn.Dropout(dropout)
        self.ebm = EnergyBasedModel(n_embed)

    # ------------------------------------------------------------------ helpers
    def _pad_or_trim(self, ids):
        B, T = ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if T >= self.block_size:
            return ids[:, -self.block_size :]
        pad = torch.full(
            (B, self.block_size - T), pad_id, dtype=ids.dtype, device=ids.device
        )
        return torch.cat([ids, pad], 1)

    # ------------------------------------------------------------------ losses
    def forward_next_token_efficient(
        self, input_ids, reduction="mean", attention_mask=None, force_bf16=False
    ):
        ids = self._pad_or_trim(input_ids)
        T = self.block_size
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x, _ = blk(x, attention_mask)
        x = self.ln_f(self.dropout(x))
        if force_bf16:
            x = x.to(torch.float16)

        with GatheredParameters(self.tok_emb.weight, modifier_rank=0):
            W = (
                getattr(self, "_gathered_weights", None)
                or self.tok_emb.weight.to(ids.device)
            )

        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return linear_cross_entropy(
            x, W, ids, ignore_index=pad_id, reduction=reduction, shift=1
        )

    # ------------------------------------------------------------------ embeds
    def get_embeddings(self, input_ids, pool=False, attention_mask=None):
        ids = self._pad_or_trim(input_ids)
        if attention_mask is None:
            pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            attention_mask = (ids != pad_id).to(ids.device)

        x = self.tok_emb(ids) + self.pos_emb(
            torch.arange(self.block_size, device=ids.device)
        )
        for blk in self.blocks:
            x, _ = blk(x, attention_mask)
        x = self.ln_f(x)

        if not pool:
            return x
        scores = self.attn_pool(x).squeeze(-1).masked_fill(~attention_mask, -1e9)
        w = F.softmax(scores, 1)
        return torch.einsum("btd,bt->bd", x, w)

    # ------------------------------------------------------------------ Coconut
    def forward_coconut(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        latent_token_id=99998,
        reduction="mean",
        force_bf16=False,
    ):
        ids = self._pad_or_trim(input_ids)
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if attention_mask is None:
            attention_mask = (ids != pad_id).to(ids.device)

        B, T = ids.shape
        base = (
            self.tok_emb(ids)
            + self.pos_emb(torch.arange(T, device=ids.device))
        ).detach()
        embeds = base.clone().requires_grad_(True)

        lat_pos = [
            (ids[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist()
            for b in range(B)
        ]
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
                x = x.to(torch.float16)
            for b, lst in enumerate(lat_pos):
                if p < len(lst) and lst[p] > 0:
                    embeds[b, lst[p]] = x[b, lst[p] - 1]

        x = embeds
        for blk in self.blocks:
            x, _ = blk(x, attention_mask)
        x = self.ln_f(x)
        if force_bf16:
            x = x.to(torch.float16)

        if labels is None:
            return x
        labels = labels[:, -T:]
        with GatheredParameters(self.tok_emb.weight, modifier_rank=0):
            W = (
                getattr(self, "_gathered_weights", None)
                or self.tok_emb.weight.to(ids.device)
            )
        return linear_cross_entropy(
            x, W, labels, ignore_index=pad_id, reduction=reduction, shift=1
        )


# -----------------------------------------------------------------------------
# helpers to extend context length
# -----------------------------------------------------------------------------
def update_model_rope_for_extended_context(model, new_len, base: float = 5e5):
    for blk in model.blocks:
        blk.sa.update_rope(new_len, base)
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
