# utils/model.py
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizerFast
from deepspeed.runtime.zero.stage3 import GatheredParameters
from cut_cross_entropy import linear_cross_entropy
from utils.config import config
from utils.ebm import EnergyBasedModel            # still used by training

# ---------------------------------------------------------------------------
#  Ring-Flash-Attention kernel (from Dao et al.)
# ---------------------------------------------------------------------------
from ring_flash_attn.ring_flash_attn_varlen import (
    ring_flash_attn_varlen_qkvpacked_func,
)

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
    """
    q, k  : (B, T, H, Dh)
    sin   : (T, Dh//2)
    cos   : (T, Dh//2)
    """
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

    sin, cos = sin.unsqueeze(0).unsqueeze(2), cos.unsqueeze(0).unsqueeze(2)
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot


# ────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention (sequence-parallel Ring-Flash)
# ────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Implements exact ring-attention with Flash-Attention kernel.
    Each rank receives only a *slice* of the full sequence (T_local) and
    rotates its K/V blocks around the ring to compute the global context.
    """
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        self.n_head   = n_head
        self.head_dim = n_embed // n_head
        assert n_embed % n_head == 0, "n_embed must be divisible by n_head"

        self.qkv  = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed,     n_embed, bias=False)

        # RoPE pre-computed trigs (CPU buffer → broadcast to GPU each call)
        self.max_seq_len = config.BLOCK_SIZE
        sin, cos = build_sin_cos(
            self.max_seq_len, self.head_dim // 2, torch.device("cpu")
        )
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

        # ring topology information (populated lazily in forward)
        self.rank         = 0
        self.world_size   = 1
        self.next_rank    = 0
        self.prev_rank    = 0

    # called externally when context is enlarged
    def update_rope_buffers(self, new_len: int, base: float = 5e5):
        self.max_seq_len = new_len
        sin, cos = build_sin_cos(
            new_len, self.head_dim // 2, torch.device("cpu"), base
        )
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    # ------------------------------------------------------------------
    # Fwd
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, seq_offset: int = 0):
        """
        x          : (B, T_local, D)
        seq_offset : starting *global* position of this local chunk
                     (needed for RoPE in sequence parallel training)
        """
        B, T_local, D = x.shape
        x = x.half()

        # ---------------- distributed context -------------------------
        if dist.is_initialized():
            if self.world_size == 1:       # populate once per process
                self.world_size = dist.get_world_size()
                self.rank       = dist.get_rank()
                self.next_rank  = (self.rank + 1) % self.world_size
                self.prev_rank  = (self.rank - 1 + self.world_size) % self.world_size
        else:
            self.world_size = 1

        # ---------------- QKV projection ------------------------------
        qkv = self.qkv(x).view(B, T_local, 3, self.n_head, self.head_dim)
        q, k_local, v_local = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # ---------------- RoPE ----------------------------------------
        pos   = torch.arange(seq_offset, seq_offset + T_local, device=x.device)
        sin_t = self.rope_sin[pos]
        cos_t = self.rope_cos[pos]
        q, k_local = apply_rope(q, k_local, sin_t, cos_t)

        # ---------------- Ring pass -----------------------------------
        # Accumulator for the context. We *sum* partial contexts, then
        # normalise once at the end with a global softmax denominator.
        ctx_acc = torch.zeros_like(q)       # (B, T_local, H, Dh)
        lse_acc = torch.full(               # log-sum-exp denominator
            (B, self.n_head, T_local), -torch.inf, device=x.device, dtype=q.dtype
        )

        cur_k, cur_v = k_local, v_local     # buffers that will rotate
        for hop in range(self.world_size):
            # 1.  Context from current K/V block ----------------------
            qkv_packed = (
                torch.stack([q, cur_k, cur_v], dim=2)
                .contiguous()
                .view(B * T_local, 3, self.n_head, self.head_dim)
            )
            cu_seqlens = torch.arange(
                0, (B + 1) * T_local, step=T_local,
                dtype=torch.int32, device=x.device
            )
            # FlashAttention returns context but also log-sum-exp; we
            # get both so we can do exact softmax across hops.
            ctx_part, lse_part = ring_flash_attn_varlen_qkvpacked_func(
                qkv_packed,
                cu_seqlens,
                max_seqlen=T_local,
                dropout_p=0.0,
                softmax_scale=None,
                causal=True,
                return_log_sum_exp=True      # (total,B*H*T_local)
            )
            ctx_part = (
                ctx_part.view(B, T_local, self.n_head, self.head_dim)
            )
            lse_part = lse_part.view(B, self.n_head, T_local)

            # 2.  Merge numerically-stable softmax pieces -------------
            lse_new = torch.maximum(lse_acc, lse_part)
            ctx_acc = (
                ctx_acc * torch.exp(lse_acc.unsqueeze(-1) - lse_new.unsqueeze(-1))
                + ctx_part * torch.exp(lse_part.unsqueeze(-1) - lse_new.unsqueeze(-1))
            )
            lse_acc = lse_new

            # 3.  Rotate K/V blocks around the ring -------------------
            if hop < self.world_size - 1:   # no need after final hop
                send_k = cur_k.contiguous()
                send_v = cur_v.contiguous()
                recv_k = torch.empty_like(k_local)
                recv_v = torch.empty_like(v_local)

                # overlap comm & compute (isend / irecv)
                req_k = dist.isend(send_k, dst=self.next_rank)
                req_v = dist.isend(send_v, dst=self.next_rank)
                dist.recv(recv_k, src=self.prev_rank)
                dist.recv(recv_v, src=self.prev_rank)
                req_k.wait(); req_v.wait()

                cur_k, cur_v = recv_k, recv_v

        # 4.  Final projection  ---------------------------------------
        out = (
            ctx_acc.view(B, T_local, self.n_head, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(B, T_local, D)
        )
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
        self.lin_logits = nn.Linear(d_model, n_exp)
        self.lin_noise  = nn.Linear(d_model, n_exp)

    def forward(self, h):
        logits = self.lin_logits(h)
        noise  = torch.randn_like(logits) * F.softplus(self.lin_noise(h))
        noisy  = logits + noise

        full_p  = F.softmax(noisy, dim=-1)
        topk, ix = noisy.topk(self.top_k, dim=-1)
        sparse   = torch.full_like(noisy, -float("inf")).scatter_(-1, ix, topk)
        route_p  = F.softmax(sparse, dim=-1)
        return route_p, ix, full_p


class SparseMoE(nn.Module):
    def __init__(self, d_model, n_exp, top_k, cap_factor=1.0):
        super().__init__()
        self.router  = NoisyTopkRouter(d_model, n_exp, top_k)
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_exp)])
        self.top_k   = top_k
        self.cap     = cap_factor
        self.n_exp   = n_exp

    def forward(self, x):
        B, T, _ = x.shape
        p, ix, full = self.router(x)

        flat_x = x.view(-1, x.size(-1))
        flat_p = p.view(-1, self.n_exp)
        updates = torch.zeros_like(flat_x)

        cap = int((B * T * self.top_k / self.n_exp) * self.cap)
        for i, exp in enumerate(self.experts):
            mask = (ix == i).any(-1).view(-1)
            sel  = torch.nonzero(mask).squeeze(-1)[:cap]
            if sel.numel():
                out  = exp(flat_x[sel])
                gate = flat_p[sel, i].unsqueeze(1)
                updates.index_add_(0, sel, out * gate)

        y = updates.view(B, T, -1)
        ent = (-full * torch.log(full.clamp_min(1e-8))).sum(-1).mean() if self.training else None
        return y, ent


class Block(nn.Module):
    def __init__(self, d_model, n_head, n_exp, top_k):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model, dtype=torch.float32)
        self.ln2  = nn.LayerNorm(d_model, dtype=torch.float32)
        self.attn = MultiHeadAttention(d_model, n_head)
        self.moe  = SparseMoE(d_model, n_exp, top_k)

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
    forward_next_token_efficient   – Cut-Xent next-token objective
    forward_coconut                – latent-reasoning Coconut loss
    get_embeddings                 – token-wise or pooled
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

        # ---------------- tokenizer --------------------------------------
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_name, model_max_length=config.BLOCK_SIZE
        )
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<bot>", "<start_latent>", "<end_latent>",
                    "<reasoning>", "</reasoning>"
                ]
            }
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---------------- embeddings -------------------------------------
        vocab = self.tokenizer.vocab_size
        self.tok_emb = nn.Embedding(vocab, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        # keep old attribute names for utils.prepare_optimizer
        self.token_embedding_table    = self.tok_emb
        self.position_embedding_table = self.pos_emb

        # ---------------- transformer ------------------------------------
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed, dtype=torch.float32)
        self.attn_pool = nn.Linear(n_embed, 1, bias=False)
        self.block_size = block_size

        # ---------------- integrated EBM ---------------------------------
        self.ebm = EnergyBasedModel(n_embed)

    # ======================================================================
    # helpers
    # ======================================================================
    def _pad_or_trim(self, ids):
        """right-trim or left-pad to self.block_size"""
        B, T = ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if T >= self.block_size:
            return ids[:, -self.block_size:]
        pad = torch.full(
            (B, self.block_size - T),
            pad_id, dtype=ids.dtype, device=ids.device
        )
        return torch.cat([ids, pad], dim=1)

    # ======================================================================
    #  Next-token objective (Cut Cross-Entropy)
    # ======================================================================
    def forward_next_token_efficient(
        self,
        ids,
        reduction="mean",
        attention_mask=None,
        offset: int = 0,
    ):
        """
        ids : (B, T_local) – local slice in sequence-parallel mode
        """
        ids = self._pad_or_trim(ids)
        B, T = ids.shape

        tok = self.tok_emb(ids)
        pos = self.pos_emb(torch.arange(T, device=ids.device))
        x   = tok + pos

        for blk in self.blocks:
            x = blk(x, offset=offset, attn_mask=attention_mask)

        x = self.ln_f(x).half()

        with GatheredParameters(self.tok_emb.weight, modifier_rank=0):
            cls_w = self.tok_emb.weight.detach().clone().half()

        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return linear_cross_entropy(
            x, cls_w, ids,
            ignore_index=pad_id, reduction=reduction, shift=1
        )

    # ======================================================================
    #  Coconut latent-reasoning forward
    # ======================================================================
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

        # 1) base embeddings
        with torch.no_grad():
            tok = self.tok_emb(ids)
            pos = self.pos_emb(torch.arange(T, device=device))
            base = tok + pos                         # (B,T,D)

        # 2) locate latent tokens
        latent_pos = [
            (ids[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist()
            for b in range(B)
        ]
        max_lat = max((len(p) for p in latent_pos), default=0)
        embeds = base.clone().requires_grad_(True)

        # 3) multi-pass latent filling
        for p in range(max_lat):
            active = [lst[p] for lst in latent_pos if p < len(lst)]
            if not active:
                continue
            cut = min(active) + 1
            x = embeds[:, :cut]
            am = attention_mask[:, :cut]
            for blk in self.blocks:
                x = blk(x, offset=offset, attn_mask=am)
            x = self.ln_f(x).half()
            for b, lst in enumerate(latent_pos):
                if p < len(lst) and lst[p] > 0:
                    embeds[b, lst[p]] = x[b, lst[p] - 1]

        # 4) full pass
        x = embeds
        for blk in self.blocks:
            x = blk(x, offset=offset, attn_mask=attention_mask)
        x = self.ln_f(x).half()

        if labels is None:
            return x
        labels = labels[:, -T:]
        with GatheredParameters(self.tok_emb.weight, modifier_rank=0):
            cls_w = self.tok_emb.weight.detach().clone().half()
        return linear_cross_entropy(
            x, cls_w, labels,
            ignore_index=pad_id, reduction=reduction, shift=1
        )

    # ======================================================================
    #  Embedding extractor (token-wise or pooled)
    # ======================================================================
    def get_embeddings(
        self,
        ids,
        pool: bool = False,
        attention_mask=None,
        offset: int = 0,
    ):
        ids = self._pad_or_trim(ids)
        B, T = ids.shape
        if attention_mask is None:
            pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            attention_mask = (ids != pad_id).to(ids.device)

        tok = self.tok_emb(ids)
        pos = self.pos_emb(torch.arange(T, device=ids.device))
        x   = tok + pos
        for blk in self.blocks:
            x = blk(x, offset=offset, attn_mask=attention_mask)
        x = self.ln_f(x)

        if not pool:
            return x                                   # (B,T,D)

        # attention-pool
        scores = self.attn_pool(x).squeeze(-1)         # (B,T)
        scores = scores.masked_fill(~attention_mask, -1e9)
        w = F.softmax(scores, dim=1)                   # (B,T)
        return torch.einsum("btd,bt->bd", x, w)        # (B,D)


# ────────────────────────────────────────────────────────────────────────────
# Helpers to extend context
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
    model.position_embedding_table = model.pos_emb  # keep alias
    model.block_size = new_len
