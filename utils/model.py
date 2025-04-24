# utils/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizerFast
from deepspeed.runtime.zero.stage3 import GatheredParameters
from cut_cross_entropy import linear_cross_entropy
from utils.config import config
from utils.ebm import EnergyBasedModel   # <-- still used by training script

# ---------------------------------------------------------------------------
#  Ring-Flash-Attention (sequence-parallel) kernel
# ---------------------------------------------------------------------------
from ring_flash_attn.ring_flash_attn_varlen import (
    ring_flash_attn_varlen_qkvpacked_func,
)

# ────────────────────────────────────────────────────────────────────────────
# Rotary helpers
# ────────────────────────────────────────────────────────────────────────────
def build_sin_cos(seq_len, dim, device, base=10_000.0):
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    inv = torch.exp(
        torch.arange(0, dim * 2, 2, dtype=torch.float32, device=device)
        * (-math.log(base) / (dim * 2))
    )
    sin = torch.sin(pos * inv)
    cos = torch.cos(pos * inv)
    return sin, cos


def apply_rope(q, k, sin, cos):
    """q,k : (B,T,H,Dh)  •  sin,cos : (T,Dh//2)"""
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

    sin, cos = sin.unsqueeze(0).unsqueeze(2), cos.unsqueeze(0).unsqueeze(2)
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot


# ────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention (sequence parallel Ring-Flash)
# ────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        assert n_embed % n_head == 0

        self.qkv  = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed,     n_embed, bias=False)

        self.max_seq_len = config.BLOCK_SIZE
        sin, cos = build_sin_cos(self.max_seq_len, self.head_dim // 2, torch.device('cpu'))
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    # called from helper when context window is enlarged
    def update_rope_buffers(self, new_len: int, base: float = 5e5):
        self.max_seq_len = new_len
        sin, cos = build_sin_cos(new_len, self.head_dim // 2, torch.device('cpu'), base)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, seq_offset: int = 0):
        """
        x         : (B,T,D)
        seq_offset: starting position of this *local* chunk in the *global*
                    sequence (needed for RoPE when doing sequence parallel)
        """
        x = x.half()                      # FP16 activations
        B, T, D = x.shape

        # ---- QKV projection -------------------------------------------------
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # ---- apply RoPE -----------------------------------------------------
        pos   = torch.arange(seq_offset, seq_offset + T, device=x.device)
        sin_t = self.rope_sin[pos]
        cos_t = self.rope_cos[pos]
        q, k  = apply_rope(q, k, sin_t, cos_t)

        # ---- pack for varlen kernel ----------------------------------------
        qkv_packed = (
            torch.stack([q, k, v], dim=2)
            .contiguous()
            .view(B * T, 3, self.n_head, self.head_dim)
        )  # (total_tokens,3,H,Dh)

        cu_seqlens = torch.arange(
            0, (B + 1) * T, step=T, dtype=torch.int32, device=x.device
        )

        # ---- Ring-Flash-Attention kernel -----------------------------------
        attn = ring_flash_attn_varlen_qkvpacked_func(
            qkv_packed,
            cu_seqlens,
            max_seqlen=T,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )  # (total_tokens, H, Dh)

        attn = (
            attn
            .view(B, T, self.n_head, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(B, T, D)
        )
        return self.proj(attn)


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
        logits  = self.lin_logits(h)
        noise   = torch.randn_like(logits) * F.softplus(self.lin_noise(h))
        noisy   = logits + noise

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
                out   = exp(flat_x[sel])
                gate  = flat_p[sel, i].unsqueeze(1)
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
    Provides:
      • forward_next_token_efficient   (Cut-Xent next-token loss)
      • forward_coconut                (latent-reasoning multi-pass)
      • get_embeddings                 (token-wise or pooled)
    All attributes referenced elsewhere (ebm, token_embedding_table, etc.) are
    still present.
    """

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

        # ---------------- tokenizer ----------------------------------------
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_name, model_max_length=config.BLOCK_SIZE
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<bot>", "<start_latent>", "<end_latent>",
                                           "<reasoning>", "</reasoning>"]}
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---------------- embeddings ---------------------------------------
        vocab = self.tokenizer.vocab_size
        self.tok_emb = nn.Embedding(vocab, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        # keep old attribute names for utils.prepare_optimizer
        self.token_embedding_table    = self.tok_emb
        self.position_embedding_table = self.pos_emb

        # ---------------- transformer --------------------------------------
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed, dtype=torch.float32)
        self.attn_pool = nn.Linear(n_embed, 1, bias=False)
        self.block_size = block_size

        # ---------------- integrated EBM -----------------------------------
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
        pad = torch.full((B, self.block_size - T), pad_id,
                         dtype=ids.dtype, device=ids.device)
        return torch.cat([ids, pad], dim=1)

    # ======================================================================
    #  Next-token objective (Cut Cross Entropy)
    # ======================================================================
    def forward_next_token_efficient(self, ids, reduction="mean",
                                     attention_mask=None, offset=0):
        """
        ids : (B,T_local)   – *local slice* when doing sequence parallelism
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
            x, cls_w, ids, ignore_index=pad_id,
            reduction=reduction, shift=1
        )

    # ======================================================================
    #  Coconut-style latent-reasoning forward
    # ======================================================================
    def forward_coconut(
        self, ids, attention_mask=None, labels=None,
        latent_token_id=99998, reduction="mean", offset: int = 0
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
            base = tok + pos                     # (B,T,D)

        # 2) locate latent tokens
        latent_pos = [
            (ids[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist()
            for b in range(B)
        ]
        max_lat = max((len(p) for p in latent_pos), default=0)
        embeds = base.clone().requires_grad_(True)

        # 3) multi-pass
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
            return x  # hidden states if caller wants
        labels = labels[:, -T:]
        with GatheredParameters(self.tok_emb.weight, modifier_rank=0):
            cls_w = self.tok_emb.weight.detach().clone().half()
        return linear_cross_entropy(
            x, cls_w, labels, ignore_index=pad_id,
            reduction=reduction, shift=1
        )

    # ======================================================================
    #  Embedding extractor (token-wise or pooled)
    # ======================================================================
    def get_embeddings(self, ids, pool=False, attention_mask=None, offset=0):
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
            return x  # (B,T,D)

        # attention-pool
        scores = self.attn_pool(x).squeeze(-1)  # (B,T)
        scores = scores.masked_fill(~attention_mask, -1e9)
        w = F.softmax(scores, dim=1)            # (B,T)
        return torch.einsum("btd,bt->bd", x, w)  # (B,D)


# ────────────────────────────────────────────────────────────────────────────
# Helpers to extend context
# ────────────────────────────────────────────────────────────────────────────
def update_model_rope_for_extended_context(model, new_len, base=5e5):
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
    model.position_embedding_table = model.pos_emb  # keep alias in sync
    model.block_size = new_len
