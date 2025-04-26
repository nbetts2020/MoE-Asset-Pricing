import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from utils.config import config
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_unpadded_qkvpacked_func,
)
from deepspeed.runtime.zero.stage3 import GatheredParameters
from cut_cross_entropy import linear_cross_entropy
from utils.ebm import EnergyBasedModel

# --------------------------------------------------------------------
# RoPE helpers
# --------------------------------------------------------------------
def build_sin_cos(seq_len: int, dim: int, device, base: float = 10_000.0):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim * 2, 2, dtype=torch.float, device=device)
        * (-math.log(base) / (dim * 2))
    )
    sin = torch.sin(pos * div)
    cos = torch.cos(pos * div)
    return sin, cos


def apply_rope(q, k, sin, cos):
    half = q.size(-1) // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot


# --------------------------------------------------------------------
# Ring-Flash Multi-Head Attention
# --------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0, "n_embed must be divisible by n_head"
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)

        self.max_seq_len = config.BLOCK_SIZE
        sin, cos = build_sin_cos(self.max_seq_len, self.head_size // 2, "cpu")
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    # ---------------- utilities ----------------
    def update_rope_buffers(self, new_len: int, *, base: float = 500_000.0):
        self.max_seq_len = new_len
        sin, cos = build_sin_cos(new_len, self.head_size // 2, "cpu", base)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    # ---------------- public forward -----------
    def forward(self, x, *, return_attn_probs: bool = False):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self._forward_flash(x, return_attn_probs)
        return self._forward_ring_flash(x)

    # ---------- single-GPU Flash ---------------
    def _forward_flash(self, x, return_attn_probs: bool):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_head, self.head_size)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        sin, cos = self.rope_sin[:T].to(x.device), self.rope_cos[:T].to(x.device)
        q, k = apply_rope(q, k, sin, cos)
        qkv[:, :, 0], qkv[:, :, 1] = q, k
        qkv_flat = qkv.view(B * T, 3, self.n_head, self.head_size)
        cu = torch.arange(0, (B + 1) * T, step=T, dtype=torch.int32, device=x.device)
        attn_out, attn_probs, _ = flash_attn_unpadded_qkvpacked_func(
            qkv_flat,
            cu,
            T,
            dropout_p=config.DROPOUT,
            softmax_scale=None,
            causal=True,
            return_attn_probs=return_attn_probs,
        )
        attn_out = (
            attn_out.view(B, T, self.n_head, self.head_size)
            .permute(0, 2, 1, 3)
            .reshape(B, T, C)
        )
        out = self.out_proj(attn_out)
        return (out, attn_probs, None) if return_attn_probs else out

    # ---------- Ring-Flash (multi-GPU) ---------
    def _forward_ring_flash(self, x):
        B, T_local, C = x.size()
        world = dist.get_world_size()
        rank = dist.get_rank()
        assert (
            T_local == config.BLOCK_SIZE
        ), "--block_size must equal tokens per GPU input length"

        # 1. project + RoPE
        qkv = self.qkv_proj(x).view(B, T_local, 3, self.n_head, self.head_size)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        sin, cos = self.rope_sin[:T_local].to(x.device), self.rope_cos[:T_local].to(
            x.device
        )
        q, k = apply_rope(q, k, sin, cos)

        # 2. comm buffers
        kv_local = torch.cat([k, v], dim=-1).contiguous()
        kv_recv = torch.empty_like(kv_local)
        attn_out = torch.zeros_like(q)

        def _flash(q_, k_, v_):
            return flash_attn_func(
                q_, k_, v_, dropout_p=config.DROPOUT, causal=False, return_attn_probs=False
            )

        # local pass
        attn_out += _flash(q, k, v)

        # ring passes
        nxt, prv = (rank + 1) % world, (rank - 1 + world) % world
        send_req = dist.isend(kv_local, dst=nxt)
        recv_req = dist.irecv(kv_recv, src=prv)

        cur_kv = kv_local
        for _ in range(world - 1):
            recv_req.wait()
            k_r, v_r = torch.split(kv_recv, self.head_size, dim=-1)
            attn_out += _flash(q, k_r, v_r)
            cur_kv, kv_recv = kv_recv, cur_kv
            send_req = dist.isend(cur_kv, dst=nxt)
            recv_req = dist.irecv(kv_recv, src=prv)
        send_req.wait()

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T_local, C)
        return self.out_proj(attn_out)


# --------------------------------------------------------------------
# Router / MoE components (unchanged)
# --------------------------------------------------------------------
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

    def forward(self, x):
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        full_probs = F.softmax(noisy_logits, dim=-1)
        top_k_logits, idx = noisy_logits.topk(self.top_k, dim=-1)
        sparse_logits = torch.full_like(noisy_logits, float("-inf")).scatter_(
            -1, idx, top_k_logits
        )
        router_out = F.softmax(sparse_logits, dim=-1)
        return router_out, idx, full_probs


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
        route, idx, full_probs = self.router(x)
        final = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_route = route.view(-1, route.size(-1))
        cap = int((B * T * self.top_k / self.num_experts) * self.capacity_factor)
        upd = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            mask = (idx == i).any(dim=-1).view(-1)
            sel = torch.nonzero(mask).squeeze(-1)[:cap]
            if sel.numel():
                out = expert(flat_x[sel])
                gate = flat_route[sel, i].unsqueeze(1)
                upd.index_add_(0, sel, out * gate)

        final += upd.view(B, T, -1)
        entropy_loss = (
            (-torch.sum(full_probs * torch.log(full_probs + 1e-8), dim=-1)).mean()
            if self.training
            else None
        )
        return final, entropy_loss


class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.sa = MultiHeadAttention(n_embed, n_head)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)

    def forward(self, x, attention_mask=None):
        x_masked = x * attention_mask.unsqueeze(-1) if attention_mask is not None else x
        x = x + self.sa(self.ln1(x_masked))
        moe_out, _ = self.smoe(self.ln2(x))
        return x + moe_out, _


# --------------------------------------------------------------------
# SparseMoE Language Model
# --------------------------------------------------------------------
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
            tokenizer_name, model_max_length=config.CONTEXT_WINDOW
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
        vocab_size = self.tokenizer.vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.attn_pool = nn.Linear(n_embed, 1, bias=False)
        self.block_size = block_size
        self.ebm = EnergyBasedModel(n_embed)

    # ---------------- helpers -----------------
    def preprocess_input(self, input_ids):
        B, T = input_ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if T >= self.block_size:
            return input_ids[:, -self.block_size :]
        pad_len = self.block_size - T
        pad = torch.full((B, pad_len), pad_id, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, pad], dim=1)

    # ---------------- next-token CE ----------
    def forward_next_token_efficient(
        self, input_ids, reduction="mean", attention_mask=None, force_bf16=False
    ):
        device = input_ids.device
        input_ids = self.preprocess_input(input_ids)
        T = self.block_size
        tok = self.token_embedding_table(input_ids)
        pos = self.position_embedding_table(torch.arange(T, device=device))
        x = tok + pos

        for blk in self.blocks:
            x, _ = blk(x, attention_mask)

        x = self.ln_f(x).to(torch.bfloat16) if force_bf16 else self.ln_f(x)

        with GatheredParameters(self.token_embedding_table.weight, modifier_rank=0):
            if hasattr(self, "_gathered_weights"):
                cls_w = self._gathered_weights
            else:
                cls_w = self.token_embedding_table.weight.clone().to(device)
                if force_bf16:
                    cls_w = cls_w.to(torch.bfloat16)

        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        loss = linear_cross_entropy(
            x, cls_w, input_ids, ignore_index=pad_id, reduction=reduction, shift=1
        )
        return loss

    # ---------------- embeddings -------------
    def get_embeddings(self, input_ids, pool=False, attention_mask=None):
        device = input_ids.device
        input_ids = self.preprocess_input(input_ids)
        if attention_mask is None:
            pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            attention_mask = (input_ids != pad_id).to(device)
        B, T = input_ids.shape

        tok = self.token_embedding_table(input_ids)
        pos = self.position_embedding_table(torch.arange(T, device=device))
        x = tok + pos

        for blk in self.blocks:
            x, _ = blk(x, attention_mask)

        x = self.ln_f(x)

        if not pool:
            return x

        scores = self.attn_pool(x).squeeze(-1).masked_fill(
            ~attention_mask, float("-1e9")
        )
        w = F.softmax(scores, dim=1)
        return torch.einsum("btd,bt->bd", x, w)

    # ---------------- coconut ----------------
    def forward_coconut(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        latent_token_id=99998,
        reduction="mean",
        force_bf16=False,
    ):
        device = input_ids.device
        input_ids = self.preprocess_input(input_ids)
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        attention_mask = (
            (input_ids != pad_id).to(device) if attention_mask is None else attention_mask
        )

        B, T = input_ids.shape
        tok = self.token_embedding_table(input_ids)
        pos = self.position_embedding_table(torch.arange(T, device=device))
        embeds = (tok + pos).requires_grad_(True)

        latent_pos = [
            (input_ids[b] == latent_token_id).nonzero(as_tuple=True)[0].tolist()
            for b in range(B)
        ]
        max_latent = max(len(lst) for lst in latent_pos) if latent_pos else 0

        for idx in range(max_latent):
            seq_cut = min(
                lst[idx] for lst in latent_pos if idx < len(lst)
            ) + 1
            x = embeds[:, :seq_cut, :]
            am = attention_mask[:, :seq_cut]
            for blk in self.blocks:
                x, _ = blk(x, am)
            x = self.ln_f(x).to(torch.bfloat16) if force_bf16 else self.ln_f(x)

            for b, lst in enumerate(latent_pos):
                if idx < len(lst) and lst[idx] != 0:
                    embeds[b, lst[idx], :] = x[b, lst[idx] - 1, :]

        x = embeds
        for blk in self.blocks:
            x, _ = blk(x, attention_mask)
        x = self.ln_f(x).to(torch.bfloat16) if force_bf16 else self.ln_f(x)

        if labels is None:
            return None

        with GatheredParameters(self.token_embedding_table.weight, modifier_rank=0):
            cls_w = (
                self._gathered_weights
                if hasattr(self, "_gathered_weights")
                else self.token_embedding_table.weight.clone().to(device)
            )
            if force_bf16:
                cls_w = cls_w.to(torch.bfloat16)

        loss = linear_cross_entropy(
            x,
            cls_w,
            labels[:, -T:],
            ignore_index=pad_id,
            reduction=reduction,
            shift=1,
        )
        return loss


# --------------------------------------------------------------------
# RoPE-update helper
# --------------------------------------------------------------------
def update_model_rope_for_extended_context(model, new_seq_len, base=500_000.0):
    for blk in model.blocks:
        blk.sa.update_rope_buffers(new_seq_len, base=base)
    return model
