# utils/model.py
# =============================================================================
#  Sequence-Parallel Unified-SP (YunChang) Attention + Sparse-MoE LM
# =============================================================================
import sys
sys.path.insert(0, "/home/ubuntu/MoE-Asset-Pricing/flash-attention")
sys.path.insert(0, "/home/ubuntu/MoE-Asset-Pricing/long-context-attention")

import torch
torch.Tensor.continous = torch.Tensor.contiguous

import yunchang.hybrid.attn_layer as _yc_attn

_SeqApply_orig = _yc_attn.SeqAllToAll4D.apply
def _SeqApply_shim(*args, **kwargs):
    # Forward all args and kwargs to the real apply
    return _SeqApply_orig(*args, **kwargs)
_yc_attn.SeqAllToAll4D.apply = _SeqApply_shim

import flash_attn.flash_attn_interface as _fai
import flash_attn                       as _fa

# ─────────── OVERRIDE VARLEN TO USE PACKED ───────────
# Force both the varlen and packed entrypoints to the same (packed) kernel
_fai.flash_attn_varlen_qkvpacked_func = _fai.flash_attn_qkvpacked_func
_fai.flash_attn_varlen_func           = _fai.flash_attn_qkvpacked_func
_fa.flash_attn_varlen_func            = _fai.flash_attn_qkvpacked_func

# Override torch.ops for CUDA kernels too:
_fai._flash_attn_varlen_forward  = _fai._flash_attn_forward
_fai._flash_attn_varlen_backward = _fai._flash_attn_backward
torch.ops.flash_attn._flash_attn_varlen_forward  = _fai._flash_attn_forward
torch.ops.flash_attn._flash_attn_varlen_backward = _fai._flash_attn_backward
# ──────────────────────────────────────────────────────

# Now the default (packed) implementations remain unchanged:
_fai.flash_attn_func        = _fai.flash_attn_qkvpacked_func
_fa.flash_attn_func         = _fai.flash_attn_qkvpacked_func
_fai._flash_attn_forward    = _fai._flash_attn_forward
_fai._flash_attn_backward   = _fai._flash_attn_backward

import math, logging
from typing import Tuple, Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# project-local imports … ------------------------------------------------------
from utils.config import config
from utils.ebm    import EnergyBasedModel
from utils.data   import GLOBAL_TOKENIZER
from cut_cross_entropy import linear_cross_entropy
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

from yunchang import set_seq_parallel_pg, LongContextAttention, EXTRACT_FUNC_DICT
from yunchang.kernels import AttnType

import contextlib
import tqdm

# -----------------------------------------------------------------------------#
#  RoPE helpers
# -----------------------------------------------------------------------------#
def build_sin_cos(
    seq_len: int, half_dim: int, device: torch.device, base: float = 10_000.0, dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds sinusoidal and cosinusoidal positional embeddings."""
    pos = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half_dim * 2, 2, dtype=dtype, device=device) / (half_dim * 2))
    )
    angles = pos * inv_freq
    return torch.sin(angles), torch.cos(angles)

def apply_rope(
    q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Positional Embeddings (RoPE) to query and key tensors."""
    _, T_q, _, head_dim = q.shape
    _, T_k, _, _ = k.shape
    half_dim = head_dim // 2

    # Ensure sin/cos match sequence lengths and half_dim, cast to correct dtype
    rope_dtype = q.dtype
    sin_q = sin[:T_q, :half_dim].to(dtype=rope_dtype)
    cos_q = cos[:T_q, :half_dim].to(dtype=rope_dtype)
    sin_k = sin[:T_k, :half_dim].to(dtype=rope_dtype)
    cos_k = cos[:T_k, :half_dim].to(dtype=rope_dtype)

    # Reshape sin/cos for broadcasting: (T, D/2) -> (1, T, 1, D/2)
    sin_q, cos_q = sin_q.unsqueeze(0).unsqueeze(2), cos_q.unsqueeze(0).unsqueeze(2)
    sin_k, cos_k = sin_k.unsqueeze(0).unsqueeze(2), cos_k.unsqueeze(0).unsqueeze(2)

    q1, q2 = q.split(half_dim, dim=-1)
    k1, k2 = k.split(half_dim, dim=-1)

    q_rotated = torch.cat([q1 * cos_q - q2 * sin_q, q1 * sin_q + q2 * cos_q], dim=-1)
    k_rotated = torch.cat([k1 * cos_k - k2 * sin_k, k1 * sin_k + k2 * cos_k], dim=-1)

    return q_rotated, k_rotated

class SafeLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, clip=6.5e4):  # 65504 is bf16 max
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        self.eps    = eps
        self.clip   = clip

    def forward(self, x):
        x32 = x.float()
        mu  = x32.mean(-1, keepdim=True)
        var = x32.var( -1, keepdim=True, unbiased=False)
        y32 = (x32 - mu) * torch.rsqrt(var + self.eps)
        y32 = torch.clamp(y32, -self.clip, self.clip)   # prevent ±inf on cast
        return y32.to(x.dtype) * self.weight + self.bias

# -----------------------------------------------------------------------------#
#  Multi-Head Attention (Using YunChang USP)
# -----------------------------------------------------------------------------#
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.n_embed = n_embed

        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)

        self.max_seq = config.BLOCK_SIZE
        sin, cos = build_sin_cos(self.max_seq, self.head_dim // 2, torch.device("cpu"))
        self.register_buffer("rope_sin", sin.float(), persistent=False)
        self.register_buffer("rope_cos", cos.float(), persistent=False)

        self.usp_attn = LongContextAttention(
            scatter_idx=2,
            gather_idx=1,
            ring_impl_type="strip",
            use_pack_qkv=True,
            use_sync=True,
            attn_type=AttnType.FA,
            attn_processor=None,
        )

    def update_rope(self, new_len: int, base: float = 10_000.0):
        if new_len <= self.max_seq:
            return
        self.max_seq = new_len
        sin, cos = build_sin_cos(new_len, self.head_dim // 2, torch.device("cpu"), base=base)
        self._buffers['rope_sin'] = sin.float()
        self._buffers['rope_cos'] = cos.float()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        past_k: torch.Tensor = None,
        past_v: torch.Tensor = None,
        return_attn_probs: bool = False,
    ):
        B, T_new, C = x.shape
        device = x.device

        # ------------------------------------------------------------------
        # 0) Global-position offset for sequence-parallel training
        # ------------------------------------------------------------------
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank       = dist.get_rank()
            # assumes global context length (self.max_seq) is divisible by world_size
            T_local    = self.max_seq // world_size
            seq_offset = rank * T_local
        else:
            seq_offset = 0

        # 1) project + split
        qkv   = self.qkv(x)
        q_new, k_new, v_new = qkv.view(B, T_new, self.n_head, 3 * self.head_dim)\
                                .split(self.head_dim, dim=-1)

        # 2) RoPE (use global positions)
        start      = past_k.shape[1] if past_k is not None else 0
        total_len  = seq_offset + start + T_new
        self.update_rope(total_len)

        sin = self.rope_sin[seq_offset + start : seq_offset + start + T_new]\
                .to(device=device, dtype=x.dtype)
        cos = self.rope_cos[seq_offset + start : seq_offset + start + T_new]\
                .to(device=device, dtype=x.dtype)

        q_rope, k_rope = apply_rope(q_new, k_new, sin, cos)
        v_rope = v_new

        # 3) KV cache
        if past_k is not None:
            current_k = torch.cat([past_k, k_rope], dim=1)
            current_v = torch.cat([past_v, v_rope], dim=1)
        else:
            current_k, current_v = k_rope, v_rope

        # 4) USP attention (chunk, if desired, to save memory)
        M = 1  # number of KV chunks
        attn_accum = torch.zeros_like(q_rope)
        for k_chunk, v_chunk in zip(current_k.chunk(M, dim=1), current_v.chunk(M, dim=1)):
            out = self.usp_attn(
                q_rope, k_chunk, v_chunk,
                dropout_p = config.DROPOUT if self.training else 0.0,
                softmax_scale = None,
                causal       = True,
                window_size  = (-1, -1),
                softcap      = 0.0,
                alibi_slopes = None,
                deterministic      = True,
                return_attn_probs  = False,
            )
            attn_accum += out                           # (B, T_new, n_head, head_dim)

        attn_ctx = attn_accum.reshape(B, T_new, C)

        # 5) final projection
        y = self.proj(attn_ctx)

        if return_attn_probs:
            raise NotImplementedError("return_attn_probs not supported with chunking")
        return y, current_k, current_v

# -----------------------------------------------------------------------------#
#  Mixture-of-Experts blocks
# -----------------------------------------------------------------------------#
class Expert(nn.Module):
    """A simple MLP Expert module."""
    def __init__(self, d: int):
        super().__init__()
        hidden_dim = 4 * d
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d),
            nn.Dropout(config.DROPOUT),
        )
    def forward(self, x): return self.net(x)

class NoisyTopkRouter(nn.Module):
    """Noisy Top-k Router."""
    def __init__(self, d: int, n_exp: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate_proj = nn.Linear(d, n_exp, bias=False)
        self.noise_proj = nn.Linear(d, n_exp, bias=False)
        self.register_buffer("router_step", torch.zeros((), dtype=torch.int64))
        with torch.no_grad(): self.noise_proj.weight.normal_(0, 0.01)

    def forward(self, hidden_states):
        # logits in bf16
        logits = self.gate_proj(hidden_states)

        if self.training:
            # only draw true random noise every top_k calls to avoid runaway seeding
            if (self.router_step.item() % self.top_k) == 0:
                g = torch.Generator(device=logits.device).manual_seed(0)
                noise = torch.randn(
                    *logits.shape,
                    device=logits.device,
                    dtype=logits.dtype,
                    generator=g
                )
            else:
                noise = torch.zeros_like(logits)

            scale  = F.softplus(self.noise_proj(hidden_states))   # bf16 scale
            logits = logits + noise * scale
            self.router_step += 1
        # else: on eval, we leave logits unchanged

        gates = F.softmax(logits, dim=-1)                        # bf16 softmax
        vals, inds = torch.topk(gates, self.top_k, dim=-1, sorted=False)
        return vals, inds, gates


class SparseMoE(nn.Module):
    """Multi-GPU expert-parallel Sparse-MoE (bf16-friendly)."""

    def __init__(self, d: int, n_exp: int, top_k: int, cap_factor: float = 2):
        super().__init__()
        self.top_k       = top_k
        self.cap_factor  = cap_factor
        self.num_experts = n_exp

        # Router (computes soft-max in fp32 internally)
        self.router = NoisyTopkRouter(d, n_exp, top_k)

        # ---------- distributed setup ----------
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank       = dist.get_rank()
            self.ep_group   = dist.new_group(list(range(self.world_size)))
        else:
            self.world_size = 1
            self.rank       = 0
            self.ep_group   = None

        if n_exp % self.world_size != 0:
            raise ValueError(
                f"num_experts ({n_exp}) must be divisible by world_size ({self.world_size})"
            )

        self.exp_per_gpu     = n_exp // self.world_size
        self.first_global_id = self.rank * self.exp_per_gpu
        self.experts         = nn.ModuleList([Expert(d) for _ in range(self.exp_per_gpu)])

    # ----------------------------------------------------------------------
    def forward(self, hidden_states):
        B, T, d = hidden_states.shape
        device  = hidden_states.device
        N       = B * T
        flat    = hidden_states.view(N, d)

        # 0) Input sanity check
        assert torch.isfinite(flat).all(), "Non-finite entering SparseMoE"

        # 1) router (fp32)
        vals_fp32, top_inds, dense_fp32 = self.router(flat)
        assert torch.isfinite(vals_fp32).all(),  "Non-finite in router values"
        assert torch.isfinite(dense_fp32).all(), "Non-finite in router probabilities"
        capacity = math.ceil(self.cap_factor * N / self.num_experts)

        # 2) select tokens
        kept_tok, exp_idx, gw_list = [], [], []
        for e in range(self.num_experts):
            mask = (top_inds == e)
            if not mask.any():
                continue
            tok_i, slot_i = mask.nonzero(as_tuple=True)
            scores = vals_fp32[tok_i, slot_i]
            top_n  = min(capacity, scores.numel())
            if top_n == 0:
                continue
            best = torch.topk(scores, top_n, largest=True).indices
            kept_tok.append(tok_i[best])
            exp_idx.append(torch.full((top_n,), e, device=device, dtype=torch.long))
            gw_list.append(scores[best].unsqueeze(1))

        if not kept_tok:
            return hidden_states, torch.tensor(0.0, device=device)

        kept_tok = torch.cat(kept_tok)                            # (R,)
        exp_idx  = torch.cat(exp_idx)                             # (R,)
        gate_wgt = torch.cat(gw_list).to(flat.dtype)              # (R,1)

        # Check tokens to experts input
        inputs_to_experts = flat[kept_tok]
        assert torch.isfinite(inputs_to_experts).all(), "Non-finite tokens sent to experts"

        # single-GPU
        if self.world_size == 1:
            final = torch.zeros_like(flat)
            order = torch.argsort(exp_idx)
            kt, ei, gw = kept_tok[order], exp_idx[order], gate_wgt[order]
            seg = torch.cat([
                torch.tensor([0], device=device),
                torch.where(ei[:-1] != ei[1:])[0] + 1,
                torch.tensor([ei.size(0)], device=device)
            ])
            for s, e in zip(seg[:-1], seg[1:]):
                gid  = ei[s].item()
                lid  = gid - self.first_global_id
                idxs = kt[s:e]
                out  = self.experts[lid](flat[idxs])
                assert torch.isfinite(out).all(), f"Non-finite output from expert {lid}"
                final.index_add_(0, idxs, out * gw[s:e])
            aux = self._aux_loss(exp_idx, dense_fp32, N) if self.training else torch.tensor(0.0, device=device)
            assert torch.isfinite(final).all(), "Non-finite after single-GPU combine"
            return final.view(B, T, d), aux

        # multi-GPU path
        target_rank = exp_idx // self.exp_per_gpu
        local_id    = exp_idx %  self.exp_per_gpu

        order = torch.argsort(target_rank)
        kt, tr, lid, gw = kept_tok[order], target_rank[order], local_id[order], gate_wgt[order]
        send_embed      = flat[kt]
        assert torch.isfinite(send_embed).all(), "Non-finite send_embed"

        send_counts = torch.bincount(tr, minlength=self.world_size).to(device)

        # exchange counts
        gathered    = [torch.zeros_like(send_counts) for _ in range(self.world_size)]
        dist.all_gather(gathered, send_counts, group=self.ep_group)
        recv_counts = torch.stack(gathered, dim=0)[:, self.rank]

        # embeddings exchange
        send_chunks = list(send_embed.split(send_counts.tolist(), dim=0))
        recv_chunks = [torch.empty((cnt.item(), d), device=device, dtype=flat.dtype)
                       for cnt in recv_counts]
        dist.all_to_all(recv_chunks, send_chunks, group=self.ep_group)
        recv_embed = torch.cat(recv_chunks, dim=0)
        assert torch.isfinite(recv_embed).all(), "Non-finite recv_embed"

        # local_id exchange
        send_lid_chunks = list(lid.split(send_counts.tolist(), dim=0))
        recv_lid_chunks = [torch.empty((cnt.item(),), device=device, dtype=lid.dtype)
                           for cnt in recv_counts]
        dist.all_to_all(recv_lid_chunks, send_lid_chunks, group=self.ep_group)
        recv_lid = torch.cat(recv_lid_chunks, dim=0)

        # expert MLPs
        out_local = torch.empty_like(recv_embed)
        for e in range(self.exp_per_gpu):
            mask = (recv_lid == e)
            if mask.any():
                expert_out = self.experts[e](recv_embed[mask])
                assert torch.isfinite(expert_out).all(), f"Non-finite from expert {e}"
                out_local[mask] = expert_out

        # send back outputs
        out_chunks = list(out_local.split(recv_counts.tolist(), dim=0))
        ret_chunks = [torch.empty((cnt.item(), d), device=device, dtype=out_local.dtype)
                      for cnt in send_counts]
        dist.all_to_all(ret_chunks, out_chunks, group=self.ep_group)
        final_buffer = torch.cat(ret_chunks, dim=0)
        assert torch.isfinite(final_buffer).all(), "Non-finite final_buffer"

        # combine
        final = torch.zeros_like(flat)
        final.index_add_(0, kt, final_buffer * gw)
        assert torch.isfinite(final).all(), "Non-finite after combine"

        aux = self._aux_loss(exp_idx, dense_fp32, N) if self.training else torch.tensor(0.0, device=device)
        return final.view(B, T, d), aux

    # ------------------------------------------------------------------
    def _aux_loss(self, exp_idx, dense_probs_fp32, N_tokens):
        frac_tokens = torch.bincount(exp_idx, minlength=self.num_experts).float() / max(N_tokens, 1)
        frac_probs  = dense_probs_fp32.mean(dim=0)
        return (frac_probs * frac_tokens).sum() * self.num_experts * 0.03

class Block(nn.Module):
    """Transformer block using USP Attention and SparseMoE FFN, with KV-cache support."""
    def __init__(self, d, n_head, n_exp, top_k):
        super().__init__()
        self.ln1 = SafeLayerNorm(d)
        self.sa  = MultiHeadAttention(d, n_head)
        self.ln2 = SafeLayerNorm(d)
        self.moe = SparseMoE(d, n_exp, top_k)

    def forward(self, x, mask=None, past_k=None, past_v=None):
        # 0) Sanity-check the input
        assert not torch.isnan(x).any(), "NaN in block input"
        assert not torch.isinf(x).any(), "Inf in block input"

        # 1) Pre-attention LayerNorm
        x_norm = self.ln1(x)
        assert not torch.isnan(x_norm).any(), "NaN detected in ln1 output"
        assert not torch.isinf(x_norm).any(), "Inf detected in ln1 output"

        # 2) Self-attention
        attn_out, k, v = self.sa(x_norm, mask=mask, past_k=past_k, past_v=past_v)
        x = x + attn_out
        assert not torch.isnan(x).any(), "NaN after attention residual"
        assert not torch.isinf(x).any(), "Inf after attention residual"

        # 3) Pre-MoE LayerNorm
        ff_in = self.ln2(x)
        assert not torch.isnan(ff_in).any(), "NaN detected in ln2 output"
        assert not torch.isinf(ff_in).any(), "Inf detected in ln2 output"

        # 4) MoE feed-forward
        moe_out, aux_loss = self.moe(ff_in)
        x = x + moe_out
        assert not torch.isnan(x).any(), "NaN after MoE residual"
        assert not torch.isinf(x).any(), "Inf after MoE residual"

        return x, aux_loss, k, v

# -----------------------------------------------------------------------------#
#  Sparse-MoE Language Model
# -----------------------------------------------------------------------------#
class SparseMoELanguageModel(nn.Module):
    """Sparse MoE Transformer Language Model with USP Attention."""
    def __init__(
        self,
        n_embed,
        n_head,
        n_layer,
        block_size,
        dropout,
        num_experts,
        top_k,
        tokenizer_name=None,
    ):
        super().__init__()
        from transformers import LlamaTokenizerFast

        # Tokenizer setup
        if tokenizer_name:
            self.tokenizer = LlamaTokenizerFast.from_pretrained(
                tokenizer_name, model_max_length=block_size
            )
            special = {
                'additional_special_tokens': [
                    '<bot>', '<start_latent>', '<end_latent>', '<eot>',
                    '<reasoning>', '</reasoning>', '<STOCK PRICE 30 DAYS OUT>: ', '</STOCK PRICE 30 DAYS OUT>'
                ]
            }
            self.tokenizer.add_special_tokens(special)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = GLOBAL_TOKENIZER
            if self.tokenizer.model_max_length != block_size:
                logging.warning(
                    f"Updating tokenizer model_max_length from "
                    f"{self.tokenizer.model_max_length} to {block_size}"
                )
                self.tokenizer.model_max_length = block_size

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logging.info(f"Set pad_token_id to eos_token_id ({self.tokenizer.pad_token_id})")
            else:
                raise ValueError("Tokenizer must have a pad_token_id or eos_token_id")

        vocab_size = len(self.tokenizer)
        self.n_embed = n_embed
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embed, padding_idx=self.tokenizer.pad_token_id
        )
        self.dropout_emb = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)]
        )
        self.ln_f = SafeLayerNorm(n_embed, eps=1e-5)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying
        self.token_embedding_table.weight = self.lm_head.weight

        # EBM component
        self.ebm = EnergyBasedModel(n_embed)

        self.apply(self._init_weights)
        num_params = self.get_num_params()
        logging.info(f"Initialized SparseMoELanguageModel with {num_params/1e6:.2f}M parameters.")

    def _init_weights(self, module):
        """Initializes weights using standard Transformer practices."""
        if isinstance(module, nn.Linear):
            std_dev = 0.02
            if hasattr(module, 'is_residual_proj') and module.is_residual_proj:
                std_dev = std_dev / math.sqrt(2 * len(self.blocks))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std_dev)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_num_params(self, non_embedding=True):
        """Calculates the number of parameters, adjusting for ZeRO3 embeddings."""
        if isinstance(self.lm_head.weight, torch.nn.Parameter):
            params = list(self.parameters())
            is_zero3 = False
        else:
            params = []
            for p in self.parameters():
                if hasattr(p, 'ds_numel'):
                    params.append(p)
                elif torch.is_tensor(p):
                    params.append(p)
            is_zero3 = True

        n_params = sum(
            p.ds_numel if hasattr(p, 'ds_numel') else p.numel() for p in params
        )
        if non_embedding:
            if is_zero3:
                emb_numel = (
                    self.token_embedding_table.weight.ds_numel
                    if hasattr(self.token_embedding_table.weight, 'ds_numel')
                    else 0
                )
            else:
                emb_numel = self.token_embedding_table.weight.numel()
            n_params -= emb_numel
        return n_params

    def _pad_or_trim(self, ids, target_length):
        """Pads (left) or trims (left) input_ids to the target_length."""
        B, T = ids.shape
        pad_id = self.tokenizer.pad_token_id
        assert pad_id is not None, "Tokenizer must have a pad_token_id"
        if T == target_length:
            return ids
        elif T > target_length:
            return ids[:, -target_length:]
        else:
            padding = torch.full(
                (B, target_length - T), pad_id, dtype=ids.dtype, device=ids.device
            )
            return torch.cat([ids, padding], dim=1)

    def forward(self, input_ids, attention_mask=None, labels=None, reduction="mean"):
        """Main forward pass; uses efficient next-token loss if labels provided."""
        if labels is not None:
            return self.forward_next_token_efficient(
                input_ids=input_ids,
                reduction=reduction,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            return self.forward_embeddings_only(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

    def forward_next_token_efficient(
        self, input_ids, reduction="mean", attention_mask=None, labels=None
    ):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank       = dist.get_rank()      if dist.is_initialized() else 0
        device     = input_ids.device
        T          = self.block_size

        # 1) pad/trim to fixed context length
        ids = self._pad_or_trim(input_ids, T)          # (B, T)
        B, T = ids.shape

        # 2) shard the sequence across ranks (sequence-parallel)
        if world_size > 1:
            if T % world_size != 0:
                raise ValueError(f"block_size {T} not divisible by world_size {world_size}")
            T_local   = T // world_size
            ids_local = ids[:, rank * T_local:(rank + 1) * T_local]
            if attention_mask is None:
                pad_id     = self.tokenizer.pad_token_id
                local_mask = (ids_local != pad_id).to(device)
            else:
                padded     = self._pad_or_trim(attention_mask.long(), T).bool()
                local_mask = padded[:, rank * T_local:(rank + 1) * T_local]
        else:
            ids_local  = ids
            T_local    = T
            local_mask = (
                (ids_local != self.tokenizer.pad_token_id).to(device)
                if attention_mask is None
                else self._pad_or_trim(attention_mask.long(), T).bool()
            )

        # 3) embed + transformer (also accumulates MoE aux loss)
        x_local   = self.dropout_emb(self.token_embedding_table(ids_local))
        total_aux = torch.tensor(0.0, device=device, dtype=x_local.dtype)
        cur       = x_local
        for blk in self.blocks:
            cur, aux, _, _ = blk(cur, mask=local_mask)
            if self.training:
                total_aux += aux
        x_processed = self.ln_f(cur)

        # 4) local cut-cross-entropy (memory-friendly, no gather)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        lm_w   = self.lm_head.weight
        if hasattr(lm_w, "ds_numel"):                   # ZeRO-3 sharded weight
            with GatheredParameters([lm_w], enabled=True):
                w = lm_w #.clone()
        else:
            w = lm_w

        raw_ce = linear_cross_entropy(
            e=x_processed,          # (B, T_local, D)
            c=w,                    # vocab weight
            targets=ids_local,      # (B, T_local)
            ignore_index=pad_id,
            reduction="sum",        # sum so we can globally average later
            shift=1
        )

        # count real targets on this shard (drop the first token per shard)
        n_local = (ids_local[:, 1:] != pad_id).sum()

        # 5) all-reduce loss and token count so every rank has the global mean
        if world_size > 1:
            with torch.no_grad():
                dist.all_reduce(raw_ce)
                dist.all_reduce(n_local)

        # 6) final loss
        loss = raw_ce / n_local.clamp(min=1)
        if self.training and total_aux.item() != 0:
            loss = loss + total_aux

        return loss

    def forward_embeddings_only(self, input_ids, attention_mask=None):
        """Forward pass returning final hidden states (before LM head)."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        device = input_ids.device
        T = self.block_size

        ids = self._pad_or_trim(input_ids, T)
        B, _ = ids.shape

        if attention_mask is None:
            pad_id = self.tokenizer.pad_token_id
            mask = (ids != pad_id).to(device)
        else:
            mask = self._pad_or_trim(attention_mask.long(), T).bool()

        # ring attention works on full context, so no split
        x = self.token_embedding_table(ids)
        x = self.dropout_emb(x)

        for blk in self.blocks:
            x, aux_loss, _, _ = blk(x, mask=mask)

        x = self.ln_f(x)

        if world_size > 1:
            shards = [torch.empty_like(x) for _ in range(world_size)]
            dist.all_gather(shards, x.contiguous())
            x_full = torch.cat(shards, dim=1)
        else:
            x_full = x
        return x_full

    # ------------------------------------------------------------------ #
    #  Coconut latent-reasoning forward  (k = 1, memory-friendly)         #
    # ------------------------------------------------------------------ #
    def forward_coconut(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        labels=None,
        latent_token_id=None,
        reduction: str = "mean",
        show_progress: bool = True,
    ):
        assert latent_token_id is not None  # Ensure latent_token_id is provided

        world      = dist.is_initialized()
        world_size = dist.get_world_size() if world else 1
        rank       = dist.get_rank()      if world else 0
        device     = input_ids.device
        T_glb      = self.block_size
        B          = input_ids.size(0)

        # Pad/trim & shard
        ids_full = self._pad_or_trim(input_ids, T_glb)  # (B, T_glb)
        if world_size > 1:
            T_loc = T_glb // world_size
            start_idx_loc = rank * T_loc
            end_idx_loc   = start_idx_loc + T_loc
        else:
            T_loc = T_glb
            start_idx_loc = 0
            end_idx_loc   = T_glb
        ids_loc = ids_full[:, start_idx_loc:end_idx_loc]  # (B, T_loc)

        pad_id = self.tokenizer.pad_token_id or -1
        eot_id = self.tokenizer.convert_tokens_to_ids("</reasoning>")
        bot_id = latent_token_id

        if attention_mask is None:
            full_attention_mask = (ids_full != pad_id)
        else:
            full_attention_mask = self._pad_or_trim(attention_mask.long(), T_glb).bool()
        loc_mask = full_attention_mask[:, start_idx_loc:end_idx_loc].to(device)

        # Find spans & compute max_ct_steps_in_batch
        spans_full_global_indices = []
        max_ct_steps_in_batch = 0
        for b_idx in range(B):
            bot_indices = (ids_full[b_idx] == bot_id).nonzero(as_tuple=True)[0]
            eot_indices = (ids_full[b_idx] == eot_id).nonzero(as_tuple=True)[0]
            current_batch_spans = []
            for b_g_idx in bot_indices:
                later_eots = eot_indices[eot_indices > b_g_idx]
                if later_eots.numel() > 0:
                    e_g_idx = later_eots[0].item()
                    num_cts = e_g_idx - b_g_idx.item() - 1
                    if num_cts >= 0:
                        current_batch_spans.append((b_g_idx.item(), e_g_idx))
                        max_ct_steps_in_batch = max(max_ct_steps_in_batch, num_cts)
            spans_full_global_indices.append(current_batch_spans)
        if world:
            tmp = torch.tensor(max_ct_steps_in_batch, device=device)
            dist.all_reduce(tmp, op=dist.ReduceOp.MAX)
            max_ct_steps_in_batch = int(tmp.item())

        # --- MODIFICATION: Cap max_ct_steps_in_batch to prevent OOM ---
        HARD_LIMIT_MAX_CTS = getattr(config, "COCONUT_MAX_ITERATIONS", 50)
        if max_ct_steps_in_batch > HARD_LIMIT_MAX_CTS:
            if rank == 0 and show_progress:
                print(f"WARNING: Capping max_ct_steps_in_batch from {max_ct_steps_in_batch} to {HARD_LIMIT_MAX_CTS}")
            max_ct_steps_in_batch = HARD_LIMIT_MAX_CTS
        # --- END MODIFICATION ---

        # Prepare storage
        current_g_embeddings = self.token_embedding_table(ids_full).detach().clone()  # (B, T_glb, D)
        total_aux_loss = torch.zeros(1, device=device)
        n_layers = len(self.blocks)
        past_ks = [None] * n_layers
        past_vs = [None] * n_layers

        # INITIAL FULL PASS TO BUILD CACHE
        x_loc = self.dropout_emb(current_g_embeddings[:, start_idx_loc:end_idx_loc])
        for i, blk in enumerate(self.blocks):
            out, aux, k, v = blk(x_loc, mask=loc_mask, past_k=None, past_v=None)
            total_aux_loss += aux
            past_ks[i], past_vs[i] = k, v
            x_loc = out

        # Gather full hidden to select initial CT input
        if world:
            shards = [torch.empty_like(x_loc) for _ in range(world_size)]
            dist.all_gather(shards, x_loc.contiguous())
            h_full_global = torch.cat(shards, dim=1)
        else:
            h_full_global = x_loc

        # Pick the first '<reasoning>' hidden as 'cur'
        cur = torch.stack([
            h_full_global[b, spans_full_global_indices[b][0][0], :]
            for b in range(B)
        ], dim=0).unsqueeze(1)  # (B,1,D)

        # ITERATIVE CT PASSES WITH KV CACHE
        for ct_iter in range(max_ct_steps_in_batch):
            if show_progress and rank == 0:
                print(f"  CT Gen Pass {ct_iter+1}/{max_ct_steps_in_batch}")
            x_step = self.dropout_emb(cur)  # (B,1,D)
            for i, blk in enumerate(self.blocks):
                out, aux, k, v = blk(
                    x_step,
                    mask=None,
                    past_k=past_ks[i],
                    past_v=past_vs[i],
                )
                total_aux_loss += aux
                past_ks[i], past_vs[i] = k, v
                x_step = out

            # Write the new hidden into current_g_embeddings
            for b_idx in range(B):
                bot_g_idx, eot_g_idx = spans_full_global_indices[b_idx][0]
                src_idx = bot_g_idx + ct_iter
                tgt_idx = src_idx + 1
                if tgt_idx < eot_g_idx:
                    current_g_embeddings[b_idx, tgt_idx, :] = x_step[b_idx, 0, :]
            cur = x_step

        if show_progress and rank == 0:
            print(f"[Coconut] Final Logits Pass ({max_ct_steps_in_batch+1} total effective passes)")

        # FINAL PASS & LOSS (core logic unchanged)
        final_input_embeddings_loc = current_g_embeddings[:, start_idx_loc:end_idx_loc].clone()
        x_loc = self.dropout_emb(final_input_embeddings_loc)
        iter_aux_loss = torch.zeros_like(total_aux_loss)
        for blk in self.blocks:
            x_loc, aux, _, _ = blk(x_loc, mask=loc_mask, past_k=None, past_v=None)
            iter_aux_loss += aux
        x_loc = self.ln_f(x_loc)

        if world:
            final_shards = [torch.empty_like(x_loc) for _ in range(world_size)]
            dist.all_gather(final_shards, x_loc.contiguous())
            x_full_for_logits = torch.cat(final_shards, dim=1)
        else:
            x_full_for_logits = x_loc

        loss_mask_full = torch.zeros_like(ids_full, dtype=torch.bool)
        for b_idx in range(B):
            bot_g_idx, eot_g_idx = spans_full_global_indices[b_idx][0]
            if eot_g_idx > bot_g_idx + 1:
                loss_mask_full[b_idx, bot_g_idx+1:eot_g_idx] = True

        lm_w = self.lm_head.weight
        with GatheredParameters([lm_w], enabled=hasattr(lm_w, "ds_numel")):
            w_for_ce = lm_w.clone() if hasattr(lm_w, "ds_numel") else lm_w
            targets_for_loss = ids_full.clone()
            targets_for_loss[loss_mask_full] = pad_id
            raw_ce = linear_cross_entropy(
                e=x_full_for_logits,
                c=w_for_ce,
                targets=targets_for_loss,
                ignore_index=pad_id,
                reduction="sum",
                shift=True,
            )

        n_valid = (targets_for_loss[:, 1:] != pad_id).sum()
        if world:
            with torch.no_grad():
                dist.all_reduce(raw_ce)
                dist.all_reduce(n_valid)
        mean_ce_loss = raw_ce / n_valid.clamp(min=1)
        final_loss = mean_ce_loss + (total_aux_loss + iter_aux_loss if self.training else 0.0)

        if rank == 0 and show_progress:
            print(f"[Coconut] Final Loss: {final_loss.item():.4f} (CE: {mean_ce_loss.item():.4f}, Aux: {(total_aux_loss+iter_aux_loss).item():.4f})")

        return final_loss

    def get_embeddings(self, input_ids: torch.Tensor, pool: bool = True):
        """
        Memory-efficient distributed pooling of final hidden states.
        Instead of all-gathering the full sequence,
        we compute a local sum + count then all-reduce.
        Returns:
          (B, D) if pool=True  or  (B, T, D) otherwise
        """
        # 1) run your usual forward_embeddings_only up through ln_f,
        #    but shard the sequence exactly as in forward_next_token_efficient
        world = dist.is_initialized()
        world_size = dist.get_world_size() if world else 1
        rank       = dist.get_rank()      if world else 0
        T          = self.block_size
        B, _       = input_ids.shape

        # pad/trim
        ids_full = self._pad_or_trim(input_ids, T)           # (B, T)
        if world_size > 1:
            T_loc = T // world_size
            start = rank * T_loc
            end   = start + T_loc
            ids_local = ids_full[:, start:end]               # (B, T_loc)
        else:
            ids_local = ids_full
            T_loc = T

        # build mask
        pad_id = self.tokenizer.pad_token_id
        mask_local = (ids_local != pad_id).float().unsqueeze(-1)  # (B, T_loc, 1)

        # embed + dropout + transformer blocks (local only)
        x = self.token_embedding_table(ids_local)            # (B, T_loc, D)
        x = self.dropout_emb(x)
        for blk in self.blocks:
            x, aux_loss, _, _ = blk(x, mask=mask_local.squeeze(-1))
        x = self.ln_f(x)                                     # (B, T_loc, D)

        if not pool:
            # if you want the full sequence back, you could all-gather here,
            # but since Phase 4 only ever uses pool=True, we leave this minimal.
            raise NotImplementedError("Sequence return not supported in this helper")

        # 2) local pooling
        sum_local   = (x * mask_local).sum(dim=1)             # (B, D)
        count_local =     mask_local.sum(dim=1)              # (B, 1)

        # 3) all-reduce across ranks
        if world_size > 1:
            dist.all_reduce(sum_local,   op=dist.ReduceOp.SUM)
            dist.all_reduce(count_local, op=dist.ReduceOp.SUM)

        # 4) final mean
        pooled = sum_local / count_local.clamp(min=1.0)      # (B, D)
        return pooled

    @torch.no_grad()
    def generate(  # add to SparseMoELanguageModel
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        use_cache: bool = True,      # you already support KV-cache in your blocks
    ) -> torch.LongTensor:
        """
        A minimal autoregressive sampler:
          – supports greedy (do_sample=False) or top-p sampling
          – re-uses your block’s KV-cache to avoid recomputing the full prefix
        """
        device = input_ids.device
        B, seq_len = input_ids.shape
    
        # We'll keep track of past_k/v per layer:
        past_ks = [None] * len(self.blocks)
        past_vs = [None] * len(self.blocks)
    
        # Start generation loop:
        generated = input_ids
        for _ in range(max_new_tokens):
            # 1) get final hidden states + updated past_k/v from your forward
            #    you need a helper that returns (logits, new_past_ks, new_past_vs)
            logits, past_ks, past_vs = self._step_forward(
                generated,
                past_ks,
                past_vs,
                use_cache=use_cache,
            )
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
    
            if do_sample:
                # Top-p filter
                sorted_logits, sorted_idx = next_logits.sort(descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = probs.cumsum(dim=-1)
                mask = cum_probs > top_p
                mask[..., 1:] &= ~mask[..., :-1]
                sorted_logits[mask] = -float("inf")
                next_probs = F.softmax(sorted_logits, dim=-1)
                next_token = sorted_idx.gather(
                    -1,
                    torch.multinomial(next_probs, 1),
                )
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
    
            generated = torch.cat([generated, next_token], dim=1)
    
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
    
        # pad shorter sequences if you want a fixed length
        if pad_token_id is not None and generated.size(1) < seq_len + max_new_tokens:
            pad_len = seq_len + max_new_tokens - generated.size(1)
            pad = torch.full((B, pad_len), pad_token_id, device=device, dtype=torch.long)
            generated = torch.cat([generated, pad], dim=1)
    
        return generated
    
    def _step_forward(self, 
        input_ids: torch.LongTensor,
        past_ks: list[torch.Tensor | None],
        past_vs: list[torch.Tensor | None],
        use_cache: bool,
    ):
        """
        A single forward pass returning:
          - logits over vocab (B, cur_len, V)
          - updated past_ks and past_vs for each block
        """
        # 1) pad / trim to block_size
        x = self._pad_or_trim(input_ids, self.block_size)
    
        # 2) embed + dropout
        x = self.dropout_emb(self.token_embedding_table(x))
    
        new_past_ks, new_past_vs = [], []
        # 3) run through each block with KV-cache
        for i, blk in enumerate(self.blocks):
            x, _, k, v = blk(
                x,
                mask=None,    # assumes full context
                past_k=past_ks[i],
                past_v=past_vs[i],
            )
            new_past_ks.append(k)
            new_past_vs.append(v)
    
        # 4) final LN + vocab projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
    
        return logits, new_past_ks, new_past_vs
    
# -----------------------------------------------------------------------------#
#  helpers to extend context length
# -----------------------------------------------------------------------------#
def update_model_rope_for_extended_context(model, new_seq_len, base: float = 10_000.0):
    """Updates RoPE parameters in all attention layers for a new sequence length."""
    if not hasattr(model, 'blocks'):
        logging.warning("Model has no 'blocks'; cannot update RoPE.")
        return model

    updated = False
    current_max = 0
    for blk in model.blocks:
        if hasattr(blk.sa, 'update_rope'):
            blk.sa.update_rope(new_seq_len, base)
            current_max = max(current_max, blk.sa.max_seq)
            updated = True

    if updated and hasattr(model, 'block_size'):
        if current_max != model.block_size:
            model.block_size = current_max
            logging.info(f"Model block_size updated to {model.block_size}")

    if hasattr(model, 'tokenizer') and hasattr(model, 'block_size'):
        if model.tokenizer.model_max_length != model.block_size:
            model.tokenizer.model_max_length = model.block_size
            logging.info(f"Tokenizer max_length updated to {model.block_size}")

    return model
