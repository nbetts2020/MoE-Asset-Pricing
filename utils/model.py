# utils/model.py
# =============================================================================
#  Sequence-Parallel Unified‑SP (YunChang) Attention + Sparse‑MoE LM
# =============================================================================

import torch
torch.Tensor.continous = torch.Tensor.contiguous

import yunchang.hybrid.attn_layer as _yc_attn

_SeqApply_orig = _yc_attn.SeqAllToAll4D.apply
def _SeqApply_shim(*args, **kwargs):
    # drop any kwargs and forward to real apply
    if kwargs:
        return _SeqApply_orig(*args)
    return _SeqApply_orig(*args)
_yc_attn.SeqAllToAll4D.apply = _SeqApply_shim

import sys
sys.path.insert(0, "/home/ubuntu/MoE-Asset-Pricing/flash-attention")

import flash_attn.flash_attn_interface as _fai
import flash_attn                       as _fa

_fai.flash_attn_func        = _fai.flash_attn_qkvpacked_func
_fa .flash_attn_func        = _fai.flash_attn_qkvpacked_func
_fai.flash_attn_varlen_func = _fai.flash_attn_varlen_qkvpacked_func
_fa .flash_attn_varlen_func = _fai.flash_attn_varlen_qkvpacked_func

_fai._flash_attn_forward         = _fai._flash_attn_forward
_fai._flash_attn_backward        = _fai._flash_attn_backward
_fai._flash_attn_varlen_forward  = _fai._flash_attn_varlen_forward
_fai._flash_attn_varlen_backward = _fai._flash_attn_varlen_backward

torch.ops.flash_attn._flash_attn_forward         = _fai._flash_attn_forward
torch.ops.flash_attn._flash_attn_backward        = _fai._flash_attn_backward
torch.ops.flash_attn._flash_attn_varlen_forward  = _fai._flash_attn_varlen_forward
torch.ops.flash_attn._flash_attn_varlen_backward = _fai._flash_attn_varlen_backward

import math, logging
from typing import Tuple, Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# project‑local imports … ------------------------------------------------------
from utils.config import config
from utils.ebm    import EnergyBasedModel
from utils.data   import GLOBAL_TOKENIZER
from cut_cross_entropy import linear_cross_entropy
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

from yunchang import set_seq_parallel_pg, LongContextAttention, EXTRACT_FUNC_DICT
from yunchang.kernels import AttnType

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

# -----------------------------------------------------------------------------#
#  Multi-Head Attention (Using YunChang USP)
# -----------------------------------------------------------------------------#
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.n_embed = n_embed

        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)

        # RoPE buffers
        self.max_seq = config.BLOCK_SIZE  # initial max seq length
        sin, cos = build_sin_cos(self.max_seq, self.head_dim // 2, torch.device("cpu"))
        self.register_buffer("rope_sin", sin.float(), persistent=False)
        self.register_buffer("rope_cos", cos.float(), persistent=False)

        # set up sequence-parallel process groups
        if dist.is_initialized():
            set_seq_parallel_pg(
                config.SP_ULYSSES_DEGREE,
                config.SP_RING_DEGREE,
                dist.get_rank(),
                dist.get_world_size()
            )

        # Unified Sequence-Parallel Attention
        self.usp_attn = LongContextAttention(
            scatter_idx=2,            # optional, defaults to 2
            gather_idx=1,             # optional, defaults to 1
            ring_impl_type="zigzag",
            use_pack_qkv=True,
            use_sync=True,
            attn_type=AttnType.FA,
            attn_processor=None,      # leave as default unless you have a custom processor
        )

    def update_rope(self, new_len: int, base: float = 10_000.0):
        """Extends RoPE buffers if needed."""
        if new_len <= self.max_seq:
            return
        logging.info(f"Extending RoPE max sequence length from {self.max_seq} to {new_len}")
        self.max_seq = new_len
        sin, cos = build_sin_cos(new_len, self.head_dim // 2, torch.device("cpu"), base=base)
        self._buffers['rope_sin'] = sin.float()
        self._buffers['rope_cos'] = cos.float()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attn_probs: bool = False):
        # x: (B, T_local, C)
        B, T_local, C = x.shape
        device = x.device
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # 1) Project to QKV
        qkv = self.qkv(x).view(B, T_local, self.n_head, 3 * self.head_dim)
        q_local, k_local, v_local = qkv.split(self.head_dim, dim=-1)

        # 2) Apply RoPE
        total_seq = T_local * world_size
        self.update_rope(total_seq)
        sin = self.rope_sin[:total_seq].to(device)
        cos = self.rope_cos[:total_seq].to(device)
        start = rank * T_local
        sin_l = sin[start:start + T_local]
        cos_l = cos[start:start + T_local]
        q_local_rope, k_local_rope = apply_rope(q_local, k_local, sin_l, cos_l)

        # 3) USP attention call
        out = self.usp_attn(
            q_local_rope, k_local_rope, v_local,
            dropout_p = config.DROPOUT if self.training else 0.0,
            softmax_scale=None,
            causal=True,
            window_size=(-1,-1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
            return_attn_probs=False,
        )
        attn_out = out.reshape(B, T_local, C)
        # 4) Final projection
        return self.proj(attn_out)

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
        logits = self.gate_proj(hidden_states).float()

        if self.training:
            # seed off the per‑module counter
            g = torch.Generator(device=logits.device)
            g.manual_seed(int(self.router_step.item()))

            # draw fresh noise each forward
            noise = torch.empty_like(logits)
            noise.normal_(mean=0.0, std=1.0, generator=g)
            noise = noise * F.softplus(self.noise_proj(hidden_states).float())

            # bump the counter for next time
            self.router_step += 1
        else:
            noise = torch.zeros_like(logits)

        logits = logits + noise
        gates = F.softmax(logits, dim=-1)
        vals, inds = torch.topk(gates, self.top_k, dim=-1, sorted=False)
        return vals.type_as(hidden_states), inds, gates.type_as(hidden_states)

class SparseMoE(nn.Module):
    """Sparse MoE layer."""
    def __init__(self, d: int, n_exp: int, top_k: int, cap_factor: float = 2):
        super().__init__()
        self.router = NoisyTopkRouter(d, n_exp, top_k)
        self.experts = nn.ModuleList([Expert(d) for _ in range(n_exp)])
        self.top_k = top_k
        self.cap_factor = cap_factor
        self.num_experts = n_exp

    def forward(self, hidden_states):
        """
        hidden_states: (B, T, d)
        returns: (B, T, d), aux_loss
        """
        B, T, d = hidden_states.shape
        num_tokens = B * T
        flat = hidden_states.view(num_tokens, d)

        # 1) Routing
        top_vals, top_inds, dense_probs = self.router(flat)

        # 2) Compute capacity and mask
        local_tokens = torch.tensor(B*T, device=flat.device)
        # dist.all_reduce(global_tokens)        # now equals B*T
        capacity = math.ceil(self.cap_factor * local_tokens.item() / self.num_experts)

        mask = torch.zeros(num_tokens, self.num_experts, device=flat.device, dtype=torch.bool)
        mask.scatter_(1, top_inds, True)

        # 3) Enforce capacity
        cumsum = torch.cumsum(mask.long(), dim=0)
        position = cumsum * mask - 1
        keep = (position < capacity) & mask
        kept = torch.where(keep)
        tok_idx, exp_slot = kept  # exp_slot is which expert slot

        # 4) Gather expert indices & weights
        exp_idx = exp_slot
        gate_vals = dense_probs[tok_idx, exp_slot].unsqueeze(1)

        # 5) Dispatch to experts in sorted order for efficiency
        final = torch.zeros_like(flat)
        sorted_idx = torch.argsort(exp_idx)

        seg_boundaries = torch.cat([
            torch.tensor([0], device=flat.device),
            torch.where(exp_idx[sorted_idx][:-1] != exp_idx[sorted_idx][1:])[0] + 1,
            torch.tensor([sorted_idx.numel()], device=flat.device),
        ])

        for i in range(len(seg_boundaries) - 1):
            s = seg_boundaries[i].item()
            e = seg_boundaries[i + 1].item()
            if s >= e:
                continue
            expert_i = exp_idx[sorted_idx[s]].item()
            idxs = tok_idx[sorted_idx[s:e]]
            gv = gate_vals[sorted_idx[s:e]]
            out = self.experts[expert_i](flat[idxs])
            final.index_add_(0, idxs, out * gv)

        # 6) Compute auxiliary load‐balancing loss
        aux_loss = torch.tensor(0.0, device=flat.device)
        if self.training:
            frac_tokens = torch.bincount(exp_idx, minlength=self.num_experts).float() / max(num_tokens, 1)
            frac_probs = dense_probs.mean(dim=0)
            aux_loss = (frac_probs * frac_tokens).sum() * self.num_experts * 0.01

        # 7) Reshape back to (B, T, d)
        return final.view(B, T, d), aux_loss


class Block(nn.Module):
    """Transformer block using USP Attention and SparseMoE FFN."""
    def __init__(self, d: int, n_head: int, n_exp: int, top_k: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = MultiHeadAttention(d, n_head)
        self.ln2 = nn.LayerNorm(d)
        self.moe = SparseMoE(d, n_exp, top_k)

    def forward(self, x_local, mask=None):
        attn_out = self.sa(self.ln1(x_local), mask=mask)
        x = x_local + attn_out
        moe_out, aux_loss = self.moe(self.ln2(x))
        return x + moe_out, aux_loss

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
                    '<bot>', '<start_latent>', '<end_latent>',
                    '<reasoning>', '</reasoning>', '<STOCK PRICE 30 DAYS OUT>: '
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
        self.ln_f = nn.LayerNorm(n_embed)
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
            return torch.cat([padding, ids], dim=1)

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
            cur, aux = blk(cur, mask=local_mask)
            if self.training:
                total_aux += aux
        x_processed = self.ln_f(cur)                    # (B, T_local, D)

        # 4) local cut-cross-entropy (memory-friendly, no gather)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        lm_w   = self.lm_head.weight
        if hasattr(lm_w, "ds_numel"):                   # ZeRO-3 sharded weight
            with GatheredParameters([lm_w], enabled=True):
                w = lm_w.clone()
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
            x, _ = blk(x, mask=mask)

        x = self.ln_f(x)

        if world_size > 1:
            shards = [torch.empty_like(x) for _ in range(world_size)]
            dist.all_gather(shards, x.contiguous())
            x_full = torch.cat(shards, dim=1)
        else:
            x_full = x
        return x_full

    # ------------------------------------------------------------------ #
    #  Coconut latent-reasoning forward (multi-pass)                     #
    # ------------------------------------------------------------------ #
    def forward_coconut(
        self,
        input_ids: torch.Tensor,             # (B, T_global)
        attention_mask=None,
        labels=None,                         #  ignored
        latent_token_id=None,                #  pass  self.tokenizer.convert_tokens_to_ids('<bot>')
        reduction: str = "mean",
    ):
        """
        Implements Coconut (Hao et al., 2024):
        – every <bot> token starts a *latent* step
        – its embedding is replaced by the *last* hidden state from the
          previous token
        – no CE loss is taken on the <bot> positions
        Works with sequence-parallel sharding exactly like
        `forward_next_token_efficient`.
        """
        assert latent_token_id is not None, "pass latent_token_id=<bot>-id"

        # == step-0: normal sharding boiler-plate  ────────────────────────
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank       = dist.get_rank()      if dist.is_initialized() else 0
        device     = input_ids.device
        T_glb      = self.block_size

        ids_glb = self._pad_or_trim(input_ids, T_glb)     # (B, T_glb)
        B, _     = ids_glb.shape

        if world_size > 1:
            if T_glb % world_size:
                raise ValueError("block_size not divisible by world_size")
            T_loc   = T_glb // world_size
            ids_loc = ids_glb[:, rank*T_loc:(rank+1)*T_loc]
        else:
            T_loc   = T_glb
            ids_loc = ids_glb

        # build the usual causal mask (pads→0)
        pad_id   = self.tokenizer.pad_token_id
        loc_mask = (ids_loc != pad_id).to(device) if attention_mask is None \
                   else self._pad_or_trim(attention_mask.long(), T_glb)\
                           [:, rank*T_loc:(rank+1)*T_loc].bool()

        # == step-1: clone an *embedding* matrix we can edit in-place ——───
        emb_loc = self.token_embedding_table(ids_loc)         # (B,T_loc,D)

        # we will iteratively *fill in* latent-thought embeddings, one pass
        # per <bot>.  Track which positions are still latent:
        lat_mask = (ids_loc == latent_token_id)               # (B,T_loc)
        n_lat_max = lat_mask.sum(dim=1).max().item()          # worst-case K

        total_aux = torch.tensor(0.0, device=device)          # MoE aux-loss
        dtype     = emb_loc.dtype

        # == multi-pass  loop  (K latent thoughts ⇒ K+1 forward passes) ──
        prev_h = None
        filled = torch.zeros_like(lat_mask)

        for _ in range(int(n_lat_max) + 1):

            # -- a) run one transformer pass on the *current* embeddings
            x = self.dropout_emb(emb_loc)
            for blk in self.blocks:
                x, aux = blk(x, mask=loc_mask)
                if self.training:
                    total_aux += aux
            x = self.ln_f(x)                                 # (B,T_loc,D)

            # -- b) identify the *next* latent token per sample we still
            #       need to fill; replace its embedding with the *last*
            #       hidden state before it (continuous thought)
            with torch.no_grad():
                # first unfinished <bot> in every sample
                nxt_pos = ((lat_mask & ~filled).float()      # 0/1 mask
                           .argmax(dim=1))                   # (B,)
                # if a sample is done, argmax==0, but may not be latent
                gather_idx = []
                update_idx = []
                for b in range(B):
                    p = nxt_pos[b].item()
                    if p==0 and not lat_mask[b,0]:
                        continue               # nothing left in this sample
                    update_idx.append((b, p))
                    prev_h_b = x[b, p-1 if p>0 else 0]       # last hidden
                    gather_idx.append(prev_h_b)
                if not update_idx:
                    break                     # all latent tokens filled

                h_stack = torch.stack(gather_idx, dim=0)     # (N,D)
                for (b,p), h in zip(update_idx, h_stack):
                    emb_loc[b, p] = h.to(dtype)
                    filled  [b, p] = True

        # == step-2: final CE loss  (identical to efficient NLL) ──────────
        lm_w = self.lm_head.weight
        if hasattr(lm_w, "ds_numel"):
            with GatheredParameters([lm_w], enabled=True):
                w = lm_w.clone()
        else:
            w = lm_w

        # ignore pads **and** latent tokens
        ignore = torch.tensor([pad_id, latent_token_id], device=device)

        raw_ce = linear_cross_entropy(
            e=x,                       # final hidden (B,T_loc,D)
            c=w,
            targets=ids_loc,
            ignore_index=ignore,
            reduction="sum",
            shift=1
        )

        n_local = ((~lat_mask) & (ids_loc != pad_id))[:,1:].sum()

        if world_size > 1:
            with torch.no_grad():
                dist.all_reduce(raw_ce); dist.all_reduce(n_local)

        loss = raw_ce / n_local.clamp(min=1)
        if self.training and total_aux.item() != 0:
            loss = loss + total_aux
        return loss


    def get_embeddings(self, input_ids, pool=True):
        """Gets final layer embeddings, with optional mean pooling."""
        x = self.forward_embeddings_only(input_ids, attention_mask=None)
        B, T, C = x.shape
        device = x.device

        if pool:
            ids = self._pad_or_trim(input_ids, T)
            pad_id = self.tokenizer.pad_token_id
            mask = (ids != pad_id).float().unsqueeze(-1).to(device)
            summed = (x * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-6)
            return summed / count
        else:
            return x

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

def expand_pos_embedding(*args, **kwargs):
    """Deprecated: no-op when using RoPE."""
    logging.warning("expand_pos_embedding is deprecated and does nothing.")
    pass
