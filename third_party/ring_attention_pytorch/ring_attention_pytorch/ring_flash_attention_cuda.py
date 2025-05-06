from __future__ import annotations

from math import ceil

import torch
from torch import Tensor
from torch.autograd.function import Function
from torch.amp import autocast

from ring_attention_pytorch.ring import (
    ring_pass,
    all_ring_pass,
    null_ring_pass,
    get_rank,
    get_world_size
)

from beartype import beartype
from einops import rearrange, repeat, reduce

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def exists(v):
    return v is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

# ──────────────────────────────────────────────────────────────────────────────
# Triton‐accelerated FlashAttention interface
# ──────────────────────────────────────────────────────────────────────────────
class RingFlashAttentionCUDAFunction(Function):

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None,
        causal: bool,
        bucket_size: int,
        ring_reduce_col: bool,
        striped_ring_attn: bool,
        max_lookback_seq_len: int | None,
        ring_size: int | None,
        softclamp_qk_sim: bool,
        softclamp_value: float
    ):
        # —————————————————————————————————————————————
        #  setup / validation
        # —————————————————————————————————————————————
        from ring_attention_pytorch.triton_flash_attn import flash_attn_forward

        assert k.shape[-2:] == v.shape[-2:], "K/V head dims must match"
        qh, kh = q.shape[-2], k.shape[-2]
        assert divisible_by(qh, kh), "Q heads must be divisible by KV heads"
        q_head_groups = qh // kh
        assert all(t.is_cuda for t in (q, k, v)), "Inputs must be on CUDA devices"

        orig_dtype          = q.dtype
        softmax_scale_val   = q.shape[-1] ** -0.5          # <‑‑ python float
        softmax_scale_tensor = torch.tensor(softmax_scale_val, device=q.device, dtype=q.dtype)

        # cast to fp16/bf16 if input is fp32 (Triton kernels only support 16‑bit)
        if q.dtype in (torch.float32, torch.bfloat16):
            q, k, v = q.half(), k.half(), v.half()

        ring_size          = default(ring_size, get_world_size())
        cross_attn         = q.shape[-3] != k.shape[-3]
        ring_reduce_col   &= not cross_attn
        striped_ring_attn &= not cross_attn

        per_machine_seq = k.shape[-3]
        max_ring_passes = None
        if exists(max_lookback_seq_len):
            assert causal
            assert not (ring_reduce_col and not divisible_by(per_machine_seq, bucket_size))
            max_ring_passes = ceil(max_lookback_seq_len / per_machine_seq)

        if causal:
            mask = None  # no key‑padding mask when causal

        bucket_size = min(per_machine_seq, bucket_size)
        qlen        = q.shape[1]

        # —————————————————————————————————————————————
        #  ring passes (always UN‑normalised)
        # —————————————————————————————————————————————
        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass
        kv           = torch.stack((k, v))          # (2, …)

        o = m = lse = None
        recv_kv   = None
        recv_mask = None

        for (ring_rank, (is_first, _)), ((kv_block, mask_block), (recv_kv, recv_mask)) in \
                ring_pass_fn(kv, mask,
                             receive_buffers=(recv_kv, recv_mask),
                             max_iters=max_ring_passes,
                             ring_size=ring_size):

            k_blk, v_blk = kv_block
            k_blk = repeat(k_blk, '... h d -> ... (g h) d', g=q_head_groups)
            v_blk = repeat(v_blk, '... h d -> ... (g h) d', g=q_head_groups)

            bias = None
            if exists(mask_block):
                bias = torch.where(mask_block, 0.0, float('-inf'))

            # decide causal‑mask logic for this block
            block_causal      = False
            causal_mask_diag  = False
            if causal:
                if striped_ring_attn:
                    block_causal     = True
                    causal_mask_diag = get_rank() < ring_rank
                else:                                # diagonal blocks only
                    block_causal = (get_rank() == ring_rank)
                    if get_rank() < ring_rank:      # skip upper‑tri blocks
                        continue

            o, m, lse = flash_attn_forward(
                q, k_blk, v_blk,
                bias=bias,
                causal=block_causal,
                o=o, m=m, lse=lse,
                softmax_scale=softmax_scale_tensor,
                causal_mask_diagonal=causal_mask_diag,
                return_normalized_output=False,     # always UN‑normalised
                load_accumulated=not is_first,
                softclamp_qk_sim=softclamp_qk_sim,
                softclamp_value=softclamp_value,
            )

        # —————————————————————————————————————————————
        #  python‑side normalisation (exact)
        # —————————————————————————————————————————————
        m   = m[..., :qlen]                                   # (B,H,T)
        lse = m + (lse[..., :qlen] - m).exp().sum(-1, keepdim=True).log()

        # reshape (B,H,T) → (B,T,H) so broadcast with o (B,T,H,D)
        norm = torch.exp(m.transpose(-2, -1).unsqueeze(-1) -
                         lse.transpose(-2, -1).unsqueeze(-1))        # (B,T,H,1)

        o = o[..., :qlen, :, :] * norm                            # (B,T,H,D)

        # —————————————————————————————————————————————
        #  stash for backward
        # —————————————————————————————————————————————
        ctx.args = (
            causal,                      # bool
            softmax_scale_val,           # float  (NOT a tensor)
            mask,
            bucket_size,
            ring_reduce_col,
            max_ring_passes,
            striped_ring_attn,
            ring_size,
            q_head_groups,
            softclamp_qk_sim,
            softclamp_value,
            orig_dtype
        )
        ctx.save_for_backward(q, k, v, o, lse)

        return o.to(orig_dtype)

    @staticmethod
    def backward(ctx, do: Tensor):
        from ring_attention_pytorch.triton_flash_attn import flash_attn_backward

        (causal, scale, mask, bucket_size, ring_reduce_col,
         max_passes, striped_ring_attn, ring_size, qhg,
         softclamp_qk_sim, softclamp_value, orig_dtype) = ctx.args

        q, k, v, o, lse = ctx.saved_tensors
        # ensure `do` matches `o`’s dtype before triton
        do = do.to(o.dtype)

        if causal:
            mask = None

        # prepare accumulators in FP32
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        kvdkv = torch.stack((k, v, dk, dv))
        recv_buf = recv_mask = None

        ring_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        for (rnk, _), ((kvd, mb), (rbuf, rmask)) in \
                ring_fn(kvdkv, mask, receive_buffers=(recv_buf, recv_mask),
                        max_iters=max_passes, ring_size=ring_size):
            kb, vb, dkb, dvb = kvd
            kb = repeat(kb, '... h d -> ... (g h) d', g=qhg)
            vb = repeat(vb, '... h d -> ... (g h) d', g=qhg)

            bias = None
            if exists(mb):
                bias = torch.where(mb, 0.0, float('-inf'))
                bias = rearrange(bias, 'b j -> b 1 1 j')

            # same block_causal logic as forward
            if causal and striped_ring_attn:
                need = True; blk_causal=True; diag_mask=get_rank()<rnk
            elif causal:
                need = (get_rank() >= rnk)
                blk_causal = (get_rank()==rnk)
                diag_mask=False
            else:
                need = True; blk_causal=False; diag_mask=False

            if need:
                ring_dq = torch.empty_like(q, dtype=torch.float32)
                ring_dk = torch.empty_like(k, dtype=torch.float32)
                ring_dv = torch.empty_like(v, dtype=torch.float32)

                with torch.inference_mode():
                    flash_attn_backward(
                        do, q, kb, vb, o, lse,
                        ring_dq, ring_dk, ring_dv,
                        delta=None,
                        bias=bias,
                        causal=blk_causal,
                        causal_mask_diagonal=diag_mask,
                        softmax_scale=scale,
                        softclamp_qk_sim=softclamp_qk_sim,
                        softclamp_value=softclamp_value,
                    )

                # collapse grouped heads
                ring_dk = reduce(ring_dk, '... (g h) d -> ... h d', g=qhg, reduction='sum')
                ring_dv = reduce(ring_dv, '... (g h) d -> ... h d', g=qhg, reduction='sum')

                dq.add_(ring_dq)
                dk.add_(ring_dk)
                dv.add_(ring_dv)

        # cast gradients back to original input dtype
        return (dq.to(orig_dtype),
                dk.to(orig_dtype),
                dv.to(orig_dtype),
                None, None, None, None, None, None, None, None, None)

ring_flash_attn_cuda_ = RingFlashAttentionCUDAFunction.apply

@autocast('cuda', enabled=False)
@beartype
def ring_flash_attn_cuda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    causal: bool = False,
    bucket_size: int = 1024,
    ring_reduce_col: bool = False,
    striped_ring_attn: bool = False,
    max_lookback_seq_len: int | None = None,
    ring_size: int | None = None,
    softclamp_qk_sim: bool = False,
    softclamp_value: float = 50.
):
    # If single‐GPU (no ring) fallback to torch.scaled_dot_product_attention
    if get_world_size() == 1 and not ring_reduce_col:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=causal,
            dropout_p=0.0
        )
    # otherwise call our custom Triton‐accelerated Function
    return ring_flash_attn_cuda_(
        q, k, v, mask, causal,
        bucket_size, ring_reduce_col, striped_ring_attn,
        max_lookback_seq_len, ring_size,
        softclamp_qk_sim, softclamp_value
    )
