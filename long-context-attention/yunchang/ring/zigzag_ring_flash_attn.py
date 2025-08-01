import torch
from .utils import RingComm, update_out_and_lse
from yunchang.kernels import AttnType, select_flash_attn_impl
import torch.distributed as dist

def zigzag_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: AttnType = AttnType.FA,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]

    out = None
    lse = None

    def forward_block(q_block, k_block, v_block, causal_flag):
        fn = select_flash_attn_impl(attn_type, stage="fwd-only")
        return fn(
            q_block,
            k_block,
            v_block,
            dropout_p,
            softmax_scale,
            causal=causal_flag,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=(dropout_p > 0),
        )

    for step in range(comm.world_size):
        # 1) Exchange full k, v before any slicing
        if step + 1 != comm.world_size:
            print(f"[rank {dist.get_rank()}] forward step={step} (pre-exchange), k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}")
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()
            comm.wait()
            k, v = next_k, next_v
            print(f"[rank {dist.get_rank()}] forward step={step} (post-exchange), k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}")

        # 2) Perform attention on appropriate slice
        if step == 0:
            block_out, block_lse = forward_block(q, k, v, True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_lse = forward_block(q, k0, v0, False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward_block(q1, k, v, False)
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

def zigzag_ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: AttnType = AttnType.FA,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout1 = dout.chunk(2, dim=1)[1]
    q1 = q.chunk(2, dim=1)[1]
    out1 = out.chunk(2, dim=1)[1]
    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
    block_seq_len = q.shape[1] // 2

    # buffers for backward
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def backward_step(dout_local, q_local, k_local, v_local, out_local, lse_local, causal_flag, step):
        seqlen_q = q_local.shape[1]
        seqlen_kv = k_local.shape[1]
        fn = select_flash_attn_impl(attn_type, stage="bwd-only")

        # Debug before fn: check for non-finite
        if not torch.isfinite(dq_buffer).all() or not torch.isfinite(dk_buffer).all() or not torch.isfinite(dv_buffer).all():
            print(f"[rank {dist.get_rank()}] Non-finite in buffers BEFORE backward call, step={step}")

        fn(
            dout_local,
            q_local,
            k_local,
            v_local,
            out_local,
            lse_local,
            dq_buffer[:, :seqlen_q],
            dk_buffer[:, :seqlen_kv],
            dv_buffer[:, :seqlen_kv],
            dropout_p,
            softmax_scale,
            causal_flag,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            rng_state=None,
        )

        # Debug after fn: check for non-finite
        if not torch.isfinite(dq_buffer).all():
            print(f"[rank {dist.get_rank()}] Non-finite in dq_buffer AFTER backward call, step={step}")
        if not torch.isfinite(dk_buffer).all():
            print(f"[rank {dist.get_rank()}] Non-finite in dk_buffer AFTER backward call, step={step}")
        if not torch.isfinite(dv_buffer).all():
            print(f"[rank {dist.get_rank()}] Non-finite in dv_buffer AFTER backward call, step={step}")

    for step in range(kv_comm.world_size):
        # Exchange k/v
        if step + 1 != kv_comm.world_size:
            print(f"[rank {dist.get_rank()}] backward step={step}, sending k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}")
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        # Compute local gradients
        if step == 0:
            backward_step(dout, q, k, v, out, softmax_lse, True, step)
            dq = dq_buffer.float()
            dk = dk_buffer.float()
            dv = dv_buffer.float()
        else:
            if step <= kv_comm.rank:
                backward_step(dout, q, k[:, :block_seq_len], v[:, :block_seq_len], out, softmax_lse, False, step)
                dq += dq_buffer
            else:
                backward_step(dout1, q1, k, v, out1, softmax_lse1, False, step)
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

            # Prepare gradient exchange
            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            # Accumulate
            if step <= kv_comm.rank:
                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        # Advance k/v for next step
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        # Exchange gradients
        print(f"[rank {dist.get_rank()}] backward step={step}, sending dk.shape={tuple(dk.shape)}, dv.shape={tuple(dv.shape)}")
        next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()

    # Final debug before return: check for non-finite final grads
    if not torch.isfinite(dq).all() or not torch.isfinite(dk).all() or not torch.isfinite(dv).all():
        print(f"[rank {dist.get_rank()}] Non-finite final grads: dq.norm={dq.norm().item()}, dk.norm={dk.norm().item()}, dv.norm={dv.norm().item()}")

    # Return own gradients, not the 'next_' buffers
    return dq.to(q.dtype), dk.to(q.dtype), dv.to(q.dtype)

class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_type,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = zigzag_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_type=attn_type,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.attn_type = attn_type
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = zigzag_ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            attn_type=ctx.attn_type,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return ZigZagRingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
    )


def zigzag_ring_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
    )


def zigzag_ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
    )
