# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang


import torch

from typing import Any
from torch import Tensor
from yunchang.kernels import AttnType, select_flash_attn_impl
import torch.distributed as dist
from yunchang.comm.all_to_all import SeqAllToAll4D


class UlyssesAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        use_sync (bool): whether to synchronize after all-to-all. This flag can save cuda memory but will slow down the speed.
        attn_type (AttnType): attention type enum
    """

    def __init__(
        self,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
        attn_type : AttnType = AttnType.FA,
    ) -> None:

        super(UlyssesAttention, self).__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.attn_type = attn_type

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(device)
        if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
            self.attn_type = AttnType.TORCH
        self.attn_fn = select_flash_attn_impl(self.attn_type, stage="fwd-bwd")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # scatter 2, gather 1
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx, self.use_sync)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx, self.use_sync)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx, self.use_sync)

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        context_layer = self.attn_fn(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale = softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.spg, context_layer, self.gather_idx, self.scatter_idx, self.use_sync
        )

        # out e.g., [s/p::h]
        return output

