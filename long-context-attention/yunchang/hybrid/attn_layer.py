from yunchang.comm.all_to_all import SeqAllToAll4D, SeqAllToAll5D
import torch
from typing import Any
from torch import Tensor
import torch.distributed as dist
from .utils import RING_IMPL_DICT, RING_IMPL_QKVPACKED_DICT
from yunchang.globals import PROCESS_GROUP, HAS_SPARSE_SAGE_ATTENTION
from yunchang.kernels import AttnType

class LongContextAttention(torch.nn.Module):
    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
        attn_processor: torch.nn.Module = None,
    ) -> None:
        super(LongContextAttention, self).__init__()
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG
        self.use_pack_qkv = use_pack_qkv
        self.use_sync = use_sync
        self.attn_type = attn_type
        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.attn_processor = attn_processor
        self.ring_attn_fn = RING_IMPL_DICT[ring_impl_type]
        if HAS_SPARSE_SAGE_ATTENTION:
            from spas_sage_attn.autotune import SparseAttentionMeansim
            if isinstance(attn_processor, SparseAttentionMeansim) and self.ring_pg is not None and dist.get_world_size(self.ring_pg) > 1:
                raise RuntimeError("Sparse Sage attention does not support ring degree > 1.")

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
        *args: Any,
    ) -> Tensor:
        ulysses_world_size = 1
        if self.ulysses_pg is not None:
            try:
                ulysses_world_size = dist.get_world_size(self.ulysses_pg)
            except RuntimeError: # Process group not initialized or invalid
                pass


        can_pack_qkv = self.use_pack_qkv and \
                       query.shape[1] == key.shape[1] and \
                       query.shape[1] == value.shape[1]

        if can_pack_qkv:
            qkv = torch.cat([query, key, value], dim=-1).contiguous()
            if ulysses_world_size > 1:
                qkv = SeqAllToAll4D.apply(
                    self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx, use_sync=self.use_sync
                )
            q_s, k_s, v_s = torch.chunk(qkv, 3, dim=0)
            out = self.ring_attn_fn(
                q_s,
                k_s,
                v_s,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
                attn_type=self.attn_type,
                attn_processor=self.attn_processor,
            )
        else:
            if ulysses_world_size > 1:
                query_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, query, self.scatter_idx, self.gather_idx, use_sync=self.use_sync
                )
                key_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, key, self.scatter_idx, self.gather_idx, use_sync=self.use_sync
                )
                value_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, value, self.scatter_idx, self.gather_idx, use_sync=self.use_sync
                )
            else:
                query_layer, key_layer, value_layer = query, key, value
            
            out = self.ring_attn_fn(
                query_layer,
                key_layer,
                value_layer,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
                attn_type=self.attn_type,
                attn_processor=self.attn_processor,
            )

        attn_output_val = out
        if isinstance(out, tuple): # RingFlashAttnFunc returns (out, lse, None) if return_softmax (derived from return_attn_probs)
            attn_output_val = out[0]
            # If you need to return lse or other parts of the tuple, handle it here
            # For now, just extracting the main attention output

        context_layer = attn_output_val

        if ulysses_world_size > 1:
            output = SeqAllToAll4D.apply(
                self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx, use_sync=self.use_sync
            )
        else:
            output = context_layer
        
        if return_attn_probs and isinstance(out, tuple) and len(out) > 1:
             return output, out[1] # Assuming out[1] is LSE or similar prob info
        return output

class LongContextAttentionQKVPacked(torch.nn.Module):
    def __init__(
        self,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
    ) -> None:
        super(LongContextAttentionQKVPacked, self).__init__()
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG
        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.ring_attn_fn = RING_IMPL_QKVPACKED_DICT[ring_impl_type]
        self.attn_type = attn_type
        
    def forward(
        self,
        qkv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
    ) -> Tensor:
        ulysses_world_size = 1
        if self.ulysses_pg is not None:
            try:
                ulysses_world_size = dist.get_world_size(self.ulysses_pg)
            except RuntimeError:
                pass

        if ulysses_world_size > 1:
            qkv = SeqAllToAll5D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx, self.use_sync
            )

        out = self.ring_attn_fn(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_type=self.attn_type,
        )

        output_val = out
        if isinstance(out, tuple):
            output_val = out[0]
        
        final_out = output_val
        if ulysses_world_size > 1:
            # scatter_idx - 1 for scatter_idx in SeqAllToAll4D is specific to 5D -> 4D transform's effect
            final_out = SeqAllToAll4D.apply( 
                self.ulysses_pg, output_val, self.gather_idx, self.scatter_idx - 1, self.use_sync
            )
        
        if return_attn_probs and isinstance(out, tuple) and len(out) > 1:
            return final_out, out[1]
        return final_out
