# utils/model.py
# =============================================================================
#  Sequence-Parallel Ring Flash-Attention Sparse-MoE LM
# =============================================================================
import math
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import datetime  # For potential timeout in new_group

# Project imports
from utils.config import config
from utils.ebm import EnergyBasedModel
from utils.data import GLOBAL_TOKENIZER
from cut_cross_entropy import linear_cross_entropy

from deepspeed.runtime.zero.partition_parameters import GatheredParameters

# Now import the (patched) flash‐attention binding
from ring_attention_pytorch import ring_flash_attn
from flash_attn.flash_attn_interface import (
    flash_attn_unpadded_qkvpacked_func,
    flash_attn_unpadded_kvpacked_func,
)
FLASH_ATTN_AVAILABLE = True
RING_ATTN_AVAILABLE = True
logging.info("Successfully imported ring-attention-pytorch and flash-attn.")

import os

# -----------------------------------------------------------------------------#
#  RoPE helpers
# -----------------------------------------------------------------------------#
def build_sin_cos(
    seq_len: int, half_dim: int, device: torch.device, base: float = 10_000.0, dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds sinusoidal and cosinusoidal positional embeddings."""
    pos = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim * 2, 2, dtype=dtype, device=device) / (half_dim * 2)))
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
#  Multi-Head Attention (Using ring_flash_attn)
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

        self.max_seq = config.BLOCK_SIZE # Initial max sequence length for RoPE
        # Initialize RoPE buffers on CPU with float32 for precision
        # *** Fixed TypeError Here ***
        sin, cos = build_sin_cos(self.max_seq, self.head_dim // 2, torch.device("cpu"))
        self.register_buffer("rope_sin", sin.float(), persistent=False)
        self.register_buffer("rope_cos", cos.float(), persistent=False)
        # No ring_pg attribute needed

    def update_rope(self, new_len: int, base: float = 10_000.0):
        """Updates RoPE buffers if new_len exceeds current max_seq."""
        if new_len <= self.max_seq:
            return
        logging.info(f"Extending RoPE max sequence length from {self.max_seq} to {new_len}")
        self.max_seq = new_len

        # build fresh sin/cos on CPU
        sin, cos = build_sin_cos(
            new_len,
            self.head_dim // 2,
            torch.device("cpu"),
            base=base,
            dtype=torch.float32
        )
        # overwrite the existing buffers (keeps them in self.buffers())
        self._buffers['rope_sin'] = sin.float()
        self._buffers['rope_cos'] = cos.float()

    # Inside MultiHeadAttention class
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attn_probs: bool = False):
        # x here is the LOCAL SHARD: (B, T_local, C)
        B, T_local, C = x.shape
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        device = x.device

        if return_attn_probs: raise NotImplementedError(...)
        if not FLASH_ATTN_AVAILABLE: raise RuntimeError(...)

        qkv_out = self.qkv(x) # B, T_local, 3*C
        qkv_reshaped = qkv_out.view(B, T_local, self.n_head, 3 * self.head_dim)
        # These are LOCAL q, k, v
        q_local, k_local, v_local = qkv_reshaped.split([self.head_dim, self.head_dim, self.head_dim], dim=-1)

        # Apply RoPE based on GLOBAL position
        total_seq_len = T_local * world_size # Full conceptual length
        self.update_rope(total_seq_len) # Ensure buffers are large enough
        sin = self.rope_sin[:total_seq_len].to(device=device)
        cos = self.rope_cos[:total_seq_len].to(device=device)
        local_start = rank * T_local
        # Apply RoPE using the slice corresponding to this rank's global position
        sin_for_local = sin[local_start : local_start + T_local]
        cos_for_local = cos[local_start : local_start + T_local]
        q_local_rope, k_local_rope = apply_rope(q_local, k_local, sin_for_local, cos_for_local)

        # Call ring attention with LOCAL tensors
        if is_distributed := (world_size > 1):
            attn_output = ring_flash_attn(
                q_local_rope, k_local_rope, v_local, # Pass LOCAL tensors
                mask=mask,
                bucket_size=int(config.BLOCK_SIZE/world_size/2),
                causal=True,
                ring_reduce_col=True
                # striped_ring_attn=...
            ) # Output is (B, T_local, H, Dh)
        else: # Single GPU fallback
            qkv_packed = torch.stack([q_local_rope, k_local_rope, v_local], dim=3) # Stack along new dim 3
            qkv_packed = qkv_packed.view(B * T_local, 3, self.n_head, self.head_dim).contiguous() # Reshape

            # Prepare cu_seqlens
            cu_seqlens = torch.arange(0, (B * T_local) + 1, step=T_local, dtype=torch.int32, device=device)

            # --- Call the CORRECT function ---
            attn_output_flat = flash_attn_unpadded_qkvpacked_func( # <--- CORRECT NAME
                qkv_packed, # (total, 3, nheads, headdim)
                cu_seqlens, # (batch_size + 1,)
                T_local,    # max_seqlen
                dropout_p=config.DROPOUT if self.training else 0.0,
                softmax_scale=None,
                causal=True,
                return_attn_probs=False,
            )
            # --- End correct call ---
            attn_output = attn_output_flat.view(B, T_local, self.n_head, self.head_dim) # Reshape back

        attn_output_reshaped = attn_output.view(B, T_local, C)
        proj_output = self.proj(attn_output_reshaped) # (B, T_local, C)
        return proj_output

# -----------------------------------------------------------------------------#
#  Mixture-of-Experts blocks (Keeping simplified version for now)
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
    """Noisy Top-k Router (Placeholder - check efficient implementations)."""
    def __init__(self, d: int, n_exp: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate_proj = nn.Linear(d, n_exp, bias=False)
        self.noise_proj = nn.Linear(d, n_exp, bias=False)
        with torch.no_grad(): self.noise_proj.weight.normal_(0, 0.01)

    def forward(self, hidden_states):
        logits = self.gate_proj(hidden_states).float()
        if self.training:
            noise_level = self.noise_proj(hidden_states).float()
            noise = torch.randn_like(logits) * F.softplus(noise_level)
            logits = logits + noise
        gates = F.softmax(logits, dim=-1)
        gates_values, indices = torch.topk(gates, self.top_k, dim=-1, sorted=False)
        # Return dense gates needed for typical aux loss
        return gates_values.type_as(hidden_states), indices, gates.type_as(hidden_states)

class SparseMoE(nn.Module):
    """Sparse MoE layer."""
    def __init__(self, d: int, n_exp: int, top_k: int, cap_factor: float = 1.25):
        super().__init__()
        self.router = NoisyTopkRouter(d, n_exp, top_k)
        self.experts = nn.ModuleList([Expert(d) for _ in range(n_exp)])
        self.top_k = top_k
        self.capacity_factor = cap_factor
        self.num_experts = n_exp
        self.hidden_dim = d

    def forward(self, hidden_states):
        """
        Forward pass for SparseMoE with capacity enforcement.
        """
        batch_size, sequence_length, dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        flat_hidden_states = hidden_states.view(num_tokens, dim)
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1. Get Routing Info
        top_k_gates, top_k_indices, dense_router_probs = self.router(flat_hidden_states)
        # top_k_indices shape: (num_tokens, top_k)
        # top_k_gates shape: (num_tokens, top_k)

        # 2. Calculate Capacity per Expert
        capacity = max(1, math.ceil((self.capacity_factor * num_tokens) / self.num_experts))
        logging.debug(f"MoE Capacity per expert: {capacity}")

        # 3. Determine Token Dispatch & Enforce Capacity
        expert_mask = torch.zeros(num_tokens, self.num_experts, dtype=torch.bool, device=device)
        # Mark experts chosen in top-k for each token
        expert_mask.scatter_(1, top_k_indices, True)

        # Determine position of token for each expert it's routed to
        route_cumsum = torch.cumsum(expert_mask.long(), dim=0)
        route_position = route_cumsum * expert_mask - 1 # 0-indexed position

        # Keep assignments where the position is within capacity
        keep_mask = (route_position < capacity) & expert_mask # Shape: (num_tokens, num_experts)

        # --- Correctly map capacity mask back to top-k choices ---
        # pass_cap_mask_k[n, k] is True if expert chosen as k-th choice for token n passed capacity check.
        pass_cap_mask_k = keep_mask.gather(1, top_k_indices) # Shape: (num_tokens, top_k)

        # Find the indices (token_idx, k_idx) where pass_cap_mask_k is True
        kept_indices_where = torch.where(pass_cap_mask_k)
        kept_token_indices = kept_indices_where[0] # Token indices that have at least one valid route
        k_for_kept_tokens = kept_indices_where[1]  # Which k-th choice was kept (0 to top_k-1)

        # Retrieve the corresponding expert index and gate value using the identified indices
        kept_expert_indices = top_k_indices[kept_token_indices, k_for_kept_tokens]
        kept_gate_values = top_k_gates[kept_token_indices, k_for_kept_tokens]
        # --- End mapping fix ---

        # --- Calculate actual load correctly based on kept assignments ---
        actual_load = torch.zeros(self.num_experts, device=device, dtype=torch.long)
        if kept_expert_indices.numel() > 0:
           actual_load.scatter_add_(0, kept_expert_indices, torch.ones_like(kept_expert_indices))
        logging.debug(f"MoE Actual Load: {actual_load.tolist()}")

        # 4. Dispatch tokens to experts and compute outputs
        final_hidden_states = torch.zeros_like(flat_hidden_states)

        # Sort tokens by expert index
        sorted_indices = torch.argsort(kept_expert_indices)
        sorted_kept_expert_indices = kept_expert_indices[sorted_indices]
        sorted_kept_token_indices = kept_token_indices[sorted_indices]
        sorted_kept_gate_values = kept_gate_values[sorted_indices]

        # Find boundaries between experts in the sorted list
        expert_change_indices = torch.cat([
            torch.tensor([0], device=device),
            torch.where(sorted_kept_expert_indices[:-1] != sorted_kept_expert_indices[1:])[0] + 1,
            torch.tensor([len(sorted_kept_expert_indices)], device=device)
        ])

        # Process experts one by one
        for i in range(self.num_experts):
            start_idx = expert_change_indices[i]
            end_idx = expert_change_indices[i+1]

            # Check if this segment actually corresponds to expert i and has tokens
            if start_idx < end_idx and sorted_kept_expert_indices[start_idx] == i:
                current_token_indices = sorted_kept_token_indices[start_idx:end_idx]
                current_gate_values = sorted_kept_gate_values[start_idx:end_idx].unsqueeze(1)
                current_input_tokens = flat_hidden_states[current_token_indices]

                # Run the expert
                expert_output = self.experts[i](current_input_tokens)

                # Combine expert output with gate values and add back
                weighted_output = expert_output * current_gate_values
                final_hidden_states.index_add_(0, current_token_indices, weighted_output)

        # 5. Calculate Auxiliary Loss
        if self.training:
            # Use actual load and router probs for aux loss
            fraction_tokens_routed = actual_load / max(1, num_tokens)
            fraction_prob_routed = dense_router_probs.mean(dim=0)
            aux_loss_factor = 0.01
            aux_loss = torch.sum(fraction_prob_routed * fraction_tokens_routed) * self.num_experts * aux_loss_factor
            logging.debug(f"MoE Aux Loss: {aux_loss.item()}")
        else:
            aux_loss = torch.tensor(0.0, device=device, dtype=dtype)


        return final_hidden_states.view_as(hidden_states), aux_loss

class Block(nn.Module):
    """Transformer block using Ring Attention and SparseMoE FFN."""
    def __init__(self, d: int, n_head: int, n_exp: int, top_k: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = MultiHeadAttention(d, n_head) # Now uses ring_flash_attn internally
        self.ln2 = nn.LayerNorm(d)
        self.moe = SparseMoE(d, n_exp, top_k)

    # Inside Block class
    def forward(self, x_local, mask=None): # Input x_local is (B, T_local, C)
        # Pass the local shard to attention
        attn_output = self.sa(self.ln1(x_local), mask=mask) # Input/Output is (B, T_local, C)
        x_local = x_local + attn_output
        # MoE operates on the local shard
        moe_out, aux_loss = self.moe(self.ln2(x_local)) # Input/Output is (B, T_local, C)
        out_local = x_local + moe_out
        return out_local, aux_loss # Return local shard output

# -----------------------------------------------------------------------------#
#  Sparse-MoE Language Model (using ring_flash_attn)
# -----------------------------------------------------------------------------#
class SparseMoELanguageModel(nn.Module):
    """Sparse MoE Transformer Language Model with Ring Attention."""
    def __init__(
        self,
        n_embed,
        n_head,
        n_layer,
        block_size, # Max context window size
        dropout,
        num_experts,
        top_k,
        tokenizer_name=None, # Allow passing tokenizer object directly
        # tokenizer_name="hf-internal-testing/llama-tokenizer",
    ):
        super().__init__()
        from transformers import LlamaTokenizerFast # Keep import here

        if tokenizer_name:
             try:
                  self.tokenizer = LlamaTokenizerFast.from_pretrained(
                       tokenizer_name, model_max_length=block_size
                  )
             except Exception as e:
                  logging.exception(f"Failed to load tokenizer '{tokenizer_name}'")
                  raise
        else:
             # Use tokenizer passed from data.py
             self.tokenizer = GLOBAL_TOKENIZER
             # Update tokenizer max length if model block size differs
             if self.tokenizer.model_max_length != block_size:
                  logging.warning(f"Updating tokenizer model_max_length from {self.tokenizer.model_max_length} to {block_size}")
                  self.tokenizer.model_max_length = block_size

        # Ensure pad token is set
        if self.tokenizer.pad_token_id is None:
             if self.tokenizer.eos_token_id is not None:
                  self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                  logging.info(f"Set pad_token_id to eos_token_id ({self.tokenizer.pad_token_id})")
             else:
                  raise ValueError("Tokenizer must have a pad_token_id or eos_token_id")

        vocab_size = len(self.tokenizer) # Get vocab size *after* potential token additions
        self.n_embed = n_embed
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed, padding_idx=self.tokenizer.pad_token_id)
        self.dropout_emb = nn.Dropout(dropout) # Use config dropout value

        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying
        self.token_embedding_table.weight = self.lm_head.weight

        # EBM component (assuming definition exists)
        self.ebm = EnergyBasedModel(n_embed)

        self.apply(self._init_weights)
        num_params = self.get_num_params() # Calculate after init
        logging.info(f"Initialized SparseMoELanguageModel with {num_params / 1e6:.2f}M parameters.")
        if not isinstance(self.lm_head.weight, torch.nn.Parameter):
             logging.info("Model parameters appear to be partitioned (ZeRO Stage 3 detected).")

    def _init_weights(self, module):
        """Initializes weights using standard Transformer practices."""
        if isinstance(module, nn.Linear):
            std_dev = 0.02
            # Apply scaling for residual connections (common practice)
            if hasattr(module, 'is_residual_proj') and module.is_residual_proj:
                 std_dev = std_dev / math.sqrt(2 * len(self.blocks))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std_dev)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                 with torch.no_grad(): module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
             module.bias.data.zero_()
             module.weight.data.fill_(1.0)
        # Mark residual projection layers? (Optional, for scaled init)
        # if isinstance(module, Block):
        #      if hasattr(module.sa, 'proj'): module.sa.proj.is_residual_proj = True
        #      if hasattr(module.moe, ...): # If MoE has a final projection
        #           module.moe.final_proj.is_residual_proj = True # Example


    def get_num_params(self, non_embedding=True):
        """ Calculates the number of parameters in the model, attempting to handle DeepSpeed ZeRO3. """
        if isinstance(self.lm_head.weight, torch.nn.Parameter):
            params = list(self.parameters())
            is_zero3 = False
        else: # Assume ZeRO3 partitioning
            params = []
            for p in self.parameters():
                if hasattr(p, 'ds_numel'): params.append(p)
                elif torch.is_tensor(p): params.append(p)
            is_zero3 = True

        n_params = sum(p.ds_numel if hasattr(p,'ds_numel') else p.numel() for p in params)

        if non_embedding:
             if is_zero3:
                  emb_numel = self.token_embedding_table.weight.ds_numel if hasattr(self.token_embedding_table.weight, 'ds_numel') else 0
             else:
                  emb_numel = self.token_embedding_table.weight.numel()
             n_params -= emb_numel # Subtract tied embedding weight only once
        return n_params

    def _pad_or_trim(self, ids, target_length):
        """Pads (left) or trims (left) input_ids to the target_length."""
        B, T = ids.shape
        pad_id = self.tokenizer.pad_token_id
        assert pad_id is not None, "Tokenizer must have a pad_token_id"
        if T == target_length: return ids
        elif T > target_length: return ids[:, -target_length:]
        else:
            padding = torch.full((B, target_length - T), pad_id, dtype=ids.dtype, device=ids.device)
            return torch.cat([padding, ids], dim=1)

    def forward(self, input_ids, attention_mask=None, labels=None, reduction="mean"):
        """
        Main forward pass, delegates to specific methods based on needs.
        Currently set up for next token prediction loss.
        """
        # Use the efficient forward for training/loss calculation
        # Assuming labels signify training objective
        if labels is not None:
             return self.forward_next_token_efficient(
                  input_ids=input_ids,
                  reduction=reduction,
                  attention_mask=attention_mask, # Pass mask along
                  labels=labels # Pass labels to calculate loss correctly
             )
        else:
             # Handle inference or cases where only embeddings are needed
             # This part needs careful implementation if used for generation
             # Currently returns final hidden states before LM head
             return self.forward_embeddings_only(
                  input_ids=input_ids,
                  attention_mask=attention_mask
             )


    # inside SparseMoELanguageModel:

    def forward_next_token_efficient(
        self,
        input_ids,
        reduction="mean",
        attention_mask=None,
    ):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        device = input_ids.device
        T = self.block_size

        # pad/trim
        ids = self._pad_or_trim(input_ids, T)
        B, T = ids.shape

        # split sequence for model parallel
        if world_size > 1:
            if T % world_size != 0:
                raise ValueError(f"Block size {T} not divisible by world size {world_size}")
            T_local = T // world_size
            ids_local = ids[:, rank*T_local:(rank+1)*T_local]
            if attention_mask is None:
                pad_id = self.tokenizer.pad_token_id
                local_attention_mask = (ids_local != pad_id).to(device)
            else:
                padded_mask = self._pad_or_trim(attention_mask.long(), T).bool()
                local_attention_mask = padded_mask[:, rank*T_local:(rank+1)*T_local]
        else:
            ids_local = ids
            T_local = T
            if attention_mask is None:
                pad_id = self.tokenizer.pad_token_id
                local_attention_mask = (ids_local != pad_id).to(device)
            else:
                local_attention_mask = self._pad_or_trim(attention_mask.long(), T).bool()

        # embed + transformer
        x_local = self.token_embedding_table(ids_local)
        x_local = self.dropout_emb(x_local)

        total_aux_loss = torch.tensor(0.0, device=device, dtype=x_local.dtype)
        current_x = x_local
        for blk in self.blocks:
            current_x, aux_loss = blk(current_x, mask=local_attention_mask)
            if self.training and aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
        x_processed = self.ln_f(current_x)

        # gather back to full sequence
        if world_size > 1:
            gather_list = [torch.empty_like(x_processed) for _ in range(world_size)]
            dist.all_gather(gather_list, x_processed.contiguous())   # ← supports autograd
            x_full = torch.cat(gather_list, dim=1)
        else:
            x_full = x_processed

        # --- LOSS (FP32 only) ---
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        targets = ids.to(x_full.device)

        is_zero3 = hasattr(self.lm_head.weight, "ds_numel")
        if is_zero3:
            with GatheredParameters([self.lm_head.weight], enabled=True):
                full_w = self.lm_head.weight.clone()
                loss = linear_cross_entropy(
                    e=x_full,            # (B, T, C)  still float32
                    c=full_w,
                    targets=targets,     # (B, T)
                    ignore_index=pad_id,
                    reduction=reduction,
                    shift=1,
                )
        else:
            loss = linear_cross_entropy(
                e=x_full,
                c=self.lm_head.weight,
                targets=targets,
                ignore_index=pad_id,
                reduction=reduction,
                shift=1,
            )

        if self.training and total_aux_loss.item() != 0:
            loss = loss + total_aux_loss

        # cleanup
        del x_full, targets, x_processed
        if world_size > 1:
            del current_x

        return loss

    def forward_embeddings_only(self, input_ids, attention_mask=None):
         """ Forward pass returning final hidden states (before LM head). """
         # Similar logic to forward_next_token_efficient but stops before LM head/loss
         world_size = dist.get_world_size() if dist.is_initialized() else 1
         rank = dist.get_rank() if dist.is_initialized() else 0
         device = input_ids.device
         current_block_size = self.block_size

         ids = self._pad_or_trim(input_ids, current_block_size)
         B, T = ids.shape

         if attention_mask is None:
             pad_id = self.tokenizer.pad_token_id
             attention_mask = (ids != pad_id).to(device)
         else:
             attention_mask = self._pad_or_trim(attention_mask.long(), current_block_size).bool()

         ids_local = ids # Ring attention works on full sequence context
         x_local = self.token_embedding_table(ids_local)
         x_local = self.dropout_emb(x_local)

         for blk in self.blocks:
             # Pass mask if needed by MoE
             x_local, _ = blk(x_local, mask=attention_mask) # Ignore aux loss

         x_local = self.ln_f(x_local)

         # Gather results if distributed
         if world_size > 1:
             x_full_shards = [torch.empty_like(x_local) for _ in range(world_size)]
             dist.all_gather(x_full_shards, x_local.contiguous(), group=None)
             x_full = torch.cat(x_full_shards, dim=1)
         else:
             x_full = x_local

         return x_full # Return (B, T, C) final hidden states

    # ------------- Coconut latent masking ------------- #
    def forward_coconut(self, *args, **kwargs):
        """ Placeholder for official COCONUT mechanism. Requires implementation. """
        logging.error("`forward_coconut` called, but official COCONUT logic is not implemented yet.")
        raise NotImplementedError("Implement official COCONUT multi-pass forward logic here.")

    def get_embeddings(self, input_ids, pool=True):
         """ Gets final layer embeddings, handles pooling & sequence parallelism. """
         # Uses forward_embeddings_only to get final states
         x = self.forward_embeddings_only(input_ids, attention_mask=None) # Assumes no mask needed here
         B, T, C = x.shape
         device = x.device

         if pool:
              # Need original padded ids to create correct mask for pooling
              ids = self._pad_or_trim(input_ids, T) # Ensure ids match sequence length T
              pad_id = self.tokenizer.pad_token_id
              pad_mask = (ids != pad_id).float().unsqueeze(-1).to(device) # (B, T, 1)
              masked_sum = (x * pad_mask).sum(dim=1) # (B, C)
              num_non_padding = pad_mask.sum(dim=1).clamp(min=1e-6)
              pooled_embeddings = masked_sum / num_non_padding
              return pooled_embeddings
         else:
              return x # Return full sequence embeddings (B, T, C)

# -----------------------------------------------------------------------------#
#  helpers to extend context length
# -----------------------------------------------------------------------------#
def update_model_rope_for_extended_context(model, new_seq_len, base: float = 10_000.0):
    """Updates RoPE parameters in all attention layers for a new sequence length."""
    if not hasattr(model, 'blocks'):
         logging.warning("Model does not have 'blocks' attribute, cannot update RoPE.")
         return model

    logging.info(f"Updating RoPE for new max sequence length: {new_seq_len}, base: {base}")
    updated = False
    current_max = 0
    for i, blk in enumerate(model.blocks):
        if hasattr(blk, 'sa') and hasattr(blk.sa, 'update_rope'):
            blk.sa.update_rope(new_seq_len, base)
            current_max = max(current_max, blk.sa.max_seq)
            updated = True
        # else: logging.warning(...) # Optional warning if layer doesn't have update_rope

    if updated and hasattr(model, 'block_size'):
        new_model_block_size = current_max # Set based on actual buffer size
        if new_model_block_size != model.block_size:
             model.block_size = new_model_block_size
             logging.info(f"Model block_size updated to {model.block_size}")
    elif updated:
         logging.warning("Model block_size attribute not found, RoPE updated but model size not tracked.")

    # Update tokenizer max length if model block size changed
    if hasattr(model, 'tokenizer') and hasattr(model, 'block_size'):
         if model.tokenizer.model_max_length != model.block_size:
              logging.info(f"Updating tokenizer model_max_length to {model.block_size}")
              model.tokenizer.model_max_length = model.block_size

    return model


def expand_pos_embedding(*args, **kwargs):
    """ Deprecated function - No longer needed with RoPE. """
    logging.warning("`expand_pos_embedding` is deprecated and does nothing when using RoPE.")
    pass
