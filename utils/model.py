import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.config import config
from utils.data import *
from transformers import AutoTokenizer, LlamaTokenizerFast
from torch.utils.checkpoint import checkpoint
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from utils.ebm import EnergyBasedModel  # Still kept for potential future use
from deepspeed.runtime.zero.stage3 import GatheredParameters
from cut_cross_entropy import linear_cross_entropy  # For efficient next-token loss

cuda_available = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################
# Minimal RoPE Helper
#############################################
def build_sin_cos(seq_len, dim, device, base=10000.0):
    """
    Build sine/cosine curves for minimal RoPE.
    seq_len: maximum sequence length
    dim: half of the head dimension (since we chunk in 2)
    device: torch device
    base: RoPE base frequency
    Returns:
      sin, cos => shape (seq_len, dim)
    """
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim * 2, 2, dtype=torch.float, device=device) * (-math.log(base) / (dim * 2))
    )
    sin = torch.sin(position * div_term)
    cos = torch.cos(position * div_term)
    return sin, cos

def apply_rope(q, k, sin, cos):
    """
    Applies rotary embeddings to q and k.
    q, k => (B, T, n_head, head_size)
    sin, cos => (T, head_size//2)
    """
    B, T, n_head, head_size = q.shape
    half = head_size // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    sin_ = sin.unsqueeze(0).unsqueeze(2)
    cos_ = cos.unsqueeze(0).unsqueeze(2)

    q_out_1 = q1 * cos_ - q2 * sin_
    q_out_2 = q2 * cos_ + q1 * sin_
    k_out_1 = k1 * cos_ - k2 * sin_
    k_out_2 = k2 * cos_ + k1 * sin_
    q_rotated = torch.cat([q_out_1, q_out_2], dim=-1)
    k_rotated = torch.cat([k_out_1, k_out_2], dim=-1)

    return q_rotated, k_rotated

#############################################
# MultiHeadAttention
#############################################
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        assert n_embed % n_head == 0, "Embedding dim must be divisible by the number of heads"
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.max_seq_len = config.BLOCK_SIZE
        # Build initial RoPE buffers using the default base
        sin, cos = build_sin_cos(self.max_seq_len, self.head_size // 2, device='cpu')
        self.register_buffer('rope_sin', sin, persistent=False)
        self.register_buffer('rope_cos', cos, persistent=False)
        
    def update_rope_buffers(self, new_seq_len, base=500000.0):
        """
        Rebuilds the RoPE buffers to cover extended context lengths.
        
        Parameters:
           new_seq_len (int): The new maximum sequence length (e.g., 64k).
           base (float): The RoPE base frequency to use (often increased for longer contexts).
        """
        self.max_seq_len = new_seq_len
        sin, cos = build_sin_cos(new_seq_len, self.head_size // 2, device='cpu', base=base)
        # Re-register the buffers to ensure they're updated during training/inference.
        self.register_buffer('rope_sin', sin, persistent=False)
        self.register_buffer('rope_cos', cos, persistent=False)
        
    def forward(self, x, return_attn_probs=False):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        qkv_reshape = qkv.view(B, T, 3, self.n_head, self.head_size)
        q = qkv_reshape[:, :, 0, :, :]
        k = qkv_reshape[:, :, 1, :, :]
        v = qkv_reshape[:, :, 2, :, :]

        sin_t = self.rope_sin[:T, :].to(x.device)
        cos_t = self.rope_cos[:T, :].to(x.device)
        q, k = apply_rope(q, k, sin_t, cos_t)

        # Update the qkv_reshape with rotated q and k
        qkv_reshape[:, :, 0, :, :] = q
        qkv_reshape[:, :, 1, :, :] = k
        qkv_reshape[:, :, 2, :, :] = v
        qkv = qkv_reshape.view(B * T, 3, self.n_head, self.head_size)

        cu_seqlens = torch.arange(0, (B + 1) * T, step=T, dtype=torch.int32, device=x.device)
        max_seqlen = T
        outputs = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen,
            dropout_p=config.DROPOUT, softmax_scale=None, causal=True,
            return_attn_probs=True
        )
        attn_output, attn_probs, full_attn = outputs
        attn_output = attn_output.view(B, T, self.n_head, self.head_size)\
                         .permute(0, 2, 1, 3).reshape(B, T, C)

        out = self.out_proj(attn_output)
        if return_attn_probs:
            return out, attn_probs, full_attn
        else:
            return out

#############################################
# Expert, NoisyTopkRouter, SparseMoE, Block
#############################################
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
    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        full_router_probs = F.softmax(noisy_logits, dim=-1)
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices, full_router_probs

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
        router_output, indices, full_router_probs = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_router_output = router_output.view(-1, router_output.size(-1))

        expert_capacity = int((B * T * self.top_k / self.num_experts) * self.capacity_factor)
        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1).view(-1)
            selected_indices = torch.nonzero(expert_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity]
            if limited_indices.numel() > 0:
                expert_out = expert(flat_x[limited_indices])
                gating = flat_router_output[limited_indices, i].unsqueeze(1)
                updates.index_add_(0, limited_indices, expert_out * gating)

        final_output += updates.view(B, T, -1)
        if self.training:
            entropy = -torch.sum(full_router_probs * torch.log(full_router_probs + 1e-8), dim=-1)
            entropy_loss = entropy.mean()
        else:
            entropy_loss = None

        return final_output, entropy_loss

class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.sa = MultiHeadAttention(n_embed, n_head)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            x_masked = x * attention_mask.unsqueeze(-1)
        else:
            x_masked = x

        attn_out = self.sa(self.ln1(x_masked))
        x = x + attn_out
        moe_out, ent_loss = self.smoe(self.ln2(x))
        x = x + moe_out
        return x, ent_loss

#############################################
# Updated SparseMoELanguageModel (Next-Token Only)
#############################################
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
        tokenizer_name="hf-internal-testing/llama-tokenizer"
    ):
        super().__init__()
        # We only need components for next-token prediction.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
        tokenizer_name, 
        model_max_length=config.CONTEXT_WINDOW
        )
        special_tokens = {
            'additional_special_tokens': ['<bot>', '<start_latent>', '<end_latent>', '<reasoning>', '</reasoning>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        vocab_size = self.tokenizer.vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.attn_pool = nn.Linear(n_embed, 1, bias=False)  # new!
        self.block_size = block_size
        # Retain EBM as an argument for potential online learning or logging.
        self.ebm = EnergyBasedModel(n_embed)

    def preprocess_input(self, input_ids, mode='next_token'):
        """
        For next-token prediction mode, simply truncate (or pad) input_ids to block_size.
        Always take the rightmost tokens.
        """
        device = input_ids.device
        B, T = input_ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)

        if T >= self.block_size:
            trimmed = input_ids[:, -self.block_size:]
        else:
            pad_amount = self.block_size - T
            pad = torch.full((B, pad_amount), pad_id, dtype=input_ids.dtype, device=device)
            trimmed = torch.cat([input_ids, pad], dim=1)
        return trimmed

    def forward_next_token_efficient(self, input_ids, reduction="mean", attention_mask=None, force_bf16=False):
        """
        Next-token prediction mode using Cut Cross-Entropy with built-in shift.
        If the sequence is longer than block_size, it is truncated from the right.
        """
        device = input_ids.device
        B, T = input_ids.shape
        if T > self.block_size:
            input_ids = input_ids[:, -self.block_size:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.block_size:]
            T = self.block_size

        tok_emb = self.token_embedding_table(input_ids)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x, _ = block(x, attention_mask)

        x = self.ln_f(x)
        if force_bf16:
            x = x.to(torch.bfloat16)

        with GatheredParameters(self.token_embedding_table.weight, modifier_rank=0):
            if hasattr(self, '_gathered_weights'):
                classifier = self._gathered_weights
            else:
                classifier = self.token_embedding_table.weight.data.clone().to(device)
                if force_bf16:
                    classifier = classifier.to(torch.bfloat16)

        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        loss = linear_cross_entropy(
            x,               # Transformer outputs in BF16 if forced
            classifier,      # Classifier weights
            input_ids,       # Target token ids
            ignore_index=pad_id,
            reduction=reduction,
            shift=1         # Automatically shift embeddings and targets
        )
        print("forward_next_token_efficient: loss value:", loss.item())
        return loss

    def get_embeddings(self, input_ids, pool=False, attention_mask=None):
        """
        Returns encoder embeddings.
        In next-token prediction mode, the input is first truncated (or padded) to block_size, then
        passed through the embedding layers, transformer blocks, and final layer norm.
        If `pool` is True, returns the mean-pooled embedding (one vector per sample).
        Otherwise, returns the tokenwise output.
        """
        device = input_ids.device
        input_ids = self.preprocess_input(input_ids, mode='next_token')
        if attention_mask is None:
            pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            attention_mask = (input_ids != pad_id).to(device)
        B, T = input_ids.shape

        tok_emb = self.token_embedding_table(input_ids)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x, _ = block(x, attention_mask)

        x = self.ln_f(x)

        if pool:
            # 1) compute raw scores: (B, T, 1) -> (B, T)
            scores = self.attn_pool(x).squeeze(-1)

            # 2) mask out padding tokens
            if attention_mask is None:
                pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
                attention_mask = (input_ids != pad_id).to(x.device)
            scores = scores.masked_fill(~attention_mask, float("-1e9"))

            # 3) softmax over T -> (B, T)
            weights = F.softmax(scores, dim=1)

            # 4) weighted sum -> (B, D)
            return torch.einsum("btd,bt->bd", x, weights)

        else:
            return x

    #####################################################
    # Coconut-like Forward for Latent Reasoning
    #####################################################
    def forward_coconut(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        latent_token_id=99998,
        reduction="mean",
        force_bf16=False
    ):
        """
        A naive Coconut-like multi-pass forward:
         1. Identify positions of 'latent_token_id' in each sequence.
         2. For each latent token, run partial forward pass to get the
            hidden state from the previous position. Replace the latent
            token's embedding with that hidden state.
         3. Finally, run one full pass to compute cross-entropy on the
            entire sequence (shift=1).
        """
        device = input_ids.device
        B, T = input_ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)

        # Build a default attention_mask if none is given
        if attention_mask is None:
            attention_mask = (input_ids != pad_id).to(device)

        # Preprocess input (truncate/pad if needed)
        input_ids_trimmed = self.preprocess_input(input_ids)
        Btrim, Ttrim = input_ids_trimmed.shape

        # Initial embeddings
        with torch.no_grad():
            tok_emb = self.token_embedding_table(input_ids_trimmed)
            pos_emb = self.position_embedding_table(torch.arange(Ttrim, device=device))
            base_embeds = tok_emb + pos_emb  # (Btrim, Ttrim, n_embed)

        # Collect all latent token positions (for each batch element)
        latent_positions = []
        for b_idx in range(Btrim):
            idxs = (input_ids_trimmed[b_idx] == latent_token_id).nonzero(as_tuple=True)
            latent_positions.append(idxs[0].tolist())

        # The max number of latent tokens across the batch
        max_latent = max(len(pos_list) for pos_list in latent_positions) if latent_positions else 0
        embeddings_for_pass = base_embeds.clone().requires_grad_(True)

        # Handle each latent token in a naive multi-pass manner
        for pass_idx in range(max_latent):
            # Figure out the earliest latent position for pass_idx
            positions = []
            for b_idx, lat_list in enumerate(latent_positions):
                if pass_idx < len(lat_list):
                    positions.append(lat_list[pass_idx])
            if len(positions) == 0:
                continue

            earliest_latent = min(positions)
            current_seq_len = earliest_latent + 1

            # truncated forward pass
            truncated_embeds = embeddings_for_pass[:, :current_seq_len, :]
            truncated_am = attention_mask[:, :current_seq_len]

            x = truncated_embeds
            for block in self.blocks:
                x, _ = block(x, truncated_am)
            x = self.ln_f(x)
            if force_bf16:
                x = x.to(torch.bfloat16)

            # Overwrite the embedding at the latent position with the hidden state from (pos-1)
            for b_idx, lat_list in enumerate(latent_positions):
                if pass_idx < len(lat_list):
                    lat_pos = lat_list[pass_idx]
                    if lat_pos == 0:
                        # can't replace if latent is at position 0
                        continue
                    prev_h = x[b_idx, lat_pos - 1, :]
                    embeddings_for_pass[b_idx, lat_pos, :] = prev_h

        # Finally, do a full forward pass with updated embeddings
        full_am = attention_mask[:, :Ttrim]
        x = embeddings_for_pass
        for block in self.blocks:
            x, _ = block(x, full_am)
        x = self.ln_f(x)
        if force_bf16:
            x = x.to(torch.bfloat16)

        with GatheredParameters(self.token_embedding_table.weight, modifier_rank=0):
            if hasattr(self, '_gathered_weights'):
                classifier = self._gathered_weights
            else:
                classifier = self.token_embedding_table.weight.data.clone().to(device)
                if force_bf16:
                    classifier = classifier.to(torch.bfloat16)

        # Compute cross-entropy for next-token prediction
        final_loss = None
        if labels is not None:
            # if needed, trim labels as well
            if labels.shape[1] > Ttrim:
                labels = labels[:, -Ttrim:]

            final_loss = linear_cross_entropy(
                x,  # (Btrim, Ttrim, n_embed)
                classifier,
                labels,
                ignore_index=pad_id,
                reduction=reduction,
                shift=1
            )
        return final_loss

#############################################
# Helper Function for Extended-Context RoPE
#############################################
def update_model_rope_for_extended_context(model, new_seq_len, base=500000.0):
    """
    Iterates through each transformer block in the model and updates its RoPE buffers to cover new_seq_len.
    
    Parameters:
      model: An instance of SparseMoELanguageModel.
      new_seq_len (int): The new target sequence length (e.g. 64000 for 64k tokens).
      base (float): The new RoPE base frequency to use.
    
    Returns:
      model: The updated model with extended-context RoPE buffers.
    """
    for block in model.blocks:
        # Update the RoPE buffers in the MultiHeadAttention module of each block.
        block.sa.update_rope_buffers(new_seq_len, base=base)
    return model
