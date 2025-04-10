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
from utils.ebm import EnergyBasedModel  # Assuming ebm.py is in the same directory
from deepspeed.runtime.zero.stage3 import GatheredParameters
from cut_cross_entropy import linear_cross_entropy  # Import CCE function

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
        sin, cos = build_sin_cos(self.max_seq_len, self.head_size // 2, device='cpu')
        self.register_buffer('rope_sin', sin, persistent=False)
        self.register_buffer('rope_cos', cos, persistent=False)

    def forward(self, x, return_attn_probs=False):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        qkv_reshape = qkv.view(B, T, 3, self.n_head, self.head_size)
        q = qkv_reshape[:, :, 0, :, :]
        k = qkv_reshape[:, :, 1, :, :]
        v = qkv_reshape[:, :, 2, :, :]
        sin_t = self.rope_sin[:T, :]
        cos_t = self.rope_cos[:T, :]
        q, k = apply_rope(q, k, sin_t, cos_t)
        qkv_reshape[:, :, 0, :, :] = q
        qkv_reshape[:, :, 1, :, :] = k
        qkv_reshape[:, :, 2, :, :] = v
        qkv = qkv_reshape.view(B * T, 3, self.n_head, self.head_size)
        cu_seqlens = torch.arange(0, (B + 1) * T, step=T, dtype=torch.int32, device=x.device)
        max_seqlen = T
        outputs = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen,
            dropout_p=config.DROPOUT, softmax_scale=None, causal=False,
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
# Soft Clustering for Regression Fine-Tuning
#############################################
class SoftClusteringModule(nn.Module):
    """
    Implements soft clustering over token embeddings based on Euclidean distances.
    Used only in regression fine-tuning.
    """
    def __init__(self, max_clusters=8, embed_dim=32, cluster_temp=1.0):
        super().__init__()
        self.max_clusters = max_clusters
        self.embed_dim = embed_dim
        self.cluster_temp = cluster_temp
        self.cluster_centers = nn.Parameter(0.02 * torch.randn(self.max_clusters, self.embed_dim))
    def forward(self, token_embs):
        # token_embs: (T, E)
        T, E = token_embs.shape
        var_val = token_embs.var(dim=0).mean().detach()
        cluster_count = max(2, min(self.max_clusters, int(var_val * 5 + 2)))
        centers = self.cluster_centers[:cluster_count]
        dists = (token_embs.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=-1)
        gumbel_noise = torch.empty_like(dists).uniform_(1e-6, 1.0)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise))
        logits = -(dists / (self.cluster_temp + 1e-8)) + gumbel_noise
        cluster_assignments = F.softmax(logits, dim=-1)
        expanded = token_embs.unsqueeze(1) * cluster_assignments.unsqueeze(2)
        cluster_sums = expanded.sum(dim=0)
        cluster_mass = cluster_assignments.sum(dim=0).unsqueeze(1) + 1e-8
        cluster_reps = cluster_sums / cluster_mass
        return cluster_reps

#############################################
# HighDimAttentionPooling (New Aggregator)
#############################################
class HighDimAttentionPooling(nn.Module):
    """
    Implements single-query attention pooling.
    Learns a query vector to attend over the transformer outputs and projects the pooled result
    to a high-dimensional space (e.g. 32k).
    """
    def __init__(self, embed_dim, high_dim=32000):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, embed_dim))
        self.proj = nn.Linear(embed_dim, high_dim)
    def forward(self, x):
        # x: (B, T, embed_dim)
        B, T, E = x.shape
        q_expanded = self.q.expand(B, -1).unsqueeze(1)
        attn_scores = torch.bmm(q_expanded, x.transpose(1, 2)) / math.sqrt(E)
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights, x).squeeze(1)
        high_dim_vector = self.proj(pooled)
        return high_dim_vector

#############################################
# Updated SparseMoELanguageModel with CCE
#############################################
class SparseMoELanguageModel(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, dropout,
                 num_experts, top_k, tokenizer_name="hf-internal-testing/llama-tokenizer"):
        super().__init__()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_name, model_max_length=config.CONTEXT_WINDOW
        )
        vocab_size = self.tokenizer.vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.block_size = block_size
        # For regression fine-tuning:
        self.high_attn_pool = HighDimAttentionPooling(n_embed, high_dim=32000)
        self.regression_head = nn.Sequential(
            nn.Linear(32000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        self.ebm = EnergyBasedModel(n_embed)
        # Clustering is used only for regression fine-tuning.
        self.soft_cluster = SoftClusteringModule(
            max_clusters=8,
            embed_dim=n_embed,
            cluster_temp=1.0
        )

    def fast_cluster(self, token_embs):
        cluster_reps = self.soft_cluster(token_embs)
        cluster_count = cluster_reps.size(0)
        if cluster_count < self.block_size:
            pad_amount = self.block_size - cluster_count
            pad_emb = torch.zeros(pad_amount, cluster_reps.size(-1), device=token_embs.device)
            comp_emb = torch.cat([cluster_reps, pad_emb], dim=0)
        elif cluster_count > self.block_size:
            comp_emb = cluster_reps[:self.block_size]
        else:
            comp_emb = cluster_reps
        return comp_emb

    def preprocess_input(self, input_ids, mode='regression'):
        """
        Processes input_ids into a tensor of shape (B, block_size, n_embed).
        In 'regression' mode, if the true sequence length exceeds block_size,
        perform clustering to compress the sequence.
        In 'next_token' mode, simply truncate (or pad) the sequence to block_size.
        """
        device = input_ids.device
        B, T = input_ids.shape
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        non_pad_mask = (input_ids != pad_id)
        true_lengths = non_pad_mask.sum(dim=1)
        compressed_list = []
        for b in range(B):
            seq_len = true_lengths[b].item()
            token_embs = self.token_embedding_table(input_ids[b, :seq_len])
            if mode == 'regression':
                if seq_len <= self.block_size:
                    if seq_len < self.block_size:
                        pad_amount = self.block_size - seq_len
                        pad_emb = torch.zeros(pad_amount, token_embs.size(-1), device=device)
                        token_embs = torch.cat([token_embs, pad_emb], dim=0)
                    comp_emb = token_embs
                else:
                    comp_emb = self.fast_cluster(token_embs)
            else:
                if seq_len >= self.block_size:
                    comp_emb = token_embs[:self.block_size]
                else:
                    pad_amount = self.block_size - seq_len
                    pad_emb = torch.zeros(pad_amount, token_embs.size(-1), device=device)
                    comp_emb = torch.cat([token_embs, pad_emb], dim=0)
            compressed_list.append(comp_emb.unsqueeze(0))
        compressed_embeddings = torch.cat(compressed_list, dim=0)
        return compressed_embeddings

    def forward(self, input_ids, targets=None,
                use_entropy_reg=False, lambda_entropy=1e-2, percent_complete=0.0,
                attention_mask=None):
        """
        Regression fine-tuning mode.
        Uses clustering when the true sequence length exceeds block_size.
        Attention pooling aggregates the processed tokens into a high-dimensional vector,
        which is then passed to a regression head to produce a scalar output.
        Optionally, an attention_mask is provided to ignore padding tokens.
        """
        device = input_ids.device
        B, T = input_ids.shape
        comp_emb = self.preprocess_input(input_ids, mode='regression')
        pos_emb = self.position_embedding_table(torch.arange(comp_emb.size(1), device=device))
        x = comp_emb + pos_emb
        ent_loss_sum = torch.tensor(0.0, device=device)
        for idx, block in enumerate(self.blocks):
            if self.training:
                x, ent_loss = checkpoint(block, x, attention_mask)
            else:
                x, ent_loss = block(x, attention_mask)
            if ent_loss is not None:
                ent_loss_sum += ent_loss
            print(f"[SMELM] Block {idx} output: {x.shape}")
        x = self.ln_f(x)
        high_dim_vector = self.high_attn_pool(x)
        output = self.regression_head(high_dim_vector)
        ebm_input = x.mean(dim=1)
        ebm_energy = self.ebm(ebm_input)
        task_loss = torch.tensor(0.0, device=device)
        ebm_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        if targets is not None:
            targets_squeezed = targets.squeeze(1) if (targets.dim() == 2 and targets.size(1) == 1) else targets
            task_loss = F.mse_loss(output, targets_squeezed, reduction='mean')
            raw_mse = F.mse_loss(output, targets_squeezed, reduction='none').detach()
            ebm_loss = F.mse_loss(ebm_energy, raw_mse)
            if use_entropy_reg:
                block_count = len(self.blocks)
                entropy_loss = lambda_entropy * ent_loss_sum / block_count
        return {
            "output": output,
            "ebm_energy": ebm_energy,
            "task_loss": task_loss,
            "ebm_loss": ebm_loss,
            "entropy_loss": entropy_loss
        }

    def forward_next_token(self, input_ids, reduction="mean", attention_mask=None):
        """
        Next token prediction mode with Cut Cross-Entropy loss.
        This function has been updated to use CCE for efficient memory usage.
        """
        device = input_ids.device
        B, T = input_ids.shape
        if T > self.block_size:
            input_ids = input_ids[:, -self.block_size:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.block_size:]
            T = self.block_size

        print("forward_next_token: input_ids shape:", input_ids.shape)
        tok_emb = self.token_embedding_table(input_ids)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x, _ = block(x, attention_mask)

        x = self.ln_f(x)
        print("forward_next_token: transformer output x shape:", x.shape)

        # Get the embeddings for tokens that predict the next token
        shift_embeddings = x[:, :-1, :].reshape(-1, x.size(-1))
        shift_labels = input_ids[:, 1:].reshape(-1)
        print("forward_next_token: shift_embeddings shape:", shift_embeddings.shape)
        print("forward_next_token: shift_labels shape:", shift_labels.shape)

        # Get classifier weights (embedding table) for CCE
        with GatheredParameters(self.token_embedding_table.weight, modifier_rank=0):
            classifier = self.token_embedding_table.weight
        print("forward_next_token: classifier shape:", classifier.shape)

        # Use CCE instead of standard cross-entropy
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)

        # Replace the standard cross-entropy calculation:
        # logits = shift_embeddings @ classifier.T
        # loss = F.cross_entropy(logits, shift_labels, ignore_index=pad_id, reduction=reduction)

        # With CCE:
        loss = linear_cross_entropy(
            shift_embeddings,  # token embeddings
            classifier,        # classifier weights
            shift_labels,      # target labels
            ignore_index=pad_id,
            reduction=reduction
        )

        print("forward_next_token: loss value:", loss.item())
        return loss

    # Alternative implementation using shift parameter for more memory efficiency
    def forward_next_token_efficient(self, input_ids, reduction="mean", attention_mask=None, force_bf16=False):
        """
        Memory-optimized next token prediction using CCE's built-in shift parameter.
        This avoids creating the shifted embeddings tensor.
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

        # CRITICAL: Force conversion to BF16 for CCE backward pass compatibility
        if force_bf16:
            x = x.to(torch.bfloat16)

        # Get classifier weights
        with GatheredParameters(self.token_embedding_table.weight, modifier_rank=0):
            # Use gathered weights or get from embedding table
            if hasattr(self, '_gathered_weights'):
                classifier = self._gathered_weights  # Already in BF16
            else:
                classifier = self.token_embedding_table.weight.clone()
                if force_bf16:
                    classifier = classifier.to(torch.bfloat16)

            # Use CCE with BF16 precision
            pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            loss = linear_cross_entropy(
                x,               # Now in BF16
                classifier,      # Now in BF16
                input_ids,
                ignore_index=pad_id,
                reduction=reduction,
                shift=1          # Automatically shift embeddings and targets
            )

        return loss

    def get_embeddings(self, input_ids, return_tokenwise=False, attention_mask=None):
        device = input_ids.device
        comp_emb = self.preprocess_input(input_ids, mode='regression')
        pos_emb = self.position_embedding_table(torch.arange(comp_emb.size(1), device=device))
        x = comp_emb + pos_emb
        for idx, block in enumerate(self.blocks):
            x, _ = block(x, attention_mask)
        x = self.ln_f(x)
        high_dim_vector = self.high_attn_pool(x)
        if return_tokenwise:
            return x
        else:
            return high_dim_vector
