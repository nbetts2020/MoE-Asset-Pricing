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

cuda_available = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################
# MultiHeadAttention
#############################################
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        assert n_embed % n_head == 0, "Embedding dim must be divisible by number of heads"
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x, return_attn_probs=False):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_size).permute(0, 2, 1, 3, 4)
        qkv = qkv.reshape(B * T, 3, self.n_head, self.head_size)

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

        return (out, attn_probs, full_attn) if return_attn_probs else out


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

        # Hard top-k => replaced with top-k soft gating
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
        x = x + self.sa(self.ln1(x))
        moe_out, ent_loss = self.smoe(self.ln2(x))
        x = x + moe_out
        return x, ent_loss


#############################################
# TinyTransformer (unchanged - used for guidance)
#############################################
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embed=32, context_window=config.CONTEXT_WINDOW, n_head=1):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_window, n_embed)
        self.attn = MultiHeadAttention(n_embed, n_head)
        self.ln = nn.LayerNorm(n_embed)
        self.regression_head = nn.Linear(n_embed, 1)

    def forward(self, input_ids):
        B, T = input_ids.shape
        tok_emb = self.token_embedding_table(input_ids)
        pos_emb = self.position_embedding_table(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        x, _, full_attn = self.attn(self.ln(x), return_attn_probs=True)
        x = self.ln(x)
        pooled = x.mean(dim=1)
        output = self.regression_head(pooled)
        if full_attn.dim() == 4:  # (B, 1, T, T)
            attn_scores = full_attn.squeeze(1).sum(dim=-1)  # (B, T)
        elif full_attn.dim() == 3:  # (B, 1, T)
            attn_scores = full_attn.squeeze(1)
        else:
            attn_scores = full_attn
        print(f"[TinyTransformer] attn_scores: {attn_scores.shape}")
        return output, attn_scores


#############################################
# Gumbel-based SoftClustering for fast_cluster
#############################################
class SoftClusteringModule(nn.Module):
    """
    Replaces the old discrete cluster approach in 'fast_cluster'
    with a learned set of cluster centers. We do a single pass:
     1) Distances from each token to cluster centers
     2) Gumbel noise => soft assignment
     3) Weighted sums => cluster reps
     4) If # of cluster reps < block_size => pad
    """
    def __init__(self, max_clusters=8, embed_dim=32, cluster_temp=1.0):
        super().__init__()
        self.max_clusters = max_clusters
        self.embed_dim = embed_dim
        self.cluster_temp = cluster_temp
        self.cluster_centers = nn.Parameter(0.02 * torch.randn(self.max_clusters, self.embed_dim))

    def forward(self, token_embs):
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
# BucketTransformerAggregator
#############################################
class BucketTransformerAggregator(nn.Module):
    """
    Single layer aggregator for regression with 24 discrete buckets
    from 0..11. Transforms each (B, T, E) sequence into a single
    embedding using attention pooling, produces 24 logits, softmax them,
    then computes the weighted sum (expected value) across bucket centers.
    """
    def __init__(self, n_embed, n_head, num_layers=2, dropout=0.1, num_buckets=24, min_val=0.0, max_val=11.0):
        super().__init__()
        self.num_buckets = num_buckets
        self.bucket_centers = nn.Parameter(
            torch.linspace(min_val, max_val, steps=num_buckets), requires_grad=False
        )  # shape (24,), non-trainable
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embed,
            nhead=n_head,
            dim_feedforward=4 * n_embed,
            dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.logits_head = nn.Linear(n_embed, num_buckets)
        
        # Attention vector for pooling
        self.attn_vector = nn.Parameter(torch.randn(n_embed))

    def forward(self, x):
        """
        x: (B, T, E)
        Returns:
          output: (B,) expected value from bucket centers
          final embeddings: (B, T, E)
        """
        B, T, E = x.shape
        x_t = x.permute(1, 0, 2)  # (T, B, E)
        encoded = self.encoder(x_t)  # (T, B, E)
        x_out = encoded.permute(1, 0, 2)  # (B, T, E)

        # Attention pooling (instead of mean pooling)
        # Compute attention scores and weights across T
        attn_scores = torch.matmul(x_out, self.attn_vector)  # (B, T)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (x_out * attn_weights).sum(dim=1)  # (B, E)

        # Produce logits => shape (B, 24)
        logits = self.logits_head(pooled)

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)  # (B, 24)

        # Expected value => sum_{i=1..24} [ prob[i] * bucket_center[i] ]
        bucket_centers = self.bucket_centers.unsqueeze(0).to(pooled.device)  # (1, 24)
        output = (probs * bucket_centers).sum(dim=-1)  # (B,)

        return output, x_out

#############################################
# Updated SparseMoELanguageModel
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

        # Use the new 24-bucket aggregator
        self.final_aggregator = BucketTransformerAggregator(
            n_embed=n_embed,
            n_head=n_head,
            num_layers=2,
            dropout=dropout,
            num_buckets=24,    # you can tweak
            min_val=0.0,       # if your logged scale is 0..11
            max_val=11.0
        )

        self.ebm = EnergyBasedModel(n_embed)

        # TinyTransformer for guidance
        self.tiny_transformer = TinyTransformer(
            vocab_size, n_embed=32, context_window=config.CONTEXT_WINDOW, n_head=1
        )

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

    def preprocess_input(self, input_ids):
        device = input_ids.device
        B, T = input_ids.shape
        print(f"[preprocess_input] input_ids: {input_ids.shape}")
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        non_pad_mask = (input_ids != pad_id)
        true_lengths = non_pad_mask.sum(dim=1)
        print(f"[preprocess_input] True lengths: {true_lengths}")

        compressed_list = []
        for b in range(B):
            seq_len = true_lengths[b].item()
            token_embs = self.token_embedding_table(input_ids[b, :seq_len])
            if seq_len <= self.block_size:
                if seq_len < self.block_size:
                    pad_amount = self.block_size - seq_len
                    pad_emb = torch.zeros(pad_amount, token_embs.size(-1), device=device)
                    token_embs = torch.cat([token_embs, pad_emb], dim=0)
                comp_emb = token_embs
            else:
                comp_emb = self.fast_cluster(token_embs)
            compressed_list.append(comp_emb.unsqueeze(0))
        compressed_embeddings = torch.cat(compressed_list, dim=0)
        print(f"[preprocess_input] Compressed shape: {compressed_embeddings.shape}")
        return compressed_embeddings

    def forward(self, input_ids, targets=None,
                use_entropy_reg=False, lambda_entropy=1e-2, percent_complete=0.0):
        device = input_ids.device
        B, T = input_ids.shape
        print(f"[SMELM] input_ids: {input_ids.shape}")

        # 1) TinyTransformer for guidance
        tiny_loss = torch.tensor(0.0, device=device)
        if T > self.block_size and targets is not None:
            print("[SMELM] Running TinyTransformer for loss")
            tiny_output, _ = self.tiny_transformer(input_ids)
            targets_squeezed = targets.squeeze(1) if (targets.dim() == 2 and targets.size(1) == 1) else targets
            tiny_loss = F.mse_loss(tiny_output.squeeze(-1), targets_squeezed)
            print(f"[SMELM] Tiny loss: {tiny_loss.item()}")

        # 2) compress input
        comp_emb = self.preprocess_input(input_ids)
        # 3) add positions
        pos_emb = self.position_embedding_table(torch.arange(comp_emb.size(1), device=device))
        x = comp_emb + pos_emb

        # 4) pass through blocks
        ent_loss_sum = torch.tensor(0.0, device=device)
        for idx, block in enumerate(self.blocks):
            if self.training:
                x, ent_loss = checkpoint(block, x, use_reentrant=False)
            else:
                x, ent_loss = block(x)
            if ent_loss is not None:
                ent_loss_sum += ent_loss
            print(f"[SMELM] Block {idx} output: {x.shape}")

        # 5) final LN
        x = self.ln_f(x)

        # 6) aggregator => produces distribution over 24 buckets => expected value
        output, final_embeddings = self.final_aggregator(x)

        # 7) EBM
        ebm_input = final_embeddings.mean(dim=1)
        ebm_energy = self.ebm(ebm_input)

        # 8) compute losses
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
            "tiny_loss": tiny_loss,
            "ebm_loss": ebm_loss,
            "entropy_loss": entropy_loss
        }

    def get_embeddings(self, input_ids, return_tokenwise=False):
        device = input_ids.device
        comp_emb = self.preprocess_input(input_ids)
        pos_emb = self.position_embedding_table(torch.arange(comp_emb.size(1), device=device))
        x = comp_emb + pos_emb
        for idx, block in enumerate(self.blocks):
            x, _ = block(x)
        x = self.ln_f(x)
        _, final_embeddings = self.final_aggregator(x)
        if return_tokenwise:
            return final_embeddings
        else:
            return final_embeddings.mean(dim=1)
