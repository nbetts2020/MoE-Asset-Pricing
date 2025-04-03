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
        # Always request full attention outputs for consistent shape.
        outputs = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen,
            dropout_p=config.DROPOUT, softmax_scale=None, causal=False,
            return_attn_probs=True
        )
        attn_output, attn_probs, full_attn = outputs
        attn_output = attn_output.view(B, T, self.n_head, self.head_size).permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.out_proj(attn_output)
        return (out, attn_probs, full_attn) if return_attn_probs else out

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
        self.top_k = top_k; self.capacity_factor = capacity_factor; self.num_experts = num_experts
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
            # Print clustering summary per expert
            print(f"[SparseMoE] Expert {i} selected {limited_indices.numel()} tokens")
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

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embed=32, context_window=16384, n_head=1):
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
        attn_received = full_attn.sum(dim=2)
        attn_scores = attn_received.mean(dim=1)
        print(f"[TinyTransformer] attn_scores: {attn_scores.shape}")
        return output, attn_scores

class SparseMoELanguageModel(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, dropout, num_experts, top_k,
                 tokenizer_name="hf-internal-testing/llama-tokenizer"):
        super().__init__()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name, model_max_length=16384)
        vocab_size = self.tokenizer.vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, num_experts, top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.regression_head = nn.Linear(n_embed, 1)
        self.block_size = block_size
        self.tiny_transformer = TinyTransformer(vocab_size, n_embed=32, context_window=16384, n_head=1)
        self.ebm = EnergyBasedModel(n_embed)
    @staticmethod
    def fast_cluster(attn_scores, max_cluster_size=4096, min_clusters=4, max_clusters=8, device='cuda'):
        print(f"[fast_cluster] attn_scores: {attn_scores.shape}")
        B, T = attn_scores.shape
        labels = torch.zeros_like(attn_scores, dtype=torch.long)
        for b in range(B):
            seq_scores = attn_scores[b]
            base_clusters = (T + max_cluster_size - 1) // max_cluster_size
            num_clusters = max(min_clusters, min(max_clusters, max(base_clusters, int(T / max_cluster_size) + 1)))
            print(f"[fast_cluster] Batch {b}: num_clusters = {num_clusters}")
            cluster_size = T // num_clusters
            remainder = T % num_clusters
            boundaries = []
            start = 0
            for i in range(num_clusters):
                size = cluster_size + (1 if i < remainder else 0)
                end = min(start + min(size, max_cluster_size), T)
                boundaries.append(end)
                start = end
                if start >= T: break
            # Refine boundaries (concise print)
            refined = [0]
            current = 0
            while current < T:
                window_end = min(current + max_cluster_size, T)
                if window_end - current > 1:
                    window_scores = seq_scores[current:window_end]
                    split_idx = torch.argmin(window_scores[1:]) + 1 + current
                    refined.append(split_idx.item() if (split_idx-current<=max_cluster_size and window_end-split_idx<=max_cluster_size) else window_end)
                else:
                    refined.append(window_end)
                current = refined[-1]
                if len(refined) >= max_clusters and current < T:
                    remaining = T - current
                    step = min(max_cluster_size, remaining // (max_clusters - len(refined) + 1) + 1)
                    while current < T:
                        next_boundary = min(current + step, T)
                        refined.append(next_boundary)
                        current = next_boundary
            for i in range(len(refined) - 1):
                labels[b, refined[i]:refined[i + 1]] = i
        print(f"[fast_cluster] labels: {labels.shape}")
        return labels
    def preprocess_input(self, input_ids):
        print(f"[preprocess_input] input_ids: {input_ids.shape}")
        B, T = input_ids.shape; device = input_ids.device
        if T <= self.block_size:
            print("[preprocess_input] Padding short seq")
            if T < self.block_size:
                pad_ids = torch.full((B, self.block_size - T), self.tokenizer.pad_token_id, device=device)
                return torch.cat([input_ids, pad_ids], dim=1)
            return input_ids
        print("[preprocess_input] Running TinyTransformer for attn_scores")
        _, attn_scores = self.tiny_transformer(input_ids)
        print(f"[preprocess_input] attn_scores: {attn_scores.shape}")
        labels = self.fast_cluster(attn_scores, max_cluster_size=4096, min_clusters=4, max_clusters=8, device=device)
        unique_labels = torch.unique(labels)
        print(f"[preprocess_input] Unique labels: {unique_labels}")
        cluster_sizes = [torch.sum(labels == label, dim=1).max().item() for label in unique_labels]
        cluster_means = torch.stack([attn_scores[labels == label].mean() for label in unique_labels])
        print(f"[preprocess_input] Cluster sizes: {cluster_sizes}")
        print(f"[preprocess_input] Cluster means: {cluster_means}")
        ratios = F.softmax(cluster_means, dim=0)
        target_sizes = (ratios * self.block_size).long()
        target_sizes[-1] = self.block_size - target_sizes[:-1].sum()
        print(f"[preprocess_input] Target sizes: {target_sizes}")
        selected_tokens = []
        current_pos = 0
        for i, (label, tsize) in enumerate(zip(unique_labels, target_sizes)):
            cluster_mask = (labels == label)
            cluster_indices = torch.where(cluster_mask)[1]
            num_to_select = min(tsize.item(), cluster_sizes[i])
            selected_tokens.extend(cluster_indices[:num_to_select].tolist())
            current_pos += num_to_select
            print(f"[preprocess_input] Label {label.item()} selected {num_to_select} tokens")
            if current_pos >= self.block_size: break
        selected_tokens = sorted(selected_tokens[:self.block_size])
        print(f"[preprocess_input] Final tokens (first 10): {selected_tokens[:10]}...")
        compressed = input_ids[:, selected_tokens]
        if len(selected_tokens) < self.block_size:
            pad_len = self.block_size - len(selected_tokens)
            pad_ids = torch.full((B, pad_len), self.tokenizer.pad_token_id, device=device)
            compressed = torch.cat([compressed, pad_ids], dim=1)
        print(f"[preprocess_input] Compressed shape: {compressed.shape}")
        return compressed
    def forward(self, input_ids, targets=None, use_entropy_reg=False, lambda_entropy=0.01, percent_complete=0.0):
        print(f"[SMELM] input_ids: {input_ids.shape}")
        B, T = input_ids.shape; device = input_ids.device; tiny_loss = 0.0
        if T > self.block_size:
            print("[SMELM] Running TinyTransformer for loss")
            tiny_output, _ = self.tiny_transformer(input_ids)
            if targets is not None:
                targets = targets.squeeze(1) if (targets.dim()==2 and targets.size(1)==1) else targets
                tiny_loss = F.mse_loss(tiny_output.squeeze(-1), targets)
            print(f"[SMELM] Tiny loss: {tiny_loss.item()}")
        comp_ids = self.preprocess_input(input_ids)
        tok_emb = self.token_embedding_table(comp_ids)
        pos_emb = self.position_embedding_table(torch.arange(comp_ids.size(1), device=device))
        x = tok_emb + pos_emb
        for idx, block in enumerate(self.blocks):
            x, ent_loss = checkpoint(block, x, use_reentrant=False) if self.training else block(x)
            print(f"[SMELM] Block {idx} output: {x.shape}")
        embeddings = self.ln_f(x).mean(dim=1)
        ebm_energy = self.ebm(embeddings)
        output = self.regression_head(embeddings).squeeze(-1)
        if targets is not None:
            targets = targets.squeeze(1) if (targets.dim()==2 and targets.size(1)==1) else targets
            task_loss = F.mse_loss(output, targets, reduction='none').mean()
            ebm_loss = F.mse_loss(ebm_energy, F.mse_loss(output, targets, reduction='none').detach())
            loss = task_loss + tiny_loss + config.LAMBDA_EBM * percent_complete * ebm_loss
            if use_entropy_reg: loss += lambda_entropy * ent_loss / len(self.blocks)
            print(f"[SMELM] Loss: {loss.item()}")
        else:
            loss = None
            print("[SMELM] No loss computed")
        return output, ebm_energy, loss
    def get_embeddings(self, input_ids, pool=True):
        comp_ids = self.preprocess_input(input_ids)
        tok_emb = self.token_embedding_table(comp_ids)
        pos_emb = self.position_embedding_table(torch.arange(comp_ids.size(1), device=input_ids.device))
        x = tok_emb + pos_emb
        for idx, block in enumerate(self.blocks):
            x, _ = block(x)
        x = self.ln_f(x)
        if pool:
            x = x.mean(dim=1)
        print(f"[get_embeddings] embeddings: {x.shape}")
        return x
