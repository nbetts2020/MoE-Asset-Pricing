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

cuda_available = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiHeadAttention(nn.Module):
    """ Multi-head self-attention using FlashAttention """
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        assert n_embed % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()  # x: (B, T, n_embed)
        qkv = self.qkv_proj(x)  # (B, T, 3 * n_embed)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_size)  # (B, T, 3, n_head, head_size)
        qkv = qkv.permute(0, 2, 1, 3, 4)  # (B, 3, T, n_head, head_size)
        qkv = qkv.reshape(B * T, 3, self.n_head, self.head_size)  # (B*T, 3, n_head, head_size)
        cu_seqlens = torch.arange(0, (B + 1) * T, step=T, dtype=torch.int32, device=x.device)
        max_seqlen = T
        attn_output = flash_attn_unpadded_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=config.DROPOUT,
            softmax_scale=None,
            causal=True,
            return_attn_probs=False,
        )  # (B*T, n_head, head_size)
        attn_output = attn_output.view(B, T, self.n_head, self.head_size)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, C, T).permute(0, 2, 1)
        out = self.out_proj(attn_output)
        return out


class Expert(nn.Module):
    """ An MLP expert: a linear layer followed by a non-linearity and dropout """
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
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)  # (B, T, num_experts)
        noise_logits = self.noise_linear(mh_output)  # (B, T, num_experts)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        # Compute probabilities for all experts before top_k
        full_router_probs = F.softmax(noisy_logits, dim=-1)  # (B, T, num_experts)

        # Select top_k experts
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)  # (B, T, num_experts)

        return router_output, indices, full_router_probs


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        router_output, indices, full_router_probs = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_router_output = router_output.view(-1, router_output.size(-1))
        tokens_per_batch = batch_size * seq_len * self.top_k
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_router_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)

        final_output += updates.view(batch_size, seq_len, -1)

        # Compute entropy loss
        entropy_loss = None
        if self.training:
            entropy = -torch.sum(full_router_probs * torch.log(full_router_probs + 1e-8), dim=-1)  # (B, T)
            entropy_loss = entropy.mean()  # Scalar

        return final_output, entropy_loss


class Block(nn.Module):
    """ Transformer block with FlashAttention and SparseMoE """
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.sa = MultiHeadAttention(n_embed, n_head)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)

    def forward(self, x, attention_mask=None):
        x = x + self.sa(self.ln1(x))
        moe_output, entropy_loss = self.smoe(self.ln2(x))
        x = x + moe_output
        return x, entropy_loss


class DynamicPooling(nn.Module):
    """
    Strictly adjacent, fully dynamic pooling that compresses a sequence (B, T, n_embed)
    into (B, target_length, n_embed). For each input sequence, it computes per-token
    importance scores, then uses the normalized cumulative sum of these scores to partition
    the sequence into target_length contiguous segments. Within each segment, a soft attention-
    based weighted sum (using a sharp sigmoid) is used to pool tokens.
    """
    def __init__(self, target_length, n_embed, temperature=50.0):
        super().__init__()
        self.target_length = target_length
        self.score_proj = nn.Linear(n_embed, 1)  # Computes per-token importance score.
        self.temperature = temperature  # High temperature for nearly binary segmentation.
        # Optional: a small linear layer for additional attention within each segment.
        self.attn_proj = nn.Linear(n_embed, 1)

    def forward(self, x):
        # x: (B, T, n_embed)
        B, T, C = x.shape
        # Compute per-token importance scores.
        scores = torch.sigmoid(self.score_proj(x)).squeeze(-1)  # (B, T)
        # Compute cumulative sum of scores (monotonic increasing) per example.
        cumsum = torch.cumsum(scores, dim=1)  # (B, T)
        total = cumsum[:, -1:]  # (B, 1)
        norm_cumsum = cumsum / (total + 1e-8)  # (B, T) normalized to [0,1]

        pooled_list = []
        # For each target segment, define lower and upper bounds.
        for i in range(self.target_length):
            lower = i / self.target_length
            upper = (i + 1) / self.target_length
            # Compute soft membership weights for each token:
            # The membership is nearly 1 for tokens whose normalized cumsum is within [lower, upper],
            # and nearly 0 otherwise.
            membership = torch.sigmoid(self.temperature * (norm_cumsum - lower)) - torch.sigmoid(self.temperature * (norm_cumsum - upper))  # (B, T)
            # Normalize membership weights per example.
            membership = membership / (membership.sum(dim=1, keepdim=True) + 1e-8)  # (B, T)
            # Compute a weighted sum of tokens in this segment.
            pooled_token = torch.bmm(membership.unsqueeze(1), x).squeeze(1)  # (B, n_embed)
            pooled_list.append(pooled_token.unsqueeze(1))  # (B, 1, n_embed)
        # Concatenate pooled tokens to form the output sequence.
        pooled = torch.cat(pooled_list, dim=1)  # (B, target_length, n_embed)
        return pooled


class FinalAttentionPooling(nn.Module):
    """
    Attention-based pooling that computes a weighted sum of token embeddings.
    Instead of a simple mean, a learnable projection produces a weight per token,
    and these weights are used to form the final pooled embedding.
    """
    def __init__(self, n_embed):
        super().__init__()
        self.proj = nn.Linear(n_embed, 1)

    def forward(self, x):
        # x: (B, T, n_embed)
        weights = self.proj(x).squeeze(-1)  # (B, T)
        weights = F.softmax(weights, dim=1)  # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, n_embed)
        return pooled


class SparseMoELanguageModel(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, dropout, num_experts, top_k,
                 tokenizer_name="hf-internal-testing/llama-tokenizer"):
        super(SparseMoELanguageModel, self).__init__()
        self.block_size = block_size  # Target sequence length after pooling.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name, model_max_length=4096)
        vocab_size = self.tokenizer.vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # Dynamic pooling layer: compress input to fixed block_size using strict adjacent segmentation.
        self.dynamic_pool = DynamicPooling(block_size, n_embed)
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head=n_head, num_experts=num_experts, top_k=top_k) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        # Final attention-based pooling instead of mean pooling.
        self.final_pool = FinalAttentionPooling(n_embed)
        # Regression head for scalar prediction.
        self.regression_head = nn.Linear(n_embed, 1)

    def forward(self, input_ids, targets=None, use_entropy_reg=False, lambda_entropy=0.01):
        B, T = input_ids.shape
        tok_emb = self.token_embedding_table(input_ids)  # (B, T, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=input_ids.device))  # (T, n_embed)
        x = tok_emb + pos_emb  # (B, T, n_embed)

        # If input length exceeds block_size, compress it using dynamic pooling.
        if T > self.block_size:
            x = self.dynamic_pool(x)  # (B, block_size, n_embed)
            T = self.block_size

        total_entropy_loss = 0.0
        # Process through transformer blocks with gradient checkpointing during training.
        for block in self.blocks:
            if self.training:
                x, entropy_loss = checkpoint(block, x, use_reentrant=False)
            else:
                x, entropy_loss = block(x)
            if use_entropy_reg and entropy_loss is not None:
                total_entropy_loss += entropy_loss

        x = self.ln_f(x)  # (B, T, n_embed)
        # Use attention-based pooling instead of mean pooling.
        x = self.final_pool(x)  # (B, n_embed)
        output = self.regression_head(x)  # (B, 1)
        output = output.squeeze(-1)       # (B,)

        loss = None
        if targets is not None:
            if targets.dim() == 2 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            task_loss = F.mse_loss(output, targets)
            loss = task_loss
            if use_entropy_reg:
                loss += lambda_entropy * total_entropy_loss / len(self.blocks)
        return output, loss

    def get_embeddings(self, input_ids, pool=True):
        B, T = input_ids.shape
        tok_emb = self.token_embedding_table(input_ids)  # (B, T, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=input_ids.device))  # (T, n_embed)
        x = tok_emb + pos_emb  # (B, T, n_embed)
        # Compress via dynamic pooling if needed.
        if T > self.block_size:
            x = self.dynamic_pool(x)
        x = self.ln_f(x)
        if pool:
            x = self.final_pool(x)  # Use attention-based pooling.
        return x

    # The generate method is omitted since the model is designed for regression.
    # def generate(self, idx, max_new_tokens, stream=False, temperature=1.0):
    #     for _ in range(max_new_tokens):
    #         idx_cond = idx[:, -self.block_size:]
    #         output, _ = self(idx_cond)
    #     return idx
