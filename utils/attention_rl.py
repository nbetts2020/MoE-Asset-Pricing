# attention_rl.py

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaTokenizerFast
from torch.utils.data import Dataset, DataLoader
import logging
import random

from utils.utils import load_rl_data  # our RL data loader from chunked files
from utils.config import config

# ------------------------------
# Helper: Split formatted text into sections
# ------------------------------
def split_into_sections(formatted_text: str) -> list:
    """
    Splits the formatted text (from format_concatenated_articles) into sections.
    Any line that appears to be a header (e.g., ending with "Information:" or starting with "Last")
    is removed, and the remaining lines are grouped into sections.
    """
    lines = formatted_text.split("\n")
    sections = []
    current_section = []
    for line in lines:
        stripped = line.strip()
        # Treat lines ending with "Information:" or starting with "Last" as headers.
        if stripped.endswith("Information:") or stripped.startswith("Last"):
            if current_section:
                sections.append("\n".join(current_section).strip())
                current_section = []
            continue
        if stripped:
            current_section.append(stripped)
    if current_section:
        sections.append("\n".join(current_section).strip())
    return sections

# ------------------------------
# Learned Downsampler Module
# ------------------------------
class LearnedDownSampler(nn.Module):
    def __init__(self, num_queries: int, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        """
        Uses a fixed set of learnable query tokens to pool a variable-length sequence into a fixed-size output.
        """
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        # Learnable query tokens: (num_queries, embed_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        # Multi-head cross-attention: queries attend over the input tokens.
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
    
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape (B, T, embed_dim) – output of global attention.
            key_padding_mask: Optional mask for padded tokens.
        Returns:
            out: Downsampled tensor of shape (B, num_queries, embed_dim)
            log_prob: A scalar log probability computed from attention weights.
        """
        B = x.size(0)
        # Expand learnable queries to batch dimension: (B, num_queries, embed_dim)
        queries = self.queries.unsqueeze(0).expand(B, self.num_queries, self.embed_dim)
        out, attn_weights = self.cross_attn(queries, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        out = self.ln(out)
        # Compute a dummy log probability as the mean log probability over all queries.
        log_prob = torch.mean(torch.log(attn_weights + 1e-8))
        return out, log_prob

# ------------------------------
# Hierarchical Attention RL Module
# ------------------------------
class HierarchicalAttentionRL(nn.Module):
    def __init__(self, tokenizer_name: str, embed_dim: int, max_length: int, num_queries: int = 4096, truncate_limit: int = None):
        """
        Args:
            tokenizer_name: Pretrained tokenizer name.
            embed_dim: Embedding dimension.
            max_length: Maximum tokens per section.
            num_queries: Number of learnable query tokens for downsampling.
            truncate_limit: If set, if the concatenated token sequence length exceeds this value,
                            it will be truncated (e.g., to the last truncate_limit tokens).
        """
        super(HierarchicalAttentionRL, self).__init__()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name, model_max_length=4096)
        self.tokenizer.model_max_length = max_length  # local max per section
        self.vocab_size = self.tokenizer.vocab_size
        # Learnable embedding layer for token IDs; used only if model is not provided.
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.truncate_limit = truncate_limit  # e.g., 64000; if None, no truncation
        
        # Learned downsampler to compress the full sequence to a fixed number of tokens.
        self.downsampler = LearnedDownSampler(num_queries=num_queries, embed_dim=embed_dim)
        
        # Regression head to predict target from compressed representation.
        # We pool by averaging the downsampled tokens.
        self.reg_head = nn.Linear(embed_dim, 1)
    
    def forward(self, formatted_text: str, model: nn.Module = None):
        """
        Args:
            formatted_text: A single concatenated article (string).
            model: (Optional) The main transformer model with a get_embeddings() method.
                   If provided, its embeddings will be used for each token.
        Returns:
            compressed: Fixed-size representation from learned downsampling (1, num_queries, embed_dim)
            pred: Regression output (scalar tensor) – used during RL training.
            log_prob: Log probability from the downsampler (scalar tensor)
        """
        device = next(self.parameters()).device
        # 1. Split text into sections.
        sections = split_into_sections(formatted_text)
        if len(sections) == 0:
            dummy = torch.zeros(1, self.embed_dim, device=device)
            return dummy, self.reg_head(dummy).squeeze(0), torch.tensor(0.0, device=device)
        
        # 2. For each section, tokenize and obtain token embeddings.
        section_embeds = []
        for sec in sections:
            tokens = self.tokenizer(
                sec,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].to(device)  # (1, max_length)
            if model is not None and hasattr(model, "get_embeddings"):
                embeds = model.get_embeddings(input_ids, pool=False)
            else:
                embeds = self.embedding(input_ids)
            section_embeds.append(embeds)
        
        # 3. Concatenate token embeddings along the sequence dimension.
        full_sequence = torch.cat(section_embeds, dim=1)  # (1, num_sections * max_length, embed_dim)
        
        # 4. Optional: Truncate if sequence length exceeds truncate_limit.
        if self.truncate_limit is not None and full_sequence.size(1) > self.truncate_limit:
            full_sequence = full_sequence[:, -self.truncate_limit:, :]
        
        # 5. Global Attention with Learned Downsampler:
        downsampled, log_prob = self.downsampler(full_sequence)
        # downsampled shape: (1, num_queries, embed_dim)
        
        # 6. Pool downsampled tokens (e.g., average) to get a single vector.
        pooled = downsampled.mean(dim=1)  # (1, embed_dim)
        pred = self.reg_head(pooled).squeeze(0)  # scalar prediction
        
        return downsampled, pred, log_prob

# ------------------------------
# RL Dataset for Hierarchical Attention
# ------------------------------
class RLTextDataset(Dataset):
    def __init__(self, df, text_field="formatted_text"):
        self.df = df
        self.text_field = text_field

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        text = sample.get(self.text_field, "")
        target = sample.get("weighted_avg_720_hrs", 0.0)
        return {"text": text, "target": target}

# ------------------------------
# Simple Collate Function (for RL DataLoader)
# ------------------------------
def rl_collate_fn(batch):
    return batch  # Process one sample at a time in the forward pass

# ------------------------------
# RL Training Function
# ------------------------------
def rl_train_hAttention(args, model: nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instead of a single DataFrame, iterate over chunks.
    chunk_iterator = load_rl_data(args, rl_rows=args.rl_rows)
    
    # Initialize our Hierarchical Attention RL module.
    hAttention = HierarchicalAttentionRL(
        tokenizer_name=args.tokenizer_name,
        embed_dim=config.N_EMBED,
        max_length=config.BLOCK_SIZE,
        num_queries=4096,
        truncate_limit=None  # e.g., 64000 if desired.
    ).to(device)

    optimizer = torch.optim.Adam(hAttention.parameters(), lr=args.rl_learning_rate)
    hAttention.train()

    for epoch in range(1, args.rl_epochs + 1):
        logging.info(f"RL Epoch {epoch} starting.")
        epoch_loss = 0.0
        epoch_reward = 0.0
        num_batches = 0

        # Iterate over chunks one at a time.
        for df_chunk in chunk_iterator:
            # Create RL dataset and DataLoader from this chunk.
            rl_dataset = RLTextDataset(df_chunk, text_field="formatted_text")
            rl_loader = DataLoader(
                rl_dataset,
                batch_size=args.rl_batch_size,
                shuffle=True,
                collate_fn=rl_collate_fn,
                num_workers=0
            )

            for batch in rl_loader:
                batch_loss = 0.0
                batch_log_probs = []
                preds = []
                targets = []
                for sample in batch:
                    text = sample["text"]
                    target = torch.tensor(sample["target"], dtype=torch.float32, device=device)
                    _, pred, log_prob = hAttention(text, model=model)
                    preds.append(pred)
                    targets.append(target)
                    batch_log_probs.append(log_prob)
                preds_tensor = torch.stack(preds)
                targets_tensor = torch.stack(targets)
                mse_loss = F.mse_loss(preds_tensor, targets_tensor)
                reward = -mse_loss.detach()  # lower MSE gives higher reward
                avg_log_prob = torch.stack(batch_log_probs).mean()
                rl_loss = -avg_log_prob * reward
                loss = mse_loss + rl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_reward += reward.item()
                num_batches += 1

            # Clean up after each chunk.
            del rl_dataset, rl_loader, df_chunk
            gc.collect()

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_reward = epoch_reward / num_batches if num_batches > 0 else 0.0
        logging.info(f"RL Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")

        # Reset the chunk iterator for the next epoch.
        chunk_iterator = load_rl_data(args, rl_rows=args.rl_rows)
    
    from utils.utils import save_rl_attention
    save_rl_attention(hAttention, epoch=config.EPOCHS, save_dir=args.save_dir, args=args)
    logging.info("RL training complete. RL module saved.")
