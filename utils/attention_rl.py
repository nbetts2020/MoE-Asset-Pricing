import os
import gc
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from transformers import LlamaTokenizerFast
from torch.cuda.amp import autocast, GradScaler

from utils.config import config
from utils.utils import save_rl_attention

CHUNK_SIZES = {
    1: 50000,
    2: 25000,
    3: 25000,
    4: 25000,
    5: 25000,
    6: 25000,
    7: 25000,
    8: 25000,
    9: 25000,
    10: 25000,
    11: 25000,
    12: 25000,
    13: 13146
}
TOTAL_CHUNK_ROWS = sum(CHUNK_SIZES.values())

def print_gpu_memory_usage(tag=""):
    """Helper to log current GPU memory usage (allocated & reserved)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logging.info(f"{tag} GPU Memory: allocated={allocated:.2f} MB, reserved={reserved:.2f} MB")

def get_data_rl(chunk_idx: int) -> pd.DataFrame:
    """Loads a single Parquet chunk for RL from train_dataset_{chunk_idx}a.parquet."""
    repo_id = "nbettencourt/sc454k-preprocessed-dfs"
    filename = f"train_dataset_{chunk_idx}a.parquet"
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    df = pd.read_parquet(file_path)
    df = df.dropna(subset=["weighted_avg_720_hrs"])
    df = df[df["weighted_avg_720_hrs"] > 0]
    os.remove(file_path)
    gc.collect()
    return df

class RLTextDataset(Dataset):
    def __init__(self, df, text_field="formatted_text"):
        self.df = df
        self.text_field = text_field

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row.get(self.text_field, "")
        target = row.get("weighted_avg_720_hrs", 0.0)
        return {"text": text, "target": target}

def rl_collate_fn(batch):
    return batch

def split_into_sections(formatted_text: str):
    lines = formatted_text.split("\n")
    sections = []
    current_section = []
    for line in lines:
        stripped = line.strip()
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

class LearnedDownSampler(nn.Module):
    def __init__(self, num_queries: int, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)
        queries = self.queries.unsqueeze(0).expand(B, self.num_queries, self.embed_dim)
        out, attn_weights = self.cross_attn(queries, x, x, need_weights=True)
        out = self.ln(out)
        log_prob = torch.mean(torch.log(attn_weights + 1e-8))
        return out, log_prob

class HierarchicalAttentionRL(nn.Module):
    def __init__(self, tokenizer_name: str, embed_dim: int, max_length: int,
                 num_queries: int = 4096, truncate_limit: int = None):
        super().__init__()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name, model_max_length=4096)
        self.tokenizer.model_max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.truncate_limit = truncate_limit

        self.downsampler = LearnedDownSampler(num_queries=num_queries, embed_dim=embed_dim)
        self.reg_head = nn.Linear(embed_dim, 1)

    def forward(self, formatted_text: str):
        device = next(self.parameters()).device
        sections = split_into_sections(formatted_text)
        if len(sections) == 0:
            dummy = torch.zeros(1, self.embed_dim, device=device)
            return dummy, self.reg_head(dummy).squeeze(0), torch.tensor(0.0, device=device)

        section_embeds = []
        for sec in sections:
            tokens = self.tokenizer(sec, truncation=True, padding="max_length",
                                    max_length=self.max_length, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)
            embeds = self.embedding(input_ids)
            section_embeds.append(embeds)

        full_sequence = torch.cat(section_embeds, dim=1)
        if self.truncate_limit is not None and full_sequence.size(1) > self.truncate_limit:
            full_sequence = full_sequence[:, -self.truncate_limit:, :]
        downsampled, log_prob = self.downsampler(full_sequence)
        pooled = downsampled.mean(dim=1)
        pred = self.reg_head(pooled).squeeze(0)
        return downsampled, pred, log_prob

def rl_train_hAttention(args, model: nn.Module):
    """
    RL training loop: iterates over chunk indices one by one,
    loads each chunk from disk, creates a DataLoader, trains on that chunk,
    cleans up memory after every batch, and repeats for each rl_epoch.
    Uses mixed precision for memory savings.
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    from utils.utils import save_rl_attention  # Ensure this is defined elsewhere
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the RL module and optimizer
    hAttention = HierarchicalAttentionRL(
        tokenizer_name=args.tokenizer_name,
        embed_dim=config.N_EMBED,
        max_length=config.BLOCK_SIZE,
        num_queries=4096,
        truncate_limit=None
    ).to(device)
    optimizer = torch.optim.Adam(hAttention.parameters(), lr=args.rl_learning_rate)
    scaler = GradScaler()
    hAttention.train()

    for epoch in range(1, args.rl_epochs + 1):
        logging.info(f"RL Epoch {epoch} starting.")
        epoch_loss = 0.0
        epoch_reward = 0.0
        num_batches = 0

        for chunk_idx in range(1, 14):  # Process chunks 1 to 13
            print_gpu_memory_usage(tag=f"Before loading chunk {chunk_idx}")
            if chunk_idx not in CHUNK_SIZES:
                continue

            df_chunk = get_data_rl(chunk_idx)
            print_gpu_memory_usage(tag=f"After loading chunk {chunk_idx}")
            if df_chunk.empty:
                logging.info(f"Chunk {chunk_idx} returned empty DataFrame.")
                continue

            if args.rl_rows == -1:
                chunk_df = df_chunk
            else:
                chunk_needed = int(round((CHUNK_SIZES[chunk_idx] / TOTAL_CHUNK_ROWS) * args.rl_rows))
                if chunk_needed <= 0:
                    continue
                if chunk_needed < len(df_chunk):
                    chunk_df = df_chunk.sample(n=chunk_needed, random_state=random.randint(0, 999999))
                else:
                    chunk_df = df_chunk

            print_gpu_memory_usage(tag=f"Before DataLoader chunk {chunk_idx}")
            rl_dataset = RLTextDataset(chunk_df, text_field="formatted_text")
            rl_loader = DataLoader(
                rl_dataset,
                batch_size=args.rl_batch_size,
                shuffle=True,
                collate_fn=rl_collate_fn,
                num_workers=0
            )
            print_gpu_memory_usage(tag=f"After DataLoader chunk {chunk_idx}")

            for batch in rl_loader:
                preds = []
                targets = []
                batch_log_probs = []
                with autocast(dtype=torch.float16):
                    for sample in batch:
                        text = sample["text"]
                        target = torch.tensor(sample["target"], dtype=torch.float32, device=device)
                        _, pred, log_prob = hAttention(text)
                        preds.append(pred)
                        targets.append(target)
                        batch_log_probs.append(log_prob)
                    preds_tensor = torch.stack(preds).squeeze(-1)
                    targets_tensor = torch.stack(targets)
                    mse_loss = F.mse_loss(preds_tensor, targets_tensor)
                    reward = -mse_loss.detach()  # lower MSE => higher reward
                    avg_log_prob = torch.stack(batch_log_probs).mean()
                    rl_loss = -avg_log_prob * reward
                    loss = mse_loss + rl_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                epoch_reward += reward.item()
                num_batches += 1

                # Clean up after each batch
                del preds, targets, batch_log_probs, preds_tensor, targets_tensor, loss
                gc.collect()
                torch.cuda.empty_cache()

            # Clean up after processing the chunk
            del df_chunk, chunk_df, rl_dataset, rl_loader
            gc.collect()
            torch.cuda.empty_cache()
            print_gpu_memory_usage(tag=f"After cleanup chunk {chunk_idx}")

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_reward = epoch_reward / num_batches if num_batches > 0 else 0.0
        logging.info(f"RL Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")

    save_rl_attention(hAttention, epoch=config.EPOCHS, save_dir=args.save_dir, args=args)
    logging.info("RL training complete. RL module saved.")
