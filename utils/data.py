import os
import torch
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaTokenizerFast
from utils.config import config

# Disable tokenizer parallelism to avoid warnings after fork.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# GLOBAL TOKENIZER SETUP
# -------------------------------------------------------------------------
TOKENIZER_NAME = getattr(config, "TOKENIZER_NAME", "hf-internal-testing/llama-tokenizer")
GLOBAL_TOKENIZER = LlamaTokenizerFast.from_pretrained(TOKENIZER_NAME, model_max_length=4096)
GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token
GLOBAL_TOKENIZER.model_max_length = config.BLOCK_SIZE

# -------------------------------------------------------------------------
# PRECOMPUTED DATASET CLASS
# -------------------------------------------------------------------------
class PrecomputedDataset(Dataset):
    """
    Dataset that assumes text has been precomputed and stored in a DataFrame.
    It tokenizes the text on demand and retrieves the regression target.
    """

    def __init__(self, df, tokenizer, block_size):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx].get("text", "")
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        label = self.df.iloc[idx].get("weighted_avg_720_hrs", 0.0)  # Defaulting to 0.0 if missing
        return {"input_ids": input_ids, "label": torch.tensor(label, dtype=torch.float)}

# -------------------------------------------------------------------------
# CUSTOM COLLATE FUNCTION FOR PRECOMPUTED DATASET
# -------------------------------------------------------------------------
def custom_collate_fn(batch):
    """
    Batches input_ids and labels into tensors.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}

# -------------------------------------------------------------------------
# RUN DATASET CLASS
# -------------------------------------------------------------------------
class RunDataset(Dataset):
    """
    Dataset for run mode that processes a DataFrame with multiple candidate texts.
    It concatenates iteration_1_text to iteration_30_text with a separator.
    """
    def __init__(self, df, tokenizer, separator="<SPLIT_HERE>"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.separator = separator

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Concatenate candidate texts from iteration_1_text to iteration_30_text.
        candidates = [row.get(f"iteration_{i}_text") for i in range(1, 31) if row.get(f"iteration_{i}_text") is not None]
        concatenated_text = self.separator.join(candidates)
        encoding = self.tokenizer(
            concatenated_text,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        label = row.get("weighted_avg_720_hrs", 0.0)
        weighted_avg_0_hrs = row.get("weighted_avg_0_hrs", 0.0)
        risk_free_rate = row.get("Risk_Free_Rate", 0.0)
        sector = row.get("Sector", "Unknown")
        return {
            "input_ids": input_ids,
            "label": torch.tensor(label, dtype=torch.float),
            "weighted_avg_0_hrs": torch.tensor(weighted_avg_0_hrs, dtype=torch.float),
            "Risk_Free_Rate": torch.tensor(risk_free_rate, dtype=torch.float),
            "Sector": sector,
            "text": concatenated_text
        }

# -------------------------------------------------------------------------
# CUSTOM COLLATE FUNCTION FOR RUN DATASET
# -------------------------------------------------------------------------
def run_collate_fn(batch):
    """
    Batches the run dataset, including input_ids, labels, and additional fields.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    weighted_avg_0_hrs = torch.stack([item["weighted_avg_0_hrs"] for item in batch])
    risk_free_rate = torch.stack([item["Risk_Free_Rate"] for item in batch])
    sectors = [item["Sector"] for item in batch]
    texts = [item["text"] for item in batch]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "weighted_avg_0_hrs": weighted_avg_0_hrs,
        "Risk_Free_Rate": risk_free_rate,
        "Sector": sectors,
        "texts": texts
    }
