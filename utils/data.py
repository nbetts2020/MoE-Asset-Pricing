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
        self.tokenizer.truncation_side = 'left'
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        label = self.df.iloc[idx].get("weighted_avg_720_hrs", 0.0)  # Defaulting to 0.0 if missing
        label = torch.tensor(label, dtype=torch.float)
        label = torch.log1p(label)  # Log-scaled target
        return {"input_ids": input_ids, "label": torch.tensor(label, dtype=torch.float)}

# -------------------------------------------------------------------------
# CUSTOM COLLATE FUNCTION
# -------------------------------------------------------------------------
def custom_collate_fn(batch):
    """
    Batches input_ids and labels into tensors.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}
