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
GLOBAL_TOKENIZER = LlamaTokenizerFast.from_pretrained(TOKENIZER_NAME, model_max_length=config.BLOCK_SIZE)

# Add special tokens
special_tokens = {
    'additional_special_tokens': ['<bot>', '<start_latent>', '<end_latent>', '<reasoning>', '</reasoning>']
}
GLOBAL_TOKENIZER.add_special_tokens(special_tokens)

# Set pad token
GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token

# -------------------------------------------------------------------------
# PRECOMPUTED DATASET CLASS
# -------------------------------------------------------------------------
class PrecomputedDataset(Dataset):
    """
    Dataset that assumes text has been precomputed and stored in a DataFrame.
    It tokenizes the text on demand and retrieves the regression target.
    
    In this updated version (Option #1) we append the label as text.
    The label string is appended with a clear marker "[30 DAY LABEL]:".
    """
    def __init__(self, df, tokenizer, block_size):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx].get("text", "")
        # Get the numerical label (for example, weighted_avg_720_hrs) and prepare it.
        # Here we also log-scale it (if desired) for evaluation purposes.
        raw_label = self.df.iloc[idx].get("weighted_avg_720_hrs", 0.0)
        # Format the raw label as text with a clear marker.
        label_str = f"\n<STOCK PRICE 30 DAYS OUT>: {raw_label:.2f}"
        # Append the label string to the article text.
        new_text = text + label_str

        # Set truncation side to left so that when the text is too long, we retain the most recent tokens,
        # which now include the appended label.
        self.tokenizer.truncation_side = 'left'
        encoding = self.tokenizer(
            new_text,
            truncation=True,
            padding="max_length",
            max_length=block_size,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)

        # Optionally compute a log-scaled version of the raw label for later regression evaluation.
        # (This is independent of the text-generation objective.)
        label_tensor = torch.log1p(torch.tensor(raw_label, dtype=torch.float))

        return {"input_ids": input_ids, "label": label_tensor}

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
