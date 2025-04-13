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
# SCHEDULING FUNCTION FOR LATENT MASKING
# -------------------------------------------------------------------------
def schedule_masking(input_ids_list, reasoning_start_id, reasoning_end_id, latent_token_id, gradual=True):
    """
    Replaces tokens in the reasoning segment with the latent token.
    
    Parameters:
      input_ids_list: list of token ids (for one sample).
      reasoning_start_id: token id for the <reasoning> marker.
      reasoning_end_id: token id for the </reasoning> marker.
      latent_token_id: token id to use as the latent token (e.g. '<bot>').
      gradual (bool): if True, mask only part of the reasoning span (e.g. the first half);
                       if False, mask the entire reasoning span.
                       
    Returns:
      Modified list of token ids.
    """
    if reasoning_start_id not in input_ids_list or reasoning_end_id not in input_ids_list:
        return input_ids_list

    start_idx = input_ids_list.index(reasoning_start_id)
    end_idx = input_ids_list.index(reasoning_end_id)
    # Tokens between markers
    reasoning_span = input_ids_list[start_idx+1 : end_idx]
    span_length = len(reasoning_span)
    
    if span_length == 0:
        return input_ids_list

    if gradual:
        # For example, mask half of the reasoning tokens.
        n_to_mask = span_length // 2
    else:
        # Mask all tokens within the reasoning span.
        n_to_mask = span_length

    # Replace the first n_to_mask tokens after <reasoning> with the latent token.
    for i in range(start_idx + 1, start_idx + 1 + n_to_mask):
        if i < end_idx:
            input_ids_list[i] = latent_token_id

    return input_ids_list

# -------------------------------------------------------------------------
# PRECOMPUTED DATASET CLASS
# -------------------------------------------------------------------------
class PrecomputedDataset(Dataset):
    """
    Dataset that assumes text has been precomputed and stored in a DataFrame.
    It tokenizes the text on demand and retrieves the regression target.
    
    The label string is appended with a clear marker ("<STOCK PRICE 30 DAYS OUT>:").
    
    This updated version supports latent masking via two boolean flags:
      - gradual_attention_mask: mask a fraction of tokens in the reasoning segment.
      - full_attention_mask: mask the entire reasoning segment.
    
    Only one of these should be True at a time.
    """
    def __init__(self, df, tokenizer, block_size, gradual_attention_mask=False, full_attention_mask=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.gradual_attention_mask = gradual_attention_mask
        self.full_attention_mask = full_attention_mask
        
        # Pre-calculate special token IDs.
        self.latent_token_id = self.tokenizer.convert_tokens_to_ids('<bot>')
        self.reasoning_start_id = self.tokenizer.convert_tokens_to_ids('<reasoning>')
        self.reasoning_end_id = self.tokenizer.convert_tokens_to_ids('</reasoning>')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve the raw text and target value.
        text = self.df.iloc[idx].get("text", "")
        raw_label = self.df.iloc[idx].get("weighted_avg_720_hrs", 0.0)
        label_str = f"\n<STOCK PRICE 30 DAYS OUT>: {raw_label:.2f}"
        new_text = text + label_str

        # Set truncation side to 'left' so that the recent tokens (including the label) are retained.
        self.tokenizer.truncation_side = 'left'
        encoding = self.tokenizer(
            new_text,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)

        # Convert input_ids to a list for potential masking modification.
        input_ids_list = input_ids.tolist()

        # Apply latent masking if requested.
        if self.gradual_attention_mask:
            input_ids_list = schedule_masking(
                input_ids_list,
                self.reasoning_start_id,
                self.reasoning_end_id,
                self.latent_token_id,
                gradual=True
            )
        elif self.full_attention_mask:
            input_ids_list = schedule_masking(
                input_ids_list,
                self.reasoning_start_id,
                self.reasoning_end_id,
                self.latent_token_id,
                gradual=False
            )

        # Convert the modified list back to a tensor.
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)

        # Compute the label tensor (log-scaled) for regression evaluation.
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
