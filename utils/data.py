import os
import torch
import pandas as pd
import math
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizerFast
from utils.config import config
from huggingface_hub import hf_hub_download
from torch.utils.data.distributed import DistributedSampler

# Disable tokenizer parallelism to avoid warnings after fork.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# GLOBAL TOKENIZER SETUP
# -------------------------------------------------------------------------
TOKENIZER_NAME = getattr(config, "TOKENIZER_NAME", "hf-internal-testing/llama-tokenizer")
GLOBAL_TOKENIZER = LlamaTokenizerFast.from_pretrained(
    TOKENIZER_NAME,
    model_max_length=config.CONTEXT_WINDOW
)

# Add special tokens
special_tokens = {
    'additional_special_tokens': ['<bot>', '<start_latent>', '<end_latent>', '<eot>', '<reasoning>', '</reasoning>', '<STOCK PRICE 30 DAYS OUT>: ', ' </STOCK PRICE 30 DAYS OUT>']
}
GLOBAL_TOKENIZER.add_special_tokens(special_tokens)
# Set pad token to eos
GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token

# -------------------------------------------------------------------------
# SCHEDULING FUNCTION FOR LATENT MASKING
# -------------------------------------------------------------------------
def schedule_masking(
    input_ids_list: list[int],
    reasoning_start_id: int,
    reasoning_end_id: int,
    latent_token_id: int,
    mask_fraction: float = 1.0
) -> list[int]:
    """
    Replace a fraction of tokens in the reasoning span with the latent token.

    Args:
      input_ids_list: full token ID list
      reasoning_start_id: token ID marking the start of the reasoning span
      reasoning_end_id: token ID marking the end of the reasoning span
      latent_token_id: token ID to insert (e.g. '<bot>')
      mask_fraction: fraction of the span to mask (0.0–1.0)

    Returns:
      A new list with the first ceil(span_len * mask_fraction) tokens
      after the start marker replaced by latent_token_id.
    """
    # locate the span boundaries
    try:
        start_idx = input_ids_list.index(reasoning_start_id)
        end_idx   = input_ids_list.index(reasoning_end_id)
    except ValueError:
        return input_ids_list  # markers not both present

    span_len = end_idx - (start_idx + 1)
    if span_len <= 0 or mask_fraction <= 0.0:
        return input_ids_list

    # compute how many tokens to mask
    n_mask = min(
        span_len,
        max(1, int(math.ceil(span_len * mask_fraction)))
    )

    # apply masking in-place
    for i in range(start_idx + 1, start_idx + 1 + n_mask):
        input_ids_list[i] = latent_token_id

    return input_ids_list

def rreplace(s, old, new, occurrence=1):
    parts = s.rsplit(old, occurrence)
    return new.join(parts)

# -------------------------------------------------------------------------
# PRECOMPUTED SINGLE-TEXT DATASET
# -------------------------------------------------------------------------
class PrecomputedDataset(Dataset):
    def __init__(self, df, tokenizer, block_size, gradual_latent_mask=False, full_latent_mask=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.gradual_latent_mask = gradual_latent_mask
        self.full_latent_mask = full_latent_mask
        # cache token ids
        self.latent_id = tokenizer.convert_tokens_to_ids('<bot>')
        self.start_id = tokenizer.convert_tokens_to_ids('<reasoning>')
        self.end_id = tokenizer.convert_tokens_to_ids('</reasoning>')
        # cache BOS token id (for <s>)
        self.bos_id = tokenizer.bos_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx].get('text', '')
        raw_label = self.df.iloc[idx].get('weighted_avg_720_hrs', 0.0)
        label_str = f"\n<STOCK PRICE 30 DAYS OUT>: {raw_label:.2f} </STOCK PRICE 30 DAYS OUT>"
        new_text = text + label_str

        self.tokenizer.truncation_side = 'left'
        enc = self.tokenizer(
            new_text,
            truncation=True,
            padding='max_length',
            max_length=config.BLOCK_SIZE,
            return_tensors='pt'
        )
        ids = enc['input_ids'].squeeze(0).tolist()

        # prepend BOS (<s>) and trim to block_size
        ids = [self.bos_id] + ids[:-1]

        if self.gradual_latent_mask or self.full_latent_mask:
            ids = schedule_masking(
                ids,
                self.start_id, self.end_id,
                self.latent_id,
                gradual=self.gradual_latent_mask
            )
        input_ids = torch.tensor(ids, dtype=torch.long)
        return {'input_ids': input_ids, 'label': torch.tensor(raw_label, dtype=torch.float32)}

# -------------------------------------------------------------------------
# PRECOMPUTED BOOTSTRAP DATASET (25 TEXT ITERATIONS)
# -------------------------------------------------------------------------
class PrecomputedBootstrapDataset(Dataset):
    def __init__(self, df, tokenizer, block_size, text_columns, label_column):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_cols = text_columns
        self.label_col = label_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        texts = [row[col] for col in self.text_cols]
        label = torch.tensor(row[self.label_col], dtype=torch.float32)
        
        # tokenize each bootstrap text
        all_ids = []
        for txt in texts:
            enc = self.tokenizer(
                txt,
                truncation=True,
                padding='max_length',
                max_length=config.BLOCK_SIZE,
                return_tensors='pt'
            )
            all_ids.append(enc['input_ids'].squeeze(0))
        input_ids = torch.stack(all_ids, dim=0)  # (K, T)
        return {'input_ids': input_ids, 'label': label}

# -------------------------------------------------------------------------
# CUSTOM COLLATE FUNCTIONS
# -------------------------------------------------------------------------
def custom_collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch], dim=0)
    labels = torch.stack([b['label'] for b in batch], dim=0)
    return {'input_ids': input_ids, 'labels': labels}

def bootstrap_collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch], dim=0)  # (B,K,T)
    labels = torch.stack([b['label'] for b in batch], dim=0)         # (B,)
    return {'input_ids': input_ids, 'labels': labels}

# -------------------------------------------------------------------------
# DATALOADER FACTORY
# -------------------------------------------------------------------------
def prepare_ft_dataloader(
    tokenizer,
    block_size,
    shuffle,
    args,
    stage: int = 1,
    gradual_latent_mask: bool = False,
    full_latent_mask: bool = False,
    sampler=None
):
    """
    stage:
      1 => ft_dataset_1.parquet       (PrecomputedDataset, no label replace)
      2 => ft_dataset_2.parquet       (PrecomputedDataset, do label replace)
      3-7 => ft_dataset_{stage}.parquet (PrecomputedBootstrapDataset)
      8 => ft_dataset_8.parquet       (PrecomputedBootstrapDataset test)
    """
    filename = f"ft_dataset_{stage}.parquet"
    file_path = hf_hub_download(
        repo_id="nbettencourt/sc454k-preprocessed-dfs",
        filename=filename,
        repo_type="dataset"
    )
    df = pd.read_parquet(file_path)
    os.remove(file_path)

    if stage in (1, 2):
        df["text"] = df["text"].apply(lambda x: 
            rreplace(x, "<30 DAY LABEL>", "<STOCK PRICE 30 DAYS OUT>:", 1)
            + " </STOCK PRICE 30 DAYS OUT>"
        )

        dataset = PrecomputedDataset(
            df,
            tokenizer,
            block_size=block_size,
            gradual_latent_mask=gradual_latent_mask,
            full_latent_mask=full_latent_mask
        )
        collate = custom_collate_fn
        drop_last = True
    else:
        text_cols = [f"text_iteration_{i}" for i in range(1, 27)]
        label_col = "weighted_avg_720_hrs"
        # after loading df for stages >=3:
        label_str = df[label_col].map(lambda v: 
            f"<STOCK PRICE 30 DAYS OUT>: {v:.2f} </STOCK PRICE 30 DAYS OUT>"
        )
        for col in text_cols:
            df[col] = df[col] + " " + label_str

        dataset = PrecomputedBootstrapDataset(
            df,
            tokenizer=tokenizer,
            block_size=block_size,
            text_columns=text_cols,
            label_column=label_col
        )
        collate = bootstrap_collate_fn
        drop_last = (stage < 8)

    # **Add this block** to shard data across ranks
    if torch.distributed.is_initialized() and sampler is None:
        sampler = DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            seed=getattr(args, "random_seed", 42)
        )
        shuffle = False   # sampler will handle shuffling

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
        drop_last=drop_last
    )
