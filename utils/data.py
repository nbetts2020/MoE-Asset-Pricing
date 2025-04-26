import os
import torch
import pandas as pd
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
    'additional_special_tokens': ['<bot>', '<start_latent>', '<end_latent>', '<reasoning>', '</reasoning>']
}
GLOBAL_TOKENIZER.add_special_tokens(special_tokens)
# Set pad token to eos
GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token

# -------------------------------------------------------------------------
# SCHEDULING FUNCTION FOR LATENT MASKING
# -------------------------------------------------------------------------
def schedule_masking(input_ids_list, reasoning_start_id, reasoning_end_id, latent_token_id, gradual=True):
    if reasoning_start_id not in input_ids_list or reasoning_end_id not in input_ids_list:
        return input_ids_list
    start_idx = input_ids_list.index(reasoning_start_id)
    end_idx = input_ids_list.index(reasoning_end_id)
    span = input_ids_list[start_idx+1:end_idx]
    L = len(span)
    if L == 0:
        return input_ids_list
    n = (L // 2) if gradual else L
    for i in range(start_idx+1, start_idx+1+n):
        if i < end_idx:
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
        label_str = f"\n<STOCK PRICE 30 DAYS OUT>: {raw_label:.2f}"
        new_text = text + label_str

        self.tokenizer.truncation_side = 'left'
        enc = self.tokenizer(
            new_text,
            truncation=True,
            padding='max_length',
            max_length=config.CONTEXT_WINDOW,
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
        label = torch.log1p(torch.tensor(raw_label, dtype=torch.float))
        return {'input_ids': input_ids, 'label': label}

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
        label = torch.log1p(torch.tensor(row[self.label_col], dtype=torch.float))

        # tokenize each bootstrap text
        all_ids = []
        for txt in texts:
            enc = self.tokenizer(
                txt,
                truncation=True,
                padding='max_length',
                max_length=config.CONTEXT_WINDOW,
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
        df["text"] = df["text"].apply(
            lambda x: rreplace(x, "<30 DAY LABEL>", "<STOCK PRICE 30 DAYS OUT>", 1)
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
