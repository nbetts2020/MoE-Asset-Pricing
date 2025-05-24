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

import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

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
    """
    Parquet-backed dataset.

    • If `streaming=True`, each sample is read lazily from the Parquet
      file, so only a single row-group lives in RAM at once.
    • When `streaming=False` the full DataFrame is held in memory
      (legacy behaviour).

    The curriculum for Coconut masking is driven by the *mutable*
    attribute `mask_fraction` (0 → no masking, 1 → mask the whole
    <reasoning> span).  Phase-3 code can update that field between
    passes:

        ds.mask_fraction = 0.0   # first pass, no masking
        ds.mask_fraction = 1.0   # second pass, full masking
    """

    def __init__(
        self,
        source,                     # pandas.DataFrame 𝘰𝘳 str(path-to-parquet)
        tokenizer,
        block_size: int,
        mask_fraction: float = 1.0, # initial masking ratio
        streaming: bool = False
    ):
        self.streaming      = streaming
        self.tokenizer      = tokenizer
        self.block_size     = block_size
        self.mask_fraction  = float(mask_fraction)

        # --- cache special-token ids once ----------------------------------
        self.latent_id = tokenizer.convert_tokens_to_ids("<bot>")
        self.start_id  = tokenizer.convert_tokens_to_ids("<reasoning>")
        self.end_id    = tokenizer.convert_tokens_to_ids("</reasoning>")
        self.bos_id    = tokenizer.bos_token_id

        # --- pick storage mode --------------------------------------------
        if streaming:
            self.pq_file = pq.ParquetFile(source)          # lazy, columnar
            self._n      = self.pq_file.metadata.num_rows
        else:
            self.df = source.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._n if self.streaming else len(self.df)

    # ------------------------------------------------------------------ #
    def _get_row(self, idx: int) -> dict:
        if not self.streaming:
            return self.df.iloc[idx]

        # Which row-group contains this row?
        rg_size   = self.pq_file.metadata.row_group(0).num_rows
        rg_index  = idx // rg_size
        offset    = idx %  rg_size

        batch = (
            self.pq_file
                .read_row_group(rg_index)   # load one small row-group
                .slice(offset, 1)           # keep just the single row
        )
        return {col: batch[col][0].as_py() for col in batch.schema.names}

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row        = self._get_row(idx)
        text       = row.get("text", "")
        raw_label  = row.get("weighted_avg_720_hrs", 0.0)

        # append numeric label in the expected tag format
        label_str  = (
            f"\n<STOCK PRICE 30 DAYS OUT>: {raw_label:.2f} "
            f"</STOCK PRICE 30 DAYS OUT>"
        )
        new_text   = text + label_str

        # tokenize (left-truncation to `block_size`)
        self.tokenizer.truncation_side = "left"
        enc  = self.tokenizer(
            new_text,
            truncation   = True,
            padding      = "max_length",
            max_length   = self.block_size,
            return_tensors = "pt",
        )
        ids  = enc["input_ids"].squeeze(0).tolist()

        # prepend <s> and drop final token to keep length constant
        ids = [self.bos_id] + ids[:-1]

        # -------- Coconut masking ---------------------------------------
        if 0.0 < self.mask_fraction <= 1.0:
            ids = schedule_masking(
                ids,
                self.start_id,
                self.end_id,
                self.latent_id,
                mask_fraction = self.mask_fraction,
            )

        return {
            "input_ids": torch.tensor(ids,        dtype=torch.long),
            "label":     torch.tensor(raw_label, dtype=torch.float32),
        }

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
    sampler=None,
    streaming: bool = False,
):
    """
    stage:
      1 → ft_dataset_1.parquet  (PrecomputedDataset)
      2 → ft_dataset_2.parquet  (PrecomputedDataset)
      3–7 → ft_dataset_{k}.parquet (PrecomputedBootstrapDataset)
      8 → ft_dataset_8.parquet  (PrecomputedBootstrapDataset, val)
    """
    filename  = f"ft_dataset_{stage}.parquet"
    file_path = hf_hub_download(
        repo_id  = "nbettencourt/sc454k-preprocessed-dfs",
        filename = filename,
        repo_type= "dataset",
    )

    # ───────────────────────────────────────────────────────────────────────
    #  SINGLE-TEXT DATASETS (stages 1 & 2)                                  #
    # ───────────────────────────────────────────────────────────────────────
    if stage in (1, 2):
        # • streaming=True  → pass the path; PrecomputedDataset will read lazily
        # • streaming=False → keep legacy behaviour: load full df into RAM
        if streaming:
            df_or_path = file_path
        else:
            df_or_path = pd.read_parquet(file_path)
            os.remove(file_path)

            # label-tag replacement is only needed in-memory; the streaming
            # dataset can do it on-the-fly if you wish.
            df_or_path["text"] = df_or_path["text"].apply(
                lambda x: rreplace(
                    x,
                    "<30 DAY LABEL>",
                    "<STOCK PRICE 30 DAYS OUT>:",
                    1,
                ) + " </STOCK PRICE 30 DAYS OUT>"
            )

        dataset = PrecomputedDataset(
            df_or_path,
            tokenizer,
            block_size      = block_size,
            mask_fraction   = 0.0,       # no masking for phases 1&2
            streaming       = streaming,
        )
        collate    = custom_collate_fn
        drop_last  = True

    # ───────────────────────────────────────────────────────────────────────
    #  BOOTSTRAP (K-text) DATASETS (stages 3-8)                             #
    # ───────────────────────────────────────────────────────────────────────
    else:
        df = pd.read_parquet(file_path)
        os.remove(file_path)

        text_cols = [f"text_iteration_{i}" for i in range(1, 27)]
        label_col = "weighted_avg_720_hrs"

        label_tag = df[label_col].map(
            lambda v: f"<STOCK PRICE 30 DAYS OUT>: {v:.2f} </STOCK PRICE 30 DAYS OUT>"
        )
        for col in text_cols:
            df[col] = df[col] + " " + label_tag

        dataset   = PrecomputedBootstrapDataset(
            df,
            tokenizer   = tokenizer,
            block_size  = block_size,
            text_columns= text_cols,
            label_column= label_col,
        )
        collate    = bootstrap_collate_fn
        drop_last  = (stage < 8)

    # ───────────────────────────────────────────────────────────────────────
    #  DISTRIBUTED SAMPLING (unchanged)                                     #
    # ───────────────────────────────────────────────────────────────────────
    if torch.distributed.is_initialized() and sampler is None:
        sampler = DistributedSampler(
            dataset,
            num_replicas = torch.distributed.get_world_size(),
            rank         = torch.distributed.get_rank(),
            shuffle      = shuffle,
            seed         = getattr(args, "random_seed", 42),
        )
        shuffle = False  # sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size = config.BATCH_SIZE,
        shuffle    = (shuffle and sampler is None),
        sampler    = sampler,
        num_workers= 0,
        collate_fn = collate,
        pin_memory = True,
        drop_last  = drop_last,
    )
