import os
import torch
import pandas as pd
import math
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizerFast
from utils.config import config
from huggingface_hub import list_repo_files, hf_hub_download
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
    model_max_length=config.BLOCK_SIZE
)

# Add special tokens
special_tokens = {
    'additional_special_tokens': ['<bot>', '<start_latent>', '<end_latent>', '<eot>', '<reasoning>', '</reasoning>', '<STOCK PRICE 30 DAYS OUT>: ', '</STOCK PRICE 30 DAYS OUT>']
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
    def __init__(
        self,
        source,
        tokenizer,
        block_size: int,
        mask_fraction: float = 1.0,
        streaming: bool = False,
        stage = 0
    ):
        self.streaming      = streaming
        self.tokenizer      = tokenizer
        self.block_size     = block_size
        self.mask_fraction  = float(mask_fraction)
        self.stage = stage

        self.latent_id = tokenizer.convert_tokens_to_ids("<bot>")
        self.start_id  = tokenizer.convert_tokens_to_ids("<reasoning>")
        self.end_id    = tokenizer.convert_tokens_to_ids("</reasoning>")
        self.bos_id    = tokenizer.bos_token_id
        self.eot_id_for_opener = tokenizer.convert_tokens_to_ids("</reasoning>")

        if streaming:
            self.pq_file = pq.ParquetFile(source)
            self._n      = self.pq_file.metadata.num_rows
        else:
            self.df = source.reset_index(drop=True)
            self._n = len(self.df)

    def __len__(self) -> int:
        return self._n

    def _get_row(self, idx: int) -> dict:
        if not self.streaming:
            return self.df.iloc[idx]

        rg_size   = self.pq_file.metadata.row_group(0).num_rows
        rg_index  = idx // rg_size
        offset    = idx %  rg_size
        batch = (
            self.pq_file
                .read_row_group(rg_index)
                .slice(offset, 1)
        )
        return {col: batch[col][0].as_py() for col in batch.schema.names}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row        = self._get_row(idx)
        text       = row.get("text", "")
        raw_label  = row.get("weighted_avg_720_hrs", 0.0)
    
        if self.stage == 2:
            new_text = text + " </STOCK PRICE 30 DAYS OUT>"
        else:
            label_str  = (
                f"\n<STOCK PRICE 30 DAYS OUT>: {raw_label:.2f} "
                f"</STOCK PRICE 30 DAYS OUT>"
            )
            new_text   = text + label_str

        self.tokenizer.truncation_side = "left"
        enc  = self.tokenizer(
            new_text,
            truncation   = True,
            padding      = "max_length",
            max_length   = self.block_size,
            return_tensors = "pt",
        )
        
        ids_list  = enc["input_ids"].squeeze(0).tolist()

        if ids_list[0] != self.bos_id :
            ids_list = [self.bos_id] + ids_list[:-1]
        
        has_eot = self.eot_id_for_opener in ids_list
        has_bot = self.latent_id in ids_list

        if has_eot and not has_bot:
            ids_list = [self.latent_id] + ids_list[:-1]

        if 0.0 < self.mask_fraction <= 1.0:
            ids_list = schedule_masking(
                ids_list,
                self.start_id,
                self.end_id,
                self.latent_id,
                mask_fraction = self.mask_fraction,
            )
        
        final_ids_tensor = torch.tensor(ids_list, dtype=torch.long)

        return {
            "input_ids": final_ids_tensor,
            "label":     torch.tensor(raw_label, dtype=torch.float32),
        }

# -------------------------------------------------------------------------
# PRECOMPUTED BOOTSTRAP DATASET (25 TEXT ITERATIONS)
# -------------------------------------------------------------------------
class PrecomputedBootstrapDataset(Dataset):
    """
    Parquet-backed bootstrap dataset. If streaming=True, reads
    one row at a time from the Parquet file to keep memory low.
    Otherwise loads the full DataFrame into RAM.
    """
    def __init__(
        self,
        source,
        tokenizer,
        block_size: int,
        text_columns: list[str],
        label_column: str,
        streaming: bool = False
    ):
        self.streaming    = streaming
        self.tokenizer    = tokenizer
        self.block_size   = block_size
        self.text_cols    = text_columns
        self.label_col    = label_column

        if streaming:
            # lazy, columnar reader
            self.pq_file = pq.ParquetFile(source)
            self._n      = self.pq_file.metadata.num_rows
        else:
            # in-memory
            self.df = source.reset_index(drop=True)
            self._n = len(self.df)

    def __len__(self) -> int:
        return self._n

    def _get_row(self, idx: int) -> pd.Series:
        if not self.streaming:
            return self.df.iloc[idx]
        # streaming mode: locate the row-group and offset
        rg_size  = self.pq_file.metadata.row_group(0).num_rows
        rg_index = idx // rg_size
        offset   = idx % rg_size
        batch = (
            self.pq_file
                .read_row_group(rg_index)
                .slice(offset, 1)
        )
        # convert single-row Table to dict→Series
        data = {col: batch[col][0].as_py() for col in batch.schema.names}
        return pd.Series(data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._get_row(idx)
        # gather the K different texts for this bootstrap sample
        texts = [row[col] for col in self.text_cols]
        label = torch.tensor(row[self.label_col], dtype=torch.float32)

        # tokenize each of the K texts
        all_ids = []
        self.tokenizer.truncation_side = "left"
        for txt in texts:
            label_tag = f"<STOCK PRICE 30 DAYS OUT>: {label:.2f} </STOCK PRICE 30 DAYS OUT>"
            txt = txt + " " + label_tag
            enc = self.tokenizer(
                txt,
                truncation=True,
                padding="max_length",
                max_length=self.block_size,
                return_tensors="pt",
            )
            all_ids.append(enc["input_ids"].squeeze(0))
        # stack into shape (K, T)
        input_ids = torch.stack(all_ids, dim=0)

        return {"input_ids": input_ids, "label": label}
        
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
    mask_fraction: float = 0.0
):
    """
    stage:
      0   → ALL years (2016–2024) from cc-news-formatted/*.parquet
      1–6 → ft_dataset_{stage}.parquet (PrecomputedDataset)
      7–8 → ft_dataset_{stage}.parquet (PrecomputedBootstrapDataset)
    """
    # ───────────────────────────────────────────────────────────────────────
    #  1. Download & prep the Parquet source(s)
    # ───────────────────────────────────────────────────────────────────────
    if stage == 0:
        # list all files in the public cc-news-formatted dataset
        repo_id = "nbettencourt/cc-news-formatted"
        files   = list_repo_files(repo_id=repo_id, repo_type="dataset")
        years   = {f"{y}/" for y in range(2016, 2025)}

        # download every shard under 2016/ … 2024/
        shard_paths = [
            hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset")
            for f in files
            if any(f.startswith(y) for y in years) and f.endswith(".parquet")
        ]

        # concat into one DataFrame
        df = pd.concat((pd.read_parquet(p) for p in shard_paths), ignore_index=True)
        for p in shard_paths: os.remove(p)   # clean up temp files

        source = df
        use_bootstrap = False

    else:
        # existing ft_dataset_N.parquet logic
        repo_id  = "nbettencourt/sc454k-preprocessed-dfs"
        filename = f"ft_dataset_{stage}.parquet"
        file_path = hf_hub_download(
            repo_id   = repo_id,
            filename  = filename,
            repo_type = "dataset",
        )

        if stage <= 6:
            source = file_path if streaming else pd.read_parquet(file_path)
            if not streaming: os.remove(file_path)
            use_bootstrap = False

        else:  # stage 7 or 8 → bootstrap
            text_cols = [f"iteration_text_{i}" for i in range(1, 26)]
            label_col = "weighted_avg_720_hrs"

            if streaming:
                source = file_path
            else:
                df = pd.read_parquet(file_path)
                os.remove(file_path)
                tag = df[label_col].map(
                    lambda v: f"<STOCK PRICE 30 DAYS OUT>: {v:.2f} </STOCK PRICE 30 DAYS OUT>"
                )
                for col in text_cols:
                    df[col] = df[col] + "\n" + tag
                source = df

            use_bootstrap = True

    # ───────────────────────────────────────────────────────────────────────
    #  2. Instantiate the Dataset
    # ───────────────────────────────────────────────────────────────────────
    if not use_bootstrap:
        dataset = PrecomputedDataset(
            source        = source,
            tokenizer     = tokenizer,
            block_size    = block_size,
            mask_fraction = mask_fraction,
            streaming     = streaming,
            stage         = stage,
        )
        collate   = custom_collate_fn
        drop_last = True

    else:
        dataset = PrecomputedBootstrapDataset(
            source        = source,
            tokenizer     = tokenizer,
            block_size    = block_size,
            text_columns  = text_cols,
            label_column  = label_col,
            streaming     = streaming,
        )
        collate   = bootstrap_collate_fn
        drop_last = (stage < 8)

    # ───────────────────────────────────────────────────────────────────────
    #  3. Wrap in DataLoader (with optional DistributedSampler)
    # ───────────────────────────────────────────────────────────────────────
    if torch.distributed.is_initialized() and sampler is None:
        sampler = DistributedSampler(
            dataset,
            num_replicas = torch.distributed.get_world_size(),
            rank         = torch.distributed.get_rank(),
            shuffle      = shuffle,
            seed         = getattr(args, "random_seed", 42),
        )
        shuffle = False

    return DataLoader(
        dataset     = dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = (shuffle and sampler is None),
        sampler     = sampler,
        num_workers = 0,
        collate_fn  = collate,
        pin_memory  = True,
        drop_last   = drop_last,
    )
