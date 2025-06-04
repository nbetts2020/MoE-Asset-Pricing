# utils/utils.py

import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from muon import Muon

import pandas as pd
import pyarrow.parquet as pq
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from utils.config import config
from utils.model import SparseMoELanguageModel
from utils.data import PrecomputedDataset, custom_collate_fn

import os
import shutil
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv

import boto3
from botocore.exceptions import ClientError

from tqdm import tqdm
from transformers import AutoTokenizer

import json
from huggingface_hub import hf_hub_download

from utils.ewc import ElasticWeightConsolidation
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer

from torch.utils.data.distributed import DistributedSampler

from utils.sampling import sample_articles
from utils.metrics import OnlineMetrics

import logging
import subprocess

import inspect
import numpy as np

from multiprocessing import Pool, cpu_count
import random
import ast
from pandarallel import pandarallel
import gc

import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.zero.stage3 import GatheredParameters

from utils.ebm import EnergyBasedModel

logger = logging.getLogger(__name__)

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
TOTAL_CHUNK_ROWS = sum(CHUNK_SIZES.values())  # 50000 + 25k*11 + 13146 = 338146

def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None and m.weight.dim() >= 2:
            nn.init.xavier_uniform_(m.weight, gain=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def rreplace(s, old, new, occurrence=1):
    parts = s.rsplit(old, occurrence)
    return new.join(parts)

def prepare_ft_dataloader(tokenizer, block_size, shuffle, args, stage_1=True, gradual_latent_mask=False, full_latent_mask=False, sampler=None):
    """
    Downloads the appropriate parquet file and prepares a DataLoader based on the given parameters.

    Parameters:
      tokenizer: Tokenizer to use (e.g. GLOBAL_TOKENIZER).
      block_size (int): Sequence length.
      shuffle (bool): Whether to shuffle the dataset.
      args: Additional arguments.
      stage_1 (bool): Whether to use dataset 1 (True) or dataset 2 (False).
      gradual_latent_mask (bool): Whether to use a gradual latent mask on the reasoning segment.
      full_latent_mask (bool): Whether to use a full latent mask on the reasoning segment.
      sampler: Optional sampler (e.g. DistributedSampler).

    Returns:
      DataLoader: A PyTorch DataLoader for the precomputed dataset.
    """
    if stage_1:
        file_path = hf_hub_download(
            repo_id="nbettencourt/sc454k-preprocessed-dfs",
            filename="ft_dataset_1.parquet",
            repo_type="dataset"
        )
    else:
        file_path = hf_hub_download(
            repo_id="nbettencourt/sc454k-preprocessed-dfs",
            filename="ft_dataset_2.parquet",
            repo_type="dataset"
        )  # Ensure the closing parenthesis is here.

    # Read the parquet file into a DataFrame.
    df = pd.read_parquet(file_path)
    df["text"] = df["text"].apply(lambda x: rreplace(x, "<30 DAY LABEL>", "<STOCK PRICE 30 DAYS OUT>", 1))

    # Create the dataset instance.
    dataset = PrecomputedDataset(df, tokenizer, block_size=block_size,
                                 gradual_attention_mask=gradual_latent_mask,
                                 full_attention_mask=full_latent_mask)

    # Use the sampler if provided.
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True
    )

    return dataloader

def get_data(epoch, window_index, global_offset, global_max, args=None, cache_dir="/tmp/hf_cache_datasets"):
    """
    For run mode, loads a chunk from the original preprocessed dataset.
    For train mode, loads the entire new SC454k-formatted dataset,
    filters out rows with nan or non-positive weighted_avg_720_hrs,
    and subsamples based on args.percent_data.

    For rl mode, loads a chunk from train_dataset_{window_index}a.parquet.
    """
    os.makedirs(cache_dir, exist_ok=True)

    if args.mode == "train":
        repo_id = "nbettencourt/SC454k-formatted"
        ds = load_dataset(repo_id, split="train")
        df = ds.to_pandas()
    elif args.mode == "run":
        repo_id = "nbettencourt/sc454k-preprocessed-dfs"
        epoch_letter = chr(ord('a') + epoch - 1)
        filename = f"run_dataset_{window_index}.parquet"
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=cache_dir)
        df = pd.read_parquet(file_path)
    elif args.mode == "rl":
        # For RL mode, load chunks from train_dataset_{window_index}a.parquet.
        repo_id = "nbettencourt/sc454k-preprocessed-dfs"
        epoch_letter = "a"  # Always use 'a' for RL mode.
        filename = f"train_dataset_{window_index}{epoch_letter}.parquet"
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=cache_dir)
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Common filtering: drop rows with missing or non-positive weighted_avg_720_hrs.
    df = df.dropna(subset=['weighted_avg_720_hrs'])
    df = df[df['weighted_avg_720_hrs'] > 0]

    # For run and rl modes, apply chunk-level limits.
    if args.mode in ["run", "rl"]:
        if global_offset >= global_max:
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            return pd.DataFrame()
        chunk_rows = len(df)
        if global_offset + chunk_rows > global_max:
            needed = global_max - global_offset
            df = df.iloc[:needed]
        # Cleanup downloaded file and cache.
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
    else:
        # For train mode, subsample based on percent_data.
        num_rows = int(len(df) * (args.percent_data / 100))
        df = df.iloc[:num_rows]

    return df

def load_rl_data(args, rl_rows: int = -1, chunk_start: int = 1, chunk_end: int = 13, shuffle: bool = True):
    """
    Generator that loads RL data chunk-by-chunk from Parquet files
    (train_dataset_{chunk_index}a.parquet). It yields a DataFrame for each chunk.

    If rl_rows = -1, yields all rows from each chunk.
    Otherwise, it samples proportionally from each chunk so that the total
    approximates ~rl_rows rows.
    """
    accumulated = 0
    for chunk_idx in range(chunk_start, chunk_end + 1):
        if chunk_idx not in CHUNK_SIZES:
            logging.warning(f"Chunk {chunk_idx} not found in CHUNK_SIZES map. Skipping.")
            continue

        chunk_size = CHUNK_SIZES[chunk_idx]
        df_chunk = get_data(
            epoch=1,
            window_index=chunk_idx,
            global_offset=0,
            global_max=int(1e12),  # no limit here
            args=args
        )
        if df_chunk.empty:
            logging.info(f"Chunk {chunk_idx} returned empty DataFrame.")
            continue

        if rl_rows == -1:
            chunk_df = df_chunk
        else:
            # Proportional sample: calculate how many rows to sample from this chunk.
            chunk_needed = int(round((chunk_size / TOTAL_CHUNK_ROWS) * rl_rows))
            if chunk_needed <= 0:
                continue
            if chunk_needed < len(df_chunk):
                chunk_df = df_chunk.sample(n=chunk_needed, random_state=random.randint(0, 999999))
            else:
                chunk_df = df_chunk

        accumulated += len(chunk_df)
        yield chunk_df

        # Clean up to free memory.
        del df_chunk, chunk_df
        gc.collect()

        if rl_rows != -1 and accumulated >= rl_rows:
            break

def process_run_dataset(
        run_dataset_filename,
        tokenizer,
        model,
        args,
        device,
        batch_size     = 500,
        current_offset = 0,
        global_max     = int(1e12),
        cache_dir      = "/tmp/hf_cache_datasets_run"):

    import os, gc, shutil, torch, pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from utils.train import extract_label_value
    from utils.metrics import OnlineMetrics
            
    # ------------------------------------------------------------------ #
    #  Pre-computed tag IDs
    # ------------------------------------------------------------------ #
    PRICE_OPEN_IDS = tokenizer("<STOCK PRICE 30 DAYS OUT>: ", add_special_tokens=False).input_ids
    PRICE_TAG_LEN  = len(PRICE_OPEN_IDS)
    PRICE_CLOSE_ID = tokenizer.convert_tokens_to_ids("</STOCK PRICE 30 DAYS OUT>")
    REASON_OPEN_ID = tokenizer.convert_tokens_to_ids("<reasoning>")
    REASON_CLOSE_ID = tokenizer.convert_tokens_to_ids("</reasoning>")

    # ------------------------------------------------------------------ #
    #  Load the parquet split from HF Hub
    # ------------------------------------------------------------------ #
    os.makedirs(cache_dir, exist_ok=True)
    file_path = hf_hub_download(
        repo_id   = "nbettencourt/sc454k-preprocessed-dfs",
        filename  = run_dataset_filename,
        repo_type = "dataset",
        cache_dir = cache_dir,
    )
    pq_file = pq.ParquetFile(file_path)

    overall   = OnlineMetrics()
    by_sector = {}
    processed = 0

    for batch in pq_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        df = df.dropna(subset=["weighted_avg_720_hrs"])
        df = df[df["weighted_avg_720_hrs"] > 0].reset_index(drop=True)

        for idx, row in df.iterrows():
            if current_offset + processed + idx >= global_max:
                break
            sector = row.get("Sector", "Unknown")

            # ---------------------------------------------------------- #
            #  1. Pick best context via EBM
            # ---------------------------------------------------------- #
            ctx = ebm_select_contexts(
                df           = df,
                idx          = idx,
                model        = model,
                tokenizer    = tokenizer,
                ebm_samples  = 1 if args.no_ebm else args.ebm_num_samples,
            )
            
            # Normalise label token
            ctx = ctx.replace("<30 DAY LABEL>", "<STOCK PRICE 30 DAYS OUT>: ") \
                     + " </STOCK PRICE 30 DAYS OUT>"

            # ---------------------------------------------------------- #
            #  2. Generate reasoning
            #      prefix = ctx + '\n<reasoning>'
            # ---------------------------------------------------------- #
            tokenizer.truncation_side = "left"
            if args.no_reasoning:
                continue
            else:
                prefix_ids = tokenizer(
                    ctx + "\n<reasoning>",
                    truncation=True,
                    padding="max_length",
                    max_length=args.block_size,
                    return_tensors="pt"
                )["input_ids"].to(device)
    
                gen_reason = []
                with torch.no_grad():
                    cur = prefix_ids[:, -model.block_size // max(1, torch.distributed.get_world_size()):].clone()
                    for _ in range(1000):
                        hs     = model.forward_embeddings_only(cur)
                        logits = model.lm_head(hs)[:, -1, :]
                        nxt    = logits.argmax(-1, keepdim=True)
                        if nxt.item() == REASON_CLOSE_ID:
                            break
                        gen_reason.append(nxt)
                        cur = torch.cat([cur, nxt], dim=1)[:, -cur.size(1):]
    
                # close tag if not seen
                if not gen_reason or gen_reason[-1].item() != REASON_CLOSE_ID:
                    gen_reason.append(torch.tensor([[REASON_CLOSE_ID]], device=device))
    
                reasoning_ids = torch.cat(gen_reason, dim=1)
                reasoning_txt = tokenizer.decode(reasoning_ids[0], skip_special_tokens=False)

            # ---------------------------------------------------------- #
            # 3. Predict price
            #    prefix = ctx + '\n<reasoning>' + reasoning + '</reasoning>\n<STOCK PRICE 30 DAYS OUT>: '
            # ---------------------------------------------------------- #
            price_prefix = "\n<STOCK PRICE 30 DAYS OUT>: " if args.no_reasoning else f"{ctx}\n<reasoning>{reasoning_txt}</reasoning>\n<STOCK PRICE 30 DAYS OUT>: "
            prefix_ids = tokenizer(
                price_prefix,
                truncation=True,
                padding="max_length",
                max_length=args.block_size,
                return_tensors="pt"
            )["input_ids"].to(device)

            price_toks = []
            with torch.no_grad():
                cur = prefix_ids[:, -model.block_size // max(1, torch.distributed.get_world_size()):].clone()
                for _ in range(10):
                    hs     = model.forward_embeddings_only(cur)
                    logits = model.lm_head(hs)[:, -1, :]
                    nxt    = logits.argmax(-1, keepdim=True)
                    price_toks.append(nxt)
                    if nxt.item() == PRICE_CLOSE_ID:
                        break
                    cur = torch.cat([cur, nxt], dim=1)[:, -cur.size(1):]

            full_ids = torch.cat([prefix_ids, torch.cat(price_toks, dim=1)], dim=1)[0]
            decoded  = tokenizer.decode(full_ids, skip_special_tokens=False)
            pred_val = extract_label_value(decoded)
            if pred_val is None:
                continue

            # ---------------------------------------------------------- #
            # 4. Update running metrics
            # ---------------------------------------------------------- #
            actual   = row["weighted_avg_720_hrs"]
            oldprice = row["weighted_avg_0_hrs"]
            rf       = row["Risk_Free_Rate"]

            overall.update(pred_val, actual, oldprice, rf)
            by_sector.setdefault(sector, OnlineMetrics()) \
                     .update(pred_val, actual, oldprice, rf)

            print(f"Row {current_offset+processed+idx}: pred {pred_val:.2f}")

        processed += len(df)
        del df
        gc.collect()
        torch.cuda.empty_cache()
        if current_offset + processed >= global_max:
            break

    # tidy up
    os.remove(file_path)
    shutil.rmtree(cache_dir, ignore_errors=True)
    return overall, by_sector, processed
                            
def load_model_weights(model, weights_path, device):
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logging.info(f"Loaded model weights from '{weights_path}'.")
    else:
        logging.error(f"Weights file '{weights_path}' not found.")
        raise FileNotFoundError(f"Weights file '{weights_path}' not found.")
    model.to(device)
    return model

def download_models_from_s3(bucket):
    """
    Downloads the 'model' and 'models' directories from the specified S3 bucket,
    and additionally downloads the RL attention module file.

    Args:
        bucket (str): Name of the S3 bucket.
    """
    try:
        logging.info(f"Starting download of 'model' directory from s3://{bucket}/model")
        result = subprocess.run(
            ["aws", "s3", "sync", f"s3://{bucket}/model", "model"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"'model' directory synchronized successfully.\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error syncing 'model' directory: {e.stderr}")
        raise

    try:
        logging.info(f"Starting download of 'models' directory from s3://{bucket}/models")
        result = subprocess.run(
            ["aws", "s3", "sync", f"s3://{bucket}/models", "models"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"'models' directory synchronized successfully.\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error syncing 'models' directory: {e.stderr}")
        raise

def save_rl_attention(rl_module, epoch, save_dir="models", args=None):
    """
    Saves the RL hierarchical attention module's state dictionary to a file and,
    if a bucket is specified in args, uploads it to S3.

    Args:
        rl_module (nn.Module): The RL attention module to save.
        epoch (int): Current epoch (for logging/versioning, if desired).
        save_dir (str): Local directory to save the model.
        args (argparse.Namespace, optional): Command-line arguments; if args.bucket is provided, the file is uploaded.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "hAttention_rl.pth")
    torch.save(rl_module.state_dict(), save_path)
    print(f"RL Attention module saved to {save_path}")
    logging.info(f"RL Attention module saved to {save_path} at epoch {epoch}")

    if args is not None and hasattr(args, "bucket") and args.bucket:
        try:
            import subprocess
            s3_path = f"s3://{args.bucket}/models/hAttention_rl.pth"
            cmd = ["aws", "s3", "cp", save_path, s3_path]
            subprocess.run(cmd, check=True)
            logging.info(f"RL Attention module uploaded to {s3_path}")
        except Exception as e:
            logging.error(f"Error uploading RL Attention module to S3: {e}")

def ebm_select_contexts(df, idx, model, tokenizer, ebm_samples):
    """
    Pick the context whose EBM energy is lowest (|energy| ~= best match).

    Args
    ----
    df : pd.DataFrame            – dataset with columns iteration_{i}_text
    idx : int                    – row index to evaluate
    model : torch.nn.Module      – model whose forward returns (_, energy, _)
    tokenizer : transformers.PreTrainedTokenizer
    ebm_samples : int            – how many of the available candidates to score

    Returns
    -------
    str  – the selected context text
    """
    import random
    import numpy as np
    import torch
    from utils.config import config

    row = df.iloc[idx]
    candidates = [
        row.get(f"iteration_{i}_text")
        for i in range(1, 31)
        if row.get(f"iteration_{i}_text") is not None
    ]
    if not candidates:
        raise ValueError("No candidate contexts found for this sample.")

    sampled = random.sample(candidates, min(len(candidates), ebm_samples))

    energies = []
    device = next(model.parameters()).device
    tokenizer.truncation_side = "left"

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for text in sampled:
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=config.BLOCK_SIZE,
                return_tensors="pt"
            ).to(device)

            # forward() expected to return (logits, energy, extras)
            _, energy, _ = model(input_ids=enc["input_ids"])
            energies.append(float(energy))

    best_idx = int(np.argmin(np.abs(energies)))
    return sampled[best_idx]

def get_model_from_hf(model_repo_id, device):
    """
    Downloads the model configuration and weights from Hugging Face Hub and loads them into the SparseMoELanguageModel.

    Args:
        model_repo_id (str): The repository ID on Hugging Face Hub (e.g., "username/repo-name").
        device (torch.device): The device to map the model weights.

    Returns:
        SparseMoELanguageModel: The model loaded with weights from Hugging Face.
    """
    load_dotenv('/content/MoE-Asset-Pricing/.env')
    hf_token = os.getenv('HF_TOKEN')

    login(hf_token)

    logging.info(f"Downloading 'config.json' from Hugging Face repository '{model_repo_id}'.")
    try:
        config_path = hf_hub_download(repo_id=model_repo_id, filename="config.json")
        logging.info(f"Downloaded 'config.json' to '{config_path}'.")
    except Exception as e:
        logging.error(f"Failed to download 'config.json' from '{model_repo_id}': {e}")
        raise e

    logging.info(f"Downloading 'model_weights.pth' from Hugging Face repository '{model_repo_id}'.")
    try:
        weights_path = hf_hub_download(repo_id=model_repo_id, filename="model_weights.pth")
        logging.info(f"Downloaded 'model_weights.pth' to '{weights_path}'.")
    except Exception as e:
        logging.error(f"Failed to download 'model_weights.pth' from '{model_repo_id}': {e}")
        raise e

    # Load the configuration
    try:
        with open(config_path, 'r') as f:
            model_config_json = json.load(f)
        logging.info("Loaded model configuration from 'config.json'.")
    except Exception as e:
        logging.error(f"Failed to load configuration from '{config_path}': {e}")
        raise e

    # Filter the configuration to include only expected parameters
    expected_args = inspect.getfullargspec(SparseMoELanguageModel.__init__).args
    # Remove 'self' from the list
    if 'self' in expected_args:
        expected_args.remove('self')
    logging.info(f"Expected arguments for SparseMoELanguageModel: {expected_args}")

    # Filter config
    model_config_json = {k: config[k] for k in expected_args if k in config}
    logging.info(f"Filtered model configuration: {model_config_json}")

    # ize the model with the configuration
    try:
        model = SparseMoELanguageModel(**model_config_json)
        logging.info("Initialized SparseMoELanguageModel with configuration.")
    except Exception as e:
        logging.error(f"Failed to initialize SparseMoELanguageModel: {e}")
        raise e

    # Load the model weights
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logging.info("Loaded model weights from 'model_weights.pth'.")
    except Exception as e:
        logging.error(f"Failed to load model weights from '{weights_path}': {e}")
        raise e

    # Move the model to specified device
    model.to(device)
    logging.info(f"Model moved to device '{device}'.")

    return model

def safe_sample(data, n):
    if len(data) == 0:
        return pd.DataFrame()
    return data.sample(n=min(n, len(data)), random_state=random.randint(0, 10000))

def process_group(group_df, k=5):
    """
    Process a single group of data corresponding to a single Symbol.

    Args:
        group_df (pd.DataFrame): DataFrame group for a single Symbol.
        k (int): Number of preceding articles to include.

    Returns:
        List of tuples containing processed data.
    """
    processed = []
    recent_articles = []  # To keep track of the last k articles

    # Sort the group by Date ascending to ensure chronological order
    group_df = group_df.sort_values(by='Date', ascending=True).reset_index(drop=True)

    for idx, row in group_df.iterrows():
        current_symbol = row.get('Symbol', 'Unknown Symbol')
        current_date = row.get('Date', pd.Timestamp('1970-01-01'))

        # Build concatenated text: current article + up to k previous articles
        current_article = str(row.get('Article', 'N/A'))
        all_articles = [current_article] + recent_articles.copy()
        concatenated_text = "\n----\n".join(all_articles)

        # Append to processed list
        processed.append((
            concatenated_text,
            row.get('weighted_avg_720_hrs', 0.0),
            row.get('Sector', 'Unknown Sector'),
            current_date,
            row.get('RelatedStocksList', ''),
            row.get('weighted_avg_0_hrs', 0.0),
            current_symbol,
            row.get('Industry', 'Unknown Industry'),
            row.get('Risk_Free_Rate', 0.0)
        ))

        # Update recent_articles
        recent_articles.insert(0, current_article)  # Newest first
        if len(recent_articles) > k:
            recent_articles.pop()  # Remove the oldest

    return processed

def process_group_wrapper(args):
    return process_group(*args)

def process_data(df,
                 df_preprocessed,
                 tokenizer,
                 use_ebm_format=False,
                 numeric_only=False):
    """
    Builds final text for each row.
    - If not use_ebm_format, just do your old single-article style.
    - If use_ebm_format, gather references from df_preprocessed (economic etc.)
    """

    articles = []
    prices = []
    sectors = []
    dates = []
    related_stocks_list = []
    prices_current = []
    symbols = []
    industries = []
    risk_free_rates = []

    for i, row in df.iterrows():
        future_price = row.get('weighted_avg_720_hrs', 0.0)
        sector       = row.get('Sector', 'Unknown Sector')
        date_val     = row.get('Date', pd.Timestamp('1970-01-01'))
        old_price    = row.get('weighted_avg_0_hrs', 0.0)
        symbol       = row.get('Symbol', 'Unknown Symbol')
        industry     = row.get('Industry', 'Unknown Industry')
        rfr          = row.get('Risk_Free_Rate', 0.0)
        main_article = str(row.get('Article', 'N/A'))

        if numeric_only:
            final_text = (
                "Symbol: " + str(row.get('Symbol', 'N/A')) +
                "\nSecurity: " + str(row.get('Security', 'N/A')) +
                "\nRelated Stocks/Topics: " + str(row.get('RelatedStocksList', 'N/A')) +
                "\nStock Price 4 days before: " + str(row.get('weighted_avg_-96_hrs', 'N/A')) +
                "\nStock Price 2 days before: " + str(row.get('weighted_avg_-48_hrs', 'N/A')) +
                "\nStock Price 1 day before: " + str(row.get('weighted_avg_-24_hrs', 'N/A')) +
                "\nStock Price at release: " + str(row.get('weighted_avg_0_hrs', 'N/A')) +
                "\nRisk-Free Rate at release: " + str(row.get('Risk_Free_Rate', 'N/A'))
            )
        elif not use_ebm_format:
            # --- OLD STYLE ---
            final_text = (
                "Symbol: " + str(row.get('Symbol', 'N/A')) +
                "\nSecurity: " + str(row.get('Security', 'N/A')) +
                "\nRelated Stocks/Topics: " + str(row.get('RelatedStocksList', 'N/A')) +
                "\nArticle Content: " + str(row.get('Article', 'N/A')) +
                "\nArticle Title: " + str(row.get('Title', 'N/A')) +
                "\nArticle Type: " + str(row.get('articleType', 'N/A')) +
                "\nArticle Publication: " + str(row.get('Publication', 'N/A')) +
                "\nPublication Author: " + str(row.get('Author', 'N/A')) +
                "\nStock Price 4 days before: " + str(row.get('weighted_avg_-96_hrs', 'N/A')) +
                "\nStock Price 2 days before: " + str(row.get('weighted_avg_-48_hrs', 'N/A')) +
                "\nStock Price 1 day before: " + str(row.get('weighted_avg_-24_hrs', 'N/A')) +
                "\nStock Price at release: " + str(row.get('weighted_avg_0_hrs', 'N/A')) +
                "\nRisk-Free Rate at release: " + str(row.get('Risk_Free_Rate', 'N/A'))
            )
        else:
            # --- EBM STYLE: gather from df_preprocessed + top25 DataFrame ---
            preproc_row     = df_preprocessed.iloc[i]

            # Each of these should be a list of indices if your parquet has them as lists
            econ_idxs = preproc_row.get('use_ebm_economic', [])
            ind_idxs  = preproc_row.get('use_ebm_industry', [])
            sec_idxs  = preproc_row.get('use_ebm_sector', [])
            hist_idxs = preproc_row.get('use_ebm_historical', [])

            top25_idxs = preproc_row.get('use_ebm_top25', [])  # Example col name

            def gather_ref_text(idxs, label):
                chunks = []
                for idx_ref in idxs:
                    if 0 <= idx_ref < len(df):
                        refrow = df.iloc[idx_ref]
                        chunks.append(f"[{label}] {refrow.get('Article','N/A')}")
                return "\n".join(chunks)

            economic_part   = gather_ref_text(econ_idxs, "Economic")
            industry_part   = gather_ref_text(ind_idxs, "Industry")
            sector_part     = gather_ref_text(sec_idxs, "Sector")
            historical_part = gather_ref_text(hist_idxs, "Historical")
            top25_part      = gather_ref_text(top25_idxs, "Top25Movers")

            # Main article LAST
            final_text = (
                f"{economic_part}\n\n"
                f"{industry_part}\n\n"
                f"{sector_part}\n\n"
                f"{historical_part}\n\n"
                f"{top25_part}\n\n"
                f"MAIN ARTICLE:\n{main_article}"
            )

        # Gather final
        articles.append(final_text)
        prices.append(future_price)
        sectors.append(sector)
        dates.append(date_val)
        related_stocks_list.append(row.get('RelatedStocksList', ''))
        prices_current.append(old_price)
        symbols.append(symbol)
        industries.append(industry)
        risk_free_rates.append(rfr)

    return (
        articles, prices, sectors, dates,
        related_stocks_list, prices_current,
        symbols, industries, risk_free_rates
    )

def prepare_dataloader(epoch, window_index, tokenizer, batch_size, shuffle,
                       global_offset, global_max, args, sampler=None):
    """
    Builds a DataLoader for a given chunk based on the provided epoch and window_index.
    Uses DistributedSampler when in a torch.distributed job so each rank sees a disjoint slice.
    """
    df = get_data(epoch, window_index, global_offset, global_max, args=args)
    dataset = PrecomputedDataset(df, tokenizer, block_size=config.BLOCK_SIZE)

    # if we're in a distributed setting, replace shuffle with a DistributedSampler
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            seed=args.random_seed
        )
        shuffle = False   # sampler handles shuffling
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True
    )

def prepare_tasks(tokenizer, args, k=3):
    """
    For catastrophic-forgetting tests, pick k random sectors and build
    separate DataLoaders for each sector.
    """
    # Load everything
    df, df_preprocessed = get_data(
        percent_data=args.percent_data,
        run=False,
        update=False,
        args=args
    )

    # Grab unique sectors
    unique_sectors = df['Sector'].dropna().unique()
    selected_sectors = np.random.choice(unique_sectors, size=k, replace=False)

    tasks = []
    for sector in selected_sectors:
        df_task = df[df['Sector'] == sector].copy()
        # For df_preprocessed (row-aligned), subset by the same index
        dfp_task = df_preprocessed.loc[df_task.index].copy()

        loader = prepare_dataloader(
            df_task,
            dfp_task,
            task_top25,        # pass the entire dictionary
            tokenizer,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            args=args
        )
        tasks.append(loader)

    return tasks

def initialize_model(args, device, init_from_scratch=False):
    """
    Initialize the SparseMoELanguageModel either from scratch or by loading from consolidated checkpoint.

    Args:
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): Device to load the model on.
        init_from_scratch (bool): If True, initialize the model from scratch using config.

    Returns:
        model (SparseMoELanguageModel): The initialized model.
        initialized_from_scratch (bool): Flag indicating if the model was initialized from scratch.
    """
    if init_from_scratch:
        logging.info("Initializing model from scratch using configurations from utils/config.py.")
        model_config = {
            'n_embed': config.N_EMBED,
            'n_head': config.N_HEAD,
            'n_layer': config.N_LAYER,
            'block_size': config.BLOCK_SIZE,
            'dropout': config.DROPOUT,
            'num_experts': config.NUM_EXPERTS,
            'top_k': config.TOP_K,
            'tokenizer_name': args.tokenizer_name
        }
        model = SparseMoELanguageModel(**model_config)
        model = model.to(device)
        initialized_from_scratch = True
    else:
        consolidated_path = os.path.join(args.save_dir, "consolidated_final.pth")
        if os.path.exists(consolidated_path):
            logging.info(f"Loading consolidated checkpoint from {consolidated_path}")
            model_config = {
                'n_embed': config.N_EMBED,
                'n_head': config.N_HEAD,
                'n_layer': config.N_LAYER,
                'block_size': config.BLOCK_SIZE,
                'dropout': config.DROPOUT,
                'num_experts': config.NUM_EXPERTS,
                'top_k': config.TOP_K,
                'tokenizer_name': args.tokenizer_name
            }
            model = SparseMoELanguageModel(**model_config)
            state_dict = torch.load(consolidated_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            initialized_from_scratch = False
        else:
            logging.error(f"Consolidated checkpoint not found at {consolidated_path}")
            raise FileNotFoundError(f"Expected consolidated checkpoint at {consolidated_path}")

    return model, initialized_from_scratch

def prepare_optimizer(model, args):
    """
    Prepares optimizer parameter groups:
      - Token embeddings
      - Transformer blocks (layer‑wise decay)
      - Final LayerNorm & LM head
      - Dedicated router group
      - Dedicated EBM optimizer
      - Catch‑all for anything else *except* router & EBM
    Always uses DeepSpeedCPUAdam under ZeRO‑Offload.
    """
    LR_DECAY     = getattr(config, "LR_DECAY", 1.0)
    BASE_LR      = getattr(config, "LEARNING_RATE", 1e-4)
    weight_decay = args.lambda_l2 if getattr(args, 'use_l2', False) else 0.0

    logging.info(f"Preparing optimizer: LR={BASE_LR}, decay={LR_DECAY}, wd={weight_decay}")

    def skip_wd(name):
        return any(tok in name for tok in ("bias", "norm", "ln", "token_embedding_table"))

    optimizers     = {}
    main_groups    = []
    handled_params = set()

    # 1) Token embeddings
    if hasattr(model, 'token_embedding_table'):
        wd_p, no_wd_p = [], []
        for n, p in model.token_embedding_table.named_parameters():
            if not p.requires_grad: continue
            full = f"token_embedding_table.{n}"
            (no_wd_p if skip_wd(full) else wd_p).append(p)
            handled_params.add(p)
        if wd_p:    main_groups.append({'params': wd_p,    'lr': BASE_LR, 'weight_decay': weight_decay})
        if no_wd_p: main_groups.append({'params': no_wd_p, 'lr': BASE_LR, 'weight_decay': 0.0})

    # 2) Transformer blocks (layer‑wise decay)
    blocks = getattr(model, 'blocks', [])
    for idx, block in enumerate(blocks):
        decay = LR_DECAY ** (len(blocks) - 1 - idx)
        lr    = BASE_LR * decay
        wd_p, no_wd_p = [], []
        for n, p in block.named_parameters():
            if not p.requires_grad or p in handled_params: continue
            full = f"blocks.{idx}.{n}"
            (no_wd_p if skip_wd(full) else wd_p).append(p)
            handled_params.add(p)
        if wd_p:    main_groups.append({'params': wd_p,    'lr': lr, 'weight_decay': weight_decay})
        if no_wd_p: main_groups.append({'params': no_wd_p, 'lr': lr, 'weight_decay': 0.0})

    # 3) Final LayerNorm
    if hasattr(model, 'ln_f'):
        wd_p, no_wd_p = [], []
        for n, p in model.ln_f.named_parameters():
            if not p.requires_grad or p in handled_params: continue
            full = f"ln_f.{n}"
            (no_wd_p if skip_wd(full) else wd_p).append(p)
            handled_params.add(p)
        if wd_p:    main_groups.append({'params': wd_p,    'lr': BASE_LR, 'weight_decay': weight_decay})
        if no_wd_p: main_groups.append({'params': no_wd_p, 'lr': BASE_LR, 'weight_decay': 0.0})

    # 4) LM head
    if hasattr(model, 'lm_head'):
        wd_p, no_wd_p = [], []
        for n, p in model.lm_head.named_parameters():
            if not p.requires_grad or p in handled_params: continue
            full = f"lm_head.{n}"
            (no_wd_p if skip_wd(full) else wd_p).append(p)
            handled_params.add(p)
        if wd_p:    main_groups.append({'params': wd_p,    'lr': BASE_LR, 'weight_decay': weight_decay})
        if no_wd_p: main_groups.append({'params': no_wd_p, 'lr': BASE_LR, 'weight_decay': 0.0})

    # 5) Router group
    router_params = []
    if hasattr(model, 'moe') and hasattr(model.moe, 'router'):
        router_params = list(model.moe.router.parameters())
        if router_params:
            mult = getattr(args, "router_lr_mult", 2.0)
            main_groups.append({
                'params': router_params,
                'lr':     BASE_LR * mult,
                'weight_decay': 0.0
            })
            handled_params.update(router_params)
            logging.info(f"Added {len(router_params)} router params at LR×{mult}")

    # Build main optimizer
    optimizers['main'] = DeepSpeedCPUAdam(main_groups)

    # 6) EBM optimizer
    ebm_params = []
    if hasattr(model, 'ebm'):
        ebm_params = [p for p in model.ebm.parameters() if p.requires_grad]
        if ebm_params:
            ebm_lr = getattr(args, "ebm_lr", BASE_LR)
            optimizers['ebm'] = DeepSpeedCPUAdam([{
                'params': ebm_params,
                'lr':     ebm_lr,
                'weight_decay': weight_decay
            }])
            handled_params.update(ebm_params)
            logging.info("Created EBM optimizer")

    # 7) Catch‑all for anything left
    all_trainable = {p for p in model.parameters() if p.requires_grad}
    leftovers     = all_trainable - handled_params
    if leftovers:
        logging.warning(f"{len(leftovers)} params unassigned; adding to default group")
        optimizers['main'].add_param_group({
            'params': list(leftovers),
            'lr':     BASE_LR,
            'weight_decay': weight_decay
        })

    return optimizers

def initialize_si(model, args):
    """
    Initializes Synaptic Intelligence (SI) if specified.
    """
    si = None
    if getattr(args, 'use_si', False):
        si = SynapticIntelligence(model, lambda_si=config.LAMBDA_SI)
        si_state_path = os.path.join(args.save_dir, f'si_state_rank_{dist.get_rank()}.pth') if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1 else os.path.join(args.save_dir, 'si_state.pth')
        if os.path.exists(si_state_path):
            si.load_state(si_state_path)
            logging.info(f"Loaded Synaptic Intelligence (SI) state from '{si_state_path}'.")
        else:
            logging.info("No existing SI state found. Starting fresh SI.")
    return si

def initialize_replay_buffer(args):
    """
    Initializes or loads the replay buffer if specified.
    """
    replay_buffer = None
    if getattr(args, 'use_replay_buffer', False):
        replay_buffer_capacity = args.replay_buffer_capacity
        replay_buffer = MemoryReplayBuffer(capacity=replay_buffer_capacity)
        logging.info(f"Initialized Memory Replay Buffer with capacity {replay_buffer_capacity}.")
        replay_buffer_path = os.path.join(args.save_dir, f'replay_buffer_rank_{dist.get_rank()}.pth') if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1 else os.path.join(args.save_dir, 'replay_buffer.pth')
        if os.path.exists(replay_buffer_path):
            replay_buffer.load(replay_buffer_path)
            logging.info(f"Loaded Memory Replay Buffer from '{replay_buffer_path}'.")
        else:
            logging.info("No existing Memory Replay Buffer found. Starting fresh.")
    return replay_buffer

def save_ebm_model(ebm, epoch, save_dir="models", args=None):
    """
    Saves the EBM model's state dictionary and optionally uploads it to AWS S3.

    Args:
        ebm (torch.nn.Module): The Energy-Based Model to save.
        epoch (int): The current epoch number.
        save_dir (str): Directory where the model will be saved.
        args (argparse.Namespace, optional): Command-line arguments; if args.bucket is provided, the file is uploaded.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ebm.pt")
    torch.save(ebm.state_dict(), save_path)
    print(f"EBM model saved to {save_path}")

    # If a bucket is specified, upload the file to S3
    if hasattr(args, "bucket") and args.bucket:
        try:
            import subprocess
            s3_path = f"s3://{args.bucket}/models/ebm.pt"
            cmd = ["aws", "s3", "cp", save_path, s3_path]
            subprocess.run(cmd, check=True)
            logging.info(f"EBM model uploaded to {s3_path}")
        except Exception as e:
            logging.error(f"Error uploading EBM model to S3: {e}")

def save_model_and_states(model, si, replay_buffer, ewc_list, args):
    """
    Saves the model weights and states of SI, EWC, and Replay Buffer locally and uploads them to S3 if --bucket is provided.

    Args:
        model (nn.Module): The transformer model.
        si (SynapticIntelligence, optional): The SI instance.
        replay_buffer (MemoryReplayBuffer, optional): The replay buffer instance.
        ewc_list (list, optional): List of EWC instances.
        args (argparse.Namespace): Command-line arguments.
    """
    import subprocess
    import os
    import torch.distributed as dist

    if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1:
        rank = dist.get_rank()
    else:
        rank = 0

    # Only rank 0 saves the main model and EWC state
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        # Save model weights
        model_filename = args.save_model_name if args.save_model_name else "model_weights.pth"
        model_path = os.path.join(args.save_dir, model_filename)
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, model_path)
        logging.info(f"Model weights saved to '{model_path}'.")

        # Upload model weights to S3 if bucket is provided
        if hasattr(args, "bucket") and args.bucket:
            try:
                s3_model_path = f"s3://{args.bucket}/model/{model_filename}"
                subprocess.run(["aws", "s3", "cp", model_path, s3_model_path], check=True)
                logging.info(f"Model weights uploaded to '{s3_model_path}'.")
            except Exception as e:
                logging.error(f"Error uploading model weights to S3: {e}")

        # Save EWC state if used
        if getattr(args, 'use_ewc', False) and ewc_list is not None:
            ewc_state_path = os.path.join(args.save_dir, 'ewc_state.pth')
            ewc_states = []
            for ewc_instance in ewc_list:
                ewc_states.append({
                    'params': {n: p.cpu() for n, p in ewc_instance.params.items()},
                    'fisher': {n: f.cpu() for n, f in ewc_instance.fisher.items()}
                })
            torch.save(ewc_states, ewc_state_path)
            logging.info(f"EWC state saved to '{ewc_state_path}'.")

            # Upload EWC state to S3
            if hasattr(args, "bucket") and args.bucket:
                try:
                    s3_ewc_path = f"s3://{args.bucket}/model/ewc_state.pth"
                    subprocess.run(["aws", "s3", "cp", ewc_state_path, s3_ewc_path], check=True)
                    logging.info(f"EWC state uploaded to '{s3_ewc_path}'.")
                except Exception as e:
                    logging.error(f"Error uploading EWC state to S3: {e}")

    # Save SI state for each rank
    if getattr(args, 'use_si', False) and si is not None:
        si_filename = f'si_state_rank_{rank}.pth' if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1 else 'si_state.pth'
        si_state_path = os.path.join(args.save_dir, si_filename)
        si.save_state(si_state_path)
        logging.info(f"Rank {rank}: Synaptic Intelligence (SI) state saved to '{si_state_path}'.")

        # Upload SI state to S3
        if hasattr(args, "bucket") and args.bucket:
            try:
                s3_si_path = f"s3://{args.bucket}/model/{si_filename}"
                subprocess.run(["aws", "s3", "cp", si_state_path, s3_si_path], check=True)
                logging.info(f"Rank {rank}: SI state uploaded to '{s3_si_path}'.")
            except Exception as e:
                logging.error(f"Rank {rank}: Error uploading SI state to S3: {e}")

    # Save Replay Buffer for each rank
    if getattr(args, 'use_replay_buffer', False) and replay_buffer is not None:
        rb_filename = f'replay_buffer_rank_{rank}.pth' if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1 else 'replay_buffer.pth'
        replay_buffer_path = os.path.join(args.save_dir, rb_filename)
        replay_buffer.save(replay_buffer_path)
        logging.info(f"Rank {rank}: Replay Buffer saved to '{replay_buffer_path}'.")

        # Upload Replay Buffer to S3
        if hasattr(args, "bucket") and args.bucket:
            try:
                s3_rb_path = f"s3://{args.bucket}/model/{rb_filename}"
                subprocess.run(["aws", "s3", "cp", replay_buffer_path, s3_rb_path], check=True)
                logging.info(f"Rank {rank}: Replay Buffer uploaded to '{s3_rb_path}'.")
            except Exception as e:
                logging.error(f"Rank {rank}: Error uploading Replay Buffer to S3: {e}")

def compute_l2_loss(model):
    """
    Compute the L2 penalty (squared L2 norm) of the model parameters.

    Args:
        model (nn.Module): The model.

    Returns:
        l2_loss (torch.Tensor): L2 penalty term.
    """
    l2_loss = torch.tensor(0., device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, 2) ** 2
    return l2_loss

def evaluate_model(predictions, actuals, oldprices, riskfree, sectors):
    """
    Evaluates predictions given lists of predictions, actuals, old prices, risk-free rates, and sectors.

    Returns:
        mse, r2, sector_metrics, overall_trend_acc, sharpe_ratio, sortino_ratio,
        average_return, win_rate, profit_factor
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    oldprices = np.array(oldprices)
    riskfree = np.array(riskfree)

    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    true_trends = np.sign(actuals - oldprices)
    pred_trends = np.sign(predictions - oldprices)
    overall_trend_acc = np.mean(true_trends == pred_trends) if predictions.size > 0 else 0.0

    buy_signals = (predictions > oldprices).astype(float)
    strategy_returns = (actuals - oldprices) / (oldprices + 1e-12) * buy_signals
    excess_returns = strategy_returns - riskfree

    sr_mean = np.mean(excess_returns)
    sr_std = np.std(excess_returns, ddof=1)
    sharpe_ratio = sr_mean / sr_std if sr_std > 1e-12 else 0.0

    neg_mask = (excess_returns < 0)
    if np.any(neg_mask):
        downside_std = np.std(excess_returns[neg_mask], ddof=1)
        sortino_ratio = sr_mean / downside_std if downside_std > 1e-12 else float('inf')
    else:
        sortino_ratio = float('inf')

    average_return = np.mean(strategy_returns) if strategy_returns.size > 0 else 0.0
    wins = strategy_returns > 0
    win_rate = np.mean(wins) * 100 if wins.size > 0 else 0.0
    gross_profits = strategy_returns[strategy_returns > 0].sum()
    gross_losses = -strategy_returns[strategy_returns < 0].sum()
    profit_factor = (gross_profits / gross_losses) if gross_losses > 1e-12 else float('inf')

    # Compute per-sector metrics
    sector_metrics = {}
    unique_sectors = np.unique(sectors)
    for sec in unique_sectors:
        indices = [i for i, s in enumerate(sectors) if s == sec]
        if not indices:
            continue
        spreds = predictions[indices]
        sacts = actuals[indices]
        soldp = oldprices[indices]
        sriskf = riskfree[indices]

        sec_mse = mean_squared_error(sacts, spreds)
        sec_r2 = r2_score(sacts, spreds)
        sec_true_trends = np.sign(sacts - soldp)
        sec_pred_trends = np.sign(spreds - soldp)
        sec_trend_acc = np.mean(sec_true_trends == sec_pred_trends) if spreds.size > 0 else 0.0

        sec_signals = (spreds > soldp).astype(float)
        sec_strategy_returns = (sacts - soldp) / (soldp + 1e-12) * sec_signals
        sec_excess_returns = sec_strategy_returns - sriskf

        sec_ex_mean = np.mean(sec_excess_returns)
        sec_ex_std = np.std(sec_excess_returns, ddof=1)
        sec_sharpe = sec_ex_mean / sec_ex_std if sec_ex_std > 1e-12 else 0.0

        neg_mask_s = (sec_excess_returns < 0)
        if np.any(neg_mask_s):
            sec_downside_std = np.std(sec_excess_returns[neg_mask_s], ddof=1)
            sec_sortino = sec_ex_mean / sec_downside_std if sec_downside_std > 1e-12 else float('inf')
        else:
            sec_sortino = float('inf')

        sec_avg_return = np.mean(sec_strategy_returns) if sec_strategy_returns.size > 0 else 0.0
        sec_wins = sec_strategy_returns > 0
        sec_win_rate = np.mean(sec_wins) * 100 if sec_wins.size > 0 else 0.0
        sec_gross_profits = sec_strategy_returns[sec_strategy_returns > 0].sum()
        sec_gross_losses = -sec_strategy_returns[sec_strategy_returns < 0].sum()
        sec_profit_factor = (sec_gross_profits / sec_gross_losses) if sec_gross_losses > 1e-12 else float('inf')

        sector_metrics[sec] = {
            "mse": sec_mse,
            "r2": sec_r2,
            "trend_acc": sec_trend_acc,
            "sharpe": sec_sharpe,
            "sortino": sec_sortino,
            "average_return": sec_avg_return,
            "win_rate": sec_win_rate,
            "profit_factor": sec_profit_factor
        }

    return mse, r2, sector_metrics, overall_trend_acc, sharpe_ratio, sortino_ratio, average_return, win_rate, profit_factor

def load_checkpoint(model, optimizer, ebm=None, ebm_optimizer=None, checkpoint_path=None):
    """
    Load model and optimizer states from a checkpoint.

    Args:
        model (nn.Module): The transformer model.
        optimizer (torch.optim.Optimizer): Optimizer for the transformer model.
        ebm (nn.Module, optional): The Energy-Based Model.
        ebm_optimizer (torch.optim.Optimizer, optional): Optimizer for the EBM.
        checkpoint_path (str, optional): Path to the checkpoint file.

    Returns:
        int: The epoch to resume from.
    """
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Loaded model and optimizer state from {checkpoint_path}")

        if ebm and ebm_optimizer and 'ebm_state_dict' in checkpoint:
            ebm.load_state_dict(checkpoint['ebm_state_dict'])
            ebm_optimizer.load_state_dict(checkpoint['ebm_optimizer_state_dict'])
            logging.info(f"Loaded EBM and EBM optimizer state from {checkpoint_path}")

        epoch = checkpoint.get('epoch', 0)
        return epoch
    else:
        logging.info("No checkpoint found at specified path. Starting from scratch.")
        return 0

def save_model_weights(model, filepath, device):
    """
    Load model weights from the given file path.
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {filepath}.")
    return model

def initialize_si(model, args):
    """
    Initializes Synaptic Intelligence (SI) if specified.
    """
    si = None
    if getattr(args, 'use_si', False):
        si = SynapticIntelligence(model, lambda_si=config.LAMBDA_SI)
        si_state_path = os.path.join(args.save_dir, f'si_state_rank_{dist.get_rank()}.pth') if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1 else os.path.join(args.save_dir, 'si_state.pth')
        if os.path.exists(si_state_path):
            si.load_state(si_state_path)
            logging.info(f"Loaded Synaptic Intelligence (SI) state from '{si_state_path}'.")
        else:
            logging.info("No existing SI state found. Starting fresh SI.")
    return si

def initialize_replay_buffer(args):
    """
    Initializes or loads the replay buffer if specified.
    """
    replay_buffer = None
    if getattr(args, 'use_replay_buffer', False):
        replay_buffer_capacity = args.replay_buffer_capacity
        replay_buffer = MemoryReplayBuffer(capacity=replay_buffer_capacity)
        logging.info(f"Initialized Memory Replay Buffer with capacity {replay_buffer_capacity}.")
        replay_buffer_path = os.path.join(args.save_dir, f'replay_buffer_rank_{dist.get_rank()}.pth') if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1 else os.path.join(args.save_dir, 'replay_buffer.pth')
        if os.path.exists(replay_buffer_path):
            replay_buffer.load(replay_buffer_path)
            logging.info(f"Loaded Memory Replay Buffer from '{replay_buffer_path}'.")
        else:
            logging.info("No existing Memory Replay Buffer found. Starting fresh.")
    return replay_buffer

def consolidate_checkpoint_to_pth(checkpoint_dir: str, tag: str, output_path: str, use_subdir: bool = False) -> str:
    import os
    import glob
    import shutil
    import tempfile
    import logging
    import torch
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

    # If optimizer states are stored in a subdirectory (e.g., checkpoint_dir/tag), update the path accordingly
    base_dir = os.path.join(checkpoint_dir, tag) if use_subdir else checkpoint_dir
    final_path = os.path.join(output_path, f"consolidated_{tag}.pth")

    if os.path.exists(final_path):
        logging.info(f"Found existing consolidated checkpoint at {final_path}")
        return final_path

    # Clean up any existing optim_states files to avoid the "found 4 files" error
    optim_files = glob.glob(os.path.join(base_dir, "*_optim_states.pt"))
    logging.info(f"Found {len(optim_files)} optimizer state files in {base_dir}")

    # Back up original files
    backup_dir = tempfile.mkdtemp(prefix="ds_backup_")
    for file in optim_files:
        shutil.copy(file, backup_dir)
    logging.info(f"Backed up original optimizer files to {backup_dir}")

    # Remove all existing optimizer state files
    for file in optim_files:
        os.remove(file)
    logging.info("Removed existing optimizer state files")

    # Create exactly two empty optimizer state files
    for i in range(2):
        dummy_path = os.path.join(base_dir, f"dummy_{i}_optim_states.pt")
        torch.save({}, dummy_path)
    logging.info("Created two dummy optimizer state files")

    # Perform the conversion using the entire checkpoint_dir (not base_dir) for DeepSpeed conversion
    temp_dir = tempfile.mkdtemp(prefix="ds_conversion_")
    logging.info(f"Converting checkpoint from base_dir='{checkpoint_dir}', tag='{tag}'")

    try:
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, temp_dir, tag=tag)

        # The converted file should be named 'pytorch_model.bin'
        converted_bin = os.path.join(temp_dir, "pytorch_model.bin")
        if not os.path.isfile(converted_bin):
            raise RuntimeError(f"Conversion failed; {converted_bin} not found.")

        shutil.move(converted_bin, final_path)
        logging.info(f"Successfully created consolidated checkpoint at {final_path}")
    except Exception as e:
        logging.error(f"Error during checkpoint conversion: {str(e)}")
        # Restore original files
        for file in glob.glob(os.path.join(base_dir, "*_optim_states.pt")):
            os.remove(file)
        for file in glob.glob(os.path.join(backup_dir, "*")):
            shutil.copy(file, base_dir)
        logging.info("Restored original optimizer files")
        raise
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        shutil.rmtree(backup_dir)

    if not os.path.isfile(final_path):
        raise RuntimeError(f"Failed to create consolidated checkpoint at {final_path}")

    return final_path

def upload_checkpoint_to_s3(local_dir: str, bucket: str, remote_dir: str = "model", region: str = "us-west-1"):
    """
    Recursively upload all files under `local_dir` to s3://<bucket>/<remote_dir>/...
    """
    s3 = boto3.resource("s3", region_name=region)
    bucket_obj = s3.Bucket(bucket)

    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            # build the key by taking the path relative to local_dir
            rel_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(remote_dir, rel_path).replace(os.sep, "/")
            try:
                bucket_obj.upload_file(local_path, s3_key)
                logging.info(f"Uploaded {local_path} → s3://{bucket}/{s3_key}")
            except ClientError as e:
                logging.error(f"Error uploading {local_path}: {e}")
                raise
