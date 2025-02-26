# utils/utils.py

import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from muon import Muon

import pandas as pd
import pyarrow.parquet as pq

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from utils.config import config
from utils.data import ArticlePriceDataset, custom_collate_fn, build_candidate_context_tokens
from utils.model import SparseMoELanguageModel

import os
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv

from tqdm import tqdm
from transformers import AutoTokenizer

import json
from huggingface_hub import hf_hub_download

from utils.ewc import ElasticWeightConsolidation
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer

from torch.utils.data.distributed import DistributedSampler

from utils.sampling import sample_articles
from utils.data import format_concatenated_articles

import logging
import subprocess

import inspect
import numpy as np

from multiprocessing import Pool, cpu_count
import random
import ast
from pandarallel import pandarallel

import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam

from utils.ebm import EnergyBasedModel, scale_energy, compute_sampling_probabilities

logger = logging.getLogger(__name__)

def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

def ensure_local_dataset_files():
    """
    Checks if the main and preprocessed parquet files exist in the local "data" directory.
    If not, downloads them from Hugging Face using the correct repo type and saves them locally.
    Returns the local paths for both files.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    main_file = os.path.join(data_dir, "SC454k.parquet")
    preprocessed_file = os.path.join(data_dir, "SC454k-preprocessed.parquet")

    if not os.path.exists(main_file):
        logger.info("Main parquet file not found locally. Downloading from Hugging Face...")
        main_file = hf_hub_download(
            repo_id="nbettencourt/SC454k",
            filename="SC454k_cleaned.parquet",
            repo_type="dataset",
            cache_dir=data_dir
        )
    else:
        logger.info("Found main parquet file locally.")

    if not os.path.exists(preprocessed_file):
        logger.info("Preprocessed parquet file not found locally. Downloading from Hugging Face...")
        preprocessed_file = hf_hub_download(
            repo_id="nbettencourt/SC454k-preprocessed",
            filename="sc454k-preprocessed-updated.parquet",
            repo_type="dataset",
            cache_dir=data_dir
        )
    else:
        logger.info("Found preprocessed parquet file locally.")

    return main_file, preprocessed_file

def get_total_rows(parquet_path):
    # Read only the metadata (no columns) to get the number of rows
    table = pq.read_table(parquet_path, columns=[])
    return table.num_rows
    
def get_data(epoch, window_index, global_offset, global_max, args=None):
    """
    Loads a chunk of the train data (main only) from Hugging Face, building the file name
    as "train_dataset_{window_index}{epoch_letter}.parquet". Uses global_offset/global_max
    to limit how many rows are returned (partial data logic).

    Args:
        epoch (int): 1 -> 'a', 2 -> 'b', etc.
        window_index (int): 1..13
        global_offset (int): number of rows used so far
        global_max (int): total row limit
        args: optional arguments for e.g. filters

    Returns:
        df: DataFrame with up to (global_max - global_offset) rows from that chunk
    """
    # Convert epoch to a letter (1->'a', 2->'b', etc.)
    epoch_letter = chr(ord('a') + epoch - 1)
    filename = f"train_dataset_{window_index}{epoch_letter}.parquet"

    repo_id = "nbettencourt/sc454k-preprocessed-dfs"
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Load data
    df = pd.read_parquet(file_path).dropna(subset=['weighted_avg_720_hrs'])
    df = df[(df['weighted_avg_0_hrs'] > 0) & (df['weighted_avg_720_hrs'] > 0)]

    # If we've already hit the global_max, return empty
    if global_offset >= global_max:
        return pd.DataFrame()

    chunk_rows = len(df)
    # If adding this chunk would exceed the global_max, truncate
    if global_offset + chunk_rows > global_max:
        needed = global_max - global_offset
        df = df.iloc[:needed]

    return df
    
def get_new_data(new_data_url):
    load_dotenv('/content/MoE-Asset-Pricing/.env')
    hf_token = os.getenv('HF_TOKEN')

    login(hf_token)
    dataset = load_dataset(new_data_url)
    df = dataset['test'].to_pandas()
    return df

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
    Downloads the 'model' and 'models' directories from the specified S3 bucket.

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

def ebm_select_contexts(df, idx, text, model, ebm, tokenizer, ebm_samples):
    """
    For the sample at the given index, select the best context from the precomputed candidate columns.
    It assumes candidate context strings are stored in columns "iteration_1_text" ... "iteration_30_text".
    
    Args:
        df (pd.DataFrame): DataFrame containing the test samples with candidate context columns.
        idx (int): Index of the target sample in df.
        text (str): (Unused) The article text.
        model (torch.nn.Module): The main transformer model.
        ebm (torch.nn.Module): The Energy-Based Model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        ebm_samples (int): Number of candidate contexts to sample (max 30).
        
    Returns:
        str: The candidate context with energy closest to zero.
    """
    row = df.iloc[idx]
    # Extract candidate texts from columns iteration_1_text through iteration_30_text.
    candidates = [row.get(f"iteration_{i}_text") for i in range(1, 31) if row.get(f"iteration_{i}_text") is not None]
    if not candidates:
        raise ValueError("No candidate contexts found for this sample.")
    
    # Sample candidates if there are more than ebm_samples available.
    if len(candidates) > ebm_samples:
        sampled_candidates = random.sample(candidates, ebm_samples)
    else:
        sampled_candidates = candidates

    candidate_energies = []
    device = next(model.parameters()).device
    for candidate in sampled_candidates:
        encoding = tokenizer(
            candidate,
            truncation=True,
            padding="max_length",
            max_length=config.BLOCK_SIZE,
            return_tensors="pt"
        ).to(device)
        emb = model.get_embeddings(encoding["input_ids"]).half()  # (1, embedding_dim)
        energy = ebm(emb).item()  # (1,) -> scalar
        candidate_energies.append(energy)
    
    # Select the candidate whose energy (absolute value) is closest to zero.
    best_idx = np.argmin([abs(e) for e in candidate_energies])
    best_context = sampled_candidates[best_idx]
    return best_context
    
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

    # Initialize the model with the configuration
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

def prepare_dataloader(df, df_preprocessed, tokenizer, batch_size, shuffle, args, sampler=None):
    """
    Prepare dataloader with optional distributed sampler support
    """
    from utils.data import ArticlePriceDataset, custom_collate_fn
    from torch.utils.data import DataLoader

    dataset = ArticlePriceDataset(
        df=df,
        df_preprocessed=df_preprocessed,
        tokenizer=tokenizer,
        block_size=config.BLOCK_SIZE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),  # Only shuffle if no sampler
        sampler=sampler,
        num_workers=0,  # Set to 0 for distributed training
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    return dataloader

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
    Initialize the SparseMoELanguageModel either from scratch or by loading from Hugging Face or local weights.

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
        return model, initialized_from_scratch
    else:
        model = None
        initialized_from_scratch = False
        if hasattr(args, 'model') and args.model:
            # Attempt to load from Hugging Face
            try:
                logging.info(f"Attempting to load model from Hugging Face repository '{args.model}'.")
                model = get_model_from_hf(args.model, device)
                logging.info("Successfully loaded model from Hugging Face.")
                return model, initialized_from_scratch
            except Exception as e:
                logging.warning(f"Failed to load model from Hugging Face: {e}")
                logging.info("Attempting to load model from local 'model/model_weights.pth'.")
        else:
            logging.info("No Hugging Face model specified. Attempting to load model from local 'model/model_weights.pth'.")

        # Attempt to load from local weights
        try:
            local_config_path = os.path.join(args.save_dir, 'config.json')
            if not os.path.exists(local_config_path):
                raise FileNotFoundError(f"Local config file '{local_config_path}' not found.")
            with open(local_config_path, 'r') as f:
                local_config = json.load(f)
            logging.info(f"Loaded local model configuration from '{local_config_path}'.")

            expected_args = inspect.getfullargspec(SparseMoELanguageModel.__init__).args
            if 'self' in expected_args:
                expected_args.remove('self')
            model_config = {k: config[k] for k in expected_args if k in config}
            logging.info(f"Filtered model configuration: {model_config}")

            model = SparseMoELanguageModel(**model_config)
            model = model.to(device)
            model_weights_path = os.path.join(args.save_dir, 'model_weights.pth')
            model = load_model_weights(model, model_weights_path, device)
            logging.info(f"Successfully loaded model from local '{model_weights_path}'.")
            return model, initialized_from_scratch
        except Exception as e:
            logging.error(f"Failed to load model from local '{local_config_path}': {e}")
            logging.error("Could not load model from Hugging Face or local path.")
            raise RuntimeError("Could not load model from Hugging Face or local path.")

def prepare_optimizer(model, args):
    """
    Prepares a hybrid optimizer: 
      - Uses AdamW (or DeepSpeed’s CPUAdam if desired) for low-dimensional parameters (embeddings, regression head, etc.)
      - Uses Muon for high-dimensional parameters from the transformer blocks.
    """
    LR_DECAY = config.LR_DECAY
    LEARNING_RATE = config.LEARNING_RATE
    weight_decay = args.lambda_l2 if getattr(args, 'use_l2', False) else 0.0

    adam_param_groups = []
    muon_params = []

    # Embedding parameters (optimize with AdamW)
    embedding_params_with_decay = []
    embedding_params_without_decay = []
    for name, param in list(model.token_embedding_table.named_parameters()) + list(model.position_embedding_table.named_parameters()):
        if param.requires_grad:
            if name.endswith('bias') or 'LayerNorm.weight' in name:
                embedding_params_without_decay.append(param)
            else:
                embedding_params_with_decay.append(param)
    adam_param_groups.append({
        'params': embedding_params_with_decay,
        'lr': LEARNING_RATE * (LR_DECAY ** (len(model.blocks) + 1)),
        'weight_decay': weight_decay
    })
    adam_param_groups.append({
        'params': embedding_params_without_decay,
        'lr': LEARNING_RATE * (LR_DECAY ** (len(model.blocks) + 1)),
        'weight_decay': 0.0
    })

    # Regression head parameters (optimize with AdamW)
    reg_params_with_decay = []
    reg_params_without_decay = []
    for name, param in model.regression_head.named_parameters():
        if param.requires_grad:
            if name.endswith('bias') or 'LayerNorm.weight' in name:
                reg_params_without_decay.append(param)
            else:
                reg_params_with_decay.append(param)
    adam_param_groups.append({
        'params': reg_params_with_decay,
        'lr': LEARNING_RATE,
        'weight_decay': weight_decay
    })
    adam_param_groups.append({
        'params': reg_params_without_decay,
        'lr': LEARNING_RATE,
        'weight_decay': 0.0
    })

    # Block parameters (optimize with Muon)
    # Assume all parameters in model.blocks are high-dimensional.
    for block in model.blocks:
        for name, param in block.named_parameters():
            if param.requires_grad:
                muon_params.append(param)

    # Create the AdamW optimizer for the non-block parameters.
    # (You could replace torch.optim.AdamW with DeepSpeedCPUAdam if desired.)
    adam_optimizer = torch.optim.AdamW(
        [p for group in adam_param_groups for p in group['params']],
        lr=LEARNING_RATE,
        weight_decay=weight_decay
    )

    # Create the Muon optimizer for block parameters.
    muon_optimizer = Muon(muon_params, lr=LEARNING_RATE, momentum=0.95)

    logging.info("Initialized hybrid optimizer: Muon for block parameters and AdamW for embeddings and head.")
    return adam_optimizer, muon_optimizer
    
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

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on a validation/test set and compute metrics, including:
      - MSE, R²
      - Trend Accuracy
      - Sharpe Ratio
      - Sortino Ratio
      - Average Return
      - Win Rate
      - Profit Factor

    Returns:
        mse (float): Mean Squared Error (overall).
        r2 (float): R² Score (overall).
        sector_metrics (dict): sector -> { ... metrics ... } computed on merged data.
        overall_trend_acc (float): Overall trend accuracy.
        sharpe_ratio (float): Overall Sharpe Ratio.
        sortino_ratio (float): Overall Sortino Ratio.
        average_return (float): Average return per trade.
        win_rate (float): Win rate (percentage).
        profit_factor (float): Profit factor.
    """
    import torch.distributed as dist
    from sklearn.metrics import mean_squared_error, r2_score
    from tqdm import tqdm
    model.eval()

    # Initialize overall accumulators.
    local_overall = {
        "predictions": [],
        "actuals": [],
        "oldprices": [],
        "riskfree": []
    }
    sectors_list = []
    local_sector_data = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            old_prices = batch['old_price'].to(device)
            sectors = batch['sector']  # assume sectors come as a list
            risk_free_rate = batch['risk_free_rate'].to(device)

            with torch.amp.autocast(device_type='cuda'):
                outputs, _ = model(input_ids=input_ids)

            outputs_np = outputs.detach().cpu().numpy().flatten()
            labels_np = labels.detach().cpu().numpy()
            old_prices_np = old_prices.detach().cpu().numpy()
            rf_np = risk_free_rate.detach().cpu().numpy()

            local_overall["predictions"].extend(outputs_np.tolist())
            local_overall["actuals"].extend(labels_np.tolist())
            local_overall["oldprices"].extend(old_prices_np.tolist())
            local_overall["riskfree"].extend(rf_np.tolist())
            sectors_list.extend(sectors)

            for i, sector in enumerate(sectors):
                if sector not in local_sector_data:
                    local_sector_data[sector] = {
                        "predictions": [],
                        "actuals": [],
                        "oldprices": [],
                        "riskfree": []
                    }
                local_sector_data[sector]["predictions"].append(outputs_np[i])
                local_sector_data[sector]["actuals"].append(labels_np[i])
                local_sector_data[sector]["oldprices"].append(old_prices_np[i])
                local_sector_data[sector]["riskfree"].append(rf_np[i])

    # Merge overall data from all GPUs if using DDP.
    if dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_overall = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_overall, local_overall)
        merged_overall = {"predictions": [], "actuals": [], "oldprices": [], "riskfree": []}
        for d in gathered_overall:
            merged_overall["predictions"].extend(d["predictions"])
            merged_overall["actuals"].extend(d["actuals"])
            merged_overall["oldprices"].extend(d["oldprices"])
            merged_overall["riskfree"].extend(d["riskfree"])
    else:
        merged_overall = local_overall

    # Compute overall metrics.
    predictions = np.array(merged_overall["predictions"])
    actuals = np.array(merged_overall["actuals"])
    oldprice_arr = np.array(merged_overall["oldprices"])
    rf_arr = np.array(merged_overall["riskfree"])

    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    true_trends = np.sign(actuals - oldprice_arr)
    pred_trends = np.sign(predictions - oldprice_arr)
    overall_trend_acc = np.mean(true_trends == pred_trends) if predictions.size > 0 else 0.0

    buy_signals = (predictions > oldprice_arr).astype(float)
    strategy_returns = (actuals - oldprice_arr) / (oldprice_arr + 1e-12) * buy_signals
    excess_returns = strategy_returns - rf_arr

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

    # Merge per-sector data across GPUs.
    if dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_sector_data = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_sector_data, local_sector_data)
        merged_sector_data = {}
        for sector_data in gathered_sector_data:
            for sector, data in sector_data.items():
                if sector not in merged_sector_data:
                    merged_sector_data[sector] = {"predictions": [], "actuals": [], "oldprices": [], "riskfree": []}
                merged_sector_data[sector]["predictions"].extend(data["predictions"])
                merged_sector_data[sector]["actuals"].extend(data["actuals"])
                merged_sector_data[sector]["oldprices"].extend(data["oldprices"])
                merged_sector_data[sector]["riskfree"].extend(data["riskfree"])
    else:
        merged_sector_data = local_sector_data

    sector_metrics = {}
    for sector, vals in merged_sector_data.items():
        spreds = np.array(vals["predictions"], dtype=np.float64)
        sacts = np.array(vals["actuals"], dtype=np.float64)
        soldp = np.array(vals["oldprices"], dtype=np.float64)
        sriskf = np.array(vals["riskfree"], dtype=np.float64)

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

        sector_metrics[sector] = {
            "mse": sec_mse,
            "r2": sec_r2,
            "trend_acc": sec_trend_acc,
            "sharpe": sec_sharpe,
            "sortino": sec_sortino,
            "average_return": sec_avg_return,
            "win_rate": sec_win_rate,
            "profit_factor": sec_profit_factor
        }

    model.train()
    return (
        mse,             # Mean Squared Error
        r2,              # R² Score
        sector_metrics,  # Per-sector metrics
        overall_trend_acc,  # Overall trend accuracy
        sharpe_ratio,    # Sharpe Ratio
        sortino_ratio,   # Sortino Ratio
        average_return,  # Average Return
        win_rate,        # Win Rate
        profit_factor    # Profit Factor
    )

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

def consolidate_checkpoint_to_pth(checkpoint_dir: str, tag: str, output_path: str) -> str:
    import shutil
    import tempfile
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

    final_path = os.path.join(output_path, f"consolidated_{tag}.pth")
    if os.path.exists(final_path):
        if os.path.isdir(final_path):
            shutil.rmtree(final_path)
        else:
            os.remove(final_path)

    temp_dir = tempfile.mkdtemp(prefix="ds_conversion_")
    logging.info(f"Converting checkpoint from base_dir='{checkpoint_dir}', tag='{tag}'")
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir,
        temp_dir,
        tag=tag
    )

    converted_bin = os.path.join(temp_dir, "pytorch_model.bin")
    if not os.path.isfile(converted_bin):
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Conversion failed; {converted_bin} not found.")

    shutil.move(converted_bin, final_path)
    shutil.rmtree(temp_dir)

    if not os.path.isfile(final_path):
        raise RuntimeError(f"Failed to create consolidated checkpoint file at {final_path}")

    logging.info(f"Successfully created consolidated checkpoint at {final_path}")
    return final_path

def upload_checkpoint_to_s3(local_dir: str, bucket: str, remote_dir: str = "model"):
    try:
        logging.info(f"Uploading checkpoint from {local_dir} to s3://{bucket}/{remote_dir}")
        result = subprocess.run(
            ["aws", "s3", "sync", local_dir, f"s3://{bucket}/{remote_dir}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"Checkpoint directory synchronized successfully.\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error syncing checkpoint directory: {e.stderr}")
        raise

