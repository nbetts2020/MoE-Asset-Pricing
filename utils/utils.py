# utils/utils.py

import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from utils.config import config
from utils.data import ArticlePriceDataset
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

import torch.distributed as dist

from utils.ebm import EnergyBasedModel, scale_energy, compute_sampling_probabilities

def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

def get_data(percent_data=100.0, run=False, update=False, args=None):
    load_dotenv('/content/MoE-Asset-Pricing/.env')
    hf_token = os.getenv('HF_TOKEN')

    login(hf_token)
    dataset = load_dataset("nbettencourt/SC454k")
    df = dataset['train'].to_pandas().dropna(subset=['weighted_avg_720_hrs'])

    total_samples = len(df)
    num_samples = int((percent_data / 100.0) * total_samples)
    df.sort_values(by='Date', inplace=True)
    df = df[(df['weighted_avg_0_hrs'] > 0) & (df['weighted_avg_720_hrs'] > 0)]
    df = df.head(num_samples)

    safe_div = df['weighted_avg_720_hrs'].replace(0, np.nan)
    df['Percentage Change'] = ((df['weighted_avg_0_hrs'] - safe_div) / safe_div) * 100

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['RelatedStocksList'] = df['RelatedStocksList'].fillna('')

    df.reset_index(drop=True, inplace=True)

    # 70/15/15 split
    split1 = int(len(df) * 0.7)
    split2 = int(len(df) * 0.85)

    df_preprocessed = None
    df_preprocessed_top25 = None

    if args.use_ebm:
        dataset_preprocessed = load_dataset("nbettencourt/SC454k-preprocessed")
        df_preprocessed = dataset_preprocessed['train'].to_pandas().head(453932)
        df_preprocessed = df_preprocessed.head(num_samples)

        dataset_preprocessed_top25 = load_dataset("nbettencourt/SC454k-preprocessed-top25")
        df_preprocessed_top25 = dataset_preprocessed_top25['train'].to_dict()
        df_preprocessed_top25 = df_preprocessed_top25.head(num_samples)

    if run:
        df = df[split1:split2]
        df_preprocessed = df_preprocessed[split1:split2]
        df_preprocessed_top25 = df_preprocessed_top25[split1:split2]
    elif update:
        df = df[split2:]
        df_preprocessed = df_preprocessed[split2:]
        df_preprocessed_top25 = df_preprocessed_top25[split2:]
    else:
        df = df[:split1]
        df_preprocessed = df_preprocessed[:split1]
        df_preprocessed_top25 = df_preprocessed_top25[:split1]
    return df, df_preprocessed, df_preprocessed_top25

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

def ebm_select_contexts(df, stock, date, text, model, ebm, tokenizer, ebm_samples):
    """
    Generates and selects the best context based on EBM scoring.

    Args:
        df (pd.DataFrame): The entire dataset to sample from.
        stock (str): Stock symbol to filter relevant articles.
        date (str): Date to filter relevant articles.
        sample_count (int): Number of samples to generate.
        model (torch.nn.Module): The main transformer model.
        ebm (torch.nn.Module): The Energy-Based Model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.

    Returns:
        str: The selected context string.
    """
    # Filter the dataframe based on stock and date
    filtered_df = df[df['Date'] <= pd.to_datetime(date)]

    if filtered_df.empty:
        raise ValueError("No data available for the specified stock and date.")

    candidates = []
    for _ in range(ebm_samples):
        # sample_articles(...) returns a list of sample_dicts; grab the first
        sampled_list = sample_articles(filtered_df, index_list=None, symbol=stock)
        if not sampled_list:
            continue  # skip if no samples returned
        sample_dict = sampled_list[0]  # typically you get only one item

        # Convert sample_dict into a single string
        context_str = format_concatenated_articles(sample_dict)
        candidates.append(context_str)

    if not candidates:
        raise ValueError("No candidate contexts could be generated.")

    # Score each candidate with EBM
    scores = [] # embed each article and context together???
    device = next(model.parameters()).device  # or e.g. torch.device('cuda')
    for ctx in candidates:
        encoding = tokenizer(
            ctx + text,
            truncation=True,
            padding='max_length',
            max_length=config.BLOCK_SIZE,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            # Get embeddings from the main model
            embeddings = model.get_embeddings(encoding['input_ids'])  # shape: [1, embed_dim]
            # Score with EBM
            score = ebm(embeddings).item()  # single float
        scores.append(score)

    # Pick the best (lowest-scoring) context
    min_idx = scores.index(min(scores))
    best_context = candidates[min_idx]
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
                 df_preprocessed_top25,
                 tokenizer,
                 use_ebm_format=False):
    """
    Builds final text for each row.
    - If not use_ebm_format, just do your old single-article style.
    - If use_ebm_format, gather references from df_preprocessed (economic etc.)
      plus top25 from df_preprocessed_top25, then append the main article last.
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

        if not use_ebm_format:
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
            preproc_row_25  = df_preprocessed_top25.iloc[i] if df_preprocessed_top25 is not None else {}

            # Each of these should be a list of indices if your parquet has them as lists
            econ_idxs = preproc_row.get('use_ebm_economic', [])
            ind_idxs  = preproc_row.get('use_ebm_industry', [])
            sec_idxs  = preproc_row.get('use_ebm_sector', [])
            hist_idxs = preproc_row.get('use_ebm_historical', [])

            top25_idxs = preproc_row_25.get('use_ebm_top25', [])  # Example col name

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

def prepare_dataloader(df,
                       df_preprocessed,
                       df_preprocessed_top25,
                       tokenizer,
                       batch_size,
                       shuffle,
                       args):
    """
    Pass df, df_preprocessed, and df_preprocessed_top25 to process_data.
    Then wrap the resulting data in ArticlePriceDataset -> DataLoader.
    """
    (articles, prices, sectors, dates,
     related_stocks, prices_current,
     symbols, industries, rfr) = process_data(
         df,
         df_preprocessed,
         df_preprocessed_top25,
         tokenizer,
         use_ebm_format=args.use_ebm_format
     )

    dataset = ArticlePriceDataset(
        articles=articles,
        prices=prices,
        sectors=sectors,
        dates=dates,
        related_stocks_list=related_stocks,
        prices_current=prices_current,
        symbols=symbols,
        industries=industries,
        risk_free_rates=rfr,
        tokenizer=tokenizer,
        total_epochs=config.EPOCHS,
        use_ebm=args.use_ebm
    )

    if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def prepare_tasks(tokenizer, args, k=3):
    """
    For catastrophic-forgetting tests, pick k random sectors and build
    separate DataLoaders for each sector.
    """
    # Load everything
    df, df_preprocessed, df_preprocessed_top25 = get_data(
        percent_data=args.percent_data,
        run=False,      # or however you want
        update=False,   # or however you want
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
        
        task_top25 = df_preprocessed_top25  # Pass the full dictionary

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
    Prepares the optimizer with layer-wise learning rate decay and optional weight decay (L2 regularization).
    """
    LR_DECAY = config.LR_DECAY
    LEARNING_RATE = config.LEARNING_RATE
    weight_decay = args.lambda_l2 if getattr(args, 'use_l2', False) else 0.0

    param_groups = []

    # Embedding parameters
    embedding_params_with_decay = []
    embedding_params_without_decay = []

    for name, param in list(model.token_embedding_table.named_parameters()) + list(model.position_embedding_table.named_parameters()):
        if param.requires_grad:
            if name.endswith('bias') or 'LayerNorm.weight' in name:
                embedding_params_without_decay.append(param)
            else:
                embedding_params_with_decay.append(param)

    param_groups.append({
        'params': embedding_params_with_decay,
        'lr': LEARNING_RATE * (LR_DECAY ** (len(model.blocks) + 1)),
        'weight_decay': weight_decay
    })
    param_groups.append({
        'params': embedding_params_without_decay,
        'lr': LEARNING_RATE * (LR_DECAY ** (len(model.blocks) + 1)),
        'weight_decay': 0.0
    })

    # Regression head parameters
    reg_params_with_decay = []
    reg_params_without_decay = []

    for name, param in model.regression_head.named_parameters():
        if param.requires_grad:
            if name.endswith('bias') or 'LayerNorm.weight' in name:
                reg_params_without_decay.append(param)
            else:
                reg_params_with_decay.append(param)

    param_groups.append({
        'params': reg_params_with_decay,
        'lr': LEARNING_RATE,
        'weight_decay': weight_decay
    })
    param_groups.append({
        'params': reg_params_without_decay,
        'lr': LEARNING_RATE,
        'weight_decay': 0.0
    })

    # Block parameters with layer-wise learning rate decay
    for i, block in enumerate(model.blocks):
        block_params_with_decay = []
        block_params_without_decay = []
        for name, param in block.named_parameters():
            if param.requires_grad:
                if name.endswith('bias') or 'LayerNorm.weight' in name:
                    block_params_without_decay.append(param)
                else:
                    block_params_with_decay.append(param)
        lr = LEARNING_RATE * (LR_DECAY ** (len(model.blocks) - i))
        param_groups.append({
            'params': block_params_with_decay,
            'lr': lr,
            'weight_decay': weight_decay
        })
        param_groups.append({
            'params': block_params_without_decay,
            'lr': lr,
            'weight_decay': 0.0
        })

    optimizer = torch.optim.AdamW(param_groups)
    logging.info("Initialized AdamW optimizer with layer-wise learning rate decay and weight decay.")
    return optimizer

def prepare_data(args, tokenizer):
    """
    Loads df, df_preprocessed, and df_preprocessed_top25 from get_data().
    Returns:
        (DataLoader, dict):
          - The DataLoader for the current mode (train/run/update).
          - A dictionary containing 'df', 'df_preprocessed', 'df_preprocessed_top25'.
    """
    # Load data
    df, df_preprocessed, df_preprocessed_top25 = get_data(
        percent_data=args.percent_data,
        run=(args.mode == 'run'),
        update=(args.mode == 'update'),
        args=args
    )

    data_bundle = {
        'df': df,
        'df_preprocessed': df_preprocessed,
        'df_preprocessed_top25': df_preprocessed_top25
    }

    if args.mode == 'train':
        train_loader = prepare_dataloader(
            df,
            df_preprocessed,
            df_preprocessed_top25,
            tokenizer,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            args=args
        )
        return train_loader, data_bundle

    elif args.mode == 'run':
        run_loader = prepare_dataloader(
            df,
            df_preprocessed,
            df_preprocessed_top25,
            tokenizer,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            args=args
        )
        return run_loader, data_bundle

    elif args.mode == 'update':
        update_loader = prepare_dataloader(
            df,
            df_preprocessed,
            df_preprocessed_top25,
            tokenizer,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            args=args
        )
        # Optionally combine with old training data
        old_train_df, _, _ = get_data(percent_data=args.percent_data, run=False, update=False, args=args)
        combined_df = pd.concat([old_train_df, df])
        data_bundle['df'] = combined_df  # Update 'df' in the bundle
        return update_loader, data_bundle

    else:
        raise ValueError(f"Invalid mode: {args.mode}")

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

def save_ebm_model(ebm, epoch, save_dir="models"):
    """
    Saves the EBM model's state dictionary.

    Args:
        ebm (torch.nn.Module): The Energy-Based Model to save.
        epoch (int): The current epoch number.
        save_dir (str): Directory where the model will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    save_path = os.path.join(save_dir, f"ebm.pt")
    torch.save(ebm.state_dict(), save_path)
    print(f"EBM model saved to {save_path}")

def save_model_and_states(model, si, replay_buffer, ewc_list, args):
    """
    Saves the model weights and states of SI, EWC, and Replay Buffer.

    Args:
        model (nn.Module): The transformer model.
        si (SynapticIntelligence, optional): The SI instance.
        replay_buffer (MemoryReplayBuffer, optional): The replay buffer instance.
        ewc_list (list, optional): List of EWC instances.
        args (argparse.Namespace): Command-line arguments.
    """
    if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1:
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        # Save model weights
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(args.save_dir, args.save_model_name if args.save_model_name else "model_weights.pth"))
        logging.info(f"Model weights saved to '{os.path.join(args.save_dir, args.save_model_name if args.save_model_name else "model_weights.pth")}'.")

        # Save EWC state
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

    # Save SI state for each rank
    if getattr(args, 'use_si', False) and si is not None:
        si_state_path = os.path.join(args.save_dir, f'si_state_rank_{rank}.pth') if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1 else os.path.join(args.save_dir, 'si_state.pth')
        si.save_state(si_state_path)
        logging.info(f"Rank {rank}: Synaptic Intelligence (SI) state saved to '{si_state_path}'.")

    # Save Replay Buffer for each rank
    if getattr(args, 'use_replay_buffer', False) and replay_buffer is not None:
        replay_buffer_path = os.path.join(args.save_dir, f'replay_buffer_rank_{rank}.pth') if getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1 else os.path.join(args.save_dir, 'replay_buffer.pth')
        replay_buffer.save(replay_buffer_path)
        logging.info(f"Rank {rank}: Replay Buffer saved to '{replay_buffer_path}'.")

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

    NOTE: For meaningful evaluation, ensure the dataset is chronologically sorted if
          the strategy depends on order. The same caution applies at the sector level.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the validation/test set. Must:
          - yield 'risk_free_rate' if computing Sharpe/Sortino Ratio
          - be chronologically ordered if the strategy depends on time series order
        device (torch.device): Device to perform computations on.

    Returns:
        mse (float): Mean Squared Error (overall).
        r2 (float): R² Score (overall).
        sector_metrics (dict): sector -> {
            'mse', 'r2', 'trend_acc', 'sharpe', 'sortino',
            'average_return', 'win_rate', 'profit_factor'
        }
        overall_trend_acc (float): Overall trend accuracy across all samples.
        sharpe_ratio (float): Overall Sharpe Ratio.
        sortino_ratio (float): Overall Sortino Ratio.
        average_return (float): Average return per trade (overall).
        win_rate (float): Percentage of profitable trades (overall).
        profit_factor (float): Ratio of gross profits to gross losses (overall).
    """

    model.eval()
    predictions = []
    actuals = []
    oldprice_list = []
    riskfree_list = []
    sectors_list = []

    # sector -> dict for sector-level time series
    sector_data = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids       = batch['input_ids'].to(device)
            labels          = batch['labels'].to(device)       # Future price
            old_prices      = batch['old_price'].to(device)    # Current price
            sectors         = batch['sector']
            risk_free_rate  = batch['risk_free_rate'].to(device)

            with torch.amp.autocast(device_type='cuda'):
                outputs, _ = model(input_ids=input_ids)

            # Move data to CPU/numpy
            outputs_np     = outputs.detach().cpu().numpy().flatten()
            labels_np      = labels.detach().cpu().numpy()
            old_prices_np  = old_prices.detach().cpu().numpy()
            rf_monthly_np  = risk_free_rate.detach().cpu().numpy()

            # Collect overall
            predictions.extend(outputs_np)
            actuals.extend(labels_np)
            oldprice_list.extend(old_prices_np)
            riskfree_list.extend(rf_monthly_np)
            sectors_list.extend(sectors)

            # Accumulate sector-level data
            for i, sector in enumerate(sectors):
                if sector not in sector_data:
                    sector_data[sector] = {
                        'predictions': [],
                        'actuals':     [],
                        'oldprices':   [],
                        'riskfree':    []
                    }
                sector_data[sector]['predictions'].append(outputs_np[i])
                sector_data[sector]['actuals'].append(labels_np[i])
                sector_data[sector]['oldprices'].append(old_prices_np[i])
                sector_data[sector]['riskfree'].append(rf_monthly_np[i])

    # --------------------- Model Performance (Overall) ---------------------
    mse = mean_squared_error(actuals, predictions)
    r2  = r2_score(actuals, predictions)

    # Trend Accuracy (overall)
    true_trends = np.sign(np.array(actuals) - np.array(oldprice_list))
    pred_trends = np.sign(np.array(predictions) - np.array(oldprice_list))
    overall_trend_acc = np.mean(true_trends == pred_trends) if len(predictions) > 0 else 0.0

    # Strategy returns (overall)
    oldp_arr = np.array(oldprice_list)
    pred_arr = np.array(predictions)
    actl_arr = np.array(actuals)
    rf_arr   = np.array(riskfree_list)

    # Buy signals if predicted price > current price
    buy_signals = (pred_arr > oldp_arr).astype(float)
    # Realized returns if we buy
    strategy_returns = (actl_arr - oldp_arr) / (oldp_arr + 1e-12) * buy_signals

    # Excess returns (for Sharpe/Sortino)
    excess_returns = strategy_returns - rf_arr

    # --------------------- Sharpe Ratio (overall) ---------------------
    sr_mean = np.mean(excess_returns)
    sr_std  = np.std(excess_returns, ddof=1)
    sharpe_ratio = sr_mean / sr_std if sr_std > 1e-12 else 0.0

    # --------------------- Sortino Ratio (overall) ---------------------
    neg_mask = (excess_returns < 0)
    if np.any(neg_mask):
        downside_std = np.std(excess_returns[neg_mask], ddof=1)
        sortino_ratio = sr_mean / downside_std if downside_std > 1e-12 else float('inf')
    else:
        sortino_ratio = float('inf')

    # --------------------- Average Return, Win Rate, Profit Factor (overall) ---------------------
    # Average Return per trade
    average_return = np.mean(strategy_returns) if len(strategy_returns) > 0 else 0.0

    # Win Rate => ratio of trades with strategy_returns > 0
    wins = strategy_returns > 0
    win_rate = np.mean(wins) * 100 if len(wins) > 0 else 0.0

    # Profit Factor => ratio of gross profits to gross losses
    gross_profits = strategy_returns[strategy_returns > 0].sum()
    gross_losses  = -strategy_returns[strategy_returns < 0].sum()  # negative of negative returns
    profit_factor = (gross_profits / gross_losses) if gross_losses > 1e-12 else float('inf')

    # ------------------- Sector-level metrics -------------------
    sector_metrics = {}
    for sector, vals in sector_data.items():
        spreds  = np.array(vals['predictions'], dtype=np.float64)
        sacts   = np.array(vals['actuals'],     dtype=np.float64)
        soldp   = np.array(vals['oldprices'],   dtype=np.float64)
        sriskf  = np.array(vals['riskfree'],    dtype=np.float64)

        # MSE & R2
        sec_mse = mean_squared_error(sacts, spreds)
        sec_r2  = r2_score(sacts, spreds)

        # Trend accuracy
        sec_true_trends = np.sign(sacts - soldp)
        sec_pred_trends = np.sign(spreds - soldp)
        sec_trend_acc   = np.mean(sec_true_trends == sec_pred_trends) if len(spreds) > 0 else 0.0

        # Buy signals if spreds > soldp
        sec_signals = (spreds > soldp).astype(float)
        # Strategy returns if we buy
        sec_strategy_returns = (sacts - soldp) / (soldp + 1e-12) * sec_signals

        # Excess returns for sector
        sec_excess_returns = sec_strategy_returns - sriskf

        # Sharpe (sector)
        sec_ex_mean = np.mean(sec_excess_returns)
        sec_ex_std  = np.std(sec_excess_returns, ddof=1)
        sec_sharpe  = sec_ex_mean / sec_ex_std if sec_ex_std > 1e-12 else 0.0

        # Sortino (sector)
        neg_mask_s = (sec_excess_returns < 0)
        if np.any(neg_mask_s):
            sec_downside_std = np.std(sec_excess_returns[neg_mask_s], ddof=1)
            sec_sortino = sec_ex_mean / sec_downside_std if sec_downside_std > 1e-12 else float('inf')
        else:
            sec_sortino = float('inf')

        # Average Return (sector)
        sec_avg_return = np.mean(sec_strategy_returns) if len(sec_strategy_returns) > 0 else 0.0

        # Win Rate (sector)
        sec_wins = sec_strategy_returns > 0
        sec_win_rate = np.mean(sec_wins) * 100 if len(sec_wins) > 0 else 0.0

        # Profit Factor (sector)
        sec_gross_profits = sec_strategy_returns[sec_strategy_returns > 0].sum()
        sec_gross_losses  = -sec_strategy_returns[sec_strategy_returns < 0].sum()
        sec_profit_factor = (sec_gross_profits / sec_gross_losses) if sec_gross_losses > 1e-12 else float('inf')

        # Store in sector_metrics
        sector_metrics[sector] = {
            'mse':           sec_mse,
            'r2':            sec_r2,
            'trend_acc':     sec_trend_acc,
            'sharpe':        sec_sharpe,
            'sortino':       sec_sortino,
            'average_return':sec_avg_return,
            'win_rate':      sec_win_rate,
            'profit_factor': sec_profit_factor
        }

    model.train()
    return (
        mse,             # 0
        r2,              # 1
        sector_metrics,  # 2
        overall_trend_acc,  # 3
        sharpe_ratio,    # 4
        sortino_ratio,   # 5
        average_return,  # 6
        win_rate,        # 7
        profit_factor    # 8
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
