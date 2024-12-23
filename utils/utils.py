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

import torch.distributed as dist

from utils.ebm import EnergyBasedModel, scale_energy, compute_sampling_probabilities

def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

def get_data(percent_data=100.0, run=False, update=False):
    load_dotenv('/content/MoE-Asset-Pricing/.env')
    hf_token = os.getenv('HF_TOKEN')

    login(hf_token)
    dataset = load_dataset("nbettencourt/SC454k")
    df = dataset['train'].to_pandas().dropna(subset=['weighted_avg_720_hrs'])

    total_samples = len(df)
    num_samples = int((percent_data / 100.0) * total_samples)
    df.sort_values(by='Date', inplace=True)
    df = df.head(num_samples)

    safe_div = df['weighted_avg_720_hrs'].replace(0, np.nan)
    df['Percentage Change'] = ((df['weighted_avg_0_hrs'] - safe_div) / safe_div) * 100

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['RelatedStocksList'] = df['RelatedStocksList'].fillna('')

    df.reset_index(drop=True, inplace=True)

    split1 = int(len(df) * 0.8)
    split2 = int(len(df) * 0.9)

    if run:
        df = df[split1:split2]
    elif update:
        df = df[split2:]
    else:
        df = df[:split1]
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

def process_data(df, tokenizer):
    articles = []
    prices = []
    sectors = []
    dates = []
    related_stocks_list = []
    prices_current = []
    symbols = []
    industries = []

    grouped = df.groupby('Symbol', sort=False)

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        current_symbol = row.get('Symbol', 'Unknown Symbol')
        current_date = row.get('Date', pd.Timestamp('1970-01-01'))
        symbols.append(current_symbol)
        industries.append(row.get('Industry', 'Unknown Industry'))

        # Build the concatenated text using .get() with default values
        concatenated_text = (
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
            "\nStock Price at release: " + str(row.get('weighted_avg_0_hrs', 'N/A'))
        )

        articles.append(concatenated_text)
        prices.append(row.get('weighted_avg_720_hrs', 0.0))
        sectors.append(row.get('Sector', 'Unknown Sector'))
        dates.append(current_date)
        related_stocks_list.append(row.get('RelatedStocksList', ''))
        prices_current.append(row.get('weighted_avg_0_hrs', 0.0))

    return articles, prices, sectors, dates, related_stocks_list, prices_current, symbols, industries

def prepare_dataloader(df, tokenizer, batch_size=config.BATCH_SIZE, shuffle=True, args=None):
    articles, prices, sectors, dates, related_stocks_list, prices_current, symbols, industries = process_data(df, tokenizer)
    dataset = ArticlePriceDataset(
        articles, prices, sectors, dates, related_stocks_list, prices_current, symbols, industries, tokenizer, config.EPOCHS, use_ebm=args.use_ebm if args else False
    )

    if args and getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def prepare_tasks(tokenizer, args, k=3):
    """
    Prepare multiple tasks (DataLoaders) for testing catastrophic forgetting based on sectors.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use.
        args (argparse.Namespace): Command-line arguments.
        k (int): Number of sectors to randomly select for the tasks.

    Returns:
        tasks (list): List of DataLoaders for the selected sectors.
    """
    df = get_data(percent_data=args.percent_data)  # load your data

    # Get unique sectors from the dataset
    unique_sectors = df['Sector'].unique()

    # Randomly sample k sectors from the unique sectors
    selected_sectors = np.random.choice(unique_sectors, size=k, replace=False)

    tasks = []

    # For each selected sector, create a DataLoader
    for sector in selected_sectors:
        df_task = df[df['Sector'] == sector]  # Filter data by the selected sector
        dataloader = prepare_dataloader(df_task, tokenizer, batch_size=config.BATCH_SIZE, shuffle=True, args=args)
        tasks.append(dataloader)

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
    Prepares the data for training or updating.
    """
    percent_data = args.percent_data  # get percentage of data to use
    
    random_seed = args.random_seed

    if args.mode == 'train':
        df = get_data(percent_data=percent_data)

        train_dataloader = prepare_dataloader(
            df, tokenizer, batch_size=config.BATCH_SIZE, shuffle=True, args=args
        )
        logging.info(f"Prepared DataLoader with {len(train_dataloader.dataset)} training samples.")

        return train_dataloader, df
    elif args.mode == 'run':
        df = get_data(percent_data=percent_data, run=True)

        run_dataloader = prepare_dataloader(
            df, tokenizer, batch_size=config.BATCH_SIZE, shuffle=True, args=args
        )
        logging.info(f"Prepared update DataLoader with {len(run_dataloader.dataset)} samples.")
        
        return run_dataloader, df
    elif args.mode == 'update':
        # Load new data for updating
        if args.update_url:
            df = get_new_data(args.update_url)
        else:
            df = get_data(percent_data=percent_data, update=True)

        update_dataloader = prepare_dataloader(
            df, tokenizer, batch_size=config.BATCH_SIZE, shuffle=True, args=args
        )
        logging.info(f"Prepared update DataLoader with {len(update_dataloader.dataset)} samples.")

        return update_dataloader, df
    else:
        raise ValueError("Invalid mode specified in args.mode")

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
        torch.save(state_dict, os.path.join(args.save_dir, 'model_weights.pth'))
        logging.info(f"Model weights saved to '{os.path.join(args.save_dir, 'model_weights.pth')}'.")

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
    Evaluate the model on a validation set.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to perform computations on.

    Returns:
        mse (float): Mean Squared Error on the validation set.
        r2 (float): R-squared score on the validation set.
        sector_metrics (dict): MSE, R², and trend accuracy per sector.
        overall_trend_acc (float): Overall trend accuracy across all samples.
    """
    model.eval()
    predictions = []
    actuals = []
    oldprice_list = []  # To store current prices
    sectors_list = []

    sector_metrics = {}  # Structure: sector_metrics[sector] = {'predictions': [], 'actuals': [], 'oldprices': []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            print(batch, "batch")
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)          # Future price
            old_prices = batch['old_price'].to(device)  # Current price
            sectors = batch['sector']

            with autocast():
                outputs, _ = model(input_ids=input_ids)

            outputs = outputs.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            old_prices = old_prices.cpu().numpy()

            # Accumulate overall metrics
            predictions.extend(outputs)
            actuals.extend(labels)
            oldprice_list.extend(old_prices)
            sectors_list.extend(sectors)

            # Accumulate per-sector metrics
            for i, sector in enumerate(sectors):
                if sector not in sector_metrics:
                    sector_metrics[sector] = {
                        'predictions': [],
                        'actuals': [],
                        'oldprices': []
                    }
                sector_metrics[sector]['predictions'].append(outputs[i])
                sector_metrics[sector]['actuals'].append(labels[i])
                sector_metrics[sector]['oldprices'].append(old_prices[i])

    # Compute Overall MSE & R2
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    # Compute Overall Trend Accuracy
    correct_trend = 0
    total_samples = len(predictions)
    for pred, act, oldp in zip(predictions, actuals, oldprice_list):
        real_trend = np.sign(act - oldp)
        pred_trend = np.sign(pred - oldp)
        if real_trend == pred_trend:
            correct_trend += 1
    overall_trend_acc = correct_trend / total_samples if total_samples > 0 else 0.0

    # Compute Per-Sector Metrics
    for sector, values in sector_metrics.items():
        sector_preds = values['predictions']
        sector_acts  = values['actuals']
        sector_oldp  = values['oldprices']

        sector_mse = mean_squared_error(sector_acts, sector_preds)
        sector_r2  = r2_score(sector_acts, sector_preds)

        # Compute Trend Accuracy per Sector
        sector_correct = 0
        count = len(sector_preds)
        for spred, sact, soldp in zip(sector_preds, sector_acts, sector_oldp):
            if np.sign(sact - soldp) == np.sign(spred - soldp):
                sector_correct += 1
        trend_acc_sec = sector_correct / count if count > 0 else 0.0

        sector_metrics[sector] = {
            'mse': sector_mse,
            'r2': sector_r2,
            'trend_acc': trend_acc_sec
        }

    model.train()
    return mse, r2, sector_metrics, overall_trend_acc

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
        torch.save(state_dict, os.path.join(args.save_dir, 'model_weights.pth'))
        logging.info(f"Model weights saved to '{os.path.join(args.save_dir, 'model_weights.pth')}'.")

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

def save_model_weights(model, filepath, device):
    """
    Load model weights from the given file path.
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {filepath}.")
    return model

def prepare_tasks(tokenizer, args, k=3):
    """
    Prepare multiple tasks (DataLoaders) for testing catastrophic forgetting based on sectors.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use.
        args (argparse.Namespace): Command-line arguments.
        k (int): Number of sectors to randomly select for the tasks.

    Returns:
        tasks (list): List of DataLoaders for the selected sectors.
    """
    df = get_data(percent_data=args.percent_data)  # Load your data

    # Get unique sectors from the dataset
    unique_sectors = df['Sector'].unique()

    # Randomly sample k sectors from the unique sectors
    selected_sectors = np.random.choice(unique_sectors, size=k, replace=False)

    tasks = []

    # For each selected sector, create a DataLoader
    for sector in selected_sectors:
        df_task = df[df['Sector'] == sector]  # Filter data by the selected sector
        dataloader = prepare_dataloader(df_task, tokenizer, batch_size=config.BATCH_SIZE, shuffle=True, args=args)
        tasks.append(dataloader)

    return tasks

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
