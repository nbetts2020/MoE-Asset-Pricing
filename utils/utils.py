import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader

from utils.config import *

import os
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv

from utils.data import ArticlePriceDataset
from utils.config import BATCH_SIZE, block_size

from tqdm import tqdm
from transformers import AutoTokenizer

import json
from huggingface_hub import hf_hub_download

import logging
from utils.model import SparseMoELanguageModel

def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

def get_data():
    load_dotenv('/content/MoE-Asset-Pricing/.env')
    hf_token = os.getenv('HF_TOKEN')

    login(hf_token)
    dataset = load_dataset("nbettencourt/SC454k-valid")
    df = dataset['test'].to_pandas()
    return df#.head(1024)

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

def get_model_from_hf(model_repo_id, device):
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
            config = json.load(f)
        logging.info("Loaded model configuration from 'config.json'.")
    except Exception as e:
        logging.error(f"Failed to load configuration from '{config_path}': {e}")
        raise e

    # Initialize the model with the configuration
    try:
        model = SparseMoELanguageModel(**config)
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

    # Move the model to the specified device
    model.to(device)
    logging.info(f"Model moved to device '{device}'.")

    return model

def process_data(df, tokenizer):
    articles = []
    prices = []
    sectors = []

    grouped = df.groupby('Symbol', sort=False)

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        current_symbol = row['Symbol']
        current_date = row['Date']

        # Get all articles for the current symbol before the current date
        symbol_df = grouped.get_group(current_symbol)
        previous_articles = symbol_df[symbol_df['Date'] < current_date]

        # Get the last 10 previous articles
        last_articles = previous_articles.tail(10)

        # Build the concatenated text
        concatenated_text = ''

        # Add previous articles
        for _, prev_row in last_articles.iterrows():
            concatenated_text += (
                "\nPrevious Article Date: " + str(prev_row['Date']) +
                "\nPrevious Article Content: " + str(prev_row['Article']) +
                "\nPrevious Article Title: " + str(prev_row['Title']) +
                "\nPrevious Article Type: " + str(prev_row['articleType']) +
                "\nPrevious Article Publication: " + str(prev_row['Publication']) +
                "\nPrevious Publication Author: " + str(prev_row['Author']) +
                "\n---\n"
            )

        # Add the current article
        concatenated_text += (
            "Symbol: " + str(row['Symbol']) +
            "\nSecurity: " + str(row['Date']) +
            "\nRelated Stocks/Topics: " + str(row['RelatedStocksList']) +
            "\nArticle Content: " + str(row['Article']) +
            "\nArticle Title: " + str(row['Title']) +
            "\nArticle Type: " + str(row['articleType']) +
            "\nArticle Publication: " + str(row['Publication']) +
            "\nPublication Author: " + str(row['Author']) +
            "\nStock Price 4 days before: " + str(row['weighted_avg_-96_hrs']) +
            "\nStock Price 2 days before: " + str(row['weighted_avg_-48_hrs']) +
            "\nStock Price 1 day before: " + str(row['weighted_avg_-24_hrs']) +
            "\nStock Price at release: " + str(row['weighted_avg_0_hrs'])
        )

        articles.append(concatenated_text)
        prices.append(row['weighted_avg_720_hrs'])
        sectors.append(row['Sector'])  # Include sector

    return articles, prices, sectors

def prepare_dataloader(df, tokenizer, batch_size=BATCH_SIZE):
    articles, prices, sectors = process_data(df, tokenizer)
    dataset = ArticlePriceDataset(articles, prices, sectors, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def load_model_weights(model, filepath, device):
    """
    Load model weights from the given file path.
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {filepath}.")
    return model

def prepare_tasks(k=3):
    """
    Prepare multiple tasks (DataLoaders) for testing catastrophic forgetting based on sectors.
    
    Args:
        k (int): Number of sectors to randomly select for the tasks.
        
    Returns:
        tasks (list): List of DataLoaders for the selected sectors.
    """
    df = get_data()  # Load your data
    
    # Get the unique sectors from the dataset
    unique_sectors = df['Sector'].unique()
    
    # Randomly sample k sectors from the unique sectors
    selected_sectors = np.random.choice(unique_sectors, size=k, replace=False)
    
    tasks = []

    # For each selected sector, create a DataLoader
    for sector in selected_sectors:
        df_task = df[df['Sector'] == sector]  # Filter data by the selected sector
        dataloader = prepare_dataloader(df_task, tokenizer)  # Create DataLoader for each sector
        tasks.append(dataloader)

    return tasks

def initialize_model(args, device):
    """
    Initializes the model either from Hugging Face or locally.
    """
    if args.model_repo_id:
        # Load the model from Hugging Face
        logging.info(f"Loading model from Hugging Face repository '{args.model_repo_id}'.")
        model = get_model_from_hf(args.model_repo_id, device)
    else:
        # Initialize the model locally
        model = SparseMoELanguageModel(tokenizer_name=args.tokenizer_name)
        model = model.to(device)
    # Move model to device before wrapping with DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training.")
    return model

def prepare_optimizer(model):
    """
    Prepares the optimizer with layer-wise learning rate decay.
    """
    param_groups = [
        {'params': list(model.token_embedding_table.parameters()) + list(model.position_embedding_table.parameters()), 'lr': learning_rate * (LR_DECAY ** (len(model.blocks) + 1))},
        {'params': model.regression_head.parameters(), 'lr': learning_rate}
    ] + [
        {'params': block.parameters(), 'lr': learning_rate * (LR_DECAY ** (len(model.blocks) - i))}
        for i, block in enumerate(model.blocks)
    ]
    optimizer = torch.optim.AdamW(param_groups)
    logging.info("Initialized AdamW optimizer with layer-wise learning rate decay.")
    return optimizer

def prepare_data(args, tokenizer):
    """
    Prepares the data for training or updating.
    """
    if args.mode == 'train':
        # Load data and create DataLoader for initial training
        df = get_data()
        df = df[df['weighted_avg_720_hrs'] > 0]
        train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
        actual_batch_size = 16  # Adjust based on resources
        train_dataloader = prepare_dataloader(train_df, tokenizer, batch_size=actual_batch_size)
        logging.info(f"Prepared DataLoader with {len(train_dataloader.dataset)} training samples.")
    elif args.mode == 'update':
        # Load new data from Hugging Face URL
        new_data_url = args.update
        logging.info(f"Fetching new data from Hugging Face URL: {new_data_url}")
        df_new = get_new_data(new_data_url)
        logging.info(f"Fetched {len(df_new)} new data samples.")
        df_new = df_new[df_new['weighted_avg_720_hrs'] > 0]
        actual_batch_size = min(16, len(df_new))  # Adjust based on data size
        train_dataloader = prepare_dataloader(df_new, tokenizer, batch_size=actual_batch_size)
        logging.info(f"Prepared DataLoader with {len(train_dataloader.dataset)} new training samples.")
    else:
        return None, None

    # Determine accumulation_steps
    desired_effective_batch_size = 16  # Adjust as needed
    accumulation_steps = max(1, desired_effective_batch_size // actual_batch_size)
    logging.info(f"Using accumulation_steps={accumulation_steps} for training.")
    return train_dataloader, accumulation_steps

def initialize_si(model, args):
    """
    Initializes Synaptic Intelligence (SI) if specified.
    """
    si = None
    if args.use_si:
        si = SynapticIntelligence(model, lambda_si=LAMBDA_SI)
        if args.mode == 'update':
            si_state_path = 'model/si_state.pth'
            if os.path.exists(si_state_path):
                si.load_state(si_state_path)
                logging.info(f"Loaded Synaptic Intelligence (SI) state from '{si_state_path}'.")
            else:
                logging.info("No existing SI state found. Starting fresh SI.")
        else:
            logging.info("Initialized Synaptic Intelligence (SI) for initial training.")
    return si

def initialize_replay_buffer(args):
    """
    Initializes or loads the replay buffer if specified.
    """
    replay_buffer = None
    if args.use_replay_buffer:
        replay_buffer_capacity = 10000
        replay_buffer = MemoryReplayBuffer(capacity=replay_buffer_capacity)
        logging.info(f"Initialized Memory Replay Buffer with capacity {replay_buffer_capacity}.")
        if args.mode == 'update':
            replay_buffer_path = 'model/replay_buffer.pth'
            if os.path.exists(replay_buffer_path):
                replay_buffer.load(replay_buffer_path)
                logging.info(f"Loaded Memory Replay Buffer from '{replay_buffer_path}'.")
            else:
                logging.info("No existing Memory Replay Buffer found. Starting fresh.")
    return replay_buffer

def save_model_and_states(model, si, replay_buffer, args):
    """
    Saves the model weights, SI state, and replay buffer.
    """
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/model_weights.pth')
    logging.info("Model weights saved to 'model/model_weights.pth'.")
    if args.use_si and si is not None:
        si.save_state('model/si_state.pth')
        logging.info("Synaptic Intelligence (SI) state saved to 'model/si_state.pth'.")
    if args.use_replay_buffer and replay_buffer is not None:
        replay_buffer.save('model/replay_buffer.pth')
        logging.info("Memory Replay Buffer saved to 'model/replay_buffer.pth'.")
