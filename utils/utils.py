import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from utils.config import *
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

from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer

import logging

import inspect

def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

def get_data(percent_data=100.0):
    load_dotenv('/content/MoE-Asset-Pricing/.env')
    hf_token = os.getenv('HF_TOKEN')

    login(hf_token)
    dataset = load_dataset("nbettencourt/SC454k")
    df = dataset['test'].to_pandas()
    
    total_samples = len(df)
    num_samples = int((percent_data / 100.0) * total_samples)
    df = df.head(num_samples)
    
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
            config = json.load(f)
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
    model_config = {k: config[k] for k in expected_args if k in config}
    logging.info(f"Filtered model configuration: {model_config}")

    # Initialize the model with the configuration
    try:
        model = SparseMoELanguageModel(**model_config)
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

def prepare_dataloader(df, tokenizer, batch_size=BATCH_SIZE, shuffle=True):
    articles, prices, sectors = process_data(df, tokenizer)
    dataset = ArticlePriceDataset(articles, prices, sectors, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
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
    
    # Get unique sectors from the dataset
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

def initialize_model(args, device, init_from_scratch=False):

    if init_from_scratch:
        logging.info("Initializing model from scratch using configurations from utils/config.py.")
        model_config = {
            'n_embed': n_embed,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'dropout': dropout,
            'num_experts': num_experts,
            'top_k': top_k,
            'tokenizer_name': args.tokenizer_name
        }
        model = SparseMoELanguageModel(**model_config)
        model = model.to(device)
        initialized_from_scratch = True
        return model, initialized_from_scratch
    else:
        model = None
        initialized_from_scratch = False
        if args.model:
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
                config = json.load(f)
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
    percent_data = args.percent_data  # Get the percentage of data to use
    df = get_data(percent_data=percent_data)
    df = df.sort_values('Date')  # Ensure data is sorted by date

    # Set random seed
    random_seed = args.random_seed

    if args.mode == 'train':
        if args.update:
            # Update scenario: split into train, update, and test
            total_samples = len(df)
            train_size = int(0.6 * total_samples)
            update_size = int(0.2 * total_samples)
            test_size = total_samples - train_size - update_size

            train_df = df.iloc[:train_size]
            update_df = df.iloc[train_size:train_size + update_size]
            test_df = df.iloc[train_size + update_size:]
        else:
            # Normal training run: split into train and test
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=random_seed,
                shuffle=True
            )
            update_df = None  # no update data in normal training run

        actual_batch_size = BATCH_SIZE
        train_dataloader = prepare_dataloader(train_df, tokenizer, batch_size=actual_batch_size)
        logging.info(f"Prepared DataLoader with {len(train_dataloader.dataset)} training samples.")

        # Prepare test DataLoader
        test_dataloader = prepare_dataloader(test_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False)
        logging.info(f"Prepared test DataLoader with {len(test_dataloader.dataset)} samples.")

        # Prepare update DataLoader if in update mode
        if update_df is not None:
            update_dataloader = prepare_dataloader(update_df, tokenizer, batch_size=actual_batch_size)
            logging.info(f"Prepared update DataLoader with {len(update_dataloader.dataset)} samples.")
        else:
            update_dataloader = None

        # Determine accumulation_steps
        desired_effective_batch_size = 16  # Adjust as needed
        accumulation_steps = max(1, desired_effective_batch_size // actual_batch_size)
        logging.info(f"Using accumulation_steps={accumulation_steps} for training.")

        return train_dataloader, test_dataloader, update_dataloader, accumulation_steps

    else:
        return None, None, None, None

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
    Saves the model weights, SI state, and replay buffer to the specified save directory.
    """
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_weights.pth'))
    logging.info(f"Model weights saved to '{os.path.join(args.save_dir, 'model_weights.pth')}'.")
    if args.use_si and si is not None:
        si.save_state(os.path.join(args.save_dir, 'si_state.pth'))
        logging.info(f"Synaptic Intelligence (SI) state saved to '{os.path.join(args.save_dir, 'si_state.pth')}'.")
    if args.use_replay_buffer and replay_buffer is not None:
        replay_buffer.save(os.path.join(args.save_dir, 'replay_buffer.pth'))
        logging.info(f"Memory Replay Buffer saved to '{os.path.join(args.save_dir, 'replay_buffer.pth')}'.")
