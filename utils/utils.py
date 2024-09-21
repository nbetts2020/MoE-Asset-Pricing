import torch.nn as nn
from torch.nn import init

import os
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv

from utils.data import ArticlePriceDataset
from utils.config import BATCH_SIZE, block_size
from utils.data import process_data

from tqdm import tqdm
from transformers import AutoTokenizer

def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

def get_data():
    load_dotenv('/content/MoE-Asset-Pricing/.env')
    hf_token = os.getenv('HF_TOKEN')

    login(hf_token)
    dataset = load_dataset("nbettencourt/SC454k-valid")
    df = dataset['test'].to_pandas().drop(columns=['Unnamed: 0'])
    return df

def process_data(df, tokenizer_name="gpt2"):
    """
    Preprocess the data for article-based price prediction.
    """
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    articles = []
    prices = []

    # Group the data by symbol
    grouped = df.groupby('Symbol_x', sort=False)

    print("Processing articles and prices...")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        current_symbol = row['Symbol_x']
        current_date = row['Date_x']

        # Get all articles for the current symbol before the current date
        symbol_df = grouped.get_group(current_symbol)
        previous_articles = symbol_df[symbol_df['Date_x'] < current_date]

        # Get the last 10 previous articles
        last_articles = previous_articles.tail(10)

        # Build the concatenated text
        concatenated_text = ''

        # Add previous articles
        for _, prev_row in last_articles.iterrows():
            concatenated_text += (
                "\nPrevious Article Date: " + str(prev_row['Date_x']) +
                "\nPrevious Article Content: " + str(prev_row['Article']) +
                "\nPrevious Article Title: " + str(prev_row['Title']) +
                "\nPrevious Article Type: " + str(prev_row['articleType']) +
                "\nPrevious Article Publication: " + str(prev_row['Publication']) +
                "\nPrevious Publication Author: " + str(prev_row['Author']) +
                "\n---\n"
            )

        # Add the current article
        concatenated_text += (
            "Symbol: " + str(row['Symbol_x']) +
            "\nSecurity: " + str(row['Date_x']) +
            "\nRelated Stocks/Topics: " + str(row['RelatedStocksList']) +
            "\nArticle Content: " + str(row['Article']) +
            "\nArticle Title: " + str(row['Title']) +
            "\nArticle Type: " + str(row['articleType']) +
            "\nArticle Publication: " + str(row['Publication']) +
            "\nPublication Author: " + str(row['Author']) +
            "\nStock Price 4 days before: " + str(row['weighted_avg_-96_hrs']) +
            "\nStock Price 2 days before: " + str(row['weighted_avg_-48_hrs']) +
            "\nStock Price 1 days before: " + str(row['weighted_avg_-24_hrs']) +
            "\nStock Price 0 days before: " + str(row['weighted_avg_0_hrs'])
        )

        # Tokenize the concatenated text
        tokenized_article = tokenizer(
            concatenated_text,
            truncation=True,
            padding='max_length',
            max_length=block_size,
            return_tensors="pt"
        )

        # Append tokenized articles and prices
        articles.append(tokenized_article['input_ids'].squeeze())  # Convert to tensor
        prices.append(row['weighted_avg_720_hrs'])  # Target price for prediction

    return articles, prices

def prepare_dataloader(df, tokenizer, batch_size=BATCH_SIZE):
    """
    Prepare DataLoader for a given DataFrame and tokenizer.
    """
    articles, prices = process_data(df)
    dataset = ArticlePriceDataset(articles, prices, tokenizer)
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

def initialize_si(model, si_path, lambda_si):
    """
    Initialize Synaptic Intelligence (SI) and load its state if it exists.
    """
    si = SynapticIntelligence(model, lambda_si=lambda_si)
    if os.path.exists(si_path):
        si.load_state(si_path)
        print(f"SI state loaded from {si_path}.")
    else:
        print("No existing SI state found. Starting fresh.")
    return si

def prepare_tasks():
    """
    Prepare multiple tasks (DataLoaders) for testing catastrophic forgetting.
    """
    df = get_data()  # Load your data

    # Split your data into different tasks (this is just an example)
    df_task_1 = df[df['year'] < 2015]  # Task 1: Articles before 2015
    df_task_2 = df[(df['year'] >= 2015) & (df['year'] < 2018)]  # Task 2: Articles from 2015-2017
    df_task_3 = df[df['year'] >= 2018]  # Task 3: Articles from 2018 onwards

    tasks = []

    # Create DataLoaders for each task
    for df_task in [df_task_1, df_task_2, df_task_3]:
        dataloader = prepare_dataloader(df_task, tokenizer)
        tasks.append(dataloader)

    return tasks
