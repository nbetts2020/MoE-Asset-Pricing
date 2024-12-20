# utils/data.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import logging
import os
import concurrent.futures
import numpy as np

from utils.sampling import sample_articles
from utils.config import config

from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_concatenated_articles(sample: dict) -> str:
    """
    Formats the concatenated articles from a sample dictionary.

    Args:
        sample (dict): Sampled DataFrames for each category.

    Returns:
        str: Concatenated and formatted article string.
    """
    formatted_articles = []

    # Broader Economic Information (Markets Articles)
    formatted_articles.append("Broader Economic Information:")
    markets = sample.get('markets', pd.DataFrame()).head(5)
    for _, row in markets.iterrows():
        date = row.get('Date', pd.Timestamp('1970-01-01'))
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date, errors='coerce')
            if pd.isna(date):
                date = pd.Timestamp('1970-01-01')
        date_str = date.strftime('%Y-%m-%d')

        formatted_articles.append(
            f"Date: {date_str}\n"
            f"Title: {row.get('Title', 'N/A')}\n"
            f"Article: {row.get('Article', 'N/A')}\n"
        )

    # Broader Industry Information
    formatted_articles.append("\nBroader Industry Information:")
    industry = sample.get('industry', pd.DataFrame()).head(5)
    for _, row in industry.iterrows():
        date = row.get('Date', pd.Timestamp('1970-01-01'))
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date, errors='coerce')
            if pd.isna(date):
                date = pd.Timestamp('1970-01-01')
        date_str = date.strftime('%Y-%m-%d')

        formatted_articles.append(
            f"Date: {date_str}\n"
            f"Title: {row.get('Title', 'N/A')}\n"
            f"Article: {row.get('Article', 'N/A')}\n"
        )

    # Broader Sector Information
    formatted_articles.append("\nBroader Sector Information:")
    sector = sample.get('sector', pd.DataFrame()).head(5)
    for _, row in sector.iterrows():
        date = row.get('Date', pd.Timestamp('1970-01-01'))
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date, errors='coerce')
            if pd.isna(date):
                date = pd.Timestamp('1970-01-01')
        date_str = date.strftime('%Y-%m-%d')

        formatted_articles.append(
            f"Date: {date_str}\n"
            f"Title: {row.get('Title', 'N/A')}\n"
            f"Article: {row.get('Article', 'N/A')}\n"
        )

    # Information Indicating Significant Market Movement Related to Current Stock
    formatted_articles.append("\nInformation Potentially Indicating Significant Market Movement Related to Current Stock:")
    stock = sample.get('stock', pd.DataFrame()).nlargest(5, 'Percentage Change')
    for _, row in stock.iterrows():
        date = row.get('Date', pd.Timestamp('1970-01-01'))
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date, errors='coerce')
            if pd.isna(date):
                date = pd.Timestamp('1970-01-01')
        date_str = date.strftime('%Y-%m-%d')
        percentage_change = row.get('Percentage Change', 0.0)
        formatted_articles.append(
            f"Date: {date_str}\n"
            f"Title: {row.get('Title', 'N/A')}\n"
            f"Article: {row.get('Article', 'N/A')}\n"
            f"Percentage Change: {percentage_change:.2f}%\n"
        )

    # Last 8 Articles for Current Stock
    formatted_articles.append("\nLast 8 Articles for Current Stock:")
    last_8 = sample.get('last_8', pd.DataFrame()).head(8)
    for _, row in last_8.iterrows():
        date = row.get('Date', pd.Timestamp('1970-01-01'))
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date, errors='coerce')
            if pd.isna(date):
                date = pd.Timestamp('1970-01-01')
        date_str = date.strftime('%Y-%m-%d')

        formatted_articles.append(
            f"Symbol: {row.get('Symbol', 'Unknown Symbol')}\n"
            f"Security: {row.get('Security', 'N/A')}\n"
            f"Related Stocks/Topics: {row.get('RelatedStocksList', 'N/A')}\n"
            f"Title: {row.get('Title', 'N/A')}\n"
            f"Type: {row.get('articleType', 'N/A')}\n"
            f"Publication: {row.get('Publication', 'N/A')}\n"
            f"Publication Author: {row.get('Author', 'N/A')}\n"
            f"Date: {date_str}\n"
            f"Article: {row.get('Article', 'N/A')}\n"
            f"Stock Price 4 days before: {row.get('weighted_avg_-96_hrs', 'N/A')}\n"
            f"Stock Price 2 days before: {row.get('weighted_avg_-48_hrs', 'N/A')}\n"
            f"Stock Price 1 day before: {row.get('weighted_avg_-24_hrs', 'N/A')}\n"
            f"Stock Price at release: {row.get('weighted_avg_0_hrs', 'N/A')}\n"
        )

    concatenated_articles = "\n".join(formatted_articles)
    return concatenated_articles

def parallel_context_generation_worker(args):
    """
    Worker function that generates context on CPU only.
    No GPU operations or model calls here.
    """
    idx, df, tokenizer, total_epochs, current_epoch = args
    # Generate sampled articles
    print(idx, current_epoch, df, "ea")
    sampled = sample_articles(df, [idx])
    sampled_dict = sampled[0]
    # Format concatenated articles as a CPU-only string
    context_str = format_concatenated_articles(sampled_dict)
    print(context_str, idx, "llk")
    # Tokenize on CPU with half BLOCK_SIZE
    encoding = tokenizer(
        context_str,
        truncation=True,
        padding='max_length',
        max_length= config.BLOCK_SIZE,
        return_tensors='pt'
    )

    # Return CPU tensors only (no GPU ops)
    return encoding['input_ids'].squeeze(0)  # Still on CPU

class ArticlePriceDataset(Dataset):
    def __init__(self,
                 articles: list,
                 prices: list,
                 sectors: list,
                 dates: list,
                 related_stocks_list: list,
                 prices_current: list,
                 symbols: list,
                 industries: list,
                 tokenizer,
                 total_epochs: int,
                 use_ebm: bool=False):
        self.df = pd.DataFrame({
            'Article': articles,
            'weighted_avg_720_hrs': prices,
            'Sector': sectors,
            'Date': dates,
            'RelatedStocksList': related_stocks_list,
            'weighted_avg_0_hrs': prices_current,
            'Symbol': symbols,
            'Industry': industries
        })
        self.tokenizer = tokenizer
        self.total_epochs = total_epochs
        self.use_ebm = use_ebm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article = row.get('Article', 'N/A')
        price = row.get('weighted_avg_720_hrs', 0.0)
        sector = row.get('Sector', 'Unknown Sector')
        input_encoding = self.tokenizer(
            article,
            truncation=True,
            padding='max_length',
            max_length=config.BLOCK_SIZE,
            return_tensors='pt'
        )
        input_ids = input_encoding['input_ids'].squeeze(0)

        sample = {
            'input_ids': input_ids,
            'labels': torch.tensor(price, dtype=torch.float),
            'sector': sector,
            'idx': int(idx)
        }
        return sample

def worker_wrapper(args):
    return parallel_context_generation_worker(*args)

def custom_collate_fn(batch, df, ebm, model, tokenizer, device, use_ebm, total_epochs, current_epoch):
    """
    Custom collate function.
    CPU only in parallel processes.
    GPU ops (model.get_embeddings(), ebm) in the main process.
    """
    input_ids = []
    labels = []
    # If use_ebm, we need to generate contexts using parallel workers
    # but no GPU ops inside workers
    context_input_ids = [] if use_ebm else None

    # Collect indices for parallel processing if use_ebm
    args_list = []
    if use_ebm:
        for sample in batch:
            idx = sample['idx']
            # Note: Passing only CPU related args, no GPU ops
            args_list.append((idx, df, tokenizer, total_epochs, current_epoch))
    print("aaaA")
    # Run parallel context generation (CPU only)
    if use_ebm:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(parallel_context_generation_worker, args_list))
        # results is a list of CPU tensors (context input_ids), one per sample
    else:
        results = None
    print("bbbb")
    # Now construct the main batch
    for i, sample in enumerate(batch):
        # input_ids and labels are presumably CPU tensors or arrays
        inp = sample['input_ids']
        lbl = sample['labels']
        # Ensure they are tensors
        if not isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp, dtype=torch.long)
        if not isinstance(lbl, torch.Tensor):
            lbl = torch.tensor(lbl, dtype=torch.float)

        input_ids.append(inp)
        labels.append(lbl)

        if use_ebm:
            # Append the CPU result for context_input_ids
            context_input_ids.append(results[i])

    # Pad sequences on CPU
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_tensor = torch.stack(labels)

    if use_ebm:
        context_input_ids_padded = torch.nn.utils.rnn.pad_sequence(context_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

        # NOW move data to GPU and do GPU operations
        input_ids_padded = input_ids_padded.to(device)
        labels_tensor = labels_tensor.to(device)
        context_input_ids_padded = context_input_ids_padded.to(device)

        # Now run model.get_embeddings() here
        with torch.no_grad():
            article_embeddings = model.get_embeddings(input_ids_padded)        # GPU op
            context_embeddings = model.get_embeddings(context_input_ids_padded) # GPU op

        # Compute energies with EBM here if needed, store them in batch if you want.
        # Or just return and handle EBM in train loop. Typically done in the train loop.
        return {
            'input_ids': input_ids_padded,
            'labels': labels_tensor,
            'context_input_ids': context_input_ids_padded,
            'article_embeddings': article_embeddings,
            'context_embeddings': context_embeddings
        }
    else:
        # If no EBM, just return CPU (or move to GPU in train loop)
        # Usually you'd want to move them to GPU in train loop anyway
        return {
            'input_ids': input_ids_padded,
            'labels': labels_tensor
        }
