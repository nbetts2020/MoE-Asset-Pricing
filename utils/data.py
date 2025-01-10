# utils/data.py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
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
            f"Risk-Free Rate at release: {row.get('Risk_Free_Rate', 'N/A')}\n"
        )

    concatenated_articles = "\n".join(formatted_articles)
    return concatenated_articles

def parallel_context_generation_worker(args):
    """
    CPU-only worker. No EBM or GPU ops here.
    Returns multiple raw context strings (or CPU tensors) for each sample.
    """
    (idx, df, tokenizer, total_epochs, current_epoch, context_count, top25_dict) = args
    # We'll store multiple raw context strings
    candidate_contexts = []

    for _ in range(context_count):
        # 1) sample articles
        sampled_list = sample_articles(df, index_list=[idx], top25_dict=top25_dict)
        if not sampled_list:
            continue
        sample_dict = sampled_list[0]
        # 2) format the concatenated context as a CPU-only string
        context_str = format_concatenated_articles(sample_dict)
        # store the raw string for now
        candidate_contexts.append(context_str)
    print(len(candidate_contexts), "llal")
    # Just return all candidate strings
    return candidate_contexts

class ArticlePriceDataset(Dataset):
    def __init__(self,
                 articles: list,
                 prices: list,
                 sectors: list,
                 dates: list,
                 related_stocks_list: list,
                 prices_current: list,       # old/current price
                 symbols: list,
                 industries: list,
                 risk_free_rates: list,
                 tokenizer,
                 total_epochs: int,
                 use_ebm: bool=False):
        self.df = pd.DataFrame({
            'Article': articles,
            'weighted_avg_720_hrs': prices,  # future price
            'Sector': sectors,
            'Date': dates,
            'RelatedStocksList': related_stocks_list,
            'weighted_avg_0_hrs': prices_current,  # old/current price
            'Symbol': symbols,
            'Industry': industries,
            'Risk_Free_Rate': risk_free_rates
        })
        self.tokenizer = tokenizer
        self.total_epochs = total_epochs
        self.use_ebm = use_ebm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Original article text
        article = row.get('Article', 'N/A')

        risk_free = row.get('Risk_Free_Rate', 0.0)

        # The label is the future price
        future_price = row.get('weighted_avg_720_hrs', 0.0)

        # The old/current price we also want for e.g. trend calc
        old_price = row.get('weighted_avg_0_hrs', 0.0)

        sector = row.get('Sector', 'Unknown Sector')

        # Tokenize this updated article text
        input_encoding = self.tokenizer(
            article,
            truncation=True,
            padding='max_length',
            max_length=config.BLOCK_SIZE,
            return_tensors='pt'
        )
        input_ids = input_encoding['input_ids'].squeeze(0)

        sample = {
            'input_ids':     input_ids,
            'labels':        torch.tensor(future_price, dtype=torch.float),
            'sector':        sector if sector is not None else "Unknown Sector",
            'idx':           int(idx),
            'old_price':     torch.tensor(old_price, dtype=torch.float),
            'risk_free_rate':torch.tensor(risk_free, dtype=torch.float)
        }
        return sample

def worker_wrapper(args):
    return parallel_context_generation_worker(*args)

def custom_collate_fn(batch):
    """
    Minimal collate function: merges CPU data into a single batch,
    but does NOT do large GPU calls or EBM logic.
    """
    input_ids_list = []
    labels_list    = []
    old_price_list = []
    sector_list    = []
    idx_list       = []

    for sample in batch:
        input_ids_list.append(sample['input_ids'])
        labels_list.append(sample['labels'])
        old_price_list.append(sample['old_price'])
        sector_list.append(sample['sector'])
        idx_list.append(sample['idx'])

    # Pad input_ids
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True,
        padding_value = 50256 # hardcoded padding id token for gpt2 tokenizer
    )
    labels_tensor    = torch.stack(labels_list)
    old_price_tensor = torch.stack(old_price_list)

    return {
        'input_ids':    input_ids_padded,
        'labels':       labels_tensor,
        'old_price':    old_price_tensor,
        'sector':       sector_list,  # CPU list of strings
        'idx':          idx_list
    }
