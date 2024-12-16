# utils/data.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import logging
import os
import concurrent.futures
import numpy as np

from utils.sampling import preprocess_data, sample_articles
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_concatenated_articles(sample: pd.DataFrame) -> str:
    """
    Formats the concatenated articles from a sample DataFrame.

    Args:
        sample (pd.DataFrame): Sampled DataFrame for a single data point.

    Returns:
        str: Concatenated and formatted article string.
    """
    formatted_articles = []
    idx = sample.iloc[-1].name  # Current index

    # Broader Economic Information (Markets Articles)
    formatted_articles.append("Broader Economic Information:")
    for _, row in sample.iterrows():
        if 'Markets' in row.get('RelatedStocksList', ''):
            date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
            formatted_articles.append(
                f"Date: {date_str}\n"
                f"Title: {row.get('Title', 'N/A')}\n"
                f"Article: {row.get('Article', 'N/A')}\n"
            )

    # Broader Industry Information
    formatted_articles.append("\nBroader Industry Information:")
    for _, row in sample.iterrows():
        if row.get('Industry', 'Unknown Industry') == sample.iloc[-1].get('Industry', 'Unknown Industry'):
            date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
            formatted_articles.append(
                f"Date: {date_str}\n"
                f"Title: {row.get('Title', 'N/A')}\n"
                f"Article: {row.get('Article', 'N/A')}\n"
            )

    # Broader Sector Information
    formatted_articles.append("\nBroader Sector Information:")
    for _, row in sample.iterrows():
        if row.get('Sector', 'Unknown Sector') == sample.iloc[-1].get('Sector', 'Unknown Sector'):
            date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
            formatted_articles.append(
                f"Date: {date_str}\n"
                f"Title: {row.get('Title', 'N/A')}\n"
                f"Article: {row.get('Article', 'N/A')}\n"
            )

    # Information Indicating Significant Market Movement Related to Current Stock
    formatted_articles.append("\nInformation Potentially Indicating Significant Market Movement Related to Current Stock:")
    for _, row in sample.iterrows():
        if (row.get('Symbol', 'Unknown Symbol') == sample.iloc[-1].get('Symbol', 'Unknown Symbol')) and ('Percentage Change' in row):
            date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
            formatted_articles.append(
                f"Date: {date_str}\n"
                f"Title: {row.get('Title', 'N/A')}\n"
                f"Article: {row.get('Article', 'N/A')}\n"
                f"Percentage Change: {row.get('Percentage Change', 0.0):.2f}%\n"
            )

    # Last 8 Articles for Current Stock
    formatted_articles.append("\nLast 8 Articles for Current Stock:")
    for _, row in sample.iterrows():
        if row.get('Symbol', 'Unknown Symbol') == sample.iloc[-1].get('Symbol', 'Unknown Symbol'):
            date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
            article_details = (
                f"Symbol: {row.get('Symbol', 'N/A')}\n"
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
            formatted_articles.append(article_details)

    concatenated_articles = "\n".join(formatted_articles)
    return concatenated_articles

def parallel_context_generation_worker(args):
    """
    Worker function that generates N contexts for a single sample (idx).
    It returns the current_article_str and a list of N context strings.
    """
    idx, df, N = args
    # We'll generate N contexts by calling sample_articles N times
    contexts = []
    current_article_str = None

    for i in range(N):
        sampled = sample_articles(df, [idx])[0]  # returns one sampled DataFrame for that idx
        context_str = format_concatenated_articles(sampled)
        contexts.append(context_str)

        if current_article_str is None:
            # Extract current article info from sampled
            target_row = sampled.iloc[-1]
            current_article_str = (
                f"Symbol: {target_row.get('Symbol', 'N/A')}\n"
                f"Security: {target_row.get('Security', 'N/A')}\n"
                f"Related Stocks/Topics: {target_row.get('RelatedStocksList', 'N/A')}\n"
                f"Title: {target_row.get('Title', 'N/A')}\n"
                f"Type: {target_row.get('articleType', 'N/A')}\n"
                f"Publication: {target_row.get('Publication', 'N/A')}\n"
                f"Publication Author: {target_row.get('Author', 'N/A')}\n"
                f"Date: {target_row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')}\n"
                f"Article: {target_row.get('Article', 'N/A')}\n"
            )

    return current_article_str, contexts

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
        self.df = preprocess_data(self.df)
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

def custom_collate_fn(batch, df, ebm, model, tokenizer, device, use_ebm, total_epochs, current_epoch):
    """
    Custom collate function:
    - If use_ebm: generate N contexts per sample (CPU-only) and return them as raw strings
      along with the current_article_str.
    - Do NOT run EBM or select the best context here. Just return all contexts.
    
    We'll handle EBM scoring and context selection in train_model.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    sectors = [item['sector'] for item in batch]
    idxs = [item['idx'] for item in batch]

    N = max(total_epochs - current_epoch, 5)

    if use_ebm:
        # Generate contexts in parallel
        args_list = [(idx, df, N) for idx in idxs]
        num_workers = os.cpu_count() or 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(parallel_context_generation_worker, args_list))

        # results: list of (current_article_str, [context1, context2, ..., contextN]) per sample
        # Just return them as is. We'll handle EBM scoring and selection in train_model.
        return {
            'input_ids': input_ids,
            'labels': labels,
            'sector': sectors,
            'idx': idxs,
            'current_articles': [r[0] for r in results],   # list of current_article_str
            'all_contexts': [r[1] for r in results],       # list of lists of context strings
            'N': N
        }
    else:
        # No EBM, just return normal batch
        return {
            'input_ids': input_ids,
            'labels': labels,
            'sector': sectors,
            'idx': idxs,
            'all_contexts': None,
            'current_articles': None,
            'N': 0
        }
