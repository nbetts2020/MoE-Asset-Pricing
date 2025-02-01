# utils/data.py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import logging
import os
import concurrent.futures
import numpy as np

import random
import ast

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
    stock_df = sample.get('stock', pd.DataFrame())

    # **Check if 'Percentage Change' exists**
    if 'Percentage Change' not in stock_df.columns:
        logger.error(f"'Percentage Change' missing in stock_df. Columns: {stock_df.columns}")
        stock = stock_df.head(5)  # Fallback to top 5 without sorting
    else:
        stock = stock_df.nlargest(5, 'Percentage Change')
    print("LALALAL")
    if not stock.empty:
        for _, row in stock.iterrows():
            date = row.get('Date', pd.Timestamp('1970-01-01'))
            if not isinstance(date, pd.Timestamp):
                date = pd.to_datetime(date, errors='coerce')
                if pd.isna(date):
                    date = pd.Timestamp('1970-01-01')
            date_str = date.strftime('%Y-%m-%d')
            print("OOFA")
            percentage_change = row.get('Percentage Change', 0.0)
            print("OOLA")
            formatted_articles.append(
                f"Date: {date_str}\n"
                f"Title: {row.get('Title', 'N/A')}\n"
                f"Article: {row.get('Article', 'N/A')}\n"
                f"Percentage Change: {percentage_change:.2f}%\n"
            )
    else:
        logger.warning("No stock data available to format.")

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

    # Last for Current Stock
    formatted_articles.append("\nLast 8 Articles for Current Stock:")
    current = sample.get('current', pd.DataFrame()).head(8)
    for _, row in current.iterrows():
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
    CPU-only worker that generates multiple prompt contexts (strings) for one batch sample
    in the EBM approach. No GPU logic here.

    Args:
        args (tuple): A tuple of:
          - idx (int): index in df (the main DataFrame)
          - df (pd.DataFrame): the "master" DataFrame (with 'Article', 'Date', 'Percentage Change', etc.)
          - df_preprocessed (pd.DataFrame): row-aligned with df, containing columns like
                'use_ebm_economic', 'use_ebm_industry', 'use_ebm_sector', 'use_ebm_historical'
                (each a NumPy array of row indices referencing df).
          - total_epochs (int): total training epochs
          - current_epoch (int): current epoch
          - context_count (int): e.g., max(epochs - epoch, 5)
            (how many distinct prompt contexts to generate for this single sample)

    Returns:
        List[str]: A list of prompt strings (one for each context_count) using format_concatenated_articles.
    """
    (
        idx,
        df,
        df_preprocessed,
        total_epochs,
        current_epoch,
        context_count
    ) = args
    candidate_contexts = []

    # Step 1: Validate idx and gather main row
    if idx < 0 or idx >= len(df):
        logging.error(f"Index {idx} is out-of-bounds for the main DataFrame.")
        return candidate_contexts  # out-of-bounds => empty

    main_row = df.iloc[idx]
    preproc_row = df_preprocessed.iloc[idx]  # row-aligned with df
    symbol = main_row.get('Symbol', 'Unknown Symbol')

    # Define how many samples to take from each category
    sample_map = {
        'use_ebm_economic': 5,
        'use_ebm_industry': 5,
        'use_ebm_sector': 5,
        'use_ebm_top25': 5,
        # We do NOT sample from 'use_ebm_historical'; we take the full list.
    }

    # Step 2: Build multiple contexts, each one is used by the EBM
    for _ in range(context_count):
        # A) ECONOMIC -> "markets" in your final dict
        econ_array = preproc_row.get('use_ebm_economic', np.array([]))
        econ_needed = min(len(econ_array), sample_map['use_ebm_economic'])
        if econ_needed > 0:
            if econ_needed > len(econ_array):
                logging.warning(f"Requested {econ_needed} econ samples, but only {len(econ_array)} available.")
                econ_indices = econ_array  # Take all available
            else:
                econ_indices = np.random.choice(econ_array, size=econ_needed, replace=False)
        else:
            econ_indices = np.array([], dtype=int)
        markets_df = df.loc[econ_indices].copy() if econ_indices.size > 0 else pd.DataFrame()

        # B) INDUSTRY -> "industry"
        ind_array = preproc_row.get('use_ebm_industry', np.array([]))
        ind_needed = min(len(ind_array), sample_map['use_ebm_industry'])
        if ind_needed > 0:
            if ind_needed > len(ind_array):
                logging.warning(f"Requested {ind_needed} industry samples, but only {len(ind_array)} available.")
                ind_indices = ind_array
            else:
                ind_indices = np.random.choice(ind_array, size=ind_needed, replace=False)
        else:
            ind_indices = np.array([], dtype=int)
        industry_df = df.loc[ind_indices].copy() if ind_indices.size > 0 else pd.DataFrame()

        # C) SECTOR -> "sector"
        sec_array = preproc_row.get('use_ebm_sector', np.array([]))
        sec_needed = min(len(sec_array), sample_map['use_ebm_sector'])
        if sec_needed > 0:
            if sec_needed > len(sec_array):
                logging.warning(f"Requested {sec_needed} sector samples, but only {len(sec_array)} available.")
                sec_indices = sec_array
            else:
                sec_indices = np.random.choice(sec_array, size=sec_needed, replace=False)
        else:
            sec_indices = np.array([], dtype=int)
        sector_df = df.loc[sec_indices].copy() if sec_indices.size > 0 else pd.DataFrame()

        # D) HISTORICAL -> "last_8"
        # We take **all** references (no sampling), partial if <8 is handled by .head(8) in your formatting
        hist_array = preproc_row.get('use_ebm_historical', np.array([]))
        last_8_df = df.loc[hist_array].copy() if len(hist_array) > 0 else pd.DataFrame()

        # E) TOP25 -> "stock"
        top25_array = preproc_row.get('use_ebm_top25', np.array([]))
        top25_needed = min(len(top25_array), sample_map['use_ebm_top25'])
        if top25_needed > 0:
            if top25_needed > len(top25_array):
                logging.warning(f"Requested {top25_needed} top25 samples, but only {len(top25_array)} available.")
                top25_indices = top25_array
            else:
                top25_indices = np.random.choice(top25_array, size=top25_needed, replace=False)
        else:
            top25_indices = np.array([], dtype=int)
        stock_df = df.loc[top25_indices].copy() if top25_indices.size > 0 else pd.DataFrame()

        # F) CURRENT -> main article
        current_df = pd.DataFrame([main_row])

        # Step 3: Build dict for format_concatenated_articles
        sample_dict = {
            'markets': markets_df,
            'industry': industry_df,
            'sector': sector_df,
            'stock': stock_df,
            'last_8': last_8_df,
            'current': current_df
        }

        # Step 4: Convert to final string
        prompt_str = format_concatenated_articles(sample_dict)

        # Step 5: Add to our list of contexts
        candidate_contexts.append(prompt_str)

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
    rfr_list       = []

    for sample in batch:
        input_ids_list.append(sample['input_ids'])
        labels_list.append(sample['labels'])
        old_price_list.append(sample['old_price'])
        sector_list.append(sample['sector'])
        idx_list.append(sample['idx'])
        rfr_list.append(sample['risk_free_rate'])

    # Pad input_ids
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True,
        padding_value = 50256 # hardcoded padding id token for gpt2 tokenizer
    )
    labels_tensor    = torch.stack(labels_list)
    old_price_tensor = torch.stack(old_price_list)
    rfr_tensor       = torch.stack(rfr_list)

    return {
        'input_ids':    input_ids_padded,
        'labels':       labels_tensor,
        'old_price':    old_price_tensor,
        'sector':       sector_list,  # CPU list of strings
        'idx':          idx_list,
        'risk_free_rate': rfr_list
    }
