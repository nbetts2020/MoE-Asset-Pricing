import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F
import pandas as pd
import logging
import os
import concurrent.futures
import numpy as np
import random
import ast
from tqdm import tqdm
import pyarrow.parquet as pq  # Added for RollingWindowDataset

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

    if 'Percentage Change' not in stock_df.columns:
        logger.error(f"'Percentage Change' missing in stock_df. Columns: {stock_df.columns}")
        stock = stock_df.head(5)  # Fallback to top 5 without sorting
    else:
        stock = stock_df.nlargest(5, 'Percentage Change')
    if not stock.empty:
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
          - df (pd.DataFrame): the "master" DataFrame
          - df_preprocessed (pd.DataFrame): row-aligned with df
          - total_epochs (int): total training epochs
          - current_epoch (int): current epoch
          - context_count (int): number of distinct prompt contexts to generate for this sample

    Returns:
        tuple: (idx, List[str]) where the list contains prompt strings.
    """
    (idx, df, df_preprocessed, total_epochs, current_epoch, context_count) = args
    candidate_contexts = []

    if idx < 0 or idx >= len(df):
        logging.error(f"Index {idx} is out-of-bounds for the main DataFrame.")
        return (idx, candidate_contexts)

    main_row = df.iloc[idx]
    preproc_row = df_preprocessed.iloc[idx]  # row-aligned with df

    sample_map = {
        'use_ebm_economic': 5,
        'use_ebm_industry': 5,
        'use_ebm_sector': 5,
        'use_ebm_top25': 5,
    }

    for _ in range(context_count):
        econ_array = preproc_row.get('use_ebm_economic', np.array([]))
        econ_needed = min(len(econ_array), sample_map['use_ebm_economic'])
        if econ_needed > 0:
            econ_indices = np.random.choice(econ_array, size=econ_needed, replace=False) if econ_needed <= len(econ_array) else econ_array
        else:
            econ_indices = np.array([], dtype=int)
        markets_df = df.loc[econ_indices].copy() if econ_indices.size > 0 else pd.DataFrame()

        ind_array = preproc_row.get('use_ebm_industry', np.array([]))
        ind_needed = min(len(ind_array), sample_map['use_ebm_industry'])
        if ind_needed > 0:
            ind_indices = np.random.choice(ind_array, size=ind_needed, replace=False) if ind_needed <= len(ind_array) else ind_array
        else:
            ind_indices = np.array([], dtype=int)
        industry_df = df.loc[ind_indices].copy() if ind_indices.size > 0 else pd.DataFrame()

        sec_array = preproc_row.get('use_ebm_sector', np.array([]))
        sec_needed = min(len(sec_array), sample_map['use_ebm_sector'])
        if sec_needed > 0:
            sec_indices = np.random.choice(sec_array, size=sec_needed, replace=False) if sec_needed <= len(sec_array) else sec_array
        else:
            sec_indices = np.array([], dtype=int)
        sector_df = df.loc[sec_indices].copy() if sec_indices.size > 0 else pd.DataFrame()

        hist_array = preproc_row.get('use_ebm_historical', np.array([]))
        last_8_df = df.loc[hist_array].copy() if len(hist_array) > 0 else pd.DataFrame()

        top25_array = preproc_row.get('use_ebm_top25', np.array([]))
        top25_needed = min(len(top25_array), sample_map['use_ebm_top25'])
        if top25_needed > 0:
            top25_indices = np.random.choice(top25_array, size=top25_needed, replace=False) if top25_needed <= len(top25_array) else top25_array
        else:
            top25_indices = np.array([], dtype=int)
        stock_df = df.loc[top25_indices].copy() if top25_indices.size > 0 else pd.DataFrame()

        current_df = pd.DataFrame([main_row])

        sample_dict = {
            'markets': markets_df,
            'industry': industry_df,
            'sector': sector_df,
            'stock': stock_df,
            'last_8': last_8_df,
            'current': current_df
        }

        prompt_str = format_concatenated_articles(sample_dict)
        candidate_contexts.append(prompt_str)

    return (idx, candidate_contexts)

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

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True,
        padding_value=50256
    )
    labels_tensor    = torch.stack(labels_list)
    old_price_tensor = torch.stack(old_price_list)
    rfr_tensor       = torch.stack(rfr_list)

    return {
        'input_ids':    input_ids_padded,
        'labels':       labels_tensor,
        'old_price':    old_price_tensor,
        'sector':       sector_list,
        'idx':          idx_list,
        'risk_free_rate': rfr_tensor
    }

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
                 risk_free_rates: list,
                 tokenizer,
                 total_epochs: int,
                 use_ebm: bool = False):
        self.df = pd.DataFrame({
            'Article': articles,
            'weighted_avg_720_hrs': prices,
            'Sector': sectors,
            'Date': dates,
            'RelatedStocksList': related_stocks_list,
            'weighted_avg_0_hrs': prices_current,
            'Symbol': symbols,
            'Industry': industries,
            'Risk_Free_Rate': risk_free_rates
        })
        self.tokenizer = tokenizer
        self.total_epochs = total_epochs
        self.use_ebm = use_ebm

        print("Tokenizing articles...")
        encodings = tokenizer(
            articles,
            truncation=True,
            padding='max_length',
            max_length=config.BLOCK_SIZE,
            return_tensors='pt'
        )
        self.tokenized_articles = encodings["input_ids"]

        self.prices = prices
        self.sectors = sectors
        self.dates = dates
        self.related_stocks_list = related_stocks_list
        self.prices_current = prices_current
        self.symbols = symbols
        self.industries = industries
        self.risk_free_rates = risk_free_rates

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_ids = self.tokenized_articles[idx]
        future_price = self.prices[idx]
        sector = self.sectors[idx] if self.sectors[idx] is not None else "Unknown Sector"
        old_price = self.prices_current[idx]
        risk_free = self.risk_free_rates[idx]

        sample = {
            'input_ids': input_ids,
            'labels': torch.tensor(future_price, dtype=torch.float),
            'sector': sector,
            'idx': int(idx),
            'old_price': torch.tensor(old_price, dtype=torch.float),
            'risk_free_rate': torch.tensor(risk_free, dtype=torch.float)
        }
        return sample

class RollingWindowDataset(IterableDataset):
    def __init__(self, main_parquet_path, preprocessed_parquet_path, streaming_size, overlap, tokenizer, mode):
        self.main_parquet_path = main_parquet_path
        self.preprocessed_parquet_path = preprocessed_parquet_path
        self.streaming_size = streaming_size
        self.overlap = overlap
        self.tokenizer = tokenizer
        self.mode = mode
        self.total_rows = self.get_total_rows(main_parquet_path)
        self.current_window = 0

    def get_total_rows(self, path):
        table = pq.read_table(path, columns=[])
        return table.num_rows

    def __iter__(self):
        window_start = 0
        while window_start < self.total_rows:
            window_end = min(window_start + self.streaming_size, self.total_rows)
            logger.info(f"Loading window rows {window_start} to {window_end}")
            main_table = pq.read_table(self.main_parquet_path, row_slice=(window_start, window_end - window_start))
            pre_table = pq.read_table(self.preprocessed_parquet_path, row_slice=(window_start, window_end - window_start))
            df = main_table.to_pandas()
            for idx, row in df.iterrows():
                article = row.get('Article', 'N/A')
                future_price = row.get('weighted_avg_720_hrs', 0.0)
                encoding = self.tokenizer(
                    article,
                    truncation=True,
                    padding='max_length',
                    max_length=config.BLOCK_SIZE,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].squeeze(0)
                sample = {
                    'input_ids': input_ids,
                    'labels': torch.tensor(future_price, dtype=torch.float),
                    'idx': int(idx + window_start)
                }
                yield sample
            window_start = window_start + (self.streaming_size - self.overlap)
