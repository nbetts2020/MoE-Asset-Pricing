import os
# Disable tokenizer parallelism to avoid warnings after fork.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F
import pandas as pd
import logging
import numpy as np
import random
import ast
from tqdm import tqdm

from transformers import AutoTokenizer
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
    if "Percentage Change" in stock_df.columns:
        # Convert to numeric so that nlargest works properly.
        stock_df["Percentage Change"] = pd.to_numeric(stock_df["Percentage Change"], errors="coerce")
        stock_df = stock_df.nlargest(5, "Percentage Change")
    else:
        logger.error(f"'Percentage Change' missing in stock_df. Columns: {stock_df.columns}")
        stock_df = stock_df.head(5)
    if not stock_df.empty:
        for _, row in stock_df.iterrows():
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

    # Last 8 Articles for Current Stock (again)
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

# -------------------------------------------------------------------------
# GLOBAL TOKENIZER SETUP
# -------------------------------------------------------------------------
TOKENIZER_NAME = getattr(config, "TOKENIZER_NAME", "gpt2")
GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token
GLOBAL_TOKENIZER.model_max_length = config.BLOCK_SIZE

# -------------------------------------------------------------------------
# PRE-TOKENIZED FIXED STRINGS (HEADERS, ETC.)
# -------------------------------------------------------------------------
def pretokenize_field(value):
    """
    Convert 'value' to string and return a list of token IDs from the global tokenizer, with no special tokens.
    """
    if not isinstance(value, str):
        value = str(value)
    return GLOBAL_TOKENIZER.encode(value, add_special_tokens=False)

FIXED_TOKENS = {
    "newline": GLOBAL_TOKENIZER.encode("\n", add_special_tokens=False),
    "date_prefix": GLOBAL_TOKENIZER.encode("Date: ", add_special_tokens=False),
    "title_prefix": GLOBAL_TOKENIZER.encode("Title: ", add_special_tokens=False),
    "article_prefix": GLOBAL_TOKENIZER.encode("Article: ", add_special_tokens=False),

    "section_header_economic":
        GLOBAL_TOKENIZER.encode("Broader Economic Information:", add_special_tokens=False),
    "section_header_industry":
        GLOBAL_TOKENIZER.encode("Broader Industry Information:", add_special_tokens=False),
    "section_header_sector":
        GLOBAL_TOKENIZER.encode("Broader Sector Information:", add_special_tokens=False),
    "section_header_movement":
        GLOBAL_TOKENIZER.encode("Information Potentially Indicating Significant Market Movement Related to Current Stock:", add_special_tokens=False),
    "section_header_last8":
        GLOBAL_TOKENIZER.encode("Last 8 Articles for Current Stock:", add_special_tokens=False),

    "main_article_label":
        GLOBAL_TOKENIZER.encode("MAIN ARTICLE:\n", add_special_tokens=False),
}

# -------------------------------------------------------------------------
# BUILDING CONTEXTS AS PRE-TOKENIZED LISTS
# -------------------------------------------------------------------------
def build_candidate_context_tokens(sample: dict) -> list:
    """
    Build a candidate context by concatenating pre-tokenized components from the sample dictionary.
    Returns a single list of token IDs.
    """
    tokens = []

    # 1) ECONOMIC SECTION
    tokens += FIXED_TOKENS["section_header_economic"] + FIXED_TOKENS["newline"]
    markets = sample.get("markets", pd.DataFrame()).head(5)
    for _, row in markets.iterrows():
        date_str = _safe_date_str(row.get("Date", pd.Timestamp("1970-01-01")))
        title = row.get("Title", "N/A")
        article = row.get("Article", "N/A")
        tokens += FIXED_TOKENS["date_prefix"] + pretokenize_field(date_str) + FIXED_TOKENS["newline"]
        tokens += FIXED_TOKENS["title_prefix"] + pretokenize_field(title) + FIXED_TOKENS["newline"]
        tokens += FIXED_TOKENS["article_prefix"] + pretokenize_field(article) + FIXED_TOKENS["newline"]
    tokens += FIXED_TOKENS["newline"]

    # 2) INDUSTRY SECTION
    tokens += FIXED_TOKENS["section_header_industry"] + FIXED_TOKENS["newline"]
    industry = sample.get("industry", pd.DataFrame()).head(5)
    for _, row in industry.iterrows():
        date_str = _safe_date_str(row.get("Date", pd.Timestamp("1970-01-01")))
        title = row.get("Title", "N/A")
        article = row.get("Article", "N/A")
        tokens += FIXED_TOKENS["date_prefix"] + pretokenize_field(date_str) + FIXED_TOKENS["newline"]
        tokens += FIXED_TOKENS["title_prefix"] + pretokenize_field(title) + FIXED_TOKENS["newline"]
        tokens += FIXED_TOKENS["article_prefix"] + pretokenize_field(article) + FIXED_TOKENS["newline"]
    tokens += FIXED_TOKENS["newline"]

    # 3) SECTOR SECTION
    tokens += FIXED_TOKENS["section_header_sector"] + FIXED_TOKENS["newline"]
    sector_df = sample.get("sector", pd.DataFrame()).head(5)
    for _, row in sector_df.iterrows():
        date_str = _safe_date_str(row.get("Date", pd.Timestamp("1970-01-01")))
        title = row.get("Title", "N/A")
        article = row.get("Article", "N/A")
        tokens += FIXED_TOKENS["date_prefix"] + pretokenize_field(date_str) + FIXED_TOKENS["newline"]
        tokens += FIXED_TOKENS["title_prefix"] + pretokenize_field(title) + FIXED_TOKENS["newline"]
        tokens += FIXED_TOKENS["article_prefix"] + pretokenize_field(article) + FIXED_TOKENS["newline"]
    tokens += FIXED_TOKENS["newline"]

    # 4) MARKET MOVEMENT SECTION
    tokens += FIXED_TOKENS["section_header_movement"] + FIXED_TOKENS["newline"]
    stock_df = sample.get("stock", pd.DataFrame())
    if "Percentage Change" in stock_df.columns:
        # Convert the column to numeric to ensure nlargest works correctly.
        stock_df["Percentage Change"] = pd.to_numeric(stock_df["Percentage Change"], errors="coerce")
        stock_df = stock_df.nlargest(5, "Percentage Change")
    else:
        stock_df = stock_df.head(5)
    for _, row in stock_df.iterrows():
        date_str = _safe_date_str(row.get("Date", pd.Timestamp("1970-01-01")))
        title = row.get("Title", "N/A")
        article = row.get("Article", "N/A")
        tokens += FIXED_TOKENS["date_prefix"] + pretokenize_field(date_str) + FIXED_TOKENS["newline"]
        tokens += FIXED_TOKENS["title_prefix"] + pretokenize_field(title) + FIXED_TOKENS["newline"]
        tokens += FIXED_TOKENS["article_prefix"] + pretokenize_field(article) + FIXED_TOKENS["newline"]
    tokens += FIXED_TOKENS["newline"]

    # 5) LAST 8 ARTICLES
    tokens += FIXED_TOKENS["section_header_last8"] + FIXED_TOKENS["newline"]
    df_last8 = sample.get("last_8", pd.DataFrame()).head(8)
    for _, row in df_last8.iterrows():
        date = row.get("Date", pd.Timestamp("1970-01-01"))
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date, errors="coerce")
            if pd.isna(date):
                date = pd.Timestamp("1970-01-01")
        date_str = date.strftime("%Y-%m-%d")
        tokens += FIXED_TOKENS["date_prefix"] + pretokenize_field(date_str) + FIXED_TOKENS["newline"]
        title = row.get("Title", "N/A")
        tokens += FIXED_TOKENS["title_prefix"] + pretokenize_field(title) + FIXED_TOKENS["newline"]
        article = row.get("Article", "N/A")
        tokens += FIXED_TOKENS["article_prefix"] + pretokenize_field(article) + FIXED_TOKENS["newline"]
        sp_4d = row.get("weighted_avg_-96_hrs", "N/A")
        sp_2d = row.get("weighted_avg_-48_hrs", "N/A")
        sp_1d = row.get("weighted_avg_-24_hrs", "N/A")
        sp_release = row.get("weighted_avg_0_hrs", "N/A")
        tokens += pretokenize_field(f"Stock Price 4 days before: {sp_4d}\n")
        tokens += pretokenize_field(f"Stock Price 2 days before: {sp_2d}\n")
        tokens += pretokenize_field(f"Stock Price 1 day before: {sp_1d}\n")
        tokens += pretokenize_field(f"Stock Price at release: {sp_release}\n")
        rfr = row.get("Risk_Free_Rate", "N/A")
        tokens += pretokenize_field(f"Risk-Free Rate at release: {rfr}\n")
        tokens += FIXED_TOKENS["newline"]

    # 6) MAIN ARTICLE
    tokens += FIXED_TOKENS["main_article_label"]
    current_df = sample.get("current", pd.DataFrame()).head(1)
    if not current_df.empty:
        row = current_df.iloc[0]
        date_val = row.get("Date", pd.Timestamp("1970-01-01"))
        date_str = _safe_date_str(date_val)
        tokens += FIXED_TOKENS["date_prefix"] + pretokenize_field(date_str) + FIXED_TOKENS["newline"]
        title = row.get("Title", "N/A")
        tokens += FIXED_TOKENS["title_prefix"] + pretokenize_field(title) + FIXED_TOKENS["newline"]
        main_article = row.get("Article", "N/A")
        tokens += FIXED_TOKENS["article_prefix"] + pretokenize_field(main_article) + FIXED_TOKENS["newline"]
        sp_4d = row.get("weighted_avg_-96_hrs", "N/A")
        sp_2d = row.get("weighted_avg_-48_hrs", "N/A")
        sp_1d = row.get("weighted_avg_-24_hrs", "N/A")
        sp_release = row.get("weighted_avg_0_hrs", "N/A")
        tokens += pretokenize_field(f"Stock Price 4 days before: {sp_4d}\n")
        tokens += pretokenize_field(f"Stock Price 2 days before: {sp_2d}\n")
        tokens += pretokenize_field(f"Stock Price 1 day before: {sp_1d}\n")
        tokens += pretokenize_field(f"Stock Price at release: {sp_release}\n")
        rfr = row.get("Risk_Free_Rate", "N/A")
        tokens += pretokenize_field(f"Risk-Free Rate at release: {rfr}\n")
    else:
        tokens += pretokenize_field("N/A") + FIXED_TOKENS["newline"]

    return tokens

def _safe_date_str(date_val):
    """Convert date_val to a safe YYYY-MM-DD string."""
    if not isinstance(date_val, pd.Timestamp):
        date_val = pd.to_datetime(date_val, errors="coerce")
        if pd.isna(date_val):
            date_val = pd.Timestamp("1970-01-01")
    return date_val.strftime("%Y-%m-%d")

def parallel_context_generation_worker(args):
    """
    CPU-only worker that generates multiple candidate contexts for one sample in the EBM approach.
    Instead of returning raw strings, it returns lists of token IDs built by concatenating pre-tokenized fields.
    """
    (idx, df, df_preprocessed, total_epochs, current_epoch, context_count) = args
    # Adjust context_count based on the remaining epochs.
    candidate_contexts = []
    if idx < 0 or idx >= len(df):
        logger.error(f"Index {idx} is out-of-bounds for the main DataFrame.")
        return candidate_contexts
    main_row = df.iloc[idx]
    preproc_row = df_preprocessed.iloc[idx]

    # These are the maximum number of references we take from each category
    sample_map = {
        'use_ebm_economic': 5,
        'use_ebm_industry': 5,
        'use_ebm_sector': 5,
        'use_ebm_top25': 5,
    }

    for _ in range(context_count):
        # Get the reference lists and cast them to integers for valid indexing.
        econ_array = preproc_row.get('use_ebm_economic', [])
        ind_array = preproc_row.get('use_ebm_industry', [])
        sec_array = preproc_row.get('use_ebm_sector', [])
        hist_array = preproc_row.get('use_ebm_historical', [])
        top25_array = preproc_row.get('use_ebm_top25', [])

        if len(econ_array) > 0:
            econ_indices = np.random.choice(np.array(econ_array, dtype=int),
                                             size=min(len(econ_array), sample_map['use_ebm_economic']),
                                             replace=False)
        else:
            econ_indices = np.array([], dtype=int)
        markets_df = df.loc[econ_indices].copy() if econ_indices.size > 0 else pd.DataFrame()

        if len(ind_array) > 0:
            ind_indices = np.random.choice(np.array(ind_array, dtype=int),
                                            size=min(len(ind_array), sample_map['use_ebm_industry']),
                                            replace=False)
        else:
            ind_indices = np.array([], dtype=int)
        industry_df = df.loc[ind_indices].copy() if ind_indices.size > 0 else pd.DataFrame()

        if len(sec_array) > 0:
            sec_indices = np.random.choice(np.array(sec_array, dtype=int),
                                            size=min(len(sec_array), sample_map['use_ebm_sector']),
                                            replace=False)
        else:
            sec_indices = np.array([], dtype=int)
        sector_df = df.loc[sec_indices].copy() if sec_indices.size > 0 else pd.DataFrame()

        if len(hist_array) > 0:
            hist_indices = np.array(hist_array, dtype=int)
        else:
            hist_indices = np.array([], dtype=int)
        last_8_df = df.loc[hist_indices].copy() if hist_indices.size > 0 else pd.DataFrame()

        if len(top25_array) > 0:
            top25_indices = np.random.choice(np.array(top25_array, dtype=int),
                                             size=min(len(top25_array), sample_map['use_ebm_top25']),
                                             replace=False)
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

        token_list = build_candidate_context_tokens(sample_dict)
        candidate_contexts.append(token_list)

    return (idx, candidate_contexts)

# -------------------------------------------------------------------------
# DATASET/LOADER CLASSES
# -------------------------------------------------------------------------
class ArticlePriceDataset(Dataset):
    """
    Dataset that loads the full DataFrame into memory and pre-tokenizes each article.
    This replaces the previous rolling-window dataset.
    """
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, total_epochs: int, use_ebm: bool = False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing at least these columns:
                'Article', 'weighted_avg_720_hrs', 'weighted_avg_0_hrs', 'Sector',
                'Date', 'RelatedStocksList', 'Symbol', 'Industry', 'Risk_Free_Rate'
            tokenizer: Pretrained tokenizer (e.g. GPT2).
            total_epochs (int): Total number of training epochs.
            use_ebm (bool): Whether EBM sampling is enabled.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.total_epochs = total_epochs
        self.use_ebm = use_ebm

        logger.info("Pre-tokenizing all articles in ArticlePriceDataset...")
        # Pre-tokenize each article and store the tokenized tensors.
        self.tokenized_articles = []
        articles = self.df['Article'].tolist()
        for article in tqdm(articles, desc="Pre-tokenizing", unit="article"):
            encoding = self.tokenizer(
                article,
                truncation=True,
                padding='max_length',
                max_length=config.BLOCK_SIZE,
                return_tensors='pt'
            )
            # Squeeze out the batch dimension
            self.tokenized_articles.append(encoding['input_ids'].squeeze(0))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns a dictionary for one sample.
        """
        row = self.df.iloc[idx]
        # Retrieve the pre-tokenized article
        input_ids = self.tokenized_articles[idx]

        future_price = row.get('weighted_avg_720_hrs', 0.0)
        old_price    = row.get('weighted_avg_0_hrs', 0.0)
        sector       = row.get('Sector', 'Unknown Sector')
        risk_free    = row.get('Risk_Free_Rate', 0.0)

        sample = {
            'input_ids': input_ids,
            'labels': torch.tensor(future_price, dtype=torch.float),
            'sector': sector if sector is not None else "Unknown Sector",
            'idx': int(idx),
            'old_price': torch.tensor(old_price, dtype=torch.float),
            'risk_free_rate': torch.tensor(risk_free, dtype=torch.float)
        }
        return sample

# -------------------------------------------------------------------------
# CUSTOM COLLATE
# -------------------------------------------------------------------------
def custom_collate_fn(batch):
    """
    Minimal collate function: merges CPU data into a single batch.
    """
    input_ids_list = []
    labels_list = []
    old_price_list = []
    sector_list = []
    idx_list = []
    rfr_list = []

    for sample in batch:
        input_ids_list.append(sample['input_ids'])
        labels_list.append(sample['labels'])
        old_price_list.append(sample.get('old_price', torch.tensor(0.0)))
        sector_list.append(sample.get('sector', 'Unknown Sector'))
        idx_list.append(sample.get('idx', -1))
        rfr_list.append(sample.get('risk_free_rate', torch.tensor(0.0)))

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True,
        padding_value=GLOBAL_TOKENIZER.eos_token_id
    )
    labels_tensor = torch.stack(labels_list)
    old_price_tensor = torch.stack(old_price_list)
    rfr_tensor = torch.stack(rfr_list)

    return {
        'input_ids': input_ids_padded,
        'labels': labels_tensor,
        'old_price': old_price_tensor,
        'sector': sector_list,
        'idx': idx_list,
        'risk_free_rate': rfr_tensor
    }
