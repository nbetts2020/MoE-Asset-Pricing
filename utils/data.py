# utils/data.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import logging
import os
import concurrent.futures
import numpy as np

from utils.sampling import preprocess_data
from utils.config import config
from utils.ebm import scale_energy, compute_sampling_probabilities

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """
        Initializes the dataset.
        """
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


def _get_candidate_contexts(df, idx):
    sample = df.iloc[idx]
    target_date = sample.get('Date', pd.Timestamp('1970-01-01'))
    target_symbol = sample.get('Symbol', 'Unknown Symbol')
    target_sector = sample.get('Sector', 'Unknown Sector')
    target_industry = sample.get('Industry', 'Unknown Industry')

    start_date = target_date - pd.Timedelta(days=30)

    date_filtered = df[
        (df['Date'] >= start_date) &
        (df['Date'] < target_date)
    ]

    economic_articles = date_filtered[
        date_filtered['RelatedStocksList'].str.contains(r'\bMarkets\b', na=False)
    ]

    industry_articles = date_filtered[
        date_filtered['Industry'] == target_industry
    ]

    sector_articles = date_filtered[
        date_filtered['Sector'] == target_sector
    ]

    stock_articles_all = df[
        (df['Symbol'] == target_symbol) &
        (df['Date'] < target_date - pd.Timedelta(days=30))
    ]
    if 'Percentage Change' in stock_articles_all.columns:
        stock_articles_all = stock_articles_all.nlargest(25, 'Percentage Change')
    stock_articles = stock_articles_all

    last_8_articles = df[
        (df['Symbol'] == target_symbol) &
        (df['Date'] < target_date)
    ].sort_values(by='Date', ascending=False).head(8).sort_values(by='Date', ascending=True)

    contexts = []
    def format_context(row, prefix):
        date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
        return (
            f"{prefix}\n"
            f"Date: {date_str}\n"
            f"Title: {row.get('Title', 'N/A')}\n"
            f"Article: {row.get('Article', 'N/A')}\n"
        )

    for df_ctx, prefix in [
        (economic_articles, "Broader Economic Information"),
        (industry_articles, "Broader Industry Information"),
        (sector_articles, "Broader Sector Information"),
        (stock_articles, "Information Potentially Indicating Significant Market Movement"),
        (last_8_articles, "Last 8 Articles for Current Stock")
    ]:
        for _, row in df_ctx.iterrows():
            ctx_str = format_context(row, prefix)
            contexts.append(ctx_str)

    current_article_str = (
        f"Current Article:\n"
        f"Symbol: {sample.get('Symbol', 'N/A')}\n"
        f"Security: {sample.get('Security', 'N/A')}\n"
        f"Related Stocks/Topics: {sample.get('RelatedStocksList', 'N/A')}\n"
        f"Title: {sample.get('Title', 'N/A')}\n"
        f"Type: {sample.get('articleType', 'N/A')}\n"
        f"Publication: {sample.get('Publication', 'N/A')}\n"
        f"Publication Author: {sample.get('Author', 'N/A')}\n"
        f"Date: {sample.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')}\n"
        f"Article: {sample.get('Article', 'N/A')}\n"
    )

    return contexts, current_article_str

def parallel_context_selection_worker(args):
    """
    Worker function (CPU-only!) that returns the contexts and current_article_str for each sample.

    IMPORTANT: No GPU operations or model calls here!
    """
    idx, df = args
    contexts, current_article_str = _get_candidate_contexts(df, idx)
    # Just return raw contexts and current_article_str
    return (contexts, current_article_str)

def custom_collate_fn(batch, df, ebm, model, tokenizer, device, use_ebm):
    """
    Custom collate function:
    - First, we gather the basic batch data from __getitem__.
    - Then, if use_ebm is True, we parallelize fetching contexts via parallel_context_selection_worker (CPU-only).
    - After that, in the main process, we run the EBM logic (embedding, energies) on GPU.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    sectors = [item['sector'] for item in batch]
    idxs = [item['idx'] for item in batch]

    if use_ebm and ebm is not None and model is not None:
        # CPU-only parallel fetching of contexts
        args_list = [(idx, df) for idx in idxs]
        num_workers = os.cpu_count() or 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(parallel_context_selection_worker, args_list))

        # Now we have a list of (contexts, current_article_str) for each sample
        # Perform EBM logic here in the main process (GPU ops)
        context_input_ids_list = []
        for (contexts, current_article_str) in results:
            if not contexts:
                # No contexts available
                # Just create a dummy tensor of zeros for context_input_ids
                context_input_ids_list.append(torch.zeros(config.BLOCK_SIZE, dtype=torch.long))
                continue

            # Compute embeddings and energies now
            # Tokenize article
            article_enc = tokenizer(
                current_article_str,
                truncation=True,
                padding='max_length',
                max_length=config.BLOCK_SIZE,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                article_embedding = model.get_embeddings(article_enc['input_ids'])  # (1, embed_dim)
                article_embedding = article_embedding.squeeze(0)

            # Tokenize contexts
            encodings = tokenizer(
                contexts,
                truncation=True,
                padding=True,
                max_length=config.BLOCK_SIZE,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                context_embeddings = model.get_embeddings(encodings['input_ids']) # (num_ctx, embed_dim)

            # Compute energies with EBM
            with torch.no_grad():
                article_emb_expand = article_embedding.unsqueeze(0).repeat(len(contexts), 1)
                energies = ebm(article_emb_expand, context_embeddings)  # (num_ctx,)

            energies = energies.unsqueeze(0)
            scaled_energies = scale_energy(energies)
            probabilities = compute_sampling_probabilities(scaled_energies, temperature=1.0)
            probabilities = probabilities.squeeze(0).cpu().numpy()

            num_to_select = min(5, len(contexts))
            selected_indices = np.random.choice(len(contexts), size=num_to_select, replace=False, p=probabilities)
            selected_contexts = [contexts[i] for i in selected_indices]
            concatenated_contexts = "\n".join(selected_contexts)

            # Tokenize final selected contexts
            context_encoding = tokenizer(
                concatenated_contexts,
                truncation=True,
                padding='max_length',
                max_length=config.BLOCK_SIZE,
                return_tensors='pt'
            )
            context_input_ids = context_encoding['input_ids'].squeeze(0)
            context_input_ids_list.append(context_input_ids)

        context_input_ids = torch.stack(context_input_ids_list)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'sector': sectors,
            'context_input_ids': context_input_ids,
            'idx': idxs
        }
    else:
        # EBM not used, just return the base batch
        return {
            'input_ids': input_ids,
            'labels': labels,
            'sector': sectors,
            'idx': idxs
        }
