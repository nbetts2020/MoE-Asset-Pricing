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
        )

    concatenated_articles = "\n".join(formatted_articles)
    return concatenated_articles

def parallel_context_generation_worker(args):
    """
    CPU-only worker. No EBM or GPU ops here.
    Returns multiple raw context strings (or CPU tensors) for each sample.
    """
    (idx, df, tokenizer, total_epochs, current_epoch, context_count) = args
    # We'll store multiple raw context strings
    candidate_contexts = []

    for _ in range(context_count):
        # 1) sample articles
        sampled_list = sample_articles(df, index_list=[idx])
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

def custom_collate_fn(
    batch,
    df,
    ebm,
    model,
    tokenizer,
    device,
    use_ebm,
    total_epochs,
    current_epoch,
    context_count=5,
    temperature=0.7  # for Boltzmann
):
    """
    CPU parallel for multiple contexts, then select one via Monte Carlo on GPU.
    """
    input_ids = []
    labels = []

    # We'll store final chosen contexts here, if use_ebm is True
    chosen_context_ids = [] if use_ebm else None

    # Step A: Gather CPU-worker args if EBM is used
    if use_ebm:
        args_list = []
        for sample in batch:
            idx = sample['idx']
            args_list.append((idx, df, tokenizer, total_epochs, current_epoch, context_count))
        # Launch CPU worker to get multiple context strings
        with ProcessPoolExecutor() as executor:
            # results => list of [candidate_str1, candidate_str2, ...] per sample
            all_candidates = list(executor.map(parallel_context_generation_worker, args_list))
    else:
        all_candidates = None

    print(all_candidates[0][0], len(all_candidates), len(all_candidates[0]), "ahhhh!")

    # Step B: Convert main article input to Tensors
    for i, sample in enumerate(batch):
        inp = sample['input_ids']
        lbl = sample['labels']
        if not isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp, dtype=torch.long)
        if not isinstance(lbl, torch.Tensor):
            lbl = torch.tensor(lbl, dtype=torch.float)
        input_ids.append(inp)
        labels.append(lbl)

    # Step C: If EBM, handle multiple contexts per sample
    if use_ebm:
        # We'll do the entire selection on GPU
        # Build final (per-sample) chosen context
        for i, candidate_strings in enumerate(all_candidates):
            if not candidate_strings:
                # Fallback if no candidate contexts
                chosen_context_ids.append(torch.zeros(1, dtype=torch.long))
                continue

            # 1) Tokenize each candidate -> shape [num_candidates, seq_len]
            candidate_tensors = []
            for ctx_str in candidate_strings:
                enc = tokenizer(
                    ctx_str,
                    truncation=True,
                    padding='max_length',
                    max_length=config.BLOCK_SIZE,
                    return_tensors='pt'
                )
                candidate_tensors.append(enc['input_ids'].squeeze(0))

            candidate_tensors = torch.stack(candidate_tensors, dim=0).to(device)
              # shape: [num_candidates, seq_len]

            # 2) Compute EBM energies
            with torch.no_grad():
                article_emb = model.get_embeddings(input_ids[i].unsqueeze(0).to(device))
                  # shape [1, embed_dim], expand to match
                # Compute embeddings for each candidate context
                context_emb = model.get_embeddings(candidate_tensors) # [num_candidates, embed_dim]

                # EBM => shape [num_candidates]
                # Expand article_emb to match
                expanded_article_emb = article_emb.expand(context_emb.size(0), -1)
                energies = ebm(expanded_article_emb, context_emb) # [num_candidates]

            # 3) Scale energies (min-max) then transform into probabilities
            e_min = energies.min()
            e_max = energies.max()
            scaled = (energies - e_min) / ( (e_max - e_min) + 1e-8 )  # in [0..1]
            # Boltzmann
            probs = F.softmax(-scaled / temperature, dim=0)  # shape [num_candidates]

            # 4) Sample exactly 1 context
            sampled_idx = torch.multinomial(probs, 1).item()

            # 5) Chosen context tokens -> store
            chosen_context = candidate_tensors[sampled_idx]  # shape [seq_len]
            chosen_context_ids.append(chosen_context)
    # else no EBM => skip

    # Step D: pad sequences on CPU
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_tensor = torch.stack(labels)

    # Step E: If no EBM, just return
    if not use_ebm:
        return {
            'input_ids': input_ids_padded.to(device),
            'labels': labels_tensor.to(device)
        }

    # If EBM => also pad the single chosen contexts
    # shape => [batch_size, chosen_seq_len]
    context_input_ids_padded = torch.nn.utils.rnn.pad_sequence(chosen_context_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

    # Return final
    return {
        'input_ids': input_ids_padded.to(device),
        'labels': labels_tensor.to(device),
        'context_input_ids': context_input_ids_padded
    }
