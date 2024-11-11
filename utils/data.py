# utils/data.py

import torch
from torch.utils.data import Dataset
from utils.sampling import preprocess_data, sample_articles
from utils.config import config
import pandas as pd
import os
import concurrent.futures

class ArticlePriceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, total_epochs: int):
        self.df = preprocess_data(dataframe)
        self.tokenizer = tokenizer
        self.total_epochs = total_epochs
        self.sampled_articles = []
        self.current_epoch = 0

    def prepare_epoch(self, current_epoch: int):
        self.current_epoch = current_epoch
        num_samples = self.total_epochs - current_epoch
        if num_samples < 1:
            num_samples = 1
        index_list = self.df.index.tolist()

        self.sampled_articles = []

        # Function to generate samples for a single index
        def generate_samples(idx):
            samples = []
            for _ in range(num_samples):
                sample = sample_articles(self.df, [idx])[0]
                concatenated_articles = self.format_concatenated_articles(sample)
                current_price = sample.iloc[-1]['weighted_avg_720_hrs']
                current_sector = sample.iloc[-1]['Sector']
                samples.append({
                    'concatenated_articles': concatenated_articles,
                    'current_price': current_price,
                    'current_sector': current_sector
                })
            return samples

        # Parallel processing using ProcessPoolExecutor
        num_workers = os.cpu_count() or 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(generate_samples, idx) for idx in index_list]
            for future in concurrent.futures.as_completed(futures):
                self.sampled_articles.extend(future.result())

    def format_concatenated_articles(self, sample: pd.DataFrame) -> str:
        formatted_articles = []
        idx = sample.iloc[-1].name  # Current index

        # Broader Economic Information (Markets Articles)
        formatted_articles.append("Broader Economic Information:")
        for _, row in sample.iterrows():
            if 'Markets' in row.get('RelatedStocksList', ''):
                date_str = row['Date'].strftime('%Y-%m-%d')
                formatted_articles.append(
                    f"Date: {date_str}\n"
                    f"Title: {row['Title']}\n"
                    f"Article: {row['Article']}\n"
                )

        # Broader Industry Information
        formatted_articles.append("\nBroader Industry Information:")
        for _, row in sample.iterrows():
            if row['Industry'] == sample.iloc[-1]['Industry']:
                date_str = row['Date'].strftime('%Y-%m-%d')
                formatted_articles.append(
                    f"Date: {date_str}\n"
                    f"Title: {row['Title']}\n"
                    f"Article: {row['Article']}\n"
                )

        # Broader Sector Information
        formatted_articles.append("\nBroader Sector Information:")
        for _, row in sample.iterrows():
            if row['Sector'] == sample.iloc[-1]['Sector']:
                date_str = row['Date'].strftime('%Y-%m-%d')
                formatted_articles.append(
                    f"Date: {date_str}\n"
                    f"Title: {row['Title']}\n"
                    f"Article: {row['Article']}\n"
                )

        # Information Indicating Significant Market Movement Related to Current Stock
        formatted_articles.append("\nInformation Potentially Indicating Significant Market Movement Related to Current Stock:")
        for _, row in sample.iterrows():
            if row['Symbol'] == sample.iloc[-1]['Symbol'] and 'Percentage Change' in row:
                date_str = row['Date'].strftime('%Y-%m-%d')
                formatted_articles.append(
                    f"Date: {date_str}\n"
                    f"Title: {row['Title']}\n"
                    f"Article: {row['Article']}\n"
                    f"Percentage Change: {row['Percentage Change']:.2f}%\n"
                )

        # Last 8 Articles for Current Stock
        formatted_articles.append("\nLast 8 Articles for Current Stock:")
        for _, row in sample.iterrows():
            if row['Symbol'] == sample.iloc[-1]['Symbol']:
                date_str = row['Date'].strftime('%Y-%m-%d')
                article_details = (
                    f"Symbol: {row['Symbol']}\n"
                    f"Security: {row.get('Security', 'N/A')}\n"
                    f"Related Stocks/Topics: {row.get('RelatedStocksList', 'N/A')}\n"
                    f"Title: {row['Title']}\n"
                    f"Type: {row.get('articleType', 'N/A')}\n"
                    f"Publication: {row.get('Publication', 'N/A')}\n"
                    f"Publication Author: {row.get('Author', 'N/A')}\n"
                    f"Date: {date_str}\n"
                    f"Article: {row['Article']}\n"
                    f"Stock Price 4 days before: {row.get('weighted_avg_-96_hrs', 'N/A')}\n"
                    f"Stock Price 2 days before: {row.get('weighted_avg_-48_hrs', 'N/A')}\n"
                    f"Stock Price 1 day before: {row.get('weighted_avg_-24_hrs', 'N/A')}\n"
                    f"Stock Price at release: {row.get('weighted_avg_0_hrs', 'N/A')}\n"
                )
                formatted_articles.append(article_details)

        concatenated_articles = "\n".join(formatted_articles)
        return concatenated_articles

    def __len__(self):
        if self.sampled_articles:
            return len(self.sampled_articles)
        else:
            return len(self.df)

    def __getitem__(self, idx):
        if self.sampled_articles:
            sample = self.sampled_articles[idx]
            encoding = self.tokenizer(
                sample['concatenated_articles'],
                truncation=True,
                padding='max_length',
                max_length=config.BLOCK_SIZE,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)  # remove batch dimension
            label = torch.tensor(sample['current_price'], dtype=torch.float)
            sector = sample['current_sector']
            return {
                'input_ids': input_ids,
                'labels': label,
                'sector': sector
            }
        else:
            # Fallback to original behavior if sampling not prepared
            row = self.df.iloc[idx]
            article = row['Article']
            price = row['weighted_avg_720_hrs']
            sector = row['Sector']
            encoding = self.tokenizer(
                article,
                truncation=True,
                padding='max_length',
                max_length=config.BLOCK_SIZE,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            return {
                'input_ids': input_ids,
                'labels': torch.tensor(price, dtype=torch.float),
                'sector': sector
            }
