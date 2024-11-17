# utils/data.py

import torch
from torch.utils.data import Dataset
from utils.sampling import preprocess_data, sample_articles
from utils.config import config
import pandas as pd
import os
import concurrent.futures


class ArticlePriceDataset(Dataset):
    def __init__(self, articles: list, prices: list, sectors: list, tokenizer, total_epochs: int):
        """
        Initializes the dataset with articles, prices, sectors, tokenizer, and total_epochs.

        Args:
            articles (list): List of article texts.
            prices (list): List of corresponding prices.
            sectors (list): List of corresponding sectors.
            tokenizer: Tokenizer instance for encoding text.
            total_epochs (int): Total number of training epochs.
        """
        # Create a DataFrame from the provided lists
        self.df = pd.DataFrame({
            'Article': articles,
            'weighted_avg_720_hrs': prices,
            'Sector': sectors
        })
        self.df = preprocess_data(self.df)
        self.tokenizer = tokenizer
        self.total_epochs = total_epochs
        self.sampled_articles = []
        self.current_epoch = 0

    def prepare_epoch(self, current_epoch: int):
        """
        Prepares the dataset for the current epoch by sampling articles.

        Args:
            current_epoch (int): The current epoch number.
        """
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

    def generate_context_tuples(self, idx):
        """
        Generates context tuples for a given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            list: List of context strings.
        """
        sample = self.df.iloc[idx]
        target_date = sample['Date']
        target_symbol = sample['Symbol']
        target_sector = sample['Sector']
        target_industry = sample['Industry']

        # Define a 30-day time window before the target date
        start_date = target_date - pd.Timedelta(days=30)

        # Filter articles within the date range and before the target date
        date_filtered = self.df[(self.df['Date'] >= start_date) & (self.df['Date'] < target_date)]

        # Helper function to sample articles
        def sample_articles_subset(dataframe, n_samples):
            if len(dataframe) >= n_samples:
                return dataframe.sample(n_samples, random_state=42)
            else:
                return dataframe

        # Generate contexts from different categories
        contexts = []

        # Broader Economic Information
        economic_articles = date_filtered[date_filtered['RelatedStocksList'].str.contains(r'\bMarkets\b', na=False)]
        economic_contexts = sample_articles_subset(economic_articles, 2)['Article'].tolist()

        # Industry-Specific Information
        industry_articles = date_filtered[date_filtered['Industry'] == target_industry]
        industry_contexts = sample_articles_subset(industry_articles, 2)['Article'].tolist()

        # Sector-Specific Information
        sector_articles = date_filtered[date_filtered['Sector'] == target_sector]
        sector_contexts = sample_articles_subset(sector_articles, 2)['Article'].tolist()

        # Stock-Specific Information (Top movers)
        stock_articles = self.df[
            (self.df['Symbol'] == target_symbol) & (self.df['Date'] < target_date - pd.Timedelta(days=30))
        ].nlargest(25, 'Percentage Change')
        stock_contexts = sample_articles_subset(stock_articles, 2)['Article'].tolist()

        # Last 8 Articles
        last_8_articles = self.df[
            (self.df['Symbol'] == target_symbol) & (self.df['Date'] < target_date)
        ].sort_values(by='Date', ascending=False).head(8)
        last_8_contexts = last_8_articles['Article'].tolist()

        # Combine contexts into a list
        contexts.extend(economic_contexts)
        contexts.extend(industry_contexts)
        contexts.extend(sector_contexts)
        contexts.extend(stock_contexts)
        contexts.extend(last_8_contexts)

        return contexts

    def __len__(self):
        return len(self.sampled_articles) if self.sampled_articles else len(self.df)

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
            input_ids = encoding['input_ids'].squeeze(0)  # removing batch dimension
            label = torch.tensor(sample['current_price'], dtype=torch.float)
            sector = sample['current_sector']

            # Generate context tuples
            context_strings = self.generate_context_tuples(idx)

            # Tokenize contexts
            context_input_ids_list = []
            for context_str in context_strings:
                context_encoding = self.tokenizer(
                    context_str,
                    truncation=True,
                    padding='max_length',
                    max_length=config.BLOCK_SIZE,
                    return_tensors='pt'
                )
                context_input_ids = context_encoding['input_ids'].squeeze(0)  # removing batch dimension
                context_input_ids_list.append(context_input_ids)

            # Stack context input_ids
            if context_input_ids_list:
                context_input_ids = torch.stack(context_input_ids_list)  # shape: (num_contexts, seq_len)
            else:
                # If no contexts, create a placeholder
                context_input_ids = torch.zeros((1, config.BLOCK_SIZE), dtype=torch.long)

            return {
                'input_ids': input_ids,
                'labels': label,
                'sector': sector,
                'context_input_ids': context_input_ids  # shape: (num_contexts, seq_len)
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
