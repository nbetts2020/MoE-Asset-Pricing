# data.py

import torch
from torch.utils.data import Dataset
from utils.config import config
from utils.sampling import preprocess_data, sample_articles

class ArticlePriceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, total_epochs):
        """
        Args:
            dataframe (pd.DataFrame): The complete dataset.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding text.
            total_epochs (int): Total number of training epochs.
        """
        self.df = preprocess_data(dataframe)
        self.tokenizer = tokenizer
        self.total_epochs = total_epochs
        self.sampled_articles, self.article_indices = None, None  # to be populated each epoch

    def prepare_epoch(self, current_epoch, index_list):
        """
        Prepare sampled articles for the current epoch.
        
        Args:
            current_epoch (int): The current epoch number (0-indexed).
            index_list (List[int]): List of indices to sample.
        """
        num_samples = self.total_epochs - current_epoch
        self.sampled_articles, self.article_indices = sample_articles(self.df, index_list, num_samples)

    def __len__(self):
        return len(self.sampled_articles) if self.sampled_articles else len(self.df)

    def __getitem__(self, idx):
        if self.sampled_articles:
            sample = self.sampled_articles[idx]
            # Tokenize concatenated articles
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
            # Default behavior if sampling not prepared
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
