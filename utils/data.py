# data.py

import torch
from torch.utils.data import Dataset
from utils.config import config

class ArticlePriceDataset(Dataset):
    def __init__(self, articles, prices, sectors, tokenizer):
        self.articles = articles
        self.prices = prices
        self.sectors = sectors
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        price = self.prices[idx]
        sector = self.sectors[idx]
        encoding = self.tokenizer(
            article,
            truncation=True,
            padding='max_length',
            max_length=config.block_size,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        return {
            'input_ids': input_ids,
            'labels': torch.tensor(price, dtype=torch.float),
            'sector': sector
        }
