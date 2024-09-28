import torch
from torch.utils.data import Dataset
from utils.config import *

class ArticlePriceDataset(torch.utils.data.Dataset):
    def __init__(self, articles, prices, sectors, tokenizer):
        self.articles = articles
        self.prices = prices
        self.sectors = sectors
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.articles[idx],
            truncation=True,
            padding='max_length',
            max_length=block_size,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.prices[idx], dtype=torch.float)
        item['sector'] = self.sectors[idx]  # Include sector information
        return item
