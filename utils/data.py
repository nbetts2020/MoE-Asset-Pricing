import torch
from torch.utils.data import Dataset
from utils.config import *

class ArticlePriceDataset(torch.utils.data.Dataset):
    def __init__(self, articles, prices, tokenizer):
        self.articles = articles
        self.prices = prices
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        tokens = self.tokenizer.tokenize(article)
        if len(tokens) > block_size:
            tokens = tokens[-block_size:]  # Keep the last block_size tokens
        encoding = self.tokenizer.encode_plus(
            self.tokenizer.convert_tokens_to_string(tokens),
            truncation=True,
            padding='max_length',
            max_length=block_size,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = torch.tensor(self.prices[idx], dtype=torch.float)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}
