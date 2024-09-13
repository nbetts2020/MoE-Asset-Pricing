import torch
from torch.utils.data import Dataset

class ArticlePriceDataset(Dataset):
    def __init__(self, articles, prices, tokenizer):
        self.articles = articles
        self.prices = prices
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        price = self.prices[idx]
        encoding = self.tokenizer(
            article,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(float(price), dtype=torch.float)
        }
