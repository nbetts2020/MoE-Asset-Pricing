import torch
from utils.config import *

def train_test_split(text, encoder):
    # Train and test splits

    data = torch.tensor(encoder(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    test_data = data[n:]
    return train_data, test_data

def get_vocab_size(chars):
    vocab_size = len(chars)
    return vocab_size

def encoder(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi[c] for c in s]
    
    return encode

def decoder(chars):
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    return decode

def get_batch(split, data):
    # generate a small batch of data of inputs x and targets y
    # data = text['input_ids'].squeeze(0)  # Remove batch dimension if present

    # Train and test splits
    data = data[0]
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, device, text):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, text)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
