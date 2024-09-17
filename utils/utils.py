import torch.nn as nn
from torch.nn import init

import os
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv

def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

def get_data():
    load_dotenv('/content/MoE-Asset-Pricing/.env')
    hf_token = os.getenv('HF_TOKEN')

    login(hf_token)
    dataset = load_dataset("nbettencourt/SC454k-valid")
    df = dataset['test'].to_pandas().drop(columns=['Unnamed: 0'])
    return df
