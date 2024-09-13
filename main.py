import torch
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer
from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import kaiming_init_weights
from utils.config import *
from torch.utils.data import DataLoader
from utils.data import ArticlePriceDataset

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument('mode', choices=['train', 'run'], help="Mode: 'train' to train the model, 'run' to predict price")
    parser.add_argument('input_text', type=str, nargs='?', help="Input article text (required if mode is 'run')", default=None)
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument('--data_path', type=str, default="/content/test_df.csv", help="Path to the dataset CSV file")

    args = parser.parse_args()

    torch.manual_seed(1337)

    # Detect the available device(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    print(f"Using device: {device}, Number of GPUs: {n_gpus}")

    # Initialize the model and tokenizer
    model = SparseMoELanguageModel(tokenizer_name=args.tokenizer_name)
    tokenizer = model.tokenizer

    # **Set pad token to eos token**
    tokenizer.pad_token = tokenizer.eos_token

    if args.mode == 'train':
        model.apply(kaiming_init_weights)

        if n_gpus > 1:
            model = torch.nn.DataParallel(model)

        model = model.to(device)
        print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Load your dataframe with 'Article' and 'Price' columns
        df = pd.read_csv(args.data_path)
        df = df[df['weighted_avg_720_hrs'] > 0]
        articles = [
            "Symbol: " + row['Symbol_x'] +
            "\nSecurity: " + row['Date_x'] +
            "\nRelated Stocks/Topics: " + row['RelatedStocksList'] +
            "\nArticle Content: " + row['Article'] +
            "\nArticle Title: " + row['Title'] +
            "\nArticle Type: " + row['articleType'] +
            "\nArticle Publication: " + row['Publication'] +
            "\nPublication Author: " + row['Author']
            for index, row in df.iterrows()
        ]

        prices = df['weighted_avg_720_hrs'].tolist()

        # Optionally normalize prices
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # prices = scaler.fit_transform(np.array(prices).reshape(-1, 1)).flatten()

        # Create dataset and dataloader
        dataset = ArticlePriceDataset(articles, prices, tokenizer)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print("Training Started!")
        # Train the model
        train_model(model, optimizer, EPOCHS, device, dataloader)

        # Save the model weights
        os.makedirs('model', exist_ok=True)
        torch.save(model.state_dict(), 'model/model_weights.pth')

    elif args.mode == 'run':
        if not args.input_text:
            raise ValueError("You must provide input text when mode is 'run'")

        # Load the model weights
        directory = "model"
        filename = "model_weights.pth"
        filepath = os.path.join(directory, filename)

        model.load_state_dict(torch.load(filepath, map_location=device))
        model = model.to(device)
        model.eval()

        # Tokenize the input text
        encoding = tokenizer(
            args.input_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        ).to(device)
        input_ids = encoding["input_ids"]

        # Make prediction
        with torch.no_grad():
            prediction, _ = model(input_ids=input_ids)
        print(f"Predicted Price: {prediction.item()}")

if __name__ == "__main__":
    main()
