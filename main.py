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
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument('mode', choices=['train', 'run'], help="Mode: 'train' to train the model, 'run' to predict price")
    parser.add_argument('input_text', type=str, nargs='?', help="Input article text (required if mode is 'run' without --test)", default=None)
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument('--data_path', type=str, default="/content/test_df.csv", help="Path to the dataset CSV file")
    parser.add_argument('--test', action='store_true', help="If specified in 'run' mode, evaluate the model on the test set.")

    args = parser.parse_args()

    torch.manual_seed(1337)

    # Detect the available device(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    print(f"Using device: {device}, Number of GPUs: {n_gpus}")

    # Initialize the model and tokenizer
    model = SparseMoELanguageModel(tokenizer_name=args.tokenizer_name)
    tokenizer = model.tokenizer

    # Set pad token to eos token
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

        # Prepare the articles and prices
        articles = [
            "Symbol: " + str(row['Symbol_x']) +
            "\nSecurity: " + str(row['Date_x']) +
            "\nRelated Stocks/Topics: " + str(row['RelatedStocksList']) +
            "\nArticle Content: " + str(row['Article']) +
            "\nArticle Title: " + str(row['Title']) +
            "\nArticle Type: " + str(row['articleType']) +
            "\nArticle Publication: " + str(row['Publication']) +
            "\nPublication Author: " + str(row['Author']) + 
            "\nStock Price 4 days before: " + str(row['weighted_avg_-96_hrs']) +
            "\nStock Price 2 days before: " + str(row['weighted_avg_-48_hrs']) +
            "\nStock Price 1 days before: " + str(row['weighted_avg_-24_hrs']) +
            "\nStock Price 0 days before: " + str(row['weighted_avg_0_hrs'])
            for index, row in df.iterrows()
        ]
        print(articles[0],"aaa")
        prices = df['weighted_avg_720_hrs'].tolist()

        # Split the data
        train_articles, test_articles, train_prices, test_prices = train_test_split(
            articles, prices, test_size=0.15, random_state=42)

        # Create dataset and dataloader for training
        train_dataset = ArticlePriceDataset(train_articles, train_prices, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Train the model
        print("Training Started!")
        train_model(model, optimizer, EPOCHS, device, train_dataloader)

        # Save the model weights
        os.makedirs('model', exist_ok=True)
        torch.save(model.state_dict(), 'model/model_weights.pth')

    elif args.mode == 'run':
        if args.test:
            # Evaluate the model on the test set
            # Load the model weights
            directory = "model"
            filename = "model_weights.pth"
            filepath = os.path.join(directory, filename)
            model.load_state_dict(torch.load(filepath, map_location=device))
            model = model.to(device)
            model.eval()

            # Load and prepare the data
            df = pd.read_csv(args.data_path)
            df = df[df['weighted_avg_720_hrs'] > 0]

            articles = [
                "Symbol: " + str(row['Symbol_x']) +
                "\nSecurity: " + str(row['Date_x']) +
                "\nRelated Stocks/Topics: " + str(row['RelatedStocksList']) +
                "\nArticle Content: " + str(row['Article']) +
                "\nArticle Title: " + str(row['Title']) +
                "\nArticle Type: " + str(row['articleType']) +
                "\nArticle Publication: " + str(row['Publication']) +
                "\nPublication Author: " + str(row['Author']) + 
                "\nStock Price 4 days before: " + str(row['weighted_avg_-96_hrs']) +
                "\nStock Price 2 days before: " + str(row['weighted_avg_-48_hrs']) +
                "\nStock Price 1 days before: " + str(row['weighted_avg_-24_hrs']) +
                "\nStock Price 0 days before: " + str(row['weighted_avg_0_hrs'])
                for index, row in df.iterrows()
            ]

            prices = df['weighted_avg_720_hrs'].tolist()

            # Split the data
            _, test_articles, _, test_prices = train_test_split(
                articles, prices, test_size=0.15, random_state=42)

            # Create dataset and dataloader for testing
            test_dataset = ArticlePriceDataset(test_articles, test_prices, tokenizer)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Evaluate the model
            from sklearn.metrics import mean_absolute_error, r2_score

            predictions = []
            actuals = []
            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    outputs, _ = model(input_ids=input_ids)
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(labels.cpu().numpy())

            # Calculate metrics
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            print(f"Test MAE: {mae:.4f}, R2 Score: {r2:.4f}")

        else:
            # Existing code for predicting on a single input
            if not args.input_text:
                raise ValueError("You must provide input text when mode is 'run' without --test")

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
