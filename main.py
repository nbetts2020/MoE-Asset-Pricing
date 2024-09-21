# main.py

import torch
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer
from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import kaiming_init_weights, get_data
from utils.config import *
from torch.utils.data import DataLoader
from utils.data import ArticlePriceDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bar
from utils.si import SynapticIntelligence  # Import the SI class
from utils.test import test_forgetting

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument('mode', choices=['train', 'run', 'test_forgetting'], help="Mode: 'train' to train the model, 'run' to predict price")
    parser.add_argument('input_text', type=str, nargs='?', help="Input article text (required if mode is 'run' without --test)", default=None)
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument('--test', action='store_true', help="If specified in 'run' mode, evaluate the model on the test set.")
    parser.add_argument('--update', action='store_true', help="If specified in 'train' mode, update a pre-existing model with SI.")

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

    if args.mode == 'test_forgetting':
        # Initialize the model and tokenizer
        model = SparseMoELanguageModel(tokenizer_name=args.tokenizer_name)
        model = model.to(device)

        # Load the model weights
        model.load_state_dict(torch.load('model/model_weights.pth', map_location=device))
        print("Pre-trained model loaded.")

        # Prepare the tasks (create separate dataloaders for each task)
        task_dataloaders = prepare_tasks()  # A helper function to prepare multiple task dataloaders

        # Initialize SI if applicable
        si = SynapticIntelligence(model, lambda_si=lambda_si) if args.update else None
        if si and os.path.exists('model/si_state.pth'):
            si.load_state('model/si_state.pth')
            print("SI state loaded.")

        # Run the catastrophic forgetting test
        results = test_forgetting(model, task_dataloaders, optimizer, EPOCHS, device, si=si)

        # Print or save the results
        print(results)

    elif args.mode == 'train':
        model.apply(kaiming_init_weights)

        if n_gpus > 1:
            model = torch.nn.DataParallel(model)

        model = model.to(device)
        print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")
        base_lr = learning_rate   # From config.py
        lr_decay = LR_DECAY       # From config.py
        num_layers = len(model.blocks)

        # Collect parameters for each layer
        param_groups = []

        # Embedding layers (assign lowest learning rate)
        embedding_params = list(model.token_embedding_table.parameters()) + list(model.position_embedding_table.parameters())
        param_groups.append({
            'params': embedding_params,
            'lr': base_lr * (lr_decay ** (num_layers + 1))
        })

        # Transformer blocks (decay learning rate per layer)
        for i, block in enumerate(model.blocks):
            # Deeper layers have higher indices
            layer_lr = base_lr * (lr_decay ** (num_layers - i))
            block_params = block.parameters()
            param_groups.append({'params': block_params, 'lr': layer_lr})

        # Regression head (assign base learning rate)
        regression_params = model.regression_head.parameters()
        param_groups.append({'params': regression_params, 'lr': base_lr})

        optimizer = torch.optim.AdamW(param_groups)

        # Initialize SI if update is specified
        if args.update:
            # Ensure that a pre-trained model exists
            model_path = 'model/model_weights.pth'
            si_path = 'model/si_state.pth'
            if not os.path.exists(model_path):
                raise FileNotFoundError("Pre-trained model not found. Please train the model first before updating.")

            # Load the pre-trained model
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Pre-trained model loaded.")

            # Initialize SI
            si = SynapticIntelligence(model, lambda_si=lambda_si)  # Define lambda_si in config.py or elsewhere

            # Load SI state if exists
            if os.path.exists(si_path):
                si.load_state(si_path)
                print("SI state loaded.")
            else:
                print("No existing SI state found. Starting fresh.")

        else:
            si = None  # SI is not used

        # Load your dataframe with 'Article' and 'Price' columns
        df = get_data()
        df = df[df['weighted_avg_720_hrs'] > 0]

        # Group by symbol
        grouped = df.groupby('Symbol_x', sort=False)

        # Prepare the articles and prices
        articles = []
        prices = []
        print("Grouped and Sorted!")
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            current_symbol = row['Symbol_x']
            current_date = row['Date_x']

            # Get all articles for the current symbol before the current date
            symbol_df = grouped.get_group(current_symbol)
            previous_articles = symbol_df[symbol_df['Date_x'] < current_date]

            # Get the last 10 previous articles
            last_articles = previous_articles.tail(10)

            # Build the concatenated text
            concatenated_text = ''

            # Add previous articles
            for _, prev_row in last_articles.iterrows():
                concatenated_text += (
                    "\nPrevious Article Date: " + str(prev_row['Date_x']) +
                    "\nPrevious Article Content: " + str(prev_row['Article']) +
                    "\nPrevious Article Title: " + str(prev_row['Title']) +
                    "\nPrevious Article Type: " + str(prev_row['articleType']) +
                    "\nPrevious Article Publication: " + str(prev_row['Publication']) +
                    "\nPrevious Publication Author: " + str(prev_row['Author']) +
                    "\n---\n"
                )

            # Add the current article
            concatenated_text += (
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
            )

            # Append to lists
            articles.append(concatenated_text)
            prices.append(row['weighted_avg_720_hrs'])

        # Now, split the data
        train_articles, test_articles, train_prices, test_prices = train_test_split(
            articles, prices, test_size=0.15, random_state=42
        )

        # Create dataset and dataloader for training
        train_dataset = ArticlePriceDataset(train_articles, train_prices, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Initialize SI if updating
        if args.update:
            # SI was already initialized above
            pass
        else:
            si = None

        # Train the model
        print("Training Started!")
        train_model(model, optimizer, EPOCHS, device, train_dataloader, si=si)

        # Save the model weights
        os.makedirs('model', exist_ok=True)
        torch.save(model.state_dict(), 'model/model_weights.pth')

        # Save the SI state if SI is used
        if args.update and si is not None:
            si.save_state('model/si_state.pth')
            print("SI state saved.")

    elif args.mode == 'run':
        # Load the model weights
        directory = "model"
        filename = "model_weights.pth"
        filepath = os.path.join(directory, filename)
        model.load_state_dict(torch.load(filepath, map_location=device))
        model = model.to(device)
        model.eval()
    
        if args.test:
            # Evaluation on the test set
            predictions = []
            actuals = []
            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
    
                    # Forward pass with autocast
                    with autocast():
                        outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(batch['labels'].cpu().numpy())
    
            # Calculate metrics
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            print(f"Test MAE: {mae:.4f}, R2 Score: {r2:.4f}")
    
        else:
            # Predict on a single input
            if not args.input_text:
                raise ValueError("You must provide input text when mode is 'run' without --test")
    
            encoding = tokenizer(
                args.input_text,
                truncation=True,
                padding='max_length',
                max_length=block_size,
                return_tensors='pt'
            ).to(device)
    
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
    
            # Inference with autocast
            with torch.no_grad(), autocast():
                prediction, _ = model(input_ids=input_ids, attention_mask=attention_mask)

        print(f"Predicted Price: {prediction.item()}")
if __name__ == "__main__":
    main()
