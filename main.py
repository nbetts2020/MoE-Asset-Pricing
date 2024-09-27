import torch
import os
import argparse
from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import (
    kaiming_init_weights, 
    get_data, 
    prepare_dataloader, 
    load_model_weights, 
    initialize_si, 
    get_new_data, 
    get_model
)
from utils.config import *
from torch.utils.data import DataLoader
from utils.data import ArticlePriceDataset
from utils.si import SynapticIntelligence
from utils.test import test_forgetting
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument('mode', choices=['train', 'run', 'test_forgetting'], help="Mode: 'train', 'run', or 'test_forgetting'")
    parser.add_argument('input_text', type=str, nargs='?', help="Input article text (required if mode is 'run' without --test)", default=None)
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument('--test', action='store_true', help="If specified in 'run' mode, evaluate the model on the test set.")
    parser.add_argument('--update', type=str, help="Hugging Face URL to the new dataset for updating the model.", default=None)

    args = parser.parse_args()

    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize the model and tokenizer (always needed)
    model = SparseMoELanguageModel(tokenizer_name=args.tokenizer_name)
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # Handle different modes
    if args.mode == 'test_forgetting':
        # Load pre-trained model for testing catastrophic forgetting
        weights_path, _ = get_model("nbettencourt/NLAS-test")
        model = load_model_weights(model, weights_path, device)
        logging.info("Pre-trained model loaded for testing catastrophic forgetting.")

        # Prepare tasks
        task_dataloaders = prepare_tasks()

        # Initialize Synaptic Intelligence (SI) if updating the model
        si = initialize_si(model, 'model/si_state.pth') if args.update else None

        # Run the catastrophic forgetting test
        results = test_forgetting(model, task_dataloaders, EPOCHS, device, si=si)
        print("Catastrophic Forgetting Test Results:")
        print(results)

    elif args.mode == 'train':
        # Apply Kaiming initialization to the model
        model.apply(kaiming_init_weights)
        logging.info("Applied Kaiming initialization to the model.")

        # If updating, load the model weights
        si = None
        if args.update:
            # Load pre-trained model weights from Hugging Face
            weights_path, _ = get_model("nbettencourt/NLAS-test")
            model = load_model_weights(model, weights_path, device)
            logging.info("Loaded pre-trained model weights for updating.")

            # Initialize SI for updating
            si = initialize_si(model, 'model/si_state.pth')
            logging.info("Initialized Synaptic Intelligence (SI) from saved state.")

            # Load new data from Hugging Face
            new_data_url = args.update
            logging.info(f"Fetching new data from Hugging Face URL: {new_data_url}")
            df_new = get_new_data(new_data_url)
            logging.info(f"Fetched {len(df_new)} new data samples.")
            df_new = df_new[df_new['weighted_avg_720_hrs'] > 0]

            # Create DataLoader for the new data
            actual_batch_size = min(16, len(df_new))
            train_dataloader = prepare_dataloader(df_new, tokenizer, batch_size=actual_batch_size)
            logging.info(f"Prepared DataLoader with {len(train_dataloader.dataset)} new training samples.")
        else:
            # Train from scratch (no need to load weights)
            df = get_data()
            df = df[df['weighted_avg_720_hrs'] > 0]
            train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
            actual_batch_size = 16
            train_dataloader = prepare_dataloader(train_df, tokenizer, batch_size=actual_batch_size)
            logging.info(f"Prepared DataLoader with {len(train_dataloader.dataset)} training samples.")

        # Define the optimizer with layer-wise learning rate decay
        param_groups = [
            {'params': list(model.token_embedding_table.parameters()) + list(model.position_embedding_table.parameters()), 'lr': learning_rate * (LR_DECAY ** (len(model.blocks) + 1))},
            {'params': model.regression_head.parameters(), 'lr': learning_rate}
        ] + [
            {'params': block.parameters(), 'lr': learning_rate * (LR_DECAY ** (len(model.blocks) - i))}
            for i, block in enumerate(model.blocks)
        ]
        optimizer = torch.optim.AdamW(param_groups)
        logging.info("Initialized AdamW optimizer with layer-wise learning rate decay.")

        # Determine accumulation steps for gradient accumulation
        desired_effective_batch_size = 16
        accumulation_steps = max(1, desired_effective_batch_size // actual_batch_size)
        logging.info(f"Using accumulation_steps={accumulation_steps} for training.")

        # Start training the model
        logging.info("Starting training...")
        train_model(model, optimizer, EPOCHS, device, train_dataloader, si=si, accumulation_steps=accumulation_steps)
        logging.info("Training completed.")

        # Save model weights
        os.makedirs('model', exist_ok=True)
        torch.save(model.state_dict(), 'model/model_weights.pth')
        logging.info("Model weights saved to 'model/model_weights.pth'.")

        # Save SI state if used
        if si is not None:
            si.save_state('model/si_state.pth')
            logging.info("Synaptic Intelligence (SI) state saved to 'model/si_state.pth'.")

    elif args.mode == 'run':
        # Load the model for inference
        weights_path, _ = get_model("nbettencourt/NLAS-test")
        model = load_model_weights(model, weights_path, device)
        logging.info("Loaded model weights for inference.")

        if args.test:
            # Load data for testing
            df = get_data()
            df = df[df['weighted_avg_720_hrs'] > 0]
            _, test_df = train_test_split(df, test_size=0.15, random_state=42)
            test_dataloader = prepare_dataloader(test_df, tokenizer, batch_size=16, shuffle=False)
            logging.info(f"Prepared DataLoader with {len(test_dataloader.dataset)} test samples.")

            # Perform evaluation
            predictions, actuals = [], []
            model.eval()
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Evaluating on Test Set"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    with torch.cuda.amp.autocast():
                        outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)

                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(labels.cpu().numpy())

            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, r2_score
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            print(f"Test MAE: {mae:.4f}, R2 Score: {r2:.4f}")
            logging.info(f"Test MAE: {mae:.4f}, R2 Score: {r2:.4f}")

        else:
            # Perform inference on provided text
            if not args.input_text:
                raise ValueError("You must provide input text when mode is 'run' without --test")

            encoding = tokenizer(args.input_text, truncation=True, padding='max_length', max_length=block_size, return_tensors='pt').to(device)
            input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

            # Inference
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    prediction, _ = model(input_ids, attention_mask)
            print(f"Predicted Price: {prediction.item()}")

if __name__ == "__main__":
    main()
