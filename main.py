import torch
import os
import argparse
from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import kaiming_init_weights, get_data, prepare_dataloader, load_model_weights, initialize_si, get_new_data, initialize_model, prepare_optimizer, prepare_data, initialize_replay_buffer, save_model_and_states
from utils.config import *
from torch.utils.data import DataLoader
from utils.data import ArticlePriceDataset
from utils.si import SynapticIntelligence
from utils.test import test_forgetting
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

from utils.memory_replay_buffer import MemoryReplayBuffer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument('mode', choices=['train', 'run', 'update', 'test_forgetting'], help="Mode: 'train', 'run', 'update', or 'test_forgetting'")
    parser.add_argument('input_text', type=str, nargs='?', help="Input article text (required if mode is 'run' without --test)", default=None)
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument('--test', action='store_true', help="If specified in 'run' mode, evaluate the model on the test set.")
    parser.add_argument('--update', type=str, help="Hugging Face URL to the new dataset for updating the model.", default=None)
    parser.add_argument('--use_si', action='store_true', help="Use Synaptic Intelligence during training or updating.")
    parser.add_argument('--use_replay_buffer', action='store_true', help="Use Memory Replay Buffer during training or updating.")
    parser.add_argument('--model_repo_id', type=str, help="Hugging Face repository ID to load the model from.", default=None)

    args = parser.parse_args()

    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.mode == 'train':
        # Initialize model
        model = initialize_model(args, device)
        # Apply Kaiming initialization (since we are training from scratch)
        model.apply(kaiming_init_weights)
        logging.info("Applied Kaiming initialization to the model.")
        # Prepare optimizer
        optimizer = prepare_optimizer(model)
        # Prepare data
        train_dataloader, accumulation_steps = prepare_data(args, tokenizer)
        # Initialize SI
        si = initialize_si(model, args)
        # Initialize replay buffer
        replay_buffer = initialize_replay_buffer(args)
        # Train the model
        logging.info("Starting training...")
        train_model(
            model,
            optimizer,
            EPOCHS,
            device,
            train_dataloader,
            si=si,
            accumulation_steps=accumulation_steps,
            replay_buffer=replay_buffer
        )
        logging.info("Training completed.")
        # Save model and states
        save_model_and_states(model, si, replay_buffer, args)

    elif args.mode == 'update':
        if not args.update:
            raise ValueError("You must provide the --update argument with the Hugging Face dataset URL when in 'update' mode.")
        # Initialize model
        model = initialize_model(args, device)
        # Load existing model weights
        model = load_model_weights(model, 'model/model_weights.pth', device)
        logging.info("Loaded pre-trained model weights for updating.")
        # Prepare optimizer
        optimizer = prepare_optimizer(model)
        # Prepare data
        train_dataloader, accumulation_steps = prepare_data(args, tokenizer)
        # Initialize SI
        si = initialize_si(model, args)
        # Initialize replay buffer
        replay_buffer = initialize_replay_buffer(args)
        # Update the model
        logging.info("Starting updating...")
        train_model(
            model,
            optimizer,
            EPOCHS,
            device,
            train_dataloader,
            si=si,
            accumulation_steps=accumulation_steps,
            replay_buffer=replay_buffer
        )
        logging.info("Updating completed.")
        # Save model and states
        save_model_and_states(model, si, replay_buffer, args)

    elif args.mode == 'run':
        # Initialize model
        model = initialize_model(args, device)
        model = load_model_weights(model, 'model/model_weights.pth', device)
        logging.info("Loaded model weights for inference.")

        if args.test:
            # Evaluate on the test set
            df = get_data()
            df = df[df['weighted_avg_720_hrs'] > 0]
            _, test_df = train_test_split(df, test_size=0.15, random_state=42)
            test_dataloader = prepare_dataloader(test_df, tokenizer, batch_size=16, shuffle=False)
            logging.info(f"Prepared DataLoader with {len(test_dataloader.dataset)} test samples.")
            predictions, actuals = [], []
            model.eval()
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Evaluating on Test Set"):
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    with torch.cuda.amp.autocast():
                        outputs, _ = model(input_ids=input_ids)
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(labels.cpu().numpy())
            from sklearn.metrics import mean_absolute_error, r2_score
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            print(f"Test MAE: {mae:.4f}, R2 Score: {r2:.4f}")
            logging.info(f"Test MAE: {mae:.4f}, R2 Score: {r2:.4f}")
        else:
            if not args.input_text:
                raise ValueError("You must provide input text when mode is 'run' without --test")
            encoding = tokenizer(args.input_text, truncation=True, padding='max_length', max_length=block_size, return_tensors='pt').to(device)
            input_ids = encoding["input_ids"]
            # Inference
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    prediction, _ = model(input_ids=input_ids)
            print(f"Predicted Price: {prediction.item()}")

    elif args.mode == 'test_forgetting':
        # Initialize model
        model = initialize_model(args, device)
        model = load_model_weights(model, 'model/model_weights.pth', device)
        logging.info("Loaded model weights for testing forgetting.")
        # Test for forgetting
        test_results = test_forgetting(model, device)
        logging.info(f"Catastrophic Forgetting Test Results: {test_results}")

    else:
        raise ValueError("Invalid mode selected. Choose from 'train', 'run', 'update', or 'test_forgetting'.")

if __name__ == "__main__":
    main()
