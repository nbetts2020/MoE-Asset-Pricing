import torch
import os
import argparse
import logging
from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import (
    get_data,
    prepare_dataloader,
    get_model_from_hf,
    get_new_data,
    initialize_si,
    initialize_model,
    prepare_optimizer,
    prepare_data,
    initialize_replay_buffer,
    save_model_and_states,
    kaiming_init_weights
)
from utils.config import *
from torch.utils.data import DataLoader
from utils.data import ArticlePriceDataset
from utils.si import SynapticIntelligence
from utils.test import test_forgetting
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm

from utils.memory_replay_buffer import MemoryReplayBuffer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument('mode', choices=['train', 'run', 'update', 'test_forgetting'], help="Mode: 'train', 'run', 'update', or 'test_forgetting'")
    parser.add_argument('input_text', type=str, nargs='?', help="Input article text (required if mode is 'run' without --test)", default=None)
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument('--test', action='store_true', help="If specified in 'run' mode, evaluate the model on the test set.")
    parser.add_argument('--update', type=str, help="Hugging Face dataset URL for updating the model.", default=None)
    parser.add_argument('--use_si', action='store_true', help="Use Synaptic Intelligence during training or updating.")
    parser.add_argument('--use_replay_buffer', action='store_true', help="Use Memory Replay Buffer during training or updating.")
    parser.add_argument('--model', type=str, help="Hugging Face repository ID to load the model from.", default=None)

    args = parser.parse_args()

    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.mode == 'train':
        # Initialize model from scratch
        model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)
        if initialized_from_scratch:
            # Apply Kaiming initialization
            model.apply(kaiming_init_weights)
            logging.info("Initialized model from scratch and applied Kaiming initialization.")
        # Prepare optimizer
        optimizer = prepare_optimizer(model)
        # Prepare data
        train_dataloader, accumulation_steps = prepare_data(args, tokenizer)
        # Initialize SI - if --use_si is True
        si = initialize_si(model, args)
        # Initialize replay buffer - if --use_replay_buffer is True
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
        try:
            model, _ = initialize_model(args, device, init_from_scratch=False)
            logging.info("Loaded pre-trained model for updating.")
        except RuntimeError as e:
            logging.error(str(e))
            print("Error: Could not load model for updating.")
            return
        # Prepare optimizer
        optimizer = prepare_optimizer(model)
        # Prepare data
        train_dataloader, accumulation_steps = prepare_data(args, tokenizer)
        # Initialize SI - if --use_si is True
        si = initialize_si(model, args)
        # Initialize replay buffer - if --use_replay_buffer is True
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
        try:
            model, _ = initialize_model(args, device, init_from_scratch=False)
            logging.info("Model is ready for inference.")
        except RuntimeError as e:
            logging.error(str(e))
            print("Error: Could not load model for inference.")
            return

        if args.test:
            # Evaluate on the test set
            df = get_data()
            df = df[df['weighted_avg_720_hrs'] > 0] # checking if market data is valid
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
            encoding = tokenizer(
                args.input_text,
                truncation=True,
                padding='max_length',
                max_length=block_size,
                return_tensors='pt'
            ).to(device)
            input_ids = encoding["input_ids"]
            # Inference
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    prediction, _ = model(input_ids=input_ids)
            print(f"Predicted Price: {prediction.item()}")

    elif args.mode == 'test_forgetting':
        # Initialize model
        try:
            model, _ = initialize_model(args, device, init_from_scratch=False)
            logging.info("Loaded model weights for testing forgetting.")
        except RuntimeError as e:
            logging.error(str(e))
            print("Error: Could not load model for testing forgetting.")
            return
        # Test for forgetting
        test_results = test_forgetting(model, device)
        logging.info(f"Catastrophic Forgetting Test Results: {test_results}")

    else:
        raise ValueError("Invalid mode selected. Choose from 'train', 'run', 'update', or 'test_forgetting'.")

if __name__ == "__main__":
    main()
