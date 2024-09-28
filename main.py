import torch
import os
import argparse
from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import kaiming_init_weights, get_data, prepare_dataloader, load_model_weights, initialize_si, get_new_data
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
    parser.add_argument('mode', choices=['train', 'run', 'test_forgetting'], help="Mode: 'train', 'run', or 'test_forgetting'")
    parser.add_argument('input_text', type=str, nargs='?', help="Input article text (required if mode is 'run' without --test)", default=None)
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument('--test', action='store_true', help="If specified in 'run' mode, evaluate the model on the test set.")
    parser.add_argument('--update', type=str, help="Hugging Face URL to the new dataset for updating the model.", default=None)
    parser.add_argument('--use_si', action='store_true', help="Use Synaptic Intelligence during training or updating.")
    parser.add_argument('--use_replay_buffer', action='store_true', help="Use Memory Replay Buffer during training or updating.")

    args = parser.parse_args()

    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize the model and tokenizer
    model = SparseMoELanguageModel(tokenizer_name=args.tokenizer_name)
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    if args.mode == 'train':
        # Load pre-trained model if updating
        if args.update:
            # Load pre-trained model weights
            model = load_model_weights(model, 'model/model_weights.pth', device)
            logging.info("Loaded pre-trained model weights for updating.")
        else:
            # Apply Kaiming initialization
            model.apply(kaiming_init_weights)
            logging.info("Applied Kaiming initialization to the model.")

        # Check for multiple GPUs
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            logging.info(f"Using {torch.cuda.device_count()} GPUs for training.")

        model = model.to(device)
        logging.info(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters.")

        # Initialize optimizer with layer-wise learning rate decay
        param_groups = [
            {'params': list(model.token_embedding_table.parameters()) + list(model.position_embedding_table.parameters()), 'lr': learning_rate * (LR_DECAY ** (len(model.blocks) + 1))},
            {'params': model.regression_head.parameters(), 'lr': learning_rate}
        ] + [
            {'params': block.parameters(), 'lr': learning_rate * (LR_DECAY ** (len(model.blocks) - i))}
            for i, block in enumerate(model.blocks)
        ]
        optimizer = torch.optim.AdamW(param_groups)
        logging.info("Initialized AdamW optimizer with layer-wise learning rate decay.")

        # Initialize SI if --use_si is specified
        si = None
        if args.use_si:
            if args.update:
                # Initialize SI and load its state
                si = initialize_si(model, 'model/si_state.pth')
                logging.info("Initialized Synaptic Intelligence (SI) from saved state.")

                # Load new data from Hugging Face URL
                new_data_url = args.update
                logging.info(f"Fetching new data from Hugging Face URL: {new_data_url}")
                df_new = get_new_data(new_data_url)  # Placeholder function to implement
                logging.info(f"Fetched {len(df_new)} new data samples.")

                # Preprocess new data as needed
                df_new = df_new[df_new['weighted_avg_720_hrs'] > 0]

                # Create DataLoader for new data
                actual_batch_size = min(16, len(df_new))  # Adjust batch size based on data size
                train_dataloader = prepare_dataloader(df_new, tokenizer, batch_size=actual_batch_size)
                logging.info(f"Prepared DataLoader with {len(train_dataloader.dataset)} new training samples.")

                # Determine accumulation_steps
                desired_effective_batch_size = 16  # Set your desired effective batch size
                accumulation_steps = max(1, desired_effective_batch_size // actual_batch_size)
                logging.info(f"Using accumulation_steps={accumulation_steps} for training.")
            else:
                # Initialize SI for initial training
                si = SynapticIntelligence(model, lambda_si=LAMBDA_SI)
                logging.info("Initialized Synaptic Intelligence (SI) for initial training.")

                # Load data and create DataLoader for initial training
                df = get_data()
                df = df[df['weighted_avg_720_hrs'] > 0]
                train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
                actual_batch_size = 16  # You can adjust this based on your resources
                train_dataloader = prepare_dataloader(train_df, tokenizer, batch_size=actual_batch_size)
                logging.info(f"Prepared DataLoader with {len(train_dataloader.dataset)} training samples.")

                # Determine accumulation_steps
                desired_effective_batch_size = 16  # For initial training, this might be the same as actual_batch_size
                accumulation_steps = max(1, desired_effective_batch_size // actual_batch_size)
                logging.info(f"Using accumulation_steps={accumulation_steps} for training.")
        else:
            # If not using SI, handle update and initial training separately
            if args.update:
                # Load pre-trained model weights
                model = load_model_weights(model, 'model/model_weights.pth', device)
                logging.info("Loaded pre-trained model weights for updating.")

                # Load new data from Hugging Face URL
                new_data_url = args.update
                logging.info(f"Fetching new data from Hugging Face URL: {new_data_url}")
                df_new = get_new_data(new_data_url)  # Placeholder function to implement
                logging.info(f"Fetched {len(df_new)} new data samples.")

                # Preprocess new data as needed
                df_new = df_new[df_new['weighted_avg_720_hrs'] > 0]

                # Create DataLoader for new data
                actual_batch_size = min(16, len(df_new))  # Adjust batch size based on data size
                train_dataloader = prepare_dataloader(df_new, tokenizer, batch_size=actual_batch_size)
                logging.info(f"Prepared DataLoader with {len(train_dataloader.dataset)} new training samples.")

                # Determine accumulation_steps
                desired_effective_batch_size = 16  # Set your desired effective batch size
                accumulation_steps = max(1, desired_effective_batch_size // actual_batch_size)
                logging.info(f"Using accumulation_steps={accumulation_steps} for training.")
            else:
                # Load data and create DataLoader for initial training without SI
                df = get_data()
                df = df[df['weighted_avg_720_hrs'] > 0]
                train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
                actual_batch_size = 16  # You can adjust this based on your resources
                train_dataloader = prepare_dataloader(train_df, tokenizer, batch_size=actual_batch_size)
                logging.info(f"Prepared DataLoader with {len(train_dataloader.dataset)} training samples.")

                # Determine accumulation_steps
                desired_effective_batch_size = 16  # For initial training, this might be the same as actual_batch_size
                accumulation_steps = max(1, desired_effective_batch_size // actual_batch_size)
                logging.info(f"Using accumulation_steps={accumulation_steps} for training.")

        # Initialize or load the replay buffer if --use_replay_buffer is specified
        replay_buffer = None
        if args.use_replay_buffer:
            replay_buffer_capacity = 10000  # Adjust as needed
            replay_buffer = MemoryReplayBuffer(capacity=replay_buffer_capacity)
            logging.info(f"Initialized Memory Replay Buffer with capacity {replay_buffer_capacity}.")

            # If updating, load existing replay buffer
            if args.update:
                replay_buffer_path = 'model/replay_buffer.pth'
                if os.path.exists(replay_buffer_path):
                    replay_buffer.load(replay_buffer_path)
                    logging.info(f"Loaded Memory Replay Buffer from '{replay_buffer_path}'.")
                else:
                    logging.info("No existing Memory Replay Buffer found. Starting fresh.")

        # Train the model
        logging.info("Starting training...")
        train_model(
            model,
            optimizer,
            EPOCHS,
            device,
            train_dataloader,
            si=si,  # Pass the SI object (None if not using SI)
            accumulation_steps=accumulation_steps,
            replay_buffer=replay_buffer  # Pass the replay buffer (None if not using)
        )
        logging.info("Training completed.")

        # Save the model weights
        os.makedirs('model', exist_ok=True)
        torch.save(model.state_dict(), 'model/model_weights.pth')
        logging.info("Model weights saved to 'model/model_weights.pth'.")

        # Save SI state if using SI
        if args.use_si and si is not None:
            si.save_state('model/si_state.pth')
            logging.info("Synaptic Intelligence (SI) state saved to 'model/si_state.pth'.")

        # Save the replay buffer if using it
        if args.use_replay_buffer and replay_buffer is not None:
            replay_buffer.save('model/replay_buffer.pth')
            logging.info("Memory Replay Buffer saved to 'model/replay_buffer.pth'.")

    elif args.mode == 'run':
        # Load the model weights
        model = load_model_weights(model, 'model/model_weights.pth', device)
        logging.info("Loaded model weights for inference.")

        if args.test:
            # Ensure test_df is loaded
            df = get_data()
            df = df[df['weighted_avg_720_hrs'] > 0]
            _, test_df = train_test_split(df, test_size=0.15, random_state=42)

            # Create DataLoader for test data
            test_dataloader = prepare_dataloader(test_df, tokenizer, batch_size=16, shuffle=False)
            logging.info(f"Prepared DataLoader with {len(test_dataloader.dataset)} test samples.")

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
            if not args.input_text:
                raise ValueError("You must provide input text when mode is 'run' without --test")

            encoding = tokenizer(args.input_text, truncation=True, padding='max_length', max_length=block_size, return_tensors='pt').to(device)
            input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

            # Inference
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    prediction, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            print(f"Predicted Price: {prediction.item()}")

    elif args.mode == 'test_forgetting':
        # Implement your forgetting test here
        logging.info("Testing for catastrophic forgetting...")
        # Placeholder: Implement the test_forgetting function
        test_results = test_forgetting(model, device)
        logging.info(f"Catastrophic Forgetting Test Results: {test_results}")

    else:
        raise ValueError("Invalid mode selected. Choose from 'train', 'run', or 'test_forgetting'.")

if __name__ == "__main__":
    main()
