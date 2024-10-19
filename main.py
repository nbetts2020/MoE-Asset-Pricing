# main.py

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
from utils.test import test_forgetting
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm

import torch.distributed as dist

import numpy as np
import random
import json

from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer
from utils.ewc import ElasticWeightConsolidation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    
    # modes and corresponding args
    parser.add_argument('mode', choices=['train', 'run', 'update', 'test_forgetting'], help="Mode: 'train', 'run', 'update', or 'test_forgetting'")
    parser.add_argument('input_text', type=str, nargs='?', help="Input article text (required if mode is 'run' without --test)", default=None)
    
    # model params
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument('--model', type=str, help="Hugging Face repository ID to load the model from.", default=None)

    # secondary modes and args for general and catastrophic forgetting testing
    parser.add_argument('--test', action='store_true', help="If specified in 'run' mode, evaluate the model on the test set.")
    parser.add_argument('--update', nargs='?', const=True, default=False, help="Include this flag to perform an update. Optionally provide a dataset URL for new data.")
    parser.add_argument('--percent_data', type=float, default=100.0, help="Percentage of data to use (0 < percent_data <= 100).")
    parser.add_argument('--save_dir', type=str, default="model", help="Directory to save the model and states.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--num_tasks', type=int, default=3, help="Number of tasks (sectors) to use in catastrophic forgetting testing.")
    
    # catastrophic forgetting methods and corresponding args
    parser.add_argument('--use_si', action='store_true', help="Use Synaptic Intelligence during training or updating.")
    parser.add_argument('--use_replay_buffer', action='store_true', help="Use Memory Replay Buffer during training or updating.")
    parser.add_argument('--replay_buffer_capacity', type=int, default=10000, help="Capacity of the Memory Replay Buffer.")
    parser.add_argument('--use_l2', action='store_true', help="Use L2 regularization during training or updating.")
    parser.add_argument('--lambda_l2', type=float, default=0.01, help="Regularization strength for L2 regularization.")
    parser.add_argument('--use_entropy_reg', action='store_true', help="Use entropy regularization in expert routing.")
    parser.add_argument('--lambda_entropy', type=float, default=0.01, help="Regularization strength for entropy regularization.")
    parser.add_argument('--use_ewc', action='store_true', help="Use Elastic Weight Consolidation during training or updating.")
    parser.add_argument('--lambda_ewc', type=float, default=0.4, help="Regularization strength for Elastic Weight Consolidation.")

    # Replay buffer training args
    parser.add_argument('--replay_batch_size', type=int, default=32, help='Batch size for replay buffer samples.')
    parser.add_argument('--replay_buffer_weight', type=float, default=1.0, help='Weight for replay buffer loss.')

    # distributed training
    parser.add_argument('--use_ddp', action='store_true', help='Use DistributedDataParallel')

    args = parser.parse_args()

    # Check if running with DDP
    use_ddp = torch.cuda.device_count() > 1 and args.use_ddp

    if use_ddp:
        # Initialize the process group
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        rank = dist.get_rank()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    logging.info(f"Using random seed: {random_seed}")

    if not (0 < args.percent_data <= 100):
        raise ValueError("Invalid value for --percent_data. It must be between 0 and 100.")

    logging.info(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == 'train':
        # Initialize model from scratch
        model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)
        print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")
        if initialized_from_scratch:
            # Apply Kaiming initialization
            model.apply(kaiming_init_weights)
            logging.info("Initialized model from scratch and applied Kaiming initialization.")

        # Wrap model with DDP if necessary
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        # Prepare optimizer
        optimizer = prepare_optimizer(model, args)

        # Prepare data
        train_dataloader, test_dataloader, update_dataloader = prepare_data(args, tokenizer)

        # Initialize SI - if --use_si is True
        si = initialize_si(model, args) if args.use_si else None

        # Initialize EWC - if --use_ewc is True
        if args.use_ewc:
            ewc_instance = ElasticWeightConsolidation(model, train_dataloader, device, args)
            ewc_list = [ewc_instance]
        else:
            ewc_list = None

        # Initialize replay buffer - if --use_replay_buffer is True
        replay_buffer = initialize_replay_buffer(args) if args.use_replay_buffer else None

        # Train the model
        train_model(
            model,
            optimizer,
            EPOCHS,
            device,
            train_dataloader,
            args=args,
            si=si,
            ewc=ewc_list,
            replay_buffer=replay_buffer,
            test_dataloader=test_dataloader
        )
        logging.info("Training completed.")
    
        if args.update and update_dataloader:
            # Update the model with update data
            logging.info("Starting model update with update data...")

            # Update EWC after training on previous data
            if args.use_ewc:
                # Create new EWC instance after first training phase
                ewc_instance = ElasticWeightConsolidation(model, train_dataloader, device, args)
                ewc_list.append(ewc_instance)

            train_model(
                model,
                optimizer,
                EPOCHS,
                device,
                update_dataloader,
                args=args,
                si=si,
                ewc=ewc_list,
                replay_buffer=replay_buffer,
                test_dataloader=test_dataloader
            )
            logging.info("Model update completed.")

        # Save model and states
        save_model_and_states(model, si, replay_buffer, ewc_list, args)
        
    elif args.mode == 'update':
        if not args.update:
            raise ValueError("You must provide the --update argument with the Hugging Face dataset URL when in 'update' mode.")
        # Initialize model
        try:
            model, _ = initialize_model(args, device, init_from_scratch=False)
            logging.info("Loaded pre-trained model for updating.")
            print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")
        except RuntimeError as e:
            logging.error(str(e))
            print("Error: Could not load model for updating.")
            return

        # Wrap model with DDP if necessary
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        # Prepare optimizer
        optimizer = prepare_optimizer(model, args)

        # Prepare data
        train_dataloader, test_dataloader, update_dataloader = prepare_data(args, tokenizer)

        # Initialize SI - if --use_si is True
        si = initialize_si(model, args) if args.use_si else None

        # Initialize EWC - if --use_ewc is True
        if args.use_ewc:
            # Load previous EWC state
            ewc_state_path = os.path.join(args.save_dir, 'ewc_state.pth')
            if os.path.exists(ewc_state_path):
                ewc_states = torch.load(ewc_state_path, map_location=device)
                ewc_list = []
                for state in ewc_states:
                    ewc_instance = ElasticWeightConsolidation(model, dataloader=None, device=device, args=args)
                    ewc_instance.params = {n: p.to(device) for n, p in state['params'].items()}
                    ewc_instance.fisher = {n: f.to(device) for n, f in state['fisher'].items()}
                    ewc_list.append(ewc_instance)
                logging.info(f"Loaded EWC state from '{ewc_state_path}'.")
            else:
                logging.info("No existing EWC state found. Starting fresh EWC.")
                ewc_list = []
        else:
            ewc_list = None

        # Initialize replay buffer - if --use_replay_buffer is True
        replay_buffer = initialize_replay_buffer(args) if args.use_replay_buffer else None

        # Update the model
        logging.info("Starting updating...")
        train_model(
            model,
            optimizer,
            EPOCHS,
            device,
            train_dataloader,
            args=args,
            si=si,
            ewc=ewc_list,
            replay_buffer=replay_buffer
        )
        logging.info("Updating completed.")
        # Save model and states
        save_model_and_states(model, si, replay_buffer, ewc_list, args)

    elif args.mode == 'run':
        # Initialize model
        try:
            model, _ = initialize_model(args, device, init_from_scratch=False)
            logging.info("Model is ready for inference.")
            print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")
        except RuntimeError as e:
            logging.error(str(e))
            print("Error: Could not load model for inference.")
            return

        if args.test:
            # Evaluate on the test set
            df = get_data()
            df = df[df['weighted_avg_720_hrs'] > 0] # checking if market data is valid
            _, test_df = train_test_split(df, test_size=0.15, random_state=42)
            test_dataloader = prepare_dataloader(test_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False)
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
            mse, r2, sector_metrics = evaluate_model(model, test_dataloader, device)
            print(f"Test MSE: {mse:.4f}, R² Score: {r2:.4f}")
            logging.info(f"Test MSE: {mse:.4f}, R² Score: {r2:.4f}")
            print("Per-Sector Metrics:")
            for sector, metrics in sector_metrics.items():
                print(f"Sector: {sector} - MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")
                logging.info(f"Sector: {sector} - MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")
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
        model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)
        logging.info("Initialized model for catastrophic forgetting testing.")
        print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")
    
        # Wrap model with DDP if necessary
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        # Prepare optimizer
        optimizer = prepare_optimizer(model, args)
    
        # Initialize SI if required
        si = initialize_si(model, args) if args.use_si else None
    
        # Initialize EWC if required
        if args.use_ewc:
            ewc_list = []
        else:
            ewc_list = None

        # Initialize replay buffer if required
        replay_buffer = initialize_replay_buffer(args) if args.use_replay_buffer else None
    
        test_results = test_forgetting(
            model=model,
            optimizer=optimizer,
            epochs=EPOCHS,
            device=device,
            tokenizer=tokenizer,
            args=args,
            si=si,
            ewc=ewc_list,
            replay_buffer=replay_buffer
        )
    
        # Log and save results
        logging.info(f"Catastrophic Forgetting Test Results: {test_results}")
        with open(os.path.join(args.save_dir, 'test_forgetting_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
    else:
        raise ValueError("Invalid mode selected. Choose from 'train', 'run', 'update', or 'test_forgetting'.")

    # Clean up DDP
    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
