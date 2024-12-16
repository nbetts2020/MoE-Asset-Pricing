# main.py

import torch
import os
import argparse
import logging
from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import (
    initialize_model,
    prepare_optimizer,
    initialize_si,
    initialize_replay_buffer,
    save_model_and_states,
    kaiming_init_weights,
    prepare_data
)
from utils.config import config
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch.distributed as dist
import numpy as np
import random
import pandas as pd
import json

from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer
from utils.ewc import ElasticWeightConsolidation
from utils.test import test_forgetting
from utils.train import evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")

    # Modes and corresponding args
    parser.add_argument('mode', choices=['train', 'run', 'update', 'test_forgetting'], help="Mode: 'train', 'run', 'update', or 'test_forgetting'")
    parser.add_argument('input_text', type=str, nargs='?', help="Input article text (required if mode is 'run' without --test)", default=None)

    # Model parameters
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument('--model', type=str, help="Hugging Face repository ID to load the model from.", default=None)

    # Secondary modes and args for general and catastrophic forgetting testing
    parser.add_argument('--test', action='store_true', help="If specified in 'run' mode, evaluate the model on the test set.")
    parser.add_argument('--update', action='store_true', help="Include this flag to perform an update.")
    parser.add_argument('--percent_data', type=float, default=100.0, help="Percentage of data to use (0 < percent_data <= 100).")
    parser.add_argument('--save_dir', type=str, default="model", help="Directory to save the model and states.")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Directory to save model checkpoints and states.")
    parser.add_argument('--checkpoint_path', type=str, help='Path to a checkpoint to resume training')
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--num_tasks', type=int, default=3, help="Number of tasks (sectors) to use in catastrophic forgetting testing.")

    # Catastrophic forgetting methods and corresponding args
    parser.add_argument('--use_si', action='store_true', help="Use Synaptic Intelligence during training or updating.")
    parser.add_argument('--use_replay_buffer', action='store_true', help="Use Memory Replay Buffer during training or updating.")
    parser.add_argument('--replay_buffer_capacity', type=int, default=10000, help="Capacity of the Memory Replay Buffer.")
    parser.add_argument('--use_l2', action='store_true', help="Use L2 regularization during training or updating.")
    parser.add_argument('--lambda_l2', type=float, default=0.01, help="Regularization strength for L2 regularization.")
    parser.add_argument('--use_entropy_reg', action='store_true', help="Use entropy regularization in expert routing.")
    parser.add_argument('--lambda_entropy', type=float, default=0.01, help="Regularization strength for entropy regularization.")
    parser.add_argument('--use_ewc', action='store_true', help="Use Elastic Weight Consolidation during training or updating.")
    parser.add_argument('--lambda_ewc', type=float, default=0.4, help="Regularization strength for Elastic Weight Consolidation.")

    # EBM/EBM params
    parser.add_argument('--use_ebm', action='store_true', help='Use energy-based model for prompt optimization.')
    parser.add_argument('--ebm_learning_rate', type=float, default=1e-4, help='Learning rate for the EBM')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for Monte Carlo Sampling')

    # Replay buffer training args
    parser.add_argument('--replay_batch_size', type=int, default=32, help='Batch size for replay buffer samples.')
    parser.add_argument('--replay_buffer_weight', type=float, default=1.0, help='Weight for replay buffer loss.')

    # Distributed training and early stopping
    parser.add_argument('--use_ddp', action='store_true', help='Use DistributedDataParallel')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Number of epochs with no improvement after which training will be stopped.')

    # Initialize small model for testing
    parser.add_argument('--test_model', action='store_true', help='Use a smaller model configuration for quick testing.')

    # Manually setting configs
    parser.add_argument('--n_embed', type=int, help='Embedding dimension')
    parser.add_argument('--n_head', type=int, help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, help='Number of transformer blocks')
    parser.add_argument('--block_size', type=int, help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')

    args = parser.parse_args()

    if args.test_model:
        logging.info("Test Mode Activated: Using smaller hyperparameters for faster execution.")
        config.EPOCHS = 3
        config.N_EMBED = 32
        config.N_HEAD = 4
        config.N_LAYER = 12
        config.BLOCK_SIZE = 1024

    if args.n_embed is not None:
        config.N_EMBED = args.n_embed
        logging.info(f"Overriding n_embed to {config.N_EMBED}")
    if args.n_head is not None:
        config.N_HEAD = args.n_head
        logging.info(f"Overriding n_head to {config.N_HEAD}")
    if args.n_layer is not None:
        config.N_LAYER = args.n_layer
        logging.info(f"Overriding n_layer to {config.N_LAYER}")
    if args.block_size is not None:
        config.BLOCK_SIZE = args.block_size
        logging.info(f"Overriding block_size to {config.BLOCK_SIZE}")
    if args.epochs is not None:
        config.EPOCHS = args.epochs
        logging.info(f"Overriding EPOCHS to {config.EPOCHS}")

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
        logging.info(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")
        if initialized_from_scratch:
            # Apply Kaiming initialization
            model.apply(kaiming_init_weights)
            logging.info("Initialized model from scratch and applied Kaiming initialization.")

        # Wrap model with DDP if necessary
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            logging.info("Model wrapped with DistributedDataParallel.")

        # Prepare optimizer
        optimizer = prepare_optimizer(model, args)

        # Prepare data
        train_dataloader, test_dataloader, update_dataloader, df = prepare_data(args, tokenizer)

        # Initialize EBM if using
        ebm = None
        ebm_optimizer = None
        if args.use_ebm:
            from utils.ebm import EnergyBasedModel
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED).to(device)
            ebm_optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.ebm_learning_rate * 0.1)

            from functools import partial
            from utils.data import custom_collate_fn
            train_dataset = train_dataloader.dataset
            # Re-create train_dataloader with custom_collate_fn
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=partial(
                    custom_collate_fn,
                    df=train_dataset.df,
                    ebm=ebm,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    use_ebm=args.use_ebm
                )
            )

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
            model=model,
            optimizer=optimizer,
            epochs=config.EPOCHS,
            device=device,
            dataloader=train_dataloader,
            args=args,
            si=si,
            ewc=ewc_list,
            replay_buffer=replay_buffer,
            df=df,
            ebm=ebm,
            ebm_optimizer=ebm_optimizer,
            tokenizer=tokenizer
        )
        logging.info("Training completed.")

        # Save model and states
        if rank == 0:
            save_model_and_states(model, si, replay_buffer, ewc_list, args)

    elif args.mode == 'update':
        if not args.update:
            raise ValueError("You must provide the --update argument when in 'update' mode.")
        # Load DataFrame
        df = pd.read_csv(args.csv_path)
        logging.info(f"Loaded dataset with {len(df)} rows for update.")

        # Initialize model
        try:
            model, _ = initialize_model(args, device, init_from_scratch=False)
            logging.info("Loaded pre-trained model for updating.")
            logging.info(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")
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
        train_dataloader, test_dataloader, update_dataloader, _ = prepare_data(args, tokenizer)

        # Initialize EBM if using
        ebm = None
        ebm_optimizer = None
        if args.use_ebm:
            from utils.ebm import EnergyBasedModel
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED).to(device)
            ebm_optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.ebm_learning_rate)

            from functools import partial
            from utils.data import custom_collate_fn
            train_dataset = train_dataloader.dataset
            # Re-create train_dataloader with custom_collate_fn
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=partial(
                    custom_collate_fn,
                    df=train_dataset.df,
                    ebm=ebm,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    use_ebm=args.use_ebm
                )
            )

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
                ewc_instance = ElasticWeightConsolidation(model, dataloader=train_dataloader, device=device, args=args)
                ewc_list = [ewc_instance]
        else:
            ewc_list = None

        # Initialize replay buffer - if --use_replay_buffer is True
        replay_buffer = initialize_replay_buffer(args) if args.use_replay_buffer else None

        # Update the model
        logging.info("Starting updating...")
        train_model(
            model=model,
            optimizer=optimizer,
            epochs=config.EPOCHS,
            device=device,
            dataloader=train_dataloader,
            args=args,
            si=si,
            ewc=ewc_list,
            replay_buffer=replay_buffer,
            df=df,
            ebm=ebm,
            ebm_optimizer=ebm_optimizer,
            tokenizer=tokenizer
        )
        logging.info("Updating completed.")

        # Save model and states
        if rank == 0:
            save_model_and_states(model, si, replay_buffer, ewc_list, args)

    elif args.mode == 'run':
        # Initialize model
        try:
            model, _ = initialize_model(args, device, init_from_scratch=False)
            logging.info("Model is ready for inference.")
            logging.info(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")
        except RuntimeError as e:
            logging.error(str(e))
            print("Error: Could not load model for inference.")
            return

        if args.test:
            # Load DataFrame
            df = pd.read_csv(args.csv_path)
            logging.info(f"Loaded dataset with {len(df)} rows for testing.")

            from utils.data import ArticlePriceDataset
            dataset = ArticlePriceDataset(
                articles=df['Article'].tolist(),
                prices=df['weighted_avg_720_hrs'].tolist(),
                sectors=df['Sector'].tolist(),
                dates=df['Date'].tolist(),
                related_stocks_list=df['RelatedStocksList'].tolist(),
                prices_current=df['weighted_avg_0_hrs'].tolist(),
                symbols=df['Symbol'].tolist(),
                industries=df['Industry'].tolist(),
                tokenizer=tokenizer,
                total_epochs=1,
                use_ebm=False
            )

            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

            # Evaluate on the test set
            mse, r2, sector_metrics = evaluate_model(model, dataloader, device)
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
                max_length=config.BLOCK_SIZE,
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
        # Initialize model from scratch
        model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)
        logging.info("Initialized model for catastrophic forgetting testing.")
        logging.info(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")

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

        # Note: EBM not necessarily needed for test_forgetting, but if you wanted you could initialize similarly.

        from utils.test import test_forgetting
        test_results = test_forgetting(
            model=model,
            optimizer=optimizer,
            epochs=config.EPOCHS,
            device=device,
            tokenizer=tokenizer,
            args=args,
            si=si,
            replay_buffer=replay_buffer,
            ewc=ewc_list
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
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
