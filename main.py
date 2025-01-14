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
    prepare_data,
    save_ebm_model,
    download_models_from_s3,
    evaluate_model,
    ebm_select_contexts
)
from utils.config import config
from torch.utils.data import DataLoader
from utils.data import ArticlePriceDataset, custom_collate_fn
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

from utils.ebm import EnergyBasedModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")

    # Modes and corresponding args
    parser.add_argument('mode', choices=['train', 'run', 'update', 'test_forgetting'],
                        help="Mode: 'train', 'run', 'update', or 'test_forgetting'")
    parser.add_argument('input_text', type=str, nargs='?', default=None,
                        help="Input text if mode='run' (unless --test).")

    # Model parameters
    parser.add_argument('--tokenizer_name', type=str, default="gpt2",
                        help="Name of the pretrained tokenizer to use")
    parser.add_argument('--model', type=str, default=None,
                        help="Hugging Face repository ID to load the model from.")
    parser.add_argument('--save_model_name', type=str, default=None,
                        help="Name of saved model.")

    # Secondary modes and catastrophic forgetting
    parser.add_argument('--test', action='store_true',
                        help="If specified in 'run' mode, evaluate on the test set.")
    parser.add_argument('--update', action='store_true',
                        help="Include this flag to perform an update.")
    parser.add_argument('--percent_data', type=float, default=100.0,
                        help="Percentage of data to use (0 < percent_data <= 100).")
    parser.add_argument('--save_dir', type=str, default="model",
                        help="Directory to save the model and states.")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help="Directory to save model checkpoints and states.")
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to a checkpoint to resume training')
    parser.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument('--num_tasks', type=int, default=3,
                        help="Number of tasks (sectors) for catastrophic forgetting testing.")

    # Catastrophic forgetting methods
    parser.add_argument('--use_si', action='store_true',
                        help="Use Synaptic Intelligence.")
    parser.add_argument('--use_replay_buffer', action='store_true',
                        help="Use Memory Replay Buffer.")
    parser.add_argument('--replay_buffer_capacity', type=int, default=10000,
                        help="Capacity of the Memory Replay Buffer.")
    parser.add_argument('--use_l2', action='store_true',
                        help="Use L2 regularization.")
    parser.add_argument('--lambda_l2', type=float, default=0.01,
                        help="Regularization strength for L2.")
    parser.add_argument('--use_entropy_reg', action='store_true',
                        help="Use entropy regularization in expert routing.")
    parser.add_argument('--lambda_entropy', type=float, default=0.01,
                        help="Regularization strength for entropy.")
    parser.add_argument('--use_ewc', action='store_true',
                        help="Use Elastic Weight Consolidation.")
    parser.add_argument('--lambda_ewc', type=float, default=0.4,
                        help="Regularization strength for EWC.")

    # EBM / EBM params
    parser.add_argument('--use_ebm', action='store_true',
                        help='Use energy-based model for prompt optimization.')
    parser.add_argument('--ebm_learning_rate', type=float, default=1e-4,
                        help='Learning rate for the EBM')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature parameter for Monte Carlo Sampling')
    parser.add_argument('--ebm_num_samples_train', type=int,
                        help="Number of samples the EBM generates when 'train' is active")
    parser.add_argument('--use_ebm_format', action='store_true',
                        help='Use simplified version of EBM formatting when using non-EBM model.')

    # Run params
    parser.add_argument('--ebm_num_samples', type=int, default=25,
                        help="Number of samples the EBM generates when 'run' is active")
    parser.add_argument('--stock', type=str, required=False,
                        help="Stock symbol for 'run' mode.")
    parser.add_argument('--date', type=str, required=False,
                        help="Date for 'run' mode.")
    parser.add_argument('--text', type=str, required=False,
                        help="Input article text for 'run' mode.")
    parser.add_argument('--bucket', type=str, required=False,
                        help="S3 bucket name for 'run'/'update' mode.")

    # Replay buffer training args
    parser.add_argument('--replay_batch_size', type=int, default=32,
                        help='Batch size for replay buffer samples.')
    parser.add_argument('--replay_buffer_weight', type=float, default=1.0,
                        help='Weight for replay buffer loss.')

    # Distributed training & early stopping
    parser.add_argument('--use_ddp', action='store_true',
                        help='Use DistributedDataParallel')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of epochs with no improvement before early stop.')

    # Testing a small model quickly
    parser.add_argument('--test_model', action='store_true',
                        help='Use smaller model config for quick testing.')

    # Hugging Face update data
    parser.add_argument('--update_url', type=str, required=False,
                        help="Hugging Face dataset URL for new data in 'update' mode.")

    # Manual configs
    parser.add_argument('--n_embed', type=int,
                        help='Embedding dimension override')
    parser.add_argument('--n_head', type=int,
                        help='Number of attention heads override')
    parser.add_argument('--n_layer', type=int,
                        help='Number of transformer blocks override')
    parser.add_argument('--block_size', type=int,
                        help='Max sequence length override')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs override')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')

    args = parser.parse_args()

    # Possibly override config for quick test
    if args.test_model:
        logging.info("Test Mode Activated: Using smaller hyperparameters for faster execution.")
        config.EPOCHS = 3
        config.N_EMBED = 32
        config.N_HEAD = 4
        config.N_LAYER = 12
        config.BLOCK_SIZE = 1024

    # Additional overrides
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
    use_ddp = (torch.cuda.device_count() > 1) and args.use_ddp

    if use_ddp:
        # Initialize process group
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        rank = dist.get_rank()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0

    # Set seeds
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    logging.info(f"Using random seed: {random_seed}")

    if not (0 < args.percent_data <= 100):
        raise ValueError("--percent_data must be between 0 and 100.")

    logging.info(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.save_dir, exist_ok=True)

    # -------------------------------------------
    # TRAIN MODE
    # -------------------------------------------
    if args.mode == 'train':

        # Initialize model from scratch
        model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)
        logging.info(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")
        if initialized_from_scratch:
            model.apply(kaiming_init_weights)
            logging.info("Initialized model from scratch and applied Kaiming initialization.")

        # Prepare optimizer BEFORE wrapping with DDP
        # so we can reference model.token_embedding_table, etc.
        optimizer = prepare_optimizer(model, args)

        # If using DDP, wrap AFTER param-grouping
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              output_device=local_rank, find_unused_parameters=True)
            logging.info("Model wrapped with DistributedDataParallel.")

        # Prepare data
        train_dataloader, df, top25_dict = prepare_data(args, tokenizer)

        # Initialize EBM if requested
        ebm = None
        ebm_optimizer = None
        if args.use_ebm:
            from utils.ebm import EnergyBasedModel
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED).to(device)
            ebm_optimizer = torch.optim.AdamW(ebm.parameters(),
                                              lr=args.ebm_learning_rate*0.1)

            # Re-create train_dataloader with custom collate_fn if needed
            from utils.data import custom_collate_fn
            train_dataset = train_dataloader.dataset
            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                collate_fn=custom_collate_fn
            )

        # SI
        si = initialize_si(model, args) if args.use_si else None
        # EWC
        if args.use_ewc:
            ewc_instance = ElasticWeightConsolidation(model, train_dataloader, device, args)
            ewc_list = [ewc_instance]
        else:
            ewc_list = None
        # Replay Buffer
        replay_buffer = initialize_replay_buffer(args) if args.use_replay_buffer else None

        # Train
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
            tokenizer=tokenizer,
            top25_dict=top25_dict
        )
        logging.info("Training completed.")

        if ebm and (rank == 0):
            save_ebm_model(ebm, epoch=config.EPOCHS, save_dir="models")

        # Save final model/states
        if rank == 0:
            save_model_and_states(model, si, replay_buffer, ewc_list, args)

    # -------------------------------------------
    # UPDATE MODE
    # -------------------------------------------
    elif args.mode == 'update':
        update_dataloader, df = prepare_data(args, tokenizer)

        if not all([args.bucket]):
            raise ValueError("When using EBM sampling in 'update', --bucket is required.")

        download_models_from_s3(bucket=args.bucket)

        # Initialize from scratch, then load
        model, _ = initialize_model(args, device, init_from_scratch=True)
        model.to(device)

        # Load main model weights
        model_path = os.path.join("model", args.save_model_name if args.save_model_name else "model_weights.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Main transformer model loaded from S3.")

        # Possibly load EBM
        ebm = None
        if args.use_ebm:
            from utils.ebm import EnergyBasedModel
            ebm_path = os.path.join("models", "ebm.pt")
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED)
            ebm.load_state_dict(torch.load(ebm_path, map_location=device))
            ebm.to(device)
            ebm.eval()
            logging.info("EBM model loaded from S3.")

        # Prepare optimizer (model module) BEFORE DDP
        optimizer = prepare_optimizer(model, args)

        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              output_device=local_rank)
            logging.info("Model wrapped with DistributedDataParallel.")

        # Possibly re-create update dataloader if using EBM
        ebm_optimizer = None
        if args.use_ebm:
            from utils.ebm import EnergyBasedModel
            ebm_optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.ebm_learning_rate)
            from utils.data import custom_collate_fn
            update_dataset = update_dataloader.dataset
            update_dataloader = DataLoader(
                update_dataset, batch_size=args.batch_size, shuffle=True,
                collate_fn=custom_collate_fn
            )

        si = initialize_si(model, args) if args.use_si else None

        # Possibly load EWC
        if args.use_ewc:
            ewc_state_path = os.path.join(args.save_dir, 'ewc_state.pth')
            if os.path.exists(ewc_state_path):
                ewc_states = torch.load(ewc_state_path, map_location=device)
                ewc_list = []
                for state in ewc_states:
                    ewc_instance = ElasticWeightConsolidation(model, dataloader=None,
                                                              device=device, args=args)
                    ewc_instance.params = {n: p.to(device) for n, p in state['params'].items()}
                    ewc_instance.fisher = {n: f.to(device) for n, f in state['fisher'].items()}
                    ewc_list.append(ewc_instance)
                logging.info(f"Loaded EWC state from '{ewc_state_path}'.")
            else:
                logging.info("No EWC state found. Starting fresh EWC.")
                ewc_instance = ElasticWeightConsolidation(model, update_dataloader, device, args)
                ewc_list = [ewc_instance]
        else:
            ewc_list = None

        replay_buffer = initialize_replay_buffer(args) if args.use_replay_buffer else None

        # Do the update
        logging.info("Starting updating...")
        train_model(
            model=model,
            optimizer=optimizer,
            epochs=config.EPOCHS,
            device=device,
            dataloader=update_dataloader,
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

        if ebm and (rank == 0):
            save_ebm_model(ebm, epoch=config.EPOCHS, save_dir="models")

        if rank == 0:
            save_model_and_states(model, si, replay_buffer, ewc_list, args)

    # -------------------------------------------
    # RUN MODE
    # -------------------------------------------
    elif args.mode == 'run':
        # If not just testing, ensure we have stock/date/text/bucket
        if (not args.test) and not all([args.stock, args.date, args.text, args.bucket]):
            raise ValueError("In 'run' mode with EBM (no --test), must provide --stock, --date, --text, --bucket.")
        elif args.test and not args.bucket:
            raise ValueError("When evaluating on test set, provide --bucket.")

        run_dataloader, df, top25_dict = prepare_data(args, tokenizer)
        download_models_from_s3(bucket=args.bucket)

        model, _ = initialize_model(args, device, init_from_scratch=True)
        model.to(device)
        # Load main model weights
        model_path = os.path.join("model", args.save_model_name if args.save_model_name else "model_weights.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Main transformer model loaded from S3.")

        if args.use_ebm and (not args.test):
            from utils.ebm import EnergyBasedModel
            ebm_path = os.path.join("models", "ebm.pt")
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED)
            ebm.load_state_dict(torch.load(ebm_path, map_location=device))
            ebm.to(device)
            ebm.eval()
            logging.info("EBM model loaded from S3.")

            # Possibly override sample_count
            num_samples = args.test if args.test else 25
            selected_context = ebm_select_contexts(
                df=df,
                stock=args.stock,
                date=args.date,
                sample_count=num_samples,
                model=model,
                ebm=ebm,
                tokenizer=tokenizer
            )
            final_input = f"{selected_context}\n{args.text}"

            encoding = tokenizer(
                final_input,
                truncation=True,
                padding='max_length',
                max_length=config.BLOCK_SIZE,
                return_tensors='pt'
            ).to(device)
            input_ids = encoding["input_ids"]

            with torch.no_grad():
                prediction, _ = model(input_ids=input_ids)
            print(f"Predicted Price: {prediction.item()}")

        elif not args.test:
            # Simple run without EBM
            encoding = tokenizer(
                args.text,
                truncation=True,
                padding='max_length',
                max_length=config.BLOCK_SIZE,
                return_tensors='pt'
            ).to(device)
            input_ids = encoding["input_ids"]
            with torch.no_grad():
                prediction, _ = model(input_ids=input_ids)
            print(f"Predicted Price: {prediction.item()}")

        else:
            # Evaluate on test set
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
                risk_free_rates=df['Risk_Free_Rate'].tolist(),
                tokenizer=tokenizer,
                total_epochs=1,
                use_ebm=args.use_ebm
            )
            # ---- Call evaluate_model and Unpack Metrics ----
            mse, r2, sector_metrics, overall_trend_acc, sharpe_ratio, sortino_ratio, average_return, win_rate, profit_factor = evaluate_model(model, run_dataloader, device)
            
            # ---- Print and Log Overall Metrics ----
            print(f"Test MSE: {mse:.4f}, R² Score: {r2:.4f}")
            logging.info(f"Test MSE: {mse:.4f}, R² Score: {r2:.4f}")
            
            print(f"Overall Trend Accuracy: {overall_trend_acc:.4f}")
            logging.info(f"Overall Trend Accuracy: {overall_trend_acc:.4f}")
            
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            
            print(f"Sortino Ratio: {sortino_ratio:.4f}")
            logging.info(f"Sortino Ratio: {sortino_ratio:.4f}")
            
            # ---- Print and Log New Metrics ----
            print(f"Average Return: {average_return:.4f}")
            logging.info(f"Average Return: {average_return:.4f}")
            
            print(f"Win Rate: {win_rate:.2f}%")
            logging.info(f"Win Rate: {win_rate:.2f}%")
            
            print(f"Profit Factor: {profit_factor:.4f}")
            logging.info(f"Profit Factor: {profit_factor:.4f}")
            
            # ---- Print and Log Per-Sector Metrics ----
            print("Per-Sector Metrics:")
            logging.info("Per-Sector Metrics:")
            
            for sector, metrics in sector_metrics.items():
                # Extract sector-specific metrics
                sector_mse = metrics.get('mse', 0.0)
                sector_r2 = metrics.get('r2', 0.0)
                sector_trend_acc = metrics.get('trend_acc', 0.0)
                sector_sharpe = metrics.get('sharpe', 0.0)
                sector_sortino = metrics.get('sortino', 0.0)
                sector_avg_return = metrics.get('average_return', 0.0)
                sector_win_rate = metrics.get('win_rate', 0.0)
                sector_profit_factor = metrics.get('profit_factor', 0.0)
            
                # Print sector metrics
                print(
                    f"Sector: {sector} - "
                    f"MSE: {sector_mse:.4f}, "
                    f"R²: {sector_r2:.4f}, "
                    f"Trend Accuracy: {sector_trend_acc:.4f}, "
                    f"Sharpe Ratio: {sector_sharpe:.4f}, "
                    f"Sortino Ratio: {sector_sortino:.4f}, "
                    f"Average Return: {sector_avg_return:.4f}, "
                    f"Win Rate: {sector_win_rate:.2f}%, "
                    f"Profit Factor: {sector_profit_factor:.4f}"
                )
            
                # Log sector metrics
                logging.info(
                    f"Sector: {sector} - "
                    f"MSE: {sector_mse:.4f}, "
                    f"R²: {sector_r2:.4f}, "
                    f"Trend Accuracy: {sector_trend_acc:.4f}, "
                    f"Sharpe Ratio: {sector_sharpe:.4f}, "
                    f"Sortino Ratio: {sector_sortino:.4f}, "
                    f"Average Return: {sector_avg_return:.4f}, "
                    f"Win Rate: {sector_win_rate:.2f}%, "
                    f"Profit Factor: {sector_profit_factor:.4f}"
                )
    # -------------------------------------------
    # TEST_FORGETTING
    # -------------------------------------------
    elif args.mode == 'test_forgetting':
        model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)
        logging.info("Initialized model for catastrophic forgetting testing.")
        logging.info(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")

        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              output_device=local_rank)

        optimizer = prepare_optimizer(model.module if use_ddp else model, args)
        si = initialize_si(model, args) if args.use_si else None
        ewc_list = [] if args.use_ewc else None
        replay_buffer = initialize_replay_buffer(args) if args.use_replay_buffer else None

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
        logging.info(f"Catastrophic Forgetting Test Results: {test_results}")
        with open(os.path.join(args.save_dir, 'test_forgetting_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)

    else:
        raise ValueError("Invalid mode. Choose from 'train', 'run', 'update', or 'test_forgetting'.")

    # Clean up DDP
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
