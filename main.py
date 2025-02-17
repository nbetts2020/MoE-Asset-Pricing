import torch
import os
import argparse
import logging
import numpy as np
import random
import pandas as pd
import json
from multiprocessing import cpu_count

import deepspeed  # DeepSpeed integration
from transformers import AutoTokenizer

from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import (
    initialize_model,
    prepare_optimizer,
    initialize_si,
    initialize_replay_buffer,
    save_ebm_model,
    save_model_and_states,
    kaiming_init_weights,
    prepare_data,
    download_models_from_s3,
    evaluate_model,
    ebm_select_contexts,
    get_data  # new: load dataset once
)
from utils.config import config
from torch.utils.data import DataLoader
from utils.data import ArticlePriceDataset, custom_collate_fn
from sklearn.model_selection import train_test_split

from pandarallel import pandarallel

from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer
from utils.ewc import ElasticWeightConsolidation
from utils.test import test_forgetting
from utils.ebm import EnergyBasedModel
from torch.utils.data.distributed import DistributedSampler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def print_debug_info(stage):
    # Print robust debug information about the distributed environment.
    print("=== Debug Info:", stage, "===")
    print("Environment variables:")
    print("  RANK:", os.environ.get("RANK"))
    print("  LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
    print("  WORLD_SIZE:", os.environ.get("WORLD_SIZE"))
    if torch.distributed.is_initialized():
        print("torch.distributed.get_rank():", torch.distributed.get_rank())
        print("torch.distributed.get_world_size():", torch.distributed.get_world_size())
    else:
        print("torch.distributed is NOT initialized")
    print("CUDA device count:", torch.cuda.device_count())
    print("====================================")


def main():
    # Initialize pandarallel
    pandarallel.initialize(nb_workers=cpu_count() - 1, progress_bar=True)
    logging.info("pandarallel initialized with parallel_apply.")

    # Print debug info at startup
    print_debug_info("START")

    parser = argparse.ArgumentParser(description="SparseMoE Language Model")

    # Allow DeepSpeed launcher to pass local_rank.
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank passed by DeepSpeed launcher")

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
                        help="Path to a checkpoint to resume training")
    parser.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument('--num_tasks', type=int, default=3,
                        help="Number of tasks (sectors) for catastrophic forgetting testing.")
    parser.add_argument('--numeric_only', action='store_true',
                        help="Ablation test for extracting value of text in prediction.")

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

    # DeepSpeed config path
    parser.add_argument('--deepspeed_config', type=str, default="utils/deepspeed_config.json",
                        help="Path to the DeepSpeed config file.")

    args = parser.parse_args()

    print_debug_info("AFTER ARGPARSE")

    # Possibly override config for quick test
    if args.test_model:
        logging.info("Test Mode Activated: Using smaller hyperparameters for faster execution.")
        config.EPOCHS = 3
        config.N_EMBED = 32
        config.N_HEAD = 4
        config.N_LAYER = 12
        config.BLOCK_SIZE = 1024
    if args.numeric_only:
        logging.info("Switching to numeric_only hyperparameters for faster execution.")
        config.BLOCK_SIZE = 128

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
        logging.info(f"Overriding epochs to {config.EPOCHS}")
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
        logging.info(f"Overriding batch_size to {config.BATCH_SIZE}")

    # When using DeepSpeed, distributed initialization is handled by the launcher.
    local_rank = args.local_rank
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rank = int(os.environ.get('RANK', 0))
    logging.info(f"Using device: {device}, rank: {rank}")
    print_debug_info("BEFORE SEED SETTING")

    # Set seeds
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    logging.info(f"Using random seed: {random_seed}")

    if not (0 < args.percent_data <= 100):
        raise ValueError("--percent_data must be between 0 and 100.")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.save_dir, exist_ok=True)

    print_debug_info("BEFORE MODE SWITCH")

    # ----------------------------------------------------------------
    # MAIN LOGIC (Modes)
    # ----------------------------------------------------------------
    if args.mode == 'train':
        print_debug_info("TRAIN MODE START")
        # Initialize model from config
        model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)
        logging.info(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")
        if initialized_from_scratch:
            model.apply(kaiming_init_weights)
            logging.info("Initialized model from scratch and applied Kaiming initialization.")

        # Prepare base optimizer; DeepSpeed will wrap it
        base_optimizer = prepare_optimizer(model, args)
        # For debugging, force local_rank=0 here if needed for DeepSpeed; normally, do not override:
        # args.local_rank = 0
        print("LOCAL_RANK (train):", local_rank)
        print("Available GPUs (train):", torch.cuda.device_count())

        # Initialize DeepSpeed; this call handles distributed initialization.
        engine, engine_optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=base_optimizer,
            model_parameters=model.parameters(),
            config=args.deepspeed_config
        )
        print_debug_info("AFTER DEEPSPEED INIT")
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print("Global rank (train):", global_rank)

        # Define EBM and its optimizer if needed
        if args.use_ebm:
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED).to(device)
            ebm_optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.ebm_learning_rate * 0.1)
        else:
            ebm = None
            ebm_optimizer = None

        # Build DataLoader using prepare_data (do not pass extra df arguments)
        train_dataloader, data_bundle = prepare_data(args, tokenizer)
        df = data_bundle['df']
        df_preprocessed = data_bundle['df_preprocessed']
        print("DataLoader:", train_dataloader, "Length:", len(train_dataloader), "dl")
        print("DataFrame shape:", df.shape, "Preprocessed shape:", df_preprocessed.shape)

        train_model(
            model=engine,
            optimizer=engine_optimizer,
            epochs=config.EPOCHS,
            device=device,
            dataloader=train_dataloader,
            args=args,
            si=initialize_si(engine, args) if args.use_si else None,
            ewc=[ElasticWeightConsolidation(engine, None, device, args)] if args.use_ewc else None,
            replay_buffer=initialize_replay_buffer(args) if args.use_replay_buffer else None,
            df=df,
            df_preprocessed=df_preprocessed,
            ebm=ebm,
            ebm_optimizer=ebm_optimizer,
            tokenizer=tokenizer,
            use_deepspeed=True
        )
        logging.info("Training completed.")

        if global_rank == 0:
            if args.use_ebm:
                save_ebm_model(ebm, epoch=config.EPOCHS, save_dir="models", args=args)
            save_model_and_states(engine.module, None, None, None, args)

    elif args.mode == 'update':
        print_debug_info("UPDATE MODE START")
        if not args.bucket:
            raise ValueError("When using EBM sampling in 'update', --bucket is required.")

        download_models_from_s3(bucket=args.bucket)

        model, _ = initialize_model(args, device, init_from_scratch=True)
        model_path = os.path.join("model", args.save_model_name if args.save_model_name else "model_weights.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Main transformer model loaded from S3.")

        ebm = None
        ebm_optimizer = None
        if args.use_ebm:
            ebm_path = os.path.join("models", "ebm.pt")
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED)
            ebm.load_state_dict(torch.load(ebm_path, map_location=device))
            ebm.to(device)
            ebm.eval()
            logging.info("EBM model loaded from S3.")
            ebm_optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.ebm_learning_rate)

        optimizer = prepare_optimizer(model, args)

        if args.use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank
            )
            logging.info("Model wrapped with DistributedDataParallel.")

        update_dataloader, data_bundle = prepare_data(args, tokenizer)
        df = data_bundle['df']
        df_preprocessed = data_bundle['df_preprocessed']
        print("Update DataLoader length:", len(update_dataloader))
        print("Update DataFrame shape:", df.shape, "Preprocessed shape:", df_preprocessed.shape)

        train_model(
            model=model,
            optimizer=optimizer,
            epochs=config.EPOCHS,
            device=device,
            dataloader=update_dataloader,
            args=args,
            si=initialize_si(model, args) if args.use_si else None,
            ewc=[ElasticWeightConsolidation(model, None, device, args)] if args.use_ewc else None,
            replay_buffer=initialize_replay_buffer(args) if args.use_replay_buffer else None,
            df=df,
            df_preprocessed=df_preprocessed,
            ebm=ebm,
            ebm_optimizer=ebm_optimizer,
            tokenizer=tokenizer,
            use_deepspeed=False
        )
        logging.info("Update completed.")

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            if args.use_ebm:
                save_ebm_model(ebm, epoch=config.EPOCHS, save_dir="models", args=args)
            save_model_and_states(model, None, None, None, args)

    elif args.mode == 'run':
        print_debug_info("RUN MODE START")
        if (not args.test) and not all([args.stock, args.date, args.text, args.bucket]):
            raise ValueError("In 'run' mode with EBM (no --test), must provide --stock, --date, --text, --bucket.")
        elif args.test and not args.bucket:
            raise ValueError("When evaluating on test set, provide --bucket.")

        # Build DataLoader for the test split
        run_dataloader, data_bundle = prepare_data(args, tokenizer)
        df_run = data_bundle['df']
        df_run_preprocessed = data_bundle['df_preprocessed']
        print("Run DataLoader length:", len(run_dataloader))
        print("Run DataFrame shape:", df_run.shape, "Preprocessed shape:", df_run_preprocessed.shape)

        download_models_from_s3(bucket=args.bucket)

        model, _ = initialize_model(args, device, init_from_scratch=True)
        model_path = os.path.join("model", args.save_model_name if args.save_model_name else "model_weights.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Main transformer model loaded from S3.")

        if args.use_ebm and (not args.test):
            ebm_path = os.path.join("models", "ebm.pt")
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED)
            ebm.load_state_dict(torch.load(ebm_path, map_location=device))
            ebm.to(device)
            ebm.eval()
            logging.info("EBM model loaded from S3.")

            selected_context = ebm_select_contexts(
                df=df_run,
                stock=args.stock,
                date=args.date,
                text=args.text,
                model=model,
                ebm=ebm,
                tokenizer=tokenizer,
                ebm_samples=args.ebm_num_samples
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
            mse, r2, sector_metrics, overall_trend_acc, sharpe_ratio, sortino_ratio, average_return, win_rate, profit_factor = evaluate_model(model, run_dataloader, device)
            print(f"Test MSE: {mse:.4f}, RÂ² Score: {r2:.4f}")
            logging.info(f"Test MSE: {mse:.4f}, RÂ² Score: {r2:.4f}")
            print(f"Overall Trend Accuracy: {overall_trend_acc:.4f}")
            logging.info(f"Overall Trend Accuracy: {overall_trend_acc:.4f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"Sortino Ratio: {sortino_ratio:.4f}")
            logging.info(f"Sortino Ratio: {sortino_ratio:.4f}")
            print(f"Average Return: {average_return:.4f}")
            logging.info(f"Average Return: {average_return:.4f}")
            print(f"Win Rate: {win_rate:.2f}%")
            logging.info(f"Win Rate: {win_rate:.2f}%")
            print(f"Profit Factor: {profit_factor:.4f}")
            logging.info(f"Profit Factor: {profit_factor:.4f}")
            print("Per-Sector Metrics:")
            logging.info("Per-Sector Metrics:")
            for sector, metrics in sector_metrics.items():
                print(
                    f"Sector: {sector} - "
                    f"MSE: {metrics.get('mse',0.0):.4f}, "
                    f"RÂ²: {metrics.get('r2',0.0):.4f}, "
                    f"Trend Accuracy: {metrics.get('trend_acc',0.0):.4f}, "
                    f"Sharpe Ratio: {metrics.get('sharpe',0.0):.4f}, "
                    f"Sortino Ratio: {metrics.get('sortino',0.0):.4f}, "
                    f"Average Return: {metrics.get('average_return',0.0):.4f}, "
                    f"Win Rate: {metrics.get('win_rate',0.0):.2f}%, "
                    f"Profit Factor: {metrics.get('profit_factor',0.0):.4f}"
                )
                logging.info(
                    f"Sector: {sector} - "
                    f"MSE: {metrics.get('mse',0.0):.4f}, "
                    f"RÂ²: {metrics.get('r2',0.0):.4f}, "
                    f"Trend Accuracy: {metrics.get('trend_acc',0.0):.4f}, "
                    f"Sharpe Ratio: {metrics.get('sharpe',0.0):.4f}, "
                    f"Sortino Ratio: {metrics.get('sortino',0.0):.4f}, "
                    f"Average Return: {metrics.get('average_return',0.0):.4f}, "
                    f"Win Rate: {metrics.get('win_rate',0.0):.2f}%, "
                    f"Profit Factor: {metrics.get('profit_factor',0.0):.4f}"
                )

    elif args.mode == 'test_forgetting':
        print_debug_info("TEST_FORGETTING MODE START")
        model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)
        logging.info("Initialized model for catastrophic forgetting testing.")
        logging.info(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")

        if args.use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              output_device=local_rank)
        optimizer = prepare_optimizer(model.module if args.use_ddp else model, args)
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

    if args.use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
