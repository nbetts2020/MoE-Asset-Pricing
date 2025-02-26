import torch
import os
import argparse
import logging
import numpy as np
import random
import json
import subprocess
from multiprocessing import cpu_count
from sklearn.metrics import mean_squared_error, r2_score

import deepspeed  # DeepSpeed integration
from transformers import AutoTokenizer

from utils.model import SparseMoELanguageModel, EnergyBasedModel
from utils.train import train_model, test_forgetting  # train_model now internally loads chunks
from utils.utils import (
    initialize_model,
    prepare_optimizer,           # returns (adam_optimizer, muon_optimizer)
    initialize_si,
    initialize_replay_buffer,
    save_ebm_model,
    kaiming_init_weights,
    download_models_from_s3,
    ebm_select_contexts,
    get_data,
    consolidate_checkpoint_to_pth,
    upload_checkpoint_to_s3,
    evaluate_model
)
from utils.config import config
from pandarallel import pandarallel
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer
from utils.ewc import ElasticWeightConsolidation
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def print_debug_info(stage):
    print(f"=== Debug Info: {stage} ===")
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
    pandarallel.initialize(nb_workers=cpu_count() - 1, progress_bar=True)
    logging.info("pandarallel initialized.")
    print_debug_info("START")

    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed by DeepSpeed launcher")
    parser.add_argument("mode", choices=["train", "run", "update", "test_forgetting"],
                        help="Mode: 'train', 'run', 'update', or 'test_forgetting'")
    parser.add_argument("input_text", type=str, nargs="?", default=None,
                        help="Input text if mode='run' (unless --test).")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Pretrained tokenizer name")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face repo ID to load the model from")
    parser.add_argument("--save_model_name", type=str, default=None, help="Name of saved model")
    parser.add_argument("--test", action="store_true", help="Evaluate on the test set in run mode")
    parser.add_argument("--update", action="store_true", help="Perform an update")
    parser.add_argument("--percent_data", type=float, default=100.0, help="Percentage of data to use (0 < percent_data <= 100)")
    parser.add_argument("--save_dir", type=str, default="model", help="Directory to save the model and states")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--checkpoint_path", type=str, help="Path to a checkpoint to resume training")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_tasks", type=int, default=3, help="Number of tasks for catastrophic forgetting testing")
    parser.add_argument("--numeric_only", action="store_true", help="Use numeric_only hyperparameters")
    parser.add_argument("--use_si", action="store_true", help="Use Synaptic Intelligence")
    parser.add_argument("--use_replay_buffer", action="store_true", help="Use Memory Replay Buffer")
    parser.add_argument("--replay_buffer_capacity", type=int, default=10000, help="Capacity of the replay buffer")
    parser.add_argument("--use_l2", action="store_true", help="Use L2 regularization")
    parser.add_argument("--lambda_l2", type=float, default=0.01, help="L2 regularization strength")
    parser.add_argument("--use_entropy_reg", action="store_true", help="Use entropy regularization")
    parser.add_argument("--lambda_entropy", type=float, default=0.01, help="Entropy regularization strength")
    parser.add_argument("--use_ewc", action="store_true", help="Use Elastic Weight Consolidation")
    parser.add_argument("--lambda_ewc", type=float, default=0.4, help="EWC regularization strength")
    parser.add_argument("--use_ebm", action="store_true", help="Use energy-based model for prompt optimization")
    parser.add_argument("--ebm_learning_rate", type=float, default=1e-4, help="Learning rate for the EBM")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--ebm_num_samples_train", type=int, help="Number of samples EBM generates during training")
    parser.add_argument("--use_ebm_format", action="store_true", help="Use simplified EBM formatting")
    parser.add_argument("--ebm_num_samples", type=int, default=25, help="Number of EBM samples in run mode (max 30)")
    parser.add_argument("--stock", type=str, required=False, help="Stock symbol for run mode")
    parser.add_argument("--date", type=str, required=False, help="Date for run mode (e.g., '2025-02-18')")
    parser.add_argument("--text", type=str, required=False, help="Input article text for run mode")
    parser.add_argument("--bucket", type=str, required=False, help="S3 bucket name for run/update mode")
    parser.add_argument("--replay_batch_size", type=int, default=32, help="Batch size for replay buffer samples")
    parser.add_argument("--replay_buffer_weight", type=float, default=1.0, help="Weight for replay buffer loss")
    parser.add_argument("--use_ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--test_model", action="store_true", help="Use smaller model config for testing")
    parser.add_argument("--update_url", type=str, required=False, help="Hugging Face dataset URL for update mode")
    parser.add_argument("--n_embed", type=int, help="Embedding dimension override")
    parser.add_argument("--n_head", type=int, help="Number of attention heads override")
    parser.add_argument("--n_layer", type=int, help="Number of transformer blocks override")
    parser.add_argument("--block_size", type=int, help="Max sequence length override")
    parser.add_argument("--epochs", type=int, help="Number of training epochs override")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--deepspeed_config", type=str, default="utils/deepspeed_config.json",
                        help="Path to the DeepSpeed config file")
    
    args = parser.parse_args()
    print_debug_info("AFTER ARGPARSE")

    if args.test_model:
        logging.info("Test Mode Activated: Using smaller hyperparameters for faster execution.")
        config.EPOCHS = 15  # e.g. 15 epochs (each epoch processes its set of chunks)
        config.N_EMBED = 32
        config.N_HEAD = 4
        config.N_LAYER = 12
        config.BLOCK_SIZE = 1024
    if args.numeric_only:
        logging.info("Switching to numeric_only hyperparameters for faster execution.")
        config.BLOCK_SIZE = 128

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

    local_rank = args.local_rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ.get("RANK", 0))
    logging.info(f"Using device: {device}, rank: {rank}")
    print_debug_info("BEFORE SEED SETTING")

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    logging.info(f"Using random seed: {random_seed}")

    if not (0 < args.percent_data <= 100):
        raise ValueError("--percent_data must be between 0 and 100.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.save_dir, exist_ok=True)
    print_debug_info("BEFORE MODE SWITCH")

    if args.mode == "train":
        print_debug_info("TRAIN MODE START")
        model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)
        logging.info(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")
        if initialized_from_scratch:
            model.apply(kaiming_init_weights)
            logging.info("Initialized model from scratch and applied Kaiming initialization.")

        # Prepare hybrid optimizer: returns (adam_optimizer, muon_optimizer)
        adam_optimizer, muon_optimizer = prepare_optimizer(model, args)
        logging.info(f"LOCAL_RANK (train): {local_rank}")
        logging.info(f"Available GPUs (train): {torch.cuda.device_count()}")

        # Initialize DeepSpeed with the AdamW branch.
        engine, engine_optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=adam_optimizer,
            model_parameters=model.parameters(),
            config=args.deepspeed_config
        )
        print_debug_info("AFTER DEEPSPEED INIT")
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logging.info(f"Global rank (train): {global_rank}")

        if args.use_ebm:
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED).to(device)
            ebm_optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.ebm_learning_rate * 0.1)
        else:
            ebm = None
            ebm_optimizer = None

        # Instead of passing an external dataloader, pass None.
        # The train_model function will handle loading chunks from the Hugging Face repository.
        train_model(
            model=engine,
            optimizers=(adam_optimizer, muon_optimizer),
            epochs=config.EPOCHS,
            device=device,
            dataloader=None,
            args=args,
            si=initialize_si(engine, args) if args.use_si else None,
            ewc=[ElasticWeightConsolidation(engine, None, device, args)] if args.use_ewc else None,
            replay_buffer=initialize_replay_buffer(args) if args.use_replay_buffer else None,
            ebm=ebm,
            ebm_optimizer=ebm_optimizer,
            tokenizer=tokenizer,
            use_deepspeed=True
        )
        logging.info("Training completed.")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        tag = "final"
        engine.save_checkpoint(args.save_dir, tag=tag)
        logging.info(f"DeepSpeed ZeRO checkpoint saved to {args.save_dir}, tag={tag}")
        if args.bucket:
            upload_checkpoint_to_s3(args.save_dir, args.bucket, remote_dir="model")

    elif args.mode == "update":
        print_debug_info("UPDATE MODE START")
        if not args.bucket:
            raise ValueError("When using update mode, --bucket is required.")

        download_models_from_s3(bucket=args.bucket)

        model, _ = initialize_model(args, device, init_from_scratch=True)
        model_path = os.path.join("model", args.save_model_name if args.save_model_name else "model_weights.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Main transformer model loaded from S3.")

        if args.use_ebm:
            ebm_path = os.path.join("models", "ebm.pt")
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED)
            ebm.load_state_dict(torch.load(ebm_path, map_location=device))
            ebm.to(device)
            ebm.eval()
            logging.info("EBM model loaded from S3.")
            ebm_optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.ebm_learning_rate)
        else:
            ebm = None
            ebm_optimizer = None

        adam_optimizer, muon_optimizer = prepare_optimizer(model, args)
        if args.use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank
            )
            logging.info("Model wrapped with DistributedDataParallel.")

        # For update mode, we still load data via the existing loader.
        update_dataloader, data_bundle = prepare_dataloader(args)
        logging.info(f"Update DataLoader length: {len(update_dataloader)}")

        train_model(
            model=model,
            optimizers=(adam_optimizer, muon_optimizer),
            epochs=config.EPOCHS,
            device=device,
            dataloader=update_dataloader,
            args=args,
            si=initialize_si(model, args) if args.use_si else None,
            ewc=[ElasticWeightConsolidation(model, None, device, args)] if args.use_ewc else None,
            replay_buffer=initialize_replay_buffer(args) if args.use_replay_buffer else None,
            ebm=ebm,
            ebm_optimizer=ebm_optimizer,
            tokenizer=tokenizer,
            use_deepspeed=False
        )
        logging.info("Update completed.")

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            if args.use_ebm:
                save_ebm_model(ebm, epoch=config.EPOCHS, save_dir="models", args=args)
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_weights_update.pth"))
            logging.info("Updated model weights saved (single GPU / no ZeRO partition).")

    elif args.mode == "run":
        print_debug_info("RUN MODE START")
        if not args.test and not all([args.stock, args.date, args.text, args.bucket]):
            raise ValueError("For non-test 'run' mode, provide --stock, --date, --text, and --bucket.")

        run_dataloader, data_bundle = prepare_dataloader(args)
        logging.info(f"Run DataLoader length: {len(run_dataloader)}")

        download_models_from_s3(bucket=args.bucket)

        consolidated_model_path = consolidate_checkpoint_to_pth(
            checkpoint_dir=args.save_dir,
            tag="final",
            output_path=args.save_dir
        )

        model, _ = initialize_model(args, device, init_from_scratch=True)
        model.load_state_dict(torch.load(consolidated_model_path, map_location=device), strict=False)
        model.to(device)
        model.eval()
        logging.info("Main transformer model loaded from consolidated checkpoint.")

        if args.test:
            if not args.use_ebm:
                raise ValueError("Test mode requires --use_ebm for EBM logic.")

            ebm_path = os.path.join("models", "ebm.pt")
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED)
            ebm.load_state_dict(torch.load(ebm_path, map_location=device))
            ebm.to(device)
            ebm.eval()
            logging.info("EBM model loaded from S3.")

            predictions = []
            actuals = []
            oldprices = []
            riskfree = []
            sectors = []

            # For each test sample, select the best candidate context.
            for idx, sample in run_dataloader.dataset.df.iterrows():
                best_context = ebm_select_contexts(
                    df=run_dataloader.dataset.df,
                    idx=idx,
                    text=sample['Article'],
                    model=model,
                    ebm=ebm,
                    tokenizer=tokenizer,
                    ebm_samples=args.ebm_num_samples
                )
                final_input = best_context
                encoding = tokenizer(
                    final_input,
                    truncation=True,
                    padding="max_length",
                    max_length=config.BLOCK_SIZE,
                    return_tensors="pt"
                ).to(device)
                input_ids = encoding["input_ids"]
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    pred, _ = model(input_ids=input_ids)
                predictions.append(pred.item())
                actuals.append(sample['weighted_avg_720_hrs'])
                oldprices.append(sample['weighted_avg_0_hrs'])
                riskfree.append(sample['Risk_Free_Rate'])
                sectors.append(sample.get('Sector', "Unknown"))
                print(f"Sample {idx} predicted price: {pred.item()}")

            mse, r2, sector_metrics, overall_trend_acc, sharpe_ratio, sortino_ratio, average_return, win_rate, profit_factor = evaluate_model(model, run_dataloader, device)

            print(f"Test MSE: {mse:.4f}, R² Score: {r2:.4f}")
            print(f"Overall Trend Accuracy: {overall_trend_acc:.4f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"Sortino Ratio: {sortino_ratio:.4f}")
            print(f"Average Return: {average_return:.4f}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Profit Factor: {profit_factor:.4f}")
            logging.info(f"Test MSE: {mse:.4f}, R² Score: {r2:.4f}")
            logging.info(f"Overall Trend Accuracy: {overall_trend_acc:.4f}")
            logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            logging.info(f"Sortino Ratio: {sortino_ratio:.4f}")
            logging.info(f"Average Return: {average_return:.4f}")
            logging.info(f"Win Rate: {win_rate:.2f}%")
            logging.info(f"Profit Factor: {profit_factor:.4f}")
            logging.info("Per-Sector Metrics:")
            for sector, metrics in sector_metrics.items():
                print(
                    f"Sector: {sector} - "
                    f"MSE: {metrics.get('mse', 0.0):.4f}, "
                    f"R²: {metrics.get('r2', 0.0):.4f}, "
                    f"Trend Accuracy: {metrics.get('trend_acc', 0.0):.4f}, "
                    f"Sharpe Ratio: {metrics.get('sharpe', 0.0):.4f}, "
                    f"Sortino Ratio: {metrics.get('sortino', 0.0):.4f}, "
                    f"Average Return: {metrics.get('average_return', 0.0):.4f}, "
                    f"Win Rate: {metrics.get('win_rate', 0.0):.2f}%, "
                    f"Profit Factor: {metrics.get('profit_factor', 0.0):.4f}"
                )
                logging.info(
                    f"Sector: {sector} - "
                    f"MSE: {metrics.get('mse', 0.0):.4f}, "
                    f"R²: {metrics.get('r2', 0.0):.4f}, "
                    f"Trend Accuracy: {metrics.get('trend_acc', 0.0):.4f}, "
                    f"Sharpe Ratio: {metrics.get('sharpe', 0.0):.4f}, "
                    f"Sortino Ratio: {metrics.get('sortino', 0.0):.4f}, "
                    f"Average Return: {metrics.get('average_return', 0.0):.4f}, "
                    f"Win Rate: {metrics.get('win_rate', 0.0):.2f}%, "
                    f"Profit Factor: {metrics.get('profit_factor', 0.0):.4f}"
                )
        else:
            encoding = tokenizer(
                args.text,
                truncation=True,
                padding="max_length",
                max_length=config.BLOCK_SIZE,
                return_tensors="pt"
            ).to(device)
            input_ids = encoding["input_ids"]
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                prediction, _ = model(input_ids=input_ids)
            print(f"Predicted Price: {prediction.item()}")

    elif args.mode == "test_forgetting":
        print_debug_info("TEST_FORGETTING MODE START")
        model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)
        logging.info("Initialized model for catastrophic forgetting testing.")
        logging.info(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")

        if args.use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank)
        optimizer = prepare_optimizer(model.module if args.use_ddp else model, args)
        si = initialize_si(model, args) if args.use_si else None
        ewc_list = [] if args.use_ewc else None
        replay_buffer = initialize_replay_buffer(args) if args.use_replay_buffer else None

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
        with open(os.path.join(args.save_dir, "test_forgetting_results.json"), "w") as f:
            json.dump(test_results, f, indent=4)
    else:
        raise ValueError("Invalid mode. Choose from 'train', 'run', 'update', or 'test_forgetting'.")

    if args.use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
