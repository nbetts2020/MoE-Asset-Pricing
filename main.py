# main.py (very top)
import os, sys

# 1) make sure we load your local flash-attention build first
FLASH_PATH = os.path.expanduser("~/MoE-Asset-Pricing/flash-attention")
sys.path.insert(0, FLASH_PATH)

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

import deepspeed
from transformers import AutoTokenizer, LlamaTokenizerFast

from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.test import test_forgetting
from utils.metrics import OnlineMetrics
from utils.utils import (
    initialize_model,
    prepare_optimizer,
    initialize_si,
    initialize_replay_buffer,
    kaiming_init_weights,
    download_models_from_s3,
    ebm_select_contexts,
    get_data,
    consolidate_checkpoint_to_pth,
    upload_checkpoint_to_s3,
    evaluate_model,
    prepare_dataloader,
    process_run_dataset
)
from utils.config import config
from pandarallel import pandarallel
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer
from utils.ewc import ElasticWeightConsolidation
from torch.utils.data.distributed import DistributedSampler
import gc
from deepspeed import zero

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

def gather_lists_across_ranks(local_list):
    """
    Gathers a Python list from all ranks onto rank 0.
    Returns a combined list on rank 0; on other ranks, returns an empty list.
    """
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    object_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(object_list, local_list)
    if torch.distributed.get_rank() == 0:
        combined = []
        for sublist in object_list:
            combined.extend(sublist)
        return combined
    else:
        return []

def main():
    pandarallel.initialize(nb_workers=cpu_count() - 1, progress_bar=True)
    logging.info("pandarallel initialized.")
    print_debug_info("START")

    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed by DeepSpeed launcher")
    parser.add_argument("mode", choices=["train", "run", "test_forgetting"],
                        help="Mode: 'train', 'run', or 'test_forgetting'")
    parser.add_argument("input_text", type=str, nargs="?", default=None,
                        help="Input text if mode='run' (unless --test).")
    parser.add_argument("--tokenizer_name", type=str, default="hf-internal-testing/llama-tokenizer", help="Pretrained tokenizer name")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face repo ID to load the model from")
    parser.add_argument("--save_model_name", type=str, default=None, help="Name of saved model")
    parser.add_argument("--test", action="store_true", help="Evaluate on the test set in run mode")
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
    parser.add_argument("--use_ebm", action="store_true", help="Use integrated energy-based model")
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
    parser.add_argument("--context_window", type=int, help="Max context window")
    parser.add_argument("--epochs", type=int, help="Number of training epochs override")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--deepspeed_config", type=str, default="utils/deepspeed_config.json",
                        help="Path to the DeepSpeed config file")
    parser.add_argument("--stages", type=str, default="1,2,3,4", help="Comma-separated list of training stages to execute (1–4).")

    args.stages = sorted({int(s.strip()) for s in args.stages.split(",") if s.strip()})
    if not all(1 <= s <= 4 for s in args.stages):
        raise ValueError("--stages must contain integers 1-4")

    args = parser.parse_args()
    print_debug_info("AFTER ARGPARSE")

    # Handle small/test-mode hyperparameters
    if args.test_model:
        logging.info("Test Mode Activated: Using smaller hyperparameters for faster execution.")
        config.EPOCHS = 1
        config.N_EMBED = 32
        config.N_HEAD = 4
        config.N_LAYER = 12
        config.BLOCK_SIZE = 1024
    if args.numeric_only:
        logging.info("Switching to numeric_only hyperparameters for faster execution.")
        config.BLOCK_SIZE = 128

    # CLI overrides
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
    if args.context_window is not None:
        config.CONTEXT_WINDOW = args.context_window
        logging.info(f"Overriding batch_size to {config.CONTEXT_WINDOW}")

    local_rank = args.local_rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ.get("RANK", 0))
    logging.info(f"Using device: {device}, rank: {rank}")
    print_debug_info("BEFORE SEED SETTING")

    # Reproducibility
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    logging.info(f"Using random seed: {random_seed}")

    if not (0 < args.percent_data <= 100):
        raise ValueError("--percent_data must be between 0 and 100.")

    tokenizer = LlamaTokenizerFast.from_pretrained(
    args.tokenizer_name, 
    model_max_length=config.CONTEXT_WINDOW
    )
    special_tokens = {
        'additional_special_tokens': ['<bot>', '<start_latent>', '<end_latent>', '<reasoning>', '</reasoning>', '<STOCK PRICE 30 DAYS OUT>: ']
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer vocab size:", tokenizer.vocab_size)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        current_rank = torch.distributed.get_rank()
    else:
        current_rank = 0

    os.makedirs(args.save_dir, exist_ok=True)
    print_debug_info("BEFORE MODE SWITCH")

    # ------------------ TRAIN MODE ------------------
    if args.mode == "train":
        print_debug_info("TRAIN MODE START")

        # Use ZeRO.Init so parameters are sharded across GPUs from the start
        with zero.Init(config_dict_or_path=args.deepspeed_config):
            model, initialized_from_scratch = initialize_model(args, device, init_from_scratch=True)

        # Apply custom initialization if needed
        if initialized_from_scratch:
            model.apply(kaiming_init_weights)
            logging.info("Initialized model from scratch and applied Kaiming initialization.")

        # Prepare optimizer
        optimizers = prepare_optimizer(model, args)
        logging.info(f"LOCAL_RANK (train): {local_rank}")
        logging.info(f"Available GPUs (train): {torch.cuda.device_count()}")

        # Initialize DeepSpeed Engine (ZeRO, etc.)
        engine, engine_optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizers["main"],
            model_parameters=model.parameters(),
            config=args.deepspeed_config
        )
        print_debug_info("AFTER DEEPSPEED INIT")

        # Prepare data
        train_loader = prepare_dataloader(
            epoch=1,
            window_index=1,
            tokenizer=tokenizer,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            global_offset=0,
            global_max=1e12,
            args=args
        )

        # Train
        train_model(
            model=engine,
            optimizer=optimizers,
            epochs=config.EPOCHS,
            device=device,
            dataloader=train_loader,
            args=args,
            si=initialize_si(engine, args) if args.use_si else None,
            ewc=[ElasticWeightConsolidation(engine, None, device, args)] if args.use_ewc else None,
            replay_buffer=initialize_replay_buffer(args) if args.use_replay_buffer else None,
            tokenizer=tokenizer,
            use_deepspeed=True
        )
        logging.info("Training completed.")

    # ------------------ RUN (INFERENCE) MODE ------------------
    elif args.mode == "run":
        print_debug_info("RUN MODE START")
        if not args.test and not all([args.stock, args.date, args.text, args.bucket]):
            raise ValueError("For non-test 'run' mode, provide --stock, --date, --text, and --bucket.")

        consolidated_model_path = os.path.join(args.save_dir, "consolidated_final.pth")

        # Download the consolidated checkpoint
        if current_rank == 0 and args.bucket:
            download_models_from_s3(bucket=args.bucket)
            if not os.path.exists(consolidated_model_path):
                logging.error(f"Consolidated checkpoint not found at {consolidated_model_path} after S3 download")
                raise FileNotFoundError(f"Expected consolidated checkpoint at {consolidated_model_path}")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if not os.path.exists(consolidated_model_path):
            logging.error(f"Consolidated checkpoint missing at {consolidated_model_path}")
            raise FileNotFoundError(f"Consolidated checkpoint not found at {consolidated_model_path}")

        # Basic model creation; do NOT .to(device) if we plan to use DeepSpeed inference
        model, _ = initialize_model(args, device, init_from_scratch=False)
        model.load_state_dict(torch.load(consolidated_model_path, map_location="cpu"))  # Load on CPU first

        # ------------------ Multi-GPU Inference with DeepSpeed ------------------
        # If WORLD_SIZE > 1, let's partition the model across GPUs. Otherwise it stays on one GPU.
        mp_size = int(os.environ.get("WORLD_SIZE", 1))
        # Use fp16 if your hardware supports it; otherwise, switch to bf16 or float32
        model = deepspeed.init_inference(
            model,
            mp_size=mp_size,
            dtype=torch.float16,
            replace_method="auto",
            config=args.deepspeed_config
        )
        model.eval()
        logging.info(
            f"Rank {current_rank}: Main transformer model loaded from {consolidated_model_path} "
            f"with mp_size={mp_size}"
        )

        # RL logic
        from utils.attention_rl import HierarchicalAttentionRL, rl_train_hAttention
        rl_module = None
        if args.use_rl_module:
            rl_module = HierarchicalAttentionRL(
                tokenizer_name=args.tokenizer_name,
                embed_dim=config.N_EMBED,
                max_length=config.BLOCK_SIZE,
                num_queries=config.BLOCK_SIZE,
                truncate_limit=None
            )
            rl_checkpoint_path = os.path.join(args.save_dir, "hAttention_rl.pth")
            if os.path.exists(rl_checkpoint_path):
                checkpoint = torch.load(rl_checkpoint_path, map_location="cpu")
                rl_module.load_state_dict(checkpoint, strict=False)
                logging.info("Hierarchical Attention RL module loaded from checkpoint (keys ignored where mismatched).")
            else:
                logging.warning(f"RL checkpoint not found at {rl_checkpoint_path}; proceeding with untrained RL module.")
            rl_module.eval()

        # ---------- Actual Inference Logic ----------
        if not args.test:
            if args.use_rl_module:
                with torch.no_grad():
                    downsampled, _, _ = rl_module(args.text, model=model)
                    compressed_embedding = downsampled.mean(dim=1)
                    pred = model.regression_head(compressed_embedding).squeeze(0)
                print(f"Predicted Price: {pred.item()}")
            else:
                # Simple truncation approach
                tokenizer.truncation_side = 'left'
                tokens = tokenizer(
                    args.text,
                    truncation=True,
                    max_length=config.CONTEXT_WINDOW,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    # DeepSpeed automatically handles GPU placement, so just pass CPU tensors
                    outputs = model(input_ids=tokens["input_ids"])
                    output = outputs["output"]
                print(f"Predicted Price: {output.item()}")

        else:
            # 'test' inference: iterating over data, computing metrics
            if not args.use_ebm:
                raise ValueError("Test mode requires --use_ebm for integrated EBM logic.")

            if args.percent_data < 100:
                global_max = int(0.2 * 453932 * (args.percent_data / 100))
            else:
                global_max = int(1e12)
            cumulative_offset = 0
            all_metrics = OnlineMetrics()
            all_sector_metrics = {}

            for i in range(1, 19):
                if cumulative_offset >= global_max:
                    logging.info("Global max reached; stopping further file processing.")
                    break

                run_filename = f"run_dataset_{i}.parquet"
                logging.info(f"Processing {run_filename} (cumulative offset: {cumulative_offset})...")

                metrics, sector_metrics, processed_in_file = process_run_dataset(
                    run_dataset_filename=run_filename,
                    tokenizer=tokenizer,
                    model=model,
                    rl_module=rl_module,
                    args=args,
                    device=device,  # device is typically 'cuda', but DS does the partitioning
                    batch_size=1,
                    current_offset=cumulative_offset,
                    global_max=global_max,
                    cache_dir="/tmp/hf_cache_datasets_run"
                )
                cumulative_offset += processed_in_file

                # Accumulate overall stats
                all_metrics.count += metrics.count
                all_metrics.sum_sq_error += metrics.sum_sq_error
                all_metrics.sum_y += metrics.sum_y
                all_metrics.sum_y_sq += metrics.sum_y_sq
                all_metrics.trend_correct += metrics.trend_correct
                for thresh in all_metrics.thresholds:
                    all_stats = all_metrics.stats[thresh]
                    metrics_stats = metrics.stats[thresh]
                    all_stats["excess_returns_sum"] += metrics_stats["excess_returns_sum"]
                    all_stats["excess_returns_sq_sum"] += metrics_stats["excess_returns_sq_sum"]
                    all_stats["downside_returns_sq_sum"] += metrics_stats["downside_returns_sq_sum"]
                    all_stats["downside_count"] += metrics_stats["downside_count"]
                    all_stats["strategy_returns_sum"] += metrics_stats["strategy_returns_sum"]
                    all_stats["wins"] += metrics_stats["wins"]
                    all_stats["gross_profits"] += metrics_stats["gross_profits"]
                    all_stats["gross_losses"] += metrics_stats["gross_losses"]

                # Per-sector stats
                for sector, sm in sector_metrics.items():
                    if sector not in all_sector_metrics:
                        all_sector_metrics[sector] = OnlineMetrics()
                    all_sm = all_sector_metrics[sector]
                    all_sm.count += sm.count
                    all_sm.sum_sq_error += sm.sum_sq_error
                    all_sm.sum_y += sm.sum_y
                    all_sm.sum_y_sq += sm.sum_y_sq
                    all_sm.trend_correct += sm.trend_correct
                    for thresh in all_sm.thresholds:
                        all_stats = all_sm.stats[thresh]
                        sm_stats = sm.stats[thresh]
                        all_stats["excess_returns_sum"] += sm_stats["excess_returns_sum"]
                        all_stats["excess_returns_sq_sum"] += sm_stats["excess_returns_sq_sum"]
                        all_stats["downside_returns_sq_sum"] += sm_stats["downside_returns_sq_sum"]
                        all_stats["downside_count"] += sm_stats["downside_count"]
                        all_stats["strategy_returns_sum"] += sm_stats["strategy_returns_sum"]
                        all_stats["wins"] += sm_stats["wins"]
                        all_stats["gross_profits"] += sm_stats["gross_profits"]
                        all_stats["gross_losses"] += sm_stats["gross_losses"]

                logging.info(f"After {run_filename}, cumulative offset is now {cumulative_offset}.")

            # Compute final results
            results = all_metrics.compute()
            print(f"Test MSE: {results['mse']:.4f}, R² Score: {results['r2']:.4f}")
            print(f"Overall Trend Accuracy: {results['trend_acc']:.4f}")
            for thresh in [0.0, 0.05, 0.10, 0.25, 0.50]:
                thresh_str = f"{int(thresh * 100)}%" if thresh > 0 else "Current"
                print(f"Buy at {thresh_str} higher:")
                print(f"  Sharpe Ratio: {results[f'sharpe_{thresh}']:.4f}")
                print(f"  Sortino Ratio: {results[f'sortino_{thresh}']:.4f}")
                print(f"  Average Return: {results[f'avg_return_{thresh}']:.4f}")
                print(f"  Win Rate: {results[f'win_rate_{thresh}']:.2f}%")
                print(f"  Profit Factor: {results[f'profit_factor_{thresh}']:.4f}")

            sector_results = {sector: sm.compute() for sector, sm in all_sector_metrics.items()}
            for sector, metrics in sector_results.items():
                print(f"\nSector: {sector}")
                for thresh in [0.0, 0.05, 0.10, 0.25, 0.50]:
                    thresh_str = f"{int(thresh * 100)}%" if thresh > 0 else "Current"
                    print(f"  Buy at {thresh_str} higher:")
                    print(f"    MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}, Trend Acc: {metrics['trend_acc']:.4f}")
                    print(f"    Sharpe: {metrics[f'sharpe_{thresh}']:.4f}, Sortino: {metrics[f'sortino_{thresh}']:.4f}")
                    print(f"    Avg Return: {metrics[f'avg_return_{thresh}']:.4f}, "
                          f"Win Rate: {metrics[f'win_rate_{thresh}']:.2f}%, "
                          f"Profit Factor: {metrics[f'profit_factor_{thresh}']:.4f}")

            logging.info(f"Test MSE: {results['mse']:.4f}, R² Score: {results['r2']:.4f}, Trend Acc: {results['trend_acc']:.4f}")
            for thresh in [0.0, 0.05, 0.10, 0.25, 0.50]:
                thresh_str = f"{int(thresh * 100)}%" if thresh > 0 else "Current"
                logging.info(f"Buy at {thresh_str}: Sharpe={results[f'sharpe_{thresh}']:.4f}, "
                             f"Sortino={results[f'sortino_{thresh}']:.4f}, Avg Return={results[f'avg_return_{thresh}']:.4f}, "
                             f"Win Rate={results[f'win_rate_{thresh}']:.2f}%, Profit Factor={results[f'profit_factor_{thresh}']:.4f}")

    # ------------------ TEST_FORGETTING MODE ------------------
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
        raise ValueError("Invalid mode. Choose from 'train', 'run', 'update', 'test_forgetting', or 'rl'.")

    # Destroy distributed process group if it was used
    if args.use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
