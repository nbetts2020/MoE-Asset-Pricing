import torch
import os
import argparse
import logging
import numpy as np
import random
import pandas as pd
import json
import subprocess
from multiprocessing import cpu_count
from sklearn.metrics import mean_squared_error, r2_score

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
    kaiming_init_weights,
    prepare_data,
    download_models_from_s3,
    ebm_select_contexts,
    get_data
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def print_debug_info(stage):
    """
    Print debug information about the distributed environment.
    """
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


def consolidate_checkpoint_to_pth(checkpoint_dir: str, tag: str, output_path: str) -> str:
    """
    Consolidates a DeepSpeed ZeRO checkpoint into a single .pth file
    for older DeepSpeed versions that expect (base_dir, output_dir, tag).

    When saving a checkpoint with:
        engine.save_checkpoint(save_dir, tag=tag)
    DeepSpeed creates shards in save_dir/tag.

    In this function, we pass the base directory (args.save_dir) as checkpoint_dir
    and let DeepSpeed look for the 'tag' subdirectory.
    The converted state dict is written to a temporary directory as a file
    named pytorch_model.bin, which is then moved to final_path.
    """
    import shutil
    import tempfile
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

    final_path = os.path.join(output_path, f"consolidated_{tag}.pth")
    if os.path.exists(final_path):
        if os.path.isdir(final_path):
            shutil.rmtree(final_path)
        else:
            os.remove(final_path)

    temp_dir = tempfile.mkdtemp(prefix="ds_conversion_")
    logging.info(f"Converting checkpoint from base_dir='{checkpoint_dir}', tag='{tag}'")
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir,   # base_dir (e.g., "model")
        temp_dir,         # output_dir (temporary)
        tag=tag
    )

    converted_bin = os.path.join(temp_dir, "pytorch_model.bin")
    if not os.path.isfile(converted_bin):
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Conversion failed; {converted_bin} not found.")

    shutil.move(converted_bin, final_path)
    shutil.rmtree(temp_dir)

    if not os.path.isfile(final_path):
        raise RuntimeError(f"Failed to create consolidated checkpoint file at {final_path}")

    logging.info(f"Successfully created consolidated checkpoint at {final_path}")
    return final_path


def upload_checkpoint_to_s3(local_dir: str, bucket: str, remote_dir: str = "model"):
    """
    Uploads the contents of a local directory to an S3 bucket using the AWS CLI.
    """
    try:
        logging.info(f"Uploading checkpoint from {local_dir} to s3://{bucket}/{remote_dir}")
        result = subprocess.run(
            ["aws", "s3", "sync", local_dir, f"s3://{bucket}/{remote_dir}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"Checkpoint directory synchronized successfully.\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error syncing checkpoint directory: {e.stderr}")
        raise


def main():
    pandarallel.initialize(nb_workers=cpu_count() - 1, progress_bar=True)
    logging.info("pandarallel initialized with parallel_apply.")
    print_debug_info("START")

    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed by DeepSpeed launcher")
    parser.add_argument("mode", choices=["train", "run", "update", "test_forgetting"],
                        help="Mode: 'train', 'run', 'update', or 'test_forgetting'")
    parser.add_argument("input_text", type=str, nargs="?", default=None,
                        help="Input text if mode='run' (unless --test).")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face repository ID to load the model from.")
    parser.add_argument("--save_model_name", type=str, default=None, help="Name of saved model.")
    parser.add_argument("--test", action="store_true", help="If specified in 'run' mode, evaluate on the test set.")
    parser.add_argument("--update", action="store_true", help="Include this flag to perform an update.")
    parser.add_argument("--percent_data", type=float, default=100.0, help="Percentage of data to use (0 < percent_data <= 100).")
    parser.add_argument("--save_dir", type=str, default="model", help="Directory to save the model and states.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints and states.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to a checkpoint to resume training")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_tasks", type=int, default=3, help="Number of tasks (sectors) for catastrophic forgetting testing.")
    parser.add_argument("--numeric_only", action="store_true", help="Ablation test for extracting value of text in prediction.")
    parser.add_argument("--use_si", action="store_true", help="Use Synaptic Intelligence.")
    parser.add_argument("--use_replay_buffer", action="store_true", help="Use Memory Replay Buffer.")
    parser.add_argument("--replay_buffer_capacity", type=int, default=10000, help="Capacity of the Memory Replay Buffer.")
    parser.add_argument("--use_l2", action="store_true", help="Use L2 regularization.")
    parser.add_argument("--lambda_l2", type=float, default=0.01, help="Regularization strength for L2.")
    parser.add_argument("--use_entropy_reg", action="store_true", help="Use entropy regularization in expert routing.")
    parser.add_argument("--lambda_entropy", type=float, default=0.01, help="Regularization strength for entropy.")
    parser.add_argument("--use_ewc", action="store_true", help="Use Elastic Weight Consolidation.")
    parser.add_argument("--lambda_ewc", type=float, default=0.4, help="Regularization strength for EWC.")
    parser.add_argument("--use_ebm", action="store_true", help="Use energy-based model for prompt optimization.")
    parser.add_argument("--ebm_learning_rate", type=float, default=1e-4, help="Learning rate for the EBM")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter for Sampling")
    parser.add_argument("--ebm_num_samples_train", type=int, help="Number of samples the EBM generates when 'train' is active")
    parser.add_argument("--use_ebm_format", action="store_true", help="Use simplified version of EBM formatting when using non-EBM model.")
    parser.add_argument("--ebm_num_samples", type=int, default=25, help="Number of samples the EBM generates when 'run' is active")
    parser.add_argument("--stock", type=str, required=False, help="Stock symbol for 'run' mode.")
    parser.add_argument("--date", type=str, required=False, help="Date for 'run' mode (e.g., '2025-02-18').")
    parser.add_argument("--text", type=str, required=False, help="Input article text for 'run' mode.")
    parser.add_argument("--bucket", type=str, required=False, help="S3 bucket name for 'run'/'update' mode.")
    parser.add_argument("--replay_batch_size", type=int, default=32, help="Batch size for replay buffer samples.")
    parser.add_argument("--replay_buffer_weight", type=float, default=1.0, help="Weight for replay buffer loss.")
    parser.add_argument("--use_ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs with no improvement before early stop.")
    parser.add_argument("--test_model", action="store_true", help="Use smaller model config for quick testing.")
    parser.add_argument("--update_url", type=str, required=False, help="Hugging Face dataset URL for new data in 'update' mode.")
    parser.add_argument("--n_embed", type=int, help="Embedding dimension override")
    parser.add_argument("--n_head", type=int, help="Number of attention heads override")
    parser.add_argument("--n_layer", type=int, help="Number of transformer blocks override")
    parser.add_argument("--block_size", type=int, help="Max sequence length override")
    parser.add_argument("--epochs", type=int, help="Number of training epochs override")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--deepspeed_config", type=str, default="utils/deepspeed_config.json",
                        help="Path to the DeepSpeed config file.")

    args = parser.parse_args()
    print_debug_info("AFTER ARGPARSE")

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

        base_optimizer = prepare_optimizer(model, args)
        print("LOCAL_RANK (train):", local_rank)
        print("Available GPUs (train):", torch.cuda.device_count())

        engine, engine_optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=base_optimizer,
            model_parameters=model.parameters(),
            config=args.deepspeed_config
        )
        print_debug_info("AFTER DEEPSPEED INIT")
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print("Global rank (train):", global_rank)

        if args.use_ebm:
            ebm = EnergyBasedModel(embedding_dim=config.N_EMBED).to(device)
            ebm_optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.ebm_learning_rate * 0.1)
        else:
            ebm = None
            ebm_optimizer = None

        train_dataloader, data_bundle = prepare_data(args, tokenizer)
        df = data_bundle["df"]
        df_preprocessed = data_bundle["df_preprocessed"]
        print("DataLoader:", train_dataloader, "Length:", len(train_dataloader))
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
        df = data_bundle["df"]
        df_preprocessed = data_bundle["df_preprocessed"]
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
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_weights_update.pth"))
            logging.info("Updated model weights saved (single GPU / no ZeRO partition).")

    elif args.mode == "run":
        print_debug_info("RUN MODE START")
        # For single prediction mode, require stock, date, text, and bucket.
        if not args.test and not all([args.stock, args.date, args.text, args.bucket]):
            raise ValueError("For non-test 'run' mode, provide --stock, --date, --text, and --bucket.")

        run_dataloader, data_bundle = prepare_data(args, tokenizer)
        df_run = data_bundle["df"]
        df_run_preprocessed = data_bundle["df_preprocessed"]
        print("Run DataLoader length:", len(run_dataloader))
        print("Run DataFrame shape:", df_run.shape, "Preprocessed shape:", df_run_preprocessed.shape)

        download_models_from_s3(bucket=args.bucket)

        consolidated_model_path = consolidate_checkpoint_to_pth(
            checkpoint_dir=args.save_dir,  # Base directory ("model")
            tag="final",                   # Tag used when saving the checkpoint
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

            if not np.issubdtype(df_run['Date'].dtype, np.datetime64):
                df_run['Date'] = pd.to_datetime(df_run['Date'])

            # Lists to accumulate metrics
            predictions = []
            actuals = []
            oldprices = []
            riskfree = []
            sectors = []

            # Generate predictions for each row in the test set
            for idx, sample in df_run.iterrows():
                # This is the article text for the current sample
                sample_text = sample['Article']

                # Use ebm_select_contexts with index-based sampling
                # (We've updated ebm_select_contexts to accept an 'idx' argument.)
                best_context = ebm_select_contexts(
                    df=df_run,
                    idx=idx,  # pass the row index
                    text=sample_text,
                    model=model,
                    ebm=ebm,
                    tokenizer=tokenizer,
                    ebm_samples=args.ebm_num_samples
                )

                # Concatenate the best context with the article text
                final_input = f"{best_context}\n{sample_text}"

                encoding = tokenizer(
                    final_input,
                    truncation=True,
                    padding="max_length",
                    max_length=config.BLOCK_SIZE,
                    return_tensors="pt"
                ).to(device)
                input_ids = encoding["input_ids"]

                with torch.no_grad():
                    pred, _ = model(input_ids=input_ids)

                predictions.append(pred.item())
                actuals.append(sample['Price'])       # Ground truth
                oldprices.append(sample['old_price']) # Past price for trend
                riskfree.append(sample['risk_free_rate'])
                sectors.append(sample.get('sector', "Unknown"))

                print(f"Sample {idx} predicted price: {pred.item()}")

            # Compute overall metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            oldprices = np.array(oldprices)
            riskfree = np.array(riskfree)

            mse = mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)

            true_trends = np.sign(actuals - oldprices)
            pred_trends = np.sign(predictions - oldprices)
            overall_trend_acc = np.mean(true_trends == pred_trends) if predictions.size > 0 else 0.0

            buy_signals = (predictions > oldprices).astype(float)
            strategy_returns = (actuals - oldprices) / (oldprices + 1e-12) * buy_signals
            excess_returns = strategy_returns - riskfree
            sr_mean = np.mean(excess_returns)
            sr_std = np.std(excess_returns, ddof=1)
            sharpe_ratio = sr_mean / sr_std if sr_std > 1e-12 else 0.0

            neg_mask = (excess_returns < 0)
            if np.any(neg_mask):
                downside_std = np.std(excess_returns[neg_mask], ddof=1)
                sortino_ratio = sr_mean / downside_std if downside_std > 1e-12 else float('inf')
            else:
                sortino_ratio = float('inf')

            average_return = np.mean(strategy_returns) if strategy_returns.size > 0 else 0.0
            wins = strategy_returns > 0
            win_rate = np.mean(wins) * 100 if wins.size > 0 else 0.0
            gross_profits = strategy_returns[strategy_returns > 0].sum()
            gross_losses = -strategy_returns[strategy_returns < 0].sum()
            profit_factor = (gross_profits / gross_losses) if gross_losses > 1e-12 else float('inf')

            # Compute per-sector metrics
            merged_sector_data = {}
            for s, p, a, o, r in zip(sectors, predictions, actuals, oldprices, riskfree):
                if s not in merged_sector_data:
                    merged_sector_data[s] = {"predictions": [], "actuals": [], "oldprices": [], "riskfree": []}
                merged_sector_data[s]["predictions"].append(p)
                merged_sector_data[s]["actuals"].append(a)
                merged_sector_data[s]["oldprices"].append(o)
                merged_sector_data[s]["riskfree"].append(r)

            sector_metrics = {}
            for sector, vals in merged_sector_data.items():
                spreds = np.array(vals["predictions"], dtype=np.float64)
                sacts = np.array(vals["actuals"], dtype=np.float64)
                soldp = np.array(vals["oldprices"], dtype=np.float64)
                sriskf = np.array(vals["riskfree"], dtype=np.float64)

                sec_mse = mean_squared_error(sacts, spreds)
                sec_r2 = r2_score(sacts, spreds)
                sec_true_trends = np.sign(sacts - soldp)
                sec_pred_trends = np.sign(spreds - soldp)
                sec_trend_acc = np.mean(sec_true_trends == sec_pred_trends) if spreds.size > 0 else 0.0

                sec_signals = (spreds > soldp).astype(float)
                sec_strategy_returns = (sacts - soldp) / (soldp + 1e-12) * sec_signals
                sec_excess_returns = sec_strategy_returns - sriskf

                sec_ex_mean = np.mean(sec_excess_returns)
                sec_ex_std = np.std(sec_excess_returns, ddof=1)
                sec_sharpe = sec_ex_mean / sec_ex_std if sec_ex_std > 1e-12 else 0.0

                neg_mask_s = (sec_excess_returns < 0)
                if np.any(neg_mask_s):
                    sec_downside_std = np.std(sec_excess_returns[neg_mask_s], ddof=1)
                    sec_sortino = sec_ex_mean / sec_downside_std if sec_downside_std > 1e-12 else float('inf')
                else:
                    sec_sortino = float('inf')

                sec_avg_return = np.mean(sec_strategy_returns) if sec_strategy_returns.size > 0 else 0.0
                sec_wins = sec_strategy_returns > 0
                sec_win_rate = np.mean(sec_wins) * 100 if sec_wins.size > 0 else 0.0
                sec_gross_profits = sec_strategy_returns[sec_strategy_returns > 0].sum()
                sec_gross_losses = -sec_strategy_returns[sec_strategy_returns < 0].sum()
                sec_profit_factor = (sec_gross_profits / sec_gross_losses) if sec_gross_losses > 1e-12 else float('inf')

                sector_metrics[sector] = {
                    "mse": sec_mse,
                    "r2": sec_r2,
                    "trend_acc": sec_trend_acc,
                    "sharpe": sec_sharpe,
                    "sortino": sec_sortino,
                    "average_return": sec_avg_return,
                    "win_rate": sec_win_rate,
                    "profit_factor": sec_profit_factor
                }

            # Print overall metrics
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
            # Single prediction mode
            encoding = tokenizer(
                args.text,
                truncation=True,
                padding="max_length",
                max_length=config.BLOCK_SIZE,
                return_tensors="pt"
            ).to(device)
            input_ids = encoding["input_ids"]
            with torch.no_grad():
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
