# utils/train.py

import torch
import torch.nn.functional as F
from torch import amp
from torch.amp import GradScaler
import numpy as np
import os
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count

from sklearn.metrics import mean_squared_error, r2_score
from utils.config import config
from utils.data import parallel_context_generation_worker
from utils.utils import compute_l2_loss, load_checkpoint

# Configure logging with debug level and file handler
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs for debugging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_debug.log"),
        logging.StreamHandler()
    ]
)

def get_wrapped_model(model):
    """
    Returns the underlying model whether it's wrapped with DistributedDataParallel or not.
    """
    return model.module if hasattr(model, 'module') else model

def train_model(
    model,
    optimizer,
    epochs,
    device,
    dataloader,
    args,
    si=None,              # optional Synaptic Intelligence
    ewc=None,             # optional EWC list
    replay_buffer=None,   # optional replay buffer
    df=None,              # entire DataFrame for sampling contexts (CPU)
    df_preprocessed=None, # also CPU
    ebm=None,             # EBM model
    ebm_optimizer=None,   # EBM optimizer
    tokenizer=None
):
    """
    Single-pass approach for EBM + main model in half precision for flash-attn.

    * CPU-based DataFrame usage: 'df' and 'df_preprocessed' remain on CPU.
    * Minimal GPU usage: context tokenization + embeddings happen on CPU,
      then we only push final tensor batches to GPU.
    """

    # -------------------------------------------------------------------------
    # 1) Optionally load from checkpoint
    # -------------------------------------------------------------------------
    start_epoch = 0
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        logging.info(f"Loading checkpoint from {args.checkpoint_path}")
        start_epoch = load_checkpoint(
            model,
            optimizer,
            ebm if (ebm and getattr(args, 'use_ebm', False)) else None,
            ebm_optimizer if (ebm and getattr(args, 'use_ebm', False)) else None,
            args.checkpoint_path
        )
        logging.info(f"Resumed training from epoch {start_epoch + 1}")
    else:
        logging.info("No checkpoint found or checkpoint path missing. Starting from scratch.")

    # Determine if we are using EBM
    use_ebm = (ebm is not None) and getattr(args, 'use_ebm', False)
    if use_ebm:
        ebm.train()
        logging.info("EBM is set to training mode.")
    else:
        logging.info("EBM is not in use (either None or not requested).")

    # Put main model in train mode
    model.train()
    logging.info("Main model set to train mode.")

    # Initialize GradScaler correctly
    try:
        scaler = GradScaler()
        logging.debug("Initialized GradScaler for mixed precision training.")
    except Exception as e:
        logging.error(f"Failed to initialize GradScaler: {e}")
        raise e

    # Ensure EBM optimizer is initialized with EBM's parameters
    if use_ebm and ebm_optimizer is not None:
        if not list(ebm_optimizer.param_groups):
            logging.error("EBM optimizer has no parameter groups. Ensure it's initialized with ebm.parameters().")
            raise ValueError("EBM optimizer has no parameter groups.")
        else:
            logging.debug("EBM optimizer is correctly initialized with EBM's parameters.")

        # Ensure EBM parameters require gradients
        for name, param in ebm.named_parameters():
            if not param.requires_grad:
                logging.warning(f"EBM parameter '{name}' does not require gradients. Setting requires_grad=True.")
                param.requires_grad = True

    best_loss = float('inf')
    epochs_no_improve = 0
    patience = getattr(args, 'early_stopping_patience', 5)
    logging.info(f"Early stopping patience set to {patience} epochs with no improvement.")

    # -------------------------------------------------------------------------
    # Initialize Persistent Multiprocessing Pool
    # -------------------------------------------------------------------------
    max_workers = max(cpu_count() - 1, 1)
    pool = Pool(processes=max_workers)
    logging.debug(f"Created a persistent multiprocessing Pool with {max_workers} workers.")

    # -------------------------------------------------------------------------
    # 2) Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        logging.info(f"==== Epoch {epoch+1}/{epochs} ====")
        total_loss = 0.0
        total_count = 0

        # If using DDP, set epoch for distributed sampler
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            logging.debug(f"Set epoch {epoch} for DistributedSampler.")

        # Possibly override # of contexts
        if hasattr(args, 'ebm_num_samples_train') and args.ebm_num_samples_train is not None:
            try:
                context_count = int(args.ebm_num_samples_train)
                logging.debug(f"Using ebm_num_samples_train={context_count}")
            except (TypeError, ValueError):
                logging.warning(f"Invalid ebm_num_samples_train value ({args.ebm_num_samples_train}). Using default.")
                context_count = max(epochs - epoch, 5)
                logging.debug(f"Default context_count={context_count}")
        else:
            context_count = max(epochs - epoch, 5)
            logging.debug(f"ebm_num_samples_train not set. Using default context_count={context_count}")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Zero out grads
            optimizer.zero_grad()
            if use_ebm and ebm_optimizer:
                ebm_optimizer.zero_grad()

            # Move main article data onto GPU
            main_ids = batch['input_ids'].to(device, dtype=torch.long)
            future_vals = batch['labels'].to(device, dtype=torch.float16)

            sector_list = batch.get('sector', None)
            idx_list = batch.get('idx', None)
            B = main_ids.size(0)

            logging.debug(f"Processing batch {batch_idx+1} with size {B}.")
            logging.debug(f"main_ids shape: {main_ids.shape}, future_vals shape: {future_vals.shape}")

            # Adjust future_vals shape if needed
            if future_vals.dim() == 2 and future_vals.size(1) == 1:
                future_vals = future_vals.squeeze(1)
            elif future_vals.dim() not in (1, 2):
                future_vals = future_vals.view(-1)

            with amp.autocast(device_type='cuda', dtype=torch.float16):
                # (A) EBM => multi-context approach if use_ebm
                ebm_loss = None
                if use_ebm and idx_list is not None:
                    # Prepare context generation on CPU
                    cpu_args_list = []
                    for idx_val in idx_list:
                        cpu_args_list.append((
                            idx_val,
                            df,
                            df_preprocessed,
                            epochs,
                            epoch,
                            context_count
                        ))

                    try:
                        all_contexts_batch = pool.map(parallel_context_generation_worker, cpu_args_list)
                    except Exception as e:
                        logging.error(f"Error during context generation: {e}")
                        raise e

                    wrapped_model = get_wrapped_model(model)
                    ebm_losses = []

                    # For each sample in this batch
                    for i in range(B):
                        context_str_list = all_contexts_batch[i]
                        if not context_str_list:
                            continue  # no contexts => skip

                        # 1) CPU tokenization
                        candidate_tensors_list = []
                        for c_str in context_str_list:
                            c_str = c_str.strip()
                            if not c_str:
                                continue
                            try:
                                enc = tokenizer(
                                    c_str,
                                    truncation=True,
                                    padding='max_length',
                                    max_length=config.BLOCK_SIZE,
                                    return_tensors='pt'
                                )
                                candidate_tensors_list.append(enc['input_ids'].squeeze(0))
                            except Exception as e:
                                logging.error(f"Error tokenizing context: {e}")
                                continue

                        if not candidate_tensors_list:
                            continue

                        # 2) Move minimal data onto GPU
                        try:
                            candidate_tensors = torch.stack(candidate_tensors_list, dim=0).to(device, dtype=torch.long)
                        except Exception as e:
                            logging.error(f"Error stacking candidate_tensors: {e}")
                            continue

                        # 3) Get embeddings without updating main model
                        with torch.no_grad():
                            context_embs = wrapped_model.get_embeddings(candidate_tensors).half()
                            main_emb = wrapped_model.get_embeddings(main_ids[i].unsqueeze(0)).half()
                            main_emb = main_emb.repeat(candidate_tensors.size(0), 1)

                        # 4) EBM forward => param requires_grad
                        # Detach from main model's graph
                        context_embs = context_embs.detach()
                        main_emb = main_emb.detach()

                        try:
                            pred_mse = ebm(main_emb, context_embs).float()
                            # Simple L2 to 0 for demonstration
                            ebm_loss_i = torch.mean(pred_mse**2)
                            ebm_losses.append(ebm_loss_i)
                        except Exception as e:
                            logging.error(f"Error in EBM forward pass: {e}")
                            continue

                    if ebm_losses:
                        ebm_loss = torch.stack(ebm_losses).mean()
                        logging.debug(f"EBM loss => {ebm_loss.item():.4f}")

                # (B) Main model forward pass
                preds, main_loss = model(input_ids=main_ids, targets=future_vals)
                logging.debug(f"preds shape: {preds.shape}, main_loss: {main_loss.item():.4f}")

                # Additional penalties
                if si:
                    try:
                        main_loss += si.penalty()
                    except Exception as e:
                        logging.error(f"Error adding SI penalty: {e}")

                if ewc:
                    for ewc_obj in ewc:
                        try:
                            main_loss += args.lambda_ewc * ewc_obj.penalty(model)
                        except Exception as e:
                            logging.error(f"Error adding EWC penalty: {e}")

                if getattr(args, 'use_l2', False):
                    try:
                        main_loss += args.lambda_l2 * compute_l2_loss(model)
                    except Exception as e:
                        logging.error(f"Error adding L2 penalty: {e}")

                # Combine losses
                total_loss_batch = main_loss
                if ebm_loss is not None:
                    total_loss_batch += ebm_loss

                # Single backward
                scaler.scale(total_loss_batch).backward()

            # (C) Optimizer steps
            try:
                scaler.step(optimizer)
                if use_ebm and ebm_optimizer:
                    scaler.step(ebm_optimizer)
                scaler.update()
            except Exception as e:
                logging.error(f"Error stepping optimizers: {e}")

            # Zero out grads
            optimizer.zero_grad()
            if use_ebm and ebm_optimizer:
                ebm_optimizer.zero_grad()

            # Accumulate
            total_loss += total_loss_batch.item() * B
            total_count += B

            # Replay buffer (optional)
            if replay_buffer:
                replay_samples = []
                for i in range(B):
                    replay_samples.append({
                        'input_ids': main_ids[i].detach().cpu(),
                        'labels': future_vals[i].detach().cpu(),
                        'sector': sector_list[i] if sector_list else 'Unknown'
                    })
                try:
                    replay_buffer.add_examples(replay_samples, [0]*B)
                except Exception as e:
                    logging.error(f"Error adding to replay buffer: {e}")

        # End of epoch => average loss
        avg_loss = total_loss / float(total_count) if total_count else 0.0
        logging.info(f"Epoch {epoch+1} => train loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            logging.info(f"  New best loss => {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"  No improvement => patience {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                logging.info(f"Stopping early after {epoch+1} epochs (no improvement).")
                break

    # Close pool
    pool.close()
    pool.join()
    logging.debug("Closed the multiprocessing Pool.")
    logging.info("Training loop completed.")
