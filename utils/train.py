import torch
import torch.nn.functional as F
from torch import amp
from torch.amp import GradScaler
import numpy as np
import os
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
import math
import sys

import deepspeed  # NEW

from sklearn.metrics import mean_squared_error, r2_score
from utils.config import config
from utils.data import parallel_context_generation_worker, GLOBAL_TOKENIZER
from utils.utils import compute_l2_loss, load_checkpoint
import torch.distributed as dist

logging.basicConfig(
    level=logging.DEBUG,
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
    dataloader,  # either a DataLoader or a generator of (DataLoader, data_bundle) pairs in streaming mode
    args,
    si=None,
    ewc=None,
    replay_buffer=None,
    df=None,
    df_preprocessed=None,
    ebm=None,
    ebm_optimizer=None,
    tokenizer=None,
    use_deepspeed=False
):
    # Determine if this is the main process for distributed training
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    # Enable cuDNN autotuning
    torch.backends.cudnn.benchmark = True

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

    if ebm:
        ebm.train()
    model.train()
    logging.info("Main model set to train mode.")

    # For non-DeepSpeed path, use GradScaler for AMP.
    scaler = None
    if not use_deepspeed:
        scaler = GradScaler()

    best_loss = float('inf')
    epochs_no_improve = 0
    patience = getattr(args, 'early_stopping_patience', 5)
    logging.info(f"Early stopping patience: {patience} epochs.")

    max_workers = max(cpu_count() - 1, 1)
    pool = Pool(processes=max_workers)
    logging.debug(f"Created multiprocessing Pool with {max_workers} workers.")

    for epoch in range(start_epoch, epochs):
        logging.info(f"==== Epoch {epoch+1}/{epochs} ====")
        total_loss = 0.0
        total_count = 0

        if getattr(args, 'streaming', False):
            # Compute total batches in this epoch based on training split.
            from utils.utils import ensure_local_dataset_files, get_total_rows
            main_parquet_path, _ = ensure_local_dataset_files()
            total_rows = get_total_rows(main_parquet_path)
            # For training, assume first 70% of data
            train_rows = int(total_rows * 0.7 * (args.percent_data / 100.0))
            total_epoch_batches = math.ceil(train_rows / args.batch_size)
            if is_main_process:
                epoch_pbar = tqdm(total=total_epoch_batches, desc=f"Epoch {epoch+1} Progress",
                                  file=sys.stdout, mininterval=0.1, maxinterval=0.5, ascii=True, dynamic_ncols=False, ncols=120)
                sys.stdout.flush()
            # 'dataloader' is a generator of (window_loader, data_bundle) pairs.
            for window_idx, (window_loader, window_data) in enumerate(dataloader):
                logging.info(f"Processing streaming window {window_idx}")
                window_batches = math.ceil(len(window_data['df']) / args.batch_size)
                if is_main_process:
                    inner_pbar = tqdm(total=window_batches, desc=f"Window {window_idx}", leave=False,
                                      file=sys.stdout, mininterval=0.1, maxinterval=0.5, ascii=True, dynamic_ncols=False, ncols=120)
                    sys.stdout.flush()
                for batch_idx, batch in enumerate(window_loader):
                    if not use_deepspeed:
                        optimizer.zero_grad()
                    else:
                        model.zero_grad()
                    if ebm and ebm_optimizer:
                        ebm_optimizer.zero_grad()

                    main_ids = batch['input_ids'].to(device, dtype=torch.long)
                    future_vals = batch['labels'].to(device, dtype=torch.float16)
                    sector_list = batch.get('sector', None)
                    idx_list = batch.get('idx', None)
                    B = main_ids.size(0)

                    logging.debug(f"Processing batch {batch_idx+1} (window {window_idx}) with size {B}.")
                    logging.debug(f"main_ids shape: {main_ids.shape}, future_vals shape: {future_vals.shape}")

                    if future_vals.dim() == 2 and future_vals.size(1) == 1:
                        future_vals = future_vals.squeeze(1)
                    elif future_vals.dim() not in (1, 2):
                        future_vals = future_vals.view(-1)

                    with amp.autocast(device_type='cuda', dtype=torch.float16):
                        ebm_loss = None
                        if ebm and idx_list is not None:
                            cpu_args_list = []
                            for idx_val in idx_list:
                                cpu_args_list.append((idx_val, df, df_preprocessed, epochs, epoch, getattr(args, "ebm_num_samples_train", 5)))
                            all_contexts_batch = list(pool.map(parallel_context_generation_worker, cpu_args_list))
                            logging.debug("Context generation complete for current batch.")
                            wrapped_model = get_wrapped_model(model)
                            ebm_losses = []
                            for i in range(B):
                                candidate_contexts = all_contexts_batch[i]
                                if not candidate_contexts:
                                    continue
                                candidate_tensors_list = []
                                for token_list in candidate_contexts:
                                    try:
                                        candidate_tensor = torch.tensor(token_list, dtype=torch.long)
                                    except Exception as e:
                                        logging.error(f"Error converting candidate tokens to tensor: {e}")
                                        continue
                                    if candidate_tensor.size(0) > config.BLOCK_SIZE:
                                        candidate_tensor = candidate_tensor[-config.BLOCK_SIZE:]
                                    elif candidate_tensor.size(0) < config.BLOCK_SIZE:
                                        pad_length = config.BLOCK_SIZE - candidate_tensor.size(0)
                                        candidate_tensor = torch.nn.functional.pad(
                                            candidate_tensor, (pad_length, 0),
                                            value=GLOBAL_TOKENIZER.eos_token_id
                                        )
                                    candidate_tensors_list.append(candidate_tensor)
                                if not candidate_tensors_list:
                                    continue
                                try:
                                    candidate_tensors = torch.stack(candidate_tensors_list, dim=0).to(device)
                                except Exception as e:
                                    logging.error(f"Error stacking candidate tensors: {e}")
                                    continue
                                with torch.no_grad():
                                    context_embs = wrapped_model.get_embeddings(candidate_tensors)
                                    main_emb = wrapped_model.get_embeddings(main_ids[i].unsqueeze(0))
                                    main_emb = main_emb.repeat(candidate_tensors.size(0), 1)
                                context_embs = context_embs.detach()
                                main_emb = main_emb.detach()
                                try:
                                    wrapped_ebm = get_wrapped_model(ebm)
                                    pred_mse = wrapped_ebm(main_emb, context_embs).float()
                                    ebm_loss_i = torch.mean(pred_mse**2)
                                    ebm_losses.append(ebm_loss_i)
                                except Exception as e:
                                    logging.error(f"Error in EBM forward pass: {e}")
                                    continue
                            if ebm_losses:
                                ebm_loss = torch.stack(ebm_losses).mean() * 1e-3
                                logging.debug(f"Computed EBM loss: {ebm_loss.item():.4f}")

                        outputs, main_loss = model(input_ids=main_ids, targets=future_vals)
                        logging.debug(f"Main model forward: preds shape: {outputs.shape}, main_loss: {main_loss.item():.4f}")

                        if si:
                            try:
                                si_penalty = si.penalty()
                                main_loss += si_penalty
                                logging.debug(f"Added SI penalty: {si_penalty.item():.4f}")
                            except Exception as e:
                                logging.error(f"Error adding SI penalty: {e}")
                        if ewc:
                            for idx_ewc, ewc_obj in enumerate(ewc):
                                try:
                                    ewc_penalty = args.lambda_ewc * ewc_obj.penalty(model)
                                    main_loss += ewc_penalty
                                    logging.debug(f"Added EWC penalty {idx_ewc}: {ewc_penalty.item():.4f}")
                                except Exception as e:
                                    logging.error(f"Error adding EWC penalty {idx_ewc}: {e}")
                        if getattr(args, 'use_l2', False):
                            try:
                                l2_loss = args.lambda_l2 * compute_l2_loss(model)
                                main_loss += l2_loss
                                logging.debug(f"Added L2 loss: {l2_loss.item():.4f}")
                            except Exception as e:
                                logging.error(f"Error adding L2 loss: {e}")

                        total_loss_batch = main_loss
                        if ebm_loss is not None:
                            total_loss_batch += ebm_loss

                        if not use_deepspeed:
                            scaler.scale(total_loss_batch).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            model.backward(total_loss_batch)
                            model.step()

                    if ebm and ebm_optimizer:
                        ebm_optimizer.step()

                    if not use_deepspeed:
                        optimizer.zero_grad()
                    else:
                        model.zero_grad()
                    if ebm and ebm_optimizer:
                        ebm_optimizer.zero_grad()

                    total_loss += total_loss_batch.item() * B
                    total_count += B
                    if is_main_process:
                        inner_pbar.update(1)
                        epoch_pbar.update(1)
                        sys.stdout.flush()
                if is_main_process:
                    inner_pbar.close()
            if is_main_process:
                epoch_pbar.close()
        else:
            # Non-streaming mode: dataloader is a single DataLoader.
            total_epoch_batches = len(dataloader)
            if is_main_process:
                epoch_pbar = tqdm(total=total_epoch_batches, desc=f"Epoch {epoch+1} Progress",
                                  file=sys.stdout, mininterval=0.1, maxinterval=0.5, ascii=True, dynamic_ncols=False, ncols=120)
                sys.stdout.flush()
            for batch_idx, batch in enumerate(dataloader):
                if not use_deepspeed:
                    optimizer.zero_grad()
                else:
                    model.zero_grad()
                if ebm and ebm_optimizer:
                    ebm_optimizer.zero_grad()

                main_ids = batch['input_ids'].to(device, dtype=torch.long)
                future_vals = batch['labels'].to(device, dtype=torch.float16)
                sector_list = batch.get('sector', None)
                idx_list = batch.get('idx', None)
                B = main_ids.size(0)

                logging.debug(f"Processing batch {batch_idx+1} with size {B}.")
                logging.debug(f"main_ids shape: {main_ids.shape}, future_vals shape: {future_vals.shape}")

                if future_vals.dim() == 2 and future_vals.size(1) == 1:
                    future_vals = future_vals.squeeze(1)
                elif future_vals.dim() not in (1, 2):
                    future_vals = future_vals.view(-1)

                with amp.autocast(device_type='cuda', dtype=torch.float16):
                    ebm_loss = None
                    if ebm and idx_list is not None:
                        cpu_args_list = []
                        for idx_val in idx_list:
                            cpu_args_list.append((idx_val, df, df_preprocessed, epochs, epoch, context_count))
                        all_contexts_batch = list(pool.map(parallel_context_generation_worker, cpu_args_list))
                        logging.debug("Context generation complete for current batch.")
                        wrapped_model = get_wrapped_model(model)
                        ebm_losses = []
                        for i in range(B):
                            candidate_contexts = all_contexts_batch[i]
                            if not candidate_contexts:
                                continue
                            candidate_tensors_list = []
                            for token_list in candidate_contexts:
                                try:
                                    candidate_tensor = torch.tensor(token_list, dtype=torch.long)
                                except Exception as e:
                                    logging.error(f"Error converting candidate tokens to tensor: {e}")
                                    continue
                                if candidate_tensor.size(0) > config.BLOCK_SIZE:
                                    candidate_tensor = candidate_tensor[-config.BLOCK_SIZE:]
                                elif candidate_tensor.size(0) < config.BLOCK_SIZE:
                                    pad_length = config.BLOCK_SIZE - candidate_tensor.size(0)
                                    candidate_tensor = torch.nn.functional.pad(
                                        candidate_tensor, (pad_length, 0),
                                        value=GLOBAL_TOKENIZER.eos_token_id
                                    )
                                candidate_tensors_list.append(candidate_tensor)
                            if not candidate_tensors_list:
                                continue
                            try:
                                candidate_tensors = torch.stack(candidate_tensors_list, dim=0).to(device)
                            except Exception as e:
                                logging.error(f"Error stacking candidate tensors: {e}")
                                continue
                            with torch.no_grad():
                                context_embs = wrapped_model.get_embeddings(candidate_tensors)
                                main_emb = wrapped_model.get_embeddings(main_ids[i].unsqueeze(0))
                                main_emb = main_emb.repeat(candidate_tensors.size(0), 1)
                            context_embs = context_embs.detach()
                            main_emb = main_emb.detach()
                            try:
                                wrapped_ebm = get_wrapped_model(ebm)
                                pred_mse = wrapped_ebm(main_emb, context_embs).float()
                                ebm_loss_i = torch.mean(pred_mse**2)
                                ebm_losses.append(ebm_loss_i)
                            except Exception as e:
                                logging.error(f"Error in EBM forward pass: {e}")
                                continue
                        if ebm_losses:
                            ebm_loss = torch.stack(ebm_losses).mean() * 1e-3
                            logging.debug(f"Computed EBM loss: {ebm_loss.item():.4f}")

                    outputs, main_loss = model(input_ids=main_ids, targets=future_vals)
                    logging.debug(f"Main model forward: preds shape: {outputs.shape}, main_loss: {main_loss.item():.4f}")

                    if si:
                        try:
                            si_penalty = si.penalty()
                            main_loss += si_penalty
                            logging.debug(f"Added SI penalty: {si_penalty.item():.4f}")
                        except Exception as e:
                            logging.error(f"Error adding SI penalty: {e}")
                    if ewc:
                        for idx_ewc, ewc_obj in enumerate(ewc):
                            try:
                                ewc_penalty = args.lambda_ewc * ewc_obj.penalty(model)
                                main_loss += ewc_penalty
                                logging.debug(f"Added EWC penalty {idx_ewc}: {ewc_penalty.item():.4f}")
                            except Exception as e:
                                logging.error(f"Error adding EWC penalty {idx_ewc}: {e}")
                    if getattr(args, 'use_l2', False):
                        try:
                            l2_loss = args.lambda_l2 * compute_l2_loss(model)
                            main_loss += l2_loss
                            logging.debug(f"Added L2 loss: {l2_loss.item():.4f}")
                        except Exception as e:
                            logging.error(f"Error adding L2 loss: {e}")

                    total_loss_batch = main_loss
                    if ebm_loss is not None:
                        total_loss_batch += ebm_loss

                    if not use_deepspeed:
                        scaler.scale(total_loss_batch).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        model.backward(total_loss_batch)
                        model.step()

                if ebm and ebm_optimizer:
                    ebm_optimizer.step()

                if not use_deepspeed:
                    optimizer.zero_grad()
                else:
                    model.zero_grad()
                if ebm and ebm_optimizer:
                    ebm_optimizer.zero_grad()

                total_loss += total_loss_batch.item() * B
                total_count += B
                if is_main_process:
                    epoch_pbar.update(1)
                    sys.stdout.flush()
            if is_main_process:
                epoch_pbar.close()

        avg_loss = total_loss / float(total_count) if total_count else 0.0
        logging.info(f"Epoch {epoch+1} => train loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            logging.info(f"New best loss: {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement: patience {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                logging.info(f"Stopping early after {epoch+1} epochs (no improvement).")
                break

    pool.close()
    pool.join()
    logging.debug("Closed multiprocessing Pool.")
    logging.info("Training loop completed.")
