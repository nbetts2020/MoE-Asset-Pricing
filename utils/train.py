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
import math

from sklearn.metrics import mean_squared_error, r2_score
from utils.config import config
from utils.data import parallel_context_generation_worker
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
    return model.module if hasattr(model, 'module') else model

def train_model(
    model,
    optimizer,
    epochs,
    device,
    dataloader,
    args,
    si=None,
    ewc=None,
    replay_buffer=None,
    df=None,
    df_preprocessed=None,
    ebm=None,
    ebm_optimizer=None,
    tokenizer=None
):
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

    use_ebm = (ebm is not None) and getattr(args, 'use_ebm', False)
    if use_ebm:
        ebm.train()
        logging.info("EBM set to training mode.")
    else:
        logging.info("EBM not in use.")

    model.train()
    logging.info("Main model set to train mode.")

    try:
        scaler = GradScaler()
        logging.debug("Initialized GradScaler for mixed precision training.")
    except Exception as e:
        logging.error(f"Failed to initialize GradScaler: {e}")
        raise e

    if use_ebm and ebm_optimizer is not None:
        if not list(ebm_optimizer.param_groups):
            logging.error("EBM optimizer has no parameter groups.")
            raise ValueError("EBM optimizer has no parameter groups.")
        else:
            logging.debug("EBM optimizer initialized correctly.")
        for name, param in ebm.named_parameters():
            if not param.requires_grad:
                logging.warning(f"EBM parameter '{name}' does not require gradients. Setting requires_grad=True.")
                param.requires_grad = True

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

        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        if getattr(args, 'streaming', False):
            from utils.utils import ensure_local_dataset_files, get_total_rows
            main_parquet_path, _ = ensure_local_dataset_files()
            total_rows = get_total_rows(main_parquet_path)
            train_rows = int(total_rows * 0.7 * (args.percent_data / 100.0))
            total_batches = math.ceil(train_rows / args.batch_size)
            if getattr(args, 'use_ddp', False):
                world_size = dist.get_world_size()
            else:
                world_size = 1
            batches_per_gpu = math.ceil(total_batches / world_size)
        else:
            batches_per_gpu = len(dataloader)

        if hasattr(args, 'ebm_num_samples_train') and args.ebm_num_samples_train is not None:
            try:
                context_count = int(args.ebm_num_samples_train)
                logging.debug(f"Using ebm_num_samples_train={context_count}")
            except (TypeError, ValueError):
                logging.warning(f"Invalid ebm_num_samples_train value; using default.")
                context_count = max(epochs - epoch, 5)
        else:
            context_count = max(epochs - epoch, 5)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", total=batches_per_gpu)):
            optimizer.zero_grad()
            if use_ebm and ebm_optimizer:
                ebm_optimizer.zero_grad()

            main_ids = batch['input_ids'].to(device, dtype=torch.long)
            future_vals = batch['labels'].to(device, dtype=torch.float16)
            sector_list = batch.get('sector', None)
            idx_list = batch.get('idx', None)
            B = main_ids.size(0)

            with amp.autocast(device_type='cuda', dtype=torch.float16):
                ebm_loss = None
                if use_ebm and idx_list is not None:
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
                    results = pool.imap_unordered(parallel_context_generation_worker, cpu_args_list)
                    # Reassemble results in order using the idx value
                    all_contexts_batch_unsorted = list(results)
                    all_contexts_batch = [contexts for _, contexts in sorted(all_contexts_batch_unsorted, key=lambda x: x[0])]
                    wrapped_model = get_wrapped_model(model)
                    ebm_losses = []
                    for i in range(B):
                        context_str_list = all_contexts_batch[i]
                        if not context_str_list:
                            continue
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
                        try:
                            candidate_tensors = torch.stack(candidate_tensors_list, dim=0).to(device, dtype=torch.long)
                        except Exception as e:
                            logging.error(f"Error stacking candidate tensors: {e}")
                            continue
                        with torch.no_grad():
                            context_embs = wrapped_model.get_embeddings(candidate_tensors).half()
                            main_emb = wrapped_model.get_embeddings(main_ids[i].unsqueeze(0)).half()
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
                        ebm_loss = torch.stack(ebm_losses).mean()
                preds, main_loss = model(input_ids=main_ids, targets=future_vals)
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
                total_loss_batch = main_loss
                if ebm_loss is not None:
                    total_loss_batch += ebm_loss
                scaler.scale(total_loss_batch).backward()

            try:
                scaler.step(optimizer)
                if use_ebm and ebm_optimizer:
                    scaler.step(ebm_optimizer)
                scaler.update()
            except Exception as e:
                logging.error(f"Error stepping optimizers: {e}")

            optimizer.zero_grad()
            if use_ebm and ebm_optimizer:
                ebm_optimizer.zero_grad()

            total_loss += total_loss_batch.item() * B
            total_count += B

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
