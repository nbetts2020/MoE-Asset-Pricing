import torch
import torch.nn.functional as F
from torch import amp
import numpy as np
import os
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
import math
import sys

import deepspeed

from sklearn.metrics import mean_squared_error, r2_score
from utils.config import config
from utils.utils import compute_l2_loss, load_checkpoint, prepare_dataloader, get_wrapped_model
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
    """Returns the underlying model if wrapped by DistributedDataParallel."""
    return model.module if hasattr(model, 'module') else model

def train_model(
    model,
    optimizers,   # Now a tuple: (adam_optimizer, muon_optimizer)
    epochs,
    device,
    dataloader,  # DataLoader built from the precomputed dataset.
    args,
    si=None,
    ewc=None,
    replay_buffer=None,
    ebm=None,
    ebm_optimizer=None,
    tokenizer=None,
    use_deepspeed=False
):
    torch.backends.cudnn.benchmark = True
    start_epoch = 0
    # Load checkpoint if available.
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        logging.info(f"Loading checkpoint from {args.checkpoint_path}")
        start_epoch = load_checkpoint(
            model,
            optimizers,
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
    logging.info("Set main model to train mode.")

    scaler = None
    if not use_deepspeed:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    best_loss = float('inf')
    epochs_no_improve = 0
    patience = getattr(args, 'early_stopping_patience', 5)
    logging.info(f"Early stopping patience: {patience} epochs.")

    max_workers = max(cpu_count() - 1, 1)
    pool = Pool(processes=max_workers)
    logging.debug(f"Created multiprocessing Pool with {max_workers} workers.")

    total_epoch_batches = len(dataloader)
    if torch.cuda.device_count() > 1:
        logging.info(f"Total batches per epoch: {total_epoch_batches}")

    # Unpack optimizers.
    adam_optimizer, muon_optimizer = optimizers

    for epoch in range(start_epoch, epochs):
        logging.info(f"==== Epoch {epoch+1}/{epochs} ====")
        total_loss = 0.0
        total_count = 0

        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            epoch_pbar = tqdm(total=total_epoch_batches, desc=f"Epoch {epoch+1} Progress",
                              file=sys.stdout, mininterval=0.1, maxinterval=0.5,
                              ascii=True, dynamic_ncols=False, ncols=120)
            sys.stdout.flush()

        for batch_idx, batch in enumerate(dataloader):
            # Zero gradients for both optimizers.
            adam_optimizer.zero_grad()
            muon_optimizer.zero_grad()
            if ebm and ebm_optimizer:
                ebm_optimizer.zero_grad()

            main_ids = batch['input_ids'].to(device, dtype=torch.long)
            future_vals = batch['labels'].to(device, dtype=torch.float16)
            B = main_ids.size(0)

            if future_vals.dim() == 2 and future_vals.size(1) == 1:
                future_vals = future_vals.squeeze(1)
            elif future_vals.dim() not in (1, 2):
                future_vals = future_vals.view(-1)

            with amp.autocast(device_type='cuda', dtype=torch.float16):
                # Simplified EBM branch: compute auxiliary loss from the full text embedding.
                ebm_loss = None
                if ebm is not None:
                    wrapped_ebm = get_wrapped_model(ebm)
                    full_emb = get_wrapped_model(model).get_embeddings(main_ids).half()
                    energy = wrapped_ebm(full_emb)  # shape: (batch_size,)
                    ebm_loss = torch.mean(energy**2)

                outputs, main_loss = model(input_ids=main_ids, targets=future_vals)
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

                # Backward pass: use GradScaler for the Adam branch.
                if not use_deepspeed:
                    scaler.scale(total_loss_batch).backward()
                    scaler.step(adam_optimizer)
                    scaler.update()
                    # For Muon, we assume it is stepped directly.
                    muon_optimizer.step()
                else:
                    model.backward(total_loss_batch)
                    model.step()

            if ebm and ebm_optimizer:
                ebm_optimizer.step()

            if replay_buffer:
                replay_samples = []
                for i in range(B):
                    replay_samples.append({
                        'input_ids': main_ids[i].detach().cpu(),
                        'labels': future_vals[i].detach().cpu()
                    })
                try:
                    replay_buffer.add_examples(replay_samples, [0] * B)
                except Exception as e:
                    logging.error(f"Error adding to replay buffer: {e}")

            adam_optimizer.zero_grad()
            muon_optimizer.zero_grad()
            if ebm and ebm_optimizer:
                ebm_optimizer.zero_grad()

            total_loss += total_loss_batch.item() * B
            total_count += B

            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                epoch_pbar.update(1)
                sys.stdout.flush()

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
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
