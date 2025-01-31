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
    df=None,              # entire DataFrame for sampling contexts
    df_preprocessed=None,
    ebm=None,             # EBM model
    ebm_optimizer=None,   # EBM optimizer
    tokenizer=None
):
    """
    Multi-context EBM approach in half precision for flash-attn.
    [Docstring trimmed for brevity]
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
        scaler = GradScaler()  # Using torch.amp.GradScaler
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
    if use_ebm:
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

        # Set the epoch for DistributedSampler if present
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            logging.debug(f"Set epoch {epoch} for DistributedSampler.")

        # Example: dynamic # of contexts => pyramid: e.g., max(epochs - epoch, 5)
        if hasattr(args, 'ebm_num_samples_train') and args.ebm_num_samples_train is not None:
            try:
                context_count = int(args.ebm_num_samples_train)
                logging.debug(f"Using ebm_num_samples_train={context_count}")
            except (TypeError, ValueError) as e:
                logging.warning(f"Invalid ebm_num_samples_train value ({args.ebm_num_samples_train}). Using default.")
                context_count = max(epochs - epoch, 5)
                logging.debug(f"Default context_count={context_count}")
        else:
            context_count = max(epochs - epoch, 5)
            logging.debug(f"ebm_num_samples_train not set. Using default context_count={context_count}")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Zero out gradients
            optimizer.zero_grad()
            if use_ebm and ebm_optimizer:
                ebm_optimizer.zero_grad()

            # Basic fields from collate_fn
            # Ensure 'input_ids' are of type long for embeddings
            main_ids = batch['input_ids'].to(device, dtype=torch.long)
            future_vals = batch['labels'].to(device, dtype=torch.float16)  # half precision
            sector_list = batch.get('sector', None)
            idx_list = batch.get('idx', None)
            B = main_ids.size(0)

            logging.debug(f"Processing batch {batch_idx+1} with size {B}.")
            logging.debug(f"main_ids shape: {main_ids.shape}, future_vals shape: {future_vals.shape}")

            # Adjust future_vals to match preds shape to fix MSE loss
            if future_vals.dim() == 2 and future_vals.size(1) == 1:
                future_vals = future_vals.squeeze(1)
                logging.debug(f"Squeezed future_vals to shape: {future_vals.shape}")
            elif future_vals.dim() == 1:
                # Already in the correct shape
                pass
            else:
                logging.warning(f"Unexpected shape for future_vals: {future_vals.shape}. Attempting to reshape.")
                future_vals = future_vals.view(-1)
                logging.debug(f"Reshaped future_vals to shape: {future_vals.shape}")

            # (a) If not using EBM => normal forward/back pass
            if not use_ebm:
                with amp.autocast(device_type='cuda', dtype=torch.float16):
                    try:
                        preds, main_loss = model(input_ids=main_ids, targets=future_vals)
                        logging.debug(f"  preds shape: {preds.shape}, main_loss: {main_loss.item():.4f}")
                    except Exception as e:
                        logging.error(f"Error during model forward pass: {e}")
                        raise e

                    # Add SI, EWC, L2 losses if applicable
                    if si:
                        try:
                            si_penalty = si.penalty()
                            main_loss += si_penalty
                            logging.debug(f"  Added SI penalty: {si_penalty.item():.4f}")
                        except Exception as e:
                            logging.error(f"Error adding SI penalty: {e}")
                    if ewc:
                        for idx_ewc, ewc_inst in enumerate(ewc):
                            try:
                                ewc_penalty = args.lambda_ewc * ewc_inst.penalty(model)
                                main_loss += ewc_penalty
                                logging.debug(f"  Added EWC penalty {idx_ewc}: {ewc_penalty.item():.4f}")
                            except Exception as e:
                                logging.error(f"Error adding EWC penalty {idx_ewc}: {e}")
                    if getattr(args, 'use_l2', False):
                        try:
                            l2_loss = args.lambda_l2 * compute_l2_loss(model)
                            main_loss += l2_loss
                            logging.debug(f"  Added L2 loss: {l2_loss.item():.4f}")
                        except Exception as e:
                            logging.error(f"Error adding L2 loss: {e}")

                # Backward and optimizer step
                try:
                    scaler.scale(main_loss).backward()
                    # Check gradients for main model
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            logging.debug(f"  Main model param '{name}' gradient norm: {grad_norm:.4f}")
                        else:
                            logging.warning(f"  Main model param '{name}' has no gradient.")
                    scaler.step(optimizer)
                    scaler.update()
                    logging.debug("  Completed backward and optimizer step.")
                except Exception as e:
                    logging.error(f"Error during backward or optimizer step: {e}")
                    raise e

                total_loss += main_loss.item() * B
                total_count += B

                # Replay buffer
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
                        logging.debug(f"  Added {B} samples to replay buffer.")
                    except Exception as e:
                        logging.error(f"Error adding to replay buffer: {e}")

                continue  # Proceed to next batch

            # -----------------------------------------------------------------
            # (b) If use EBM => multi-context approach
            # -----------------------------------------------------------------
            if idx_list is None:
                logging.error("idx_list is missing in batch. Cannot proceed with context generation.")
                raise KeyError("Batch missing 'idx' key required for EBM context generation.")

            # Prepare the arguments for parallel context generation
            cpu_args_list = []
            for idx in idx_list:
                cpu_args_list.append((
                    idx,
                    df,
                    df_preprocessed,
                    epochs,
                    epoch,
                    context_count
                ))

            # Utilize persistent multiprocessing Pool
            try:
                all_contexts_batch = pool.map(parallel_context_generation_worker, cpu_args_list)
                logging.debug(f"  Context generation complete for batch {batch_idx+1}.")
            except Exception as e:
                logging.error(f"  Error during context generation: {e}")
                raise e

            chosen_contexts_toks = []
            ebm_batch_loss_accum = 0.0
            ebm_count = 0

            # Step A) For each sample in the batch
            for i in range(B):
                context_str_list = all_contexts_batch[i]
                if not context_str_list:
                    # fallback => no contexts
                    chosen_contexts_toks.append(None)
                    logging.warning(f"  No contexts found for sample index {i} in batch {batch_idx+1}. Skipping EBM for this sample.")
                    continue

                # Convert each context string into token ids
                candidate_tensors = []
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
                        candidate_tensors.append(enc['input_ids'].squeeze(0))
                    except Exception as e:
                        logging.error(f"  Error tokenizing context: {e}")
                        continue

                if not candidate_tensors:
                    chosen_contexts_toks.append(None)
                    logging.warning(f"  No valid candidate tensors after tokenization for sample index {i} in batch {batch_idx+1}.")
                    continue

                try:
                    candidate_tensors = torch.stack(candidate_tensors, dim=0).to(device, dtype=torch.long)
                except Exception as e:
                    logging.error(f"  Error stacking candidate_tensors: {e}")
                    chosen_contexts_toks.append(None)
                    continue

                num_candidates = candidate_tensors.size(0)
                logging.debug(f"  Sample {i + 1}: {num_candidates} candidate contexts.")

                # Step B) compute MSE for each candidate in a single forward pass
                combined_tokens = torch.cat([candidate_tensors, main_ids[i].repeat(num_candidates, 1)], dim=1)
                # Truncate if necessary
                if combined_tokens.size(1) > config.BLOCK_SIZE:
                    combined_tokens = combined_tokens[:, :config.BLOCK_SIZE]
                    logging.debug(f"  Truncated combined tokens to BLOCK_SIZE={config.BLOCK_SIZE}.")

                with torch.no_grad(), amp.autocast(device_type='cuda', dtype=torch.float16):
                    try:
                        preds_batch, _ = model(input_ids=combined_tokens)
                        logging.debug(f"    preds_batch shape: {preds_batch.shape}")
                    except Exception as e:
                        logging.error(f"    Error during model forward pass for MSE computation: {e}")
                        preds_batch = torch.zeros(num_candidates, device=device, dtype=torch.float16)

                    # Ensure preds_batch has correct dimensions
                    if preds_batch.dim() == 2 and preds_batch.size(1) == 1:
                        preds_batch = preds_batch.squeeze(-1)  # [num_candidates]
                        logging.debug(f"    Squeezed preds_batch to shape: {preds_batch.shape}")
                    elif preds_batch.dim() != 1:
                        logging.error(f"    Unexpected preds_batch shape: {preds_batch.shape}")
                        raise ValueError(f"Unexpected preds_batch shape: {preds_batch.shape}")

                    # Compute MSE
                    mse_vals = (preds_batch - future_vals[i]) ** 2  # [num_candidates]
                    logging.debug(f"    mse_vals shape: {mse_vals.shape}")

                mse_vals_tensor = mse_vals.float()  # Already [num_candidates]

                # Step C) EBM forward => predicted MSE => L2 to actual MSE
                with amp.autocast(device_type='cuda', dtype=torch.float16):
                    try:
                        # Embed all contexts and main article
                        wrapped_model = get_wrapped_model(model)
                        context_embs = wrapped_model.get_embeddings(candidate_tensors).half()  # [num_candidates, embed_dim]
                        main_emb = wrapped_model.get_embeddings(main_ids[i].unsqueeze(0).repeat(num_candidates, 1)).half()  # [num_candidates, embed_dim]
                        logging.debug(f"    context_embs shape: {context_embs.shape}, main_emb shape: {main_emb.shape}")
                    except Exception as e:
                        logging.error(f"    Error during embedding: {e}")
                        context_embs = torch.zeros(num_candidates, config.N_EMBED, device=device, dtype=torch.float16)
                        main_emb = torch.zeros(num_candidates, config.N_EMBED, device=device, dtype=torch.float16)

                    try:
                        pred_mse = ebm(main_emb, context_embs).float().squeeze()
                        logging.debug(f"    pred_mse shape: {pred_mse.shape}")
                    except Exception as e:
                        logging.error(f"    Error during EBM forward pass: {e}")
                        pred_mse = torch.zeros(num_candidates, device=device, dtype=torch.float32)

                    # Compute L2 loss between predicted and actual MSE
                    ebm_loss_i = torch.mean((pred_mse - mse_vals_tensor) ** 2)
                    logging.debug(f"    ebm_loss_i: {ebm_loss_i.item():.4f}")

                # Backward pass for EBM loss
                try:
                    scaler.scale(ebm_loss_i).backward(retain_graph=True)
                    # Check gradients for EBM
                    for name, param in ebm.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            logging.debug(f"    EBM parameter '{name}' gradient norm: {grad_norm:.4f}")
                        else:
                            logging.warning(f"    EBM parameter '{name}' has no gradient.")
                    ebm_batch_loss_accum += ebm_loss_i.item() * num_candidates
                    ebm_count += num_candidates
                    logging.debug(f"    Backward pass completed for EBM loss.")
                except Exception as e:
                    logging.error(f"    Error during backward pass for EBM loss: {e}")
                    continue

                # Pass all contexts instead of sampling one
                chosen_contexts_toks.append(candidate_tensors)  # [num_candidates, seq_len]

            # Step E) EBM optimizer step
            if ebm_optimizer and ebm_count > 0:
                try:
                    scaler.step(ebm_optimizer)
                    scaler.update()
                    ebm_optimizer.zero_grad()
                    logging.debug("  Updated EBM optimizer.")
                except Exception as e:
                    logging.error(f"  Error during EBM optimizer step: {e}")

            # Step F) main model forward/back with all chosen contexts
            main_loss_accum = 0.0
            for i in range(B):
                context_tensors = chosen_contexts_toks[i]
                label_i = future_vals[i].unsqueeze(0)
                if context_tensors is None:
                    # no context => just main
                    combined_tokens = main_ids[i].unsqueeze(0)  # [1, seq_len]
                    logging.debug(f"  Sample {i + 1}: No contexts. Using main article only.")
                    with amp.autocast(device_type='cuda', dtype=torch.float16):
                        try:
                            preds, main_loss_i = model(input_ids=combined_tokens, targets=label_i)
                            logging.debug(f"    preds shape: {preds.shape}, main_loss_i: {main_loss_i.item():.4f}")
                        except Exception as e:
                            logging.error(f"    Error during model forward pass for sample {i} without context: {e}")
                            main_loss_i = torch.tensor(0.0, device=device, requires_grad=True)

                        # Add SI, EWC, L2 losses
                        if si:
                            try:
                                si_penalty = si.penalty()
                                main_loss_i += si_penalty
                                logging.debug(f"    Added SI penalty: {si_penalty.item():.4f}")
                            except Exception as e:
                                logging.error(f"    Error adding SI penalty: {e}")
                        if ewc:
                            for ewc_inst in ewc:
                                try:
                                    ewc_penalty = args.lambda_ewc * ewc_inst.penalty(model)
                                    main_loss_i += ewc_penalty
                                    logging.debug(f"    Added EWC penalty: {ewc_penalty.item():.4f}")
                                except Exception as e:
                                    logging.error(f"    Error adding EWC penalty: {e}")
                        if getattr(args, 'use_l2', False):
                            try:
                                l2_loss = args.lambda_l2 * compute_l2_loss(model)
                                main_loss_i += l2_loss
                                logging.debug(f"    Added L2 loss: {l2_loss.item():.4f}")
                            except Exception as e:
                                logging.error(f"    Error adding L2 loss: {e}")

                    # Backward pass
                    try:
                        scaler.scale(main_loss_i).backward()
                        main_loss_accum += main_loss_i.item()
                        logging.debug(f"    Backward pass completed for sample {i + 1}.")
                    except Exception as e:
                        logging.error(f"    Error during backward pass for sample {i}: {e}")
                else:
                    # Pass all contexts for this sample in a single batch
                    logging.debug(f"  Sample {i + 1}: Processing {context_tensors.size(0)} contexts.")
                    with amp.autocast(device_type='cuda', dtype=torch.float16):
                        # Pass all contexts to model
                        combined_tokens = torch.cat([context_tensors, main_ids[i].repeat(context_tensors.size(0), 1)], dim=1)
                        # Truncate if necessary
                        if combined_tokens.size(1) > config.BLOCK_SIZE:
                            combined_tokens = combined_tokens[:, :config.BLOCK_SIZE]
                            logging.debug(f"    Truncated combined_tokens to BLOCK_SIZE={config.BLOCK_SIZE}.")

                        # Prepare targets by repeating label_i for each context
                        expanded_labels = label_i.repeat(context_tensors.size(0), 1)  # [num_candidates, 1]
                        logging.debug(f"    combined_tokens shape: {combined_tokens.shape}, expanded_labels shape: {expanded_labels.shape}")

                        try:
                            preds, main_loss_i = model(input_ids=combined_tokens, targets=expanded_labels)
                            logging.debug(f"    preds shape: {preds.shape}, main_loss_i: {main_loss_i.item():.4f}")
                        except Exception as e:
                            logging.error(f"    Error during model forward pass with contexts for sample {i}: {e}")
                            main_loss_i = torch.tensor(0.0, device=device, requires_grad=True)

                        # Add SI, EWC, L2 losses
                        if si:
                            try:
                                si_penalty = si.penalty()
                                main_loss_i += si_penalty
                                logging.debug(f"    Added SI penalty: {si_penalty.item():.4f}")
                            except Exception as e:
                                logging.error(f"    Error adding SI penalty: {e}")
                        if ewc:
                            for ewc_inst in ewc:
                                try:
                                    ewc_penalty = args.lambda_ewc * ewc_inst.penalty(model)
                                    main_loss_i += ewc_penalty
                                    logging.debug(f"    Added EWC penalty: {ewc_penalty.item():.4f}")
                                except Exception as e:
                                    logging.error(f"    Error adding EWC penalty: {e}")
                        if getattr(args, 'use_l2', False):
                            try:
                                l2_loss = args.lambda_l2 * compute_l2_loss(model)
                                main_loss_i += l2_loss
                                logging.debug(f"    Added L2 loss: {l2_loss.item():.4f}")
                            except Exception as e:
                                logging.error(f"    Error adding L2 loss: {e}")

                    # Backward pass
                    try:
                        scaler.scale(main_loss_i).backward()
                        main_loss_accum += main_loss_i.item()
                        logging.debug(f"    Backward pass completed for sample {i + 1} with contexts.")
                    except Exception as e:
                        logging.error(f"    Error during backward pass for sample {i}: {e}")

            # Step F) main model optimizer step
            try:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                logging.debug("  Updated main model optimizer.")
            except Exception as e:
                logging.error(f"  Error during main optimizer step: {e}")

            total_loss += main_loss_accum
            total_count += B

            # Replay buffer
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
                    logging.debug(f"  Added {B} samples to replay buffer.")
                except Exception as e:
                    logging.error(f"  Error adding to replay buffer: {e}")

        # end of epoch => average loss
        avg_loss = total_loss / float(total_count) if total_count > 0 else 0.0
        logging.info(f"Epoch {epoch+1} => train loss: {avg_loss:.4f}")

        # Early stopping on this average loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            logging.info(f"  New best loss achieved => {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"  No improvement => patience {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                logging.info(f"Stopping early after {epoch+1} epochs (no improvement).")
                break

    # -------------------------------------------------------------------------
    # Close Persistent Pool
    # -------------------------------------------------------------------------
    pool.close()
    pool.join()
    logging.debug("Closed the multiprocessing Pool.")

    logging.info("Training loop completed.")
