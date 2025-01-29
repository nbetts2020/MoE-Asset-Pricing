# utils/train.py

import torch
import torch.nn.functional as F
from torch import amp
from torch.cuda.amp import GradScaler
import numpy as np
import os
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count  # Changed to multiprocessing.Pool

from sklearn.metrics import mean_squared_error, r2_score
from utils.config import config
from utils.data import parallel_context_generation_worker
from utils.utils import compute_l2_loss, load_checkpoint

logging.basicConfig(level=logging.INFO)

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
    tokenizer=None,
    numeric_only=False    # New parameter for numeric-only mode
):
    """
    Optimized Multi-context EBM approach in half precision for flash-attn.

    Additional parameter:
      - numeric_only (bool): If True, performs numeric-only training without EBM.
    """

    # -------------------------------------------------------------------------
    # 1) Optionally load from checkpoint
    # -------------------------------------------------------------------------
    start_epoch = 0
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        start_epoch = load_checkpoint(
            model,
            optimizer,
            ebm if (ebm and args.use_ebm and not numeric_only) else None,
            ebm_optimizer if (ebm and args.use_ebm and not numeric_only) else None,
            args.checkpoint_path
        )

    use_ebm = (ebm is not None) and getattr(args, 'use_ebm', False) and not numeric_only
    if use_ebm:
        ebm.train()  # EBM in train mode

    # We'll run the main model in train mode
    model.train()

    scaler = GradScaler()
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = args.early_stopping_patience

    # -------------------------------------------------------------------------
    # 2) Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        logging.info(f"==== Epoch {epoch+1}/{epochs} ====")
        total_loss = 0.0
        total_count = 0

        # Example: dynamic # of contexts => pyramid: e.g. max(epochs - epoch, 5)
        context_count = args.ebm_num_samples_train if args.ebm_num_samples_train else max(epochs - epoch, 5)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Zero out grads
            optimizer.zero_grad()
            if use_ebm and ebm_optimizer:
                ebm_optimizer.zero_grad()

            # Basic fields from collate_fn
            # Make sure 'input_ids' stays long for embeddings
            main_ids     = batch['input_ids'].to(device, dtype=torch.long)
            future_vals  = batch['labels'].to(device, dtype=torch.float16)  # half
            sector_list  = batch.get('sector', None)
            idx_list     = batch.get('idx', None)
            B = main_ids.size(0)

            if numeric_only:
                # -------------------------------------------
                # Numeric-Only Forward/Backward Pass
                # -------------------------------------------
                with amp.autocast('cuda', dtype=torch.float16):
                    preds, main_loss = model(input_ids=main_ids, targets=future_vals)

                    # add SI, EWC, L2
                    if si:
                        main_loss += si.penalty()
                    if ewc:
                        for ewc_inst in ewc:
                            main_loss += args.lambda_ewc * ewc_inst.penalty(model)
                    if getattr(args, 'use_l2', False):
                        main_loss += args.lambda_l2 * compute_l2_loss(model)

                scaler.scale(main_loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
                    replay_buffer.add_examples(replay_samples, [0]*B)

                continue  # next batch

            if not use_ebm:
                # -------------------------------------------
                # Standard Forward/Backward Pass without EBM
                # -------------------------------------------
                with amp.autocast('cuda', dtype=torch.float16):
                    preds, main_loss = model(input_ids=main_ids, targets=future_vals)

                    # add SI, EWC, L2
                    if si:
                        main_loss += si.penalty()
                    if ewc:
                        for ewc_inst in ewc:
                            main_loss += args.lambda_ewc * ewc_inst.penalty(model)
                    if getattr(args, 'use_l2', False):
                        main_loss += args.lambda_l2 * compute_l2_loss(model)

                scaler.scale(main_loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
                    replay_buffer.add_examples(replay_samples, [0]*B)

                continue  # next batch

            # -----------------------------------------------------------------
            # (b) If use EBM => multi-context approach
            # -----------------------------------------------------------------
            # Batch gather contexts using multiprocessing Pool
            cpu_args_list = [
                (
                    idx,
                    df,
                    df_preprocessed,
                    epochs,
                    epoch,
                    context_count
                )
                for idx in idx_list
            ]

            max_workers = max(cpu_count() - 1, 1)
            with Pool(processes=max_workers) as pool:
                all_contexts_batch = pool.map(parallel_context_generation_worker, cpu_args_list)
                # shape => list[B], each => list-of-strings

            chosen_contexts_toks = []
            ebm_count = 0
            ebm_batch_loss_accum = 0.0

            for i in range(B):
                context_str_list = all_contexts_batch[i]
                if not context_str_list:
                    # fallback => no contexts
                    chosen_contexts_toks.append(None)
                    continue

                # Tokenize all candidate contexts in batch
                candidate_tensors = []
                for c_str in context_str_list:
                    c_str = c_str.strip()
                    if not c_str:
                        continue
                    enc = tokenizer(
                        c_str,
                        truncation=True,
                        padding='max_length',
                        max_length=config.BLOCK_SIZE,
                        return_tensors='pt'
                    )
                    candidate_tensors.append(enc['input_ids'].squeeze(0))
                if not candidate_tensors:
                    chosen_contexts_toks.append(None)
                    continue

                candidate_tensors = torch.stack(candidate_tensors, dim=0).to(device, dtype=torch.long)
                num_candidates = candidate_tensors.size(0)

                # Step B) compute MSE for each candidate in a single forward pass
                combined_tokens = torch.cat([candidate_tensors, main_ids[i].repeat(num_candidates, 1)], dim=1)
                # Truncate if necessary
                if combined_tokens.size(1) > config.BLOCK_SIZE:
                    combined_tokens = combined_tokens[:, :config.BLOCK_SIZE]

                with torch.no_grad(), amp.autocast('cuda', dtype=torch.float16):
                    preds_batch, _ = model(input_ids=combined_tokens)
                    mse_vals = (preds_batch - future_vals[i].unsqueeze(1))**2  # Shape: [num_candidates, 1]
                    mse_vals = mse_vals.squeeze()  # Shape: [num_candidates]

                # Step C) EBM forward => predicted MSE => L2 to actual MSE
                with amp.autocast('cuda', dtype=torch.float16):
                    # Embed all contexts and main articles in batch
                    context_embs = model.get_embeddings(candidate_tensors).half()  # [num_candidates, embed_dim]
                    main_emb = model.get_embeddings(main_ids[i].unsqueeze(0).repeat(num_candidates, 1)).half()  # [num_candidates, embed_dim]

                    pred_mse = ebm(main_emb, context_embs).float().squeeze()  # [num_candidates]
                    # L2 difference with actual MSE
                    ebm_loss_i = torch.mean((pred_mse - mse_vals)**2)

                scaler.scale(ebm_loss_i).backward(retain_graph=True)
                ebm_batch_loss_accum += ebm_loss_i.item() * num_candidates
                ebm_count += num_candidates

                # **Pass all contexts instead of sampling one**
                chosen_contexts_toks.append(candidate_tensors)  # [num_candidates, seq_len]

            # Step E) EBM step
            if ebm_optimizer:
                scaler.step(ebm_optimizer)
                scaler.update()
                ebm_optimizer.zero_grad()

            # Step F) main model forward/back with all chosen contexts
            main_loss_accum = 0.0
            for i in range(B):
                context_tensors = chosen_contexts_toks[i]
                label_i = future_vals[i].unsqueeze(0)
                if context_tensors is None:
                    # no context => just main
                    combined_tokens = main_ids[i].unsqueeze(0)  # [1, seq_len]
                    with amp.autocast('cuda', dtype=torch.float16):
                        pred_val_i, main_loss_i = model(input_ids=combined_tokens, targets=label_i)
                        # add SI, EWC, L2
                        if si:
                            main_loss_i += si.penalty()
                        if ewc:
                            for ewc_inst in ewc:
                                main_loss_i += args.lambda_ewc * ewc_inst.penalty(model)
                        if getattr(args, 'use_l2', False):
                            main_loss_i += args.lambda_l2 * compute_l2_loss(model)
                    scaler.scale(main_loss_i).backward()
                    main_loss_accum += main_loss_i.item()
                else:
                    # Pass all contexts for this sample in a single batch
                    with amp.autocast('cuda', dtype=torch.float16):
                        combined_tokens = torch.cat([context_tensors, main_ids[i].repeat(context_tensors.size(0), 1)], dim=1)
                        # combined_tokens: [num_candidates, seq_len_main + seq_len_context]
                        # Truncate if necessary
                        if combined_tokens.size(1) > config.BLOCK_SIZE:
                            combined_tokens = combined_tokens[:, :config.BLOCK_SIZE]
                        # Prepare targets by repeating label_i for each context
                        expanded_labels = label_i.repeat(context_tensors.size(0), 1)  # [num_candidates, 1]

                        preds, main_loss_i = model(input_ids=combined_tokens, targets=expanded_labels)

                        # add SI, EWC, L2
                        if si:
                            main_loss_i += si.penalty()
                        if ewc:
                            for ewc_inst in ewc:
                                main_loss_i += args.lambda_ewc * ewc_inst.penalty(model)
                        if getattr(args, 'use_l2', False):
                            main_loss_i += args.lambda_l2 * compute_l2_loss(model)

                    # Assuming main_loss_i is averaged over the contexts
                    scaler.scale(main_loss_i).backward()
                    main_loss_accum += main_loss_i.item()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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
                replay_buffer.add_examples(replay_samples, [0]*B)

        # end of epoch => average loss
        avg_loss = total_loss / float(total_count) if total_count > 0 else 0.0
        logging.info(f"Epoch {epoch+1} => train loss: {avg_loss:.4f}")

        # Early stopping on this average loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Stopping early after {epoch+1} epochs (no improvement).")
                break

    logging.info("Training loop completed.")
