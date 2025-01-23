# utils/train.py

import torch
import torch.nn.functional as F
from torch import amp  # Updated import for autocast
from torch.cuda.amp import GradScaler
import numpy as np
import os
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor

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
    top25_dict=None
):
    """
    Multi-context EBM approach in half precision for flash-attn.

    Workflow:
      1) Possibly load from checkpoint.
      2) If not use_ebm: standard forward/back pass for each batch.
      3) If use_ebm:
         (A) For each sample, gather multiple contexts (CPU parallel).
         (B) For each candidate context => compute MSE => EBM regresses MSE => backward EBM loss => step.
         (C) Monte Carlo sample exactly 1 context => feed to main model => backward main model loss.
      4) Replay buffer is updated after main forward pass.
      5) EWC / SI / L2 are optionally applied.

    This code expects:
      - 'labels' field in batch => future price
      - 'input_ids' => main article tokens as long dtype
      - 'idx' => row index in df
      - 'sector' => optional string
      - If old price is used => 'old_price' in batch
    """

    # -------------------------------------------------------------------------
    # 1) Optionally load from checkpoint
    # -------------------------------------------------------------------------
    start_epoch = 0
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        start_epoch = load_checkpoint(
            model,
            optimizer,
            ebm if (ebm and args.use_ebm) else None,
            ebm_optimizer if (ebm and args.use_ebm) else None,
            args.checkpoint_path
        )

    use_ebm = (ebm is not None) and getattr(args, 'use_ebm', False)
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

            # (a) If not using EBM => normal forward/back pass
            if not use_ebm:
                with amp.autocast('cuda', dtype=torch.float16):  # Updated autocast usage
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
            # CPU gather contexts
            cpu_args_list = []
            for i in range(B):
                cpu_args_list.append((
                    idx_list[i],
                    df,
                    df_preprocessed,
                    top25_dict,
                    tokenizer,
                    epochs,
                    epoch,
                    context_count
                ))
            # parallel gather
            max_workers = max(os.cpu_count() - 1, 1)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                all_contexts_batch = list(executor.map(parallel_context_generation_worker, cpu_args_list))
                # shape => list[B], each => list-of-strings

            chosen_contexts_toks = []
            ebm_count = 0
            ebm_batch_loss_accum = 0.0

            # Step A) For each sample in the batch
            for i in range(B):
                main_article = main_ids[i].unsqueeze(0)    # [1, seq_len_main]
                label_future = future_vals[i].unsqueeze(0) # [1,]
                context_str_list = all_contexts_batch[i]
                if not context_str_list:
                    # fallback => no contexts
                    chosen_contexts_toks.append(None)
                    continue

                # build candidate_tensors
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

                # Step B) compute MSE for each candidate
                # (no grad for main model here)
                mse_vals = []
                with torch.no_grad(), amp.autocast('cuda', dtype=torch.float16):  # Updated autocast usage
                    for c_idx in range(num_candidates):
                        combined_tokens = torch.cat([
                            candidate_tensors[c_idx].unsqueeze(0),
                            main_article
                        ], dim=1)
                        if combined_tokens.size(1) > config.BLOCK_SIZE:
                            combined_tokens = combined_tokens[:, :config.BLOCK_SIZE]

                        pred_val_i, _ = model(input_ids=combined_tokens)
                        # MSE => (pred_val - label)**2
                        mse_i = (pred_val_i - label_future)**2
                        mse_vals.append(mse_i.squeeze())

                mse_vals_tensor = torch.stack(mse_vals, dim=0).float()  # shape => [num_candidates], as float

                # Step C) EBM forward => predicted MSE => L2 to actual MSE
                with amp.autocast('cuda', dtype=torch.float16):  # Updated autocast usage
                    # embed the main article once
                    main_emb = model.get_embeddings(main_article).half()  # [1, embed_dim]
                    context_embs = []
                    for c_idx in range(num_candidates):
                        ctx_ids = candidate_tensors[c_idx].unsqueeze(0)
                        ctx_emb = model.get_embeddings(ctx_ids).half()  # [1, embed_dim]
                        context_embs.append(ctx_emb.squeeze(0))
                    context_embs = torch.stack(context_embs, dim=0)  # [num_candidates, embed_dim]

                    main_emb_exp = main_emb.expand(num_candidates, -1)       # [num_candidates, embed_dim]
                    pred_mse = ebm(main_emb_exp, context_embs).float().squeeze()
                    # L2 difference with actual MSE
                    ebm_loss_i = torch.mean((pred_mse - mse_vals_tensor)**2)

                scaler.scale(ebm_loss_i).backward(retain_graph=True)
                ebm_batch_loss_accum += ebm_loss_i.item() * num_candidates
                ebm_count += num_candidates

                # Step D) sample exactly 1 context => use predicted MSE => energies
                energies = pred_mse.detach()  # shape => [num_candidates]
                e_min, e_max = energies.min(), energies.max()
                scaled_energies = (energies - e_min) / ((e_max - e_min) + 1e-8)
                temperature = getattr(args, 'temperature', 1.0)
                probs = torch.softmax(-scaled_energies / temperature, dim=0)
                print(f"Shape of probs: {probs.shape}")  # Debugging
                print(f"Probabilities: {probs}")  # Debugging

                # Handle single context candidate
                if probs.dim() == 0:
                    sampled_idx = 0
                else:
                    sampled_idx = torch.multinomial(probs, 1).item()

                chosen_contexts_toks.append(candidate_tensors[sampled_idx])

            # Step E) EBM step
            if ebm_optimizer:
                scaler.step(ebm_optimizer)
                scaler.update()
                ebm_optimizer.zero_grad()

            # Step F) main model forward/back with chosen contexts
            main_loss_accum = 0.0
            for i in range(B):
                if chosen_contexts_toks[i] is None:
                    # no context => just main
                    combined_tokens = main_ids[i].unsqueeze(0)
                else:
                    combined_tokens = torch.cat([
                        main_ids[i].unsqueeze(0),
                        chosen_contexts_toks[i].unsqueeze(0)
                    ], dim=1)
                if combined_tokens.size(1) > config.BLOCK_SIZE:
                    combined_tokens = combined_tokens[:, :config.BLOCK_SIZE]

                label_i = future_vals[i].unsqueeze(0)
                with amp.autocast('cuda', dtype=torch.float16):  # Updated autocast usage
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

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += main_loss_accum
            total_count += B

            # replay buffer
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
