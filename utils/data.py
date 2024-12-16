# utils/train.py

import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import amp
from torch.cuda.amp import GradScaler
import os
from tqdm import tqdm
import logging
import torch.distributed as dist
from functools import partial
import numpy as np

from utils.config import config
from utils.utils import (
    compute_l2_loss,
    evaluate_model,
    load_checkpoint
)
from utils.data import custom_collate_fn

logging.basicConfig(level=logging.INFO)

def train_model(model,
                optimizer,
                epochs,
                device,
                dataloader,
                args,
                si=None,
                ewc=None,
                replay_buffer=None,
                df=None,
                ebm=None,
                ebm_optimizer=None,
                tokenizer=None):
    model.train()
    scaler = GradScaler()
    logging.info("Starting training loop.")

    patience = args.early_stopping_patience
    best_loss = float('inf')
    epochs_no_improve = 0

    use_ddp = getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1
    rank = dist.get_rank() if use_ddp else 0
    use_ebm = getattr(args, 'use_ebm', False)

    # If using EBM, consider gradient clipping
    # We'll apply clipping before each optimizer step if ebm is used.
    def clip_ebm_gradients():
        if use_ebm and ebm is not None:
            torch.nn.utils.clip_grad_norm_(ebm.parameters(), max_norm=1.0)

    checkpoint_dir = args.checkpoint_dir if hasattr(args, 'checkpoint_dir') else './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer,
                                      ebm if use_ebm else None,
                                      ebm_optimizer if use_ebm else None,
                                      checkpoint_path=args.checkpoint_path)

    # Set temperature if args has temperature
    ebm_temperature = getattr(args, 'temperature', 1.0)
    if ebm is not None:
        ebm.temperature = ebm_temperature

    for epoch in range(start_epoch, epochs):
        logging.info(f"Rank {rank}: Start of Epoch {epoch + 1}/{epochs}")

        dataloader.collate_fn = partial(
            custom_collate_fn,
            df=dataloader.dataset.df,
            ebm=ebm,
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_ebm=use_ebm,
            total_epochs=epochs,
            current_epoch=epoch
        )

        total_loss = 0.0
        predictions = []
        actuals = []

        if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Rank {rank} - Epoch {epoch + 1}/{epochs}")):
            optimizer.zero_grad()
            if use_ebm and ebm_optimizer:
                ebm_optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device)
            selected_input_ids = input_ids
            ebm_loss = 0.0

            if use_ebm and ebm is not None and batch['all_contexts'] is not None:
                current_articles = batch['current_articles']
                all_contexts = batch['all_contexts']
                N = batch['N']

                best_contexts_input_ids = []
                for i in range(len(all_contexts)):
                    contexts = all_contexts[i]
                    current_article_str = current_articles[i]

                    article_enc = tokenizer(
                        current_article_str,
                        truncation=True,
                        padding='max_length',
                        max_length=config.BLOCK_SIZE,
                        return_tensors='pt'
                    ).to(device)

                    with torch.no_grad():
                        article_embedding = model.get_embeddings(article_enc['input_ids'].long())
                        article_embedding = article_embedding.squeeze(0)

                    encodings = tokenizer(
                        contexts,
                        truncation=True,
                        padding=True,
                        max_length=config.BLOCK_SIZE,
                        return_tensors='pt'
                    ).to(device)

                    context_input_ids = encodings['input_ids'].to(dtype=torch.long, device=device)

                    with torch.no_grad():
                        context_embeddings = model.get_embeddings(context_input_ids)
                        if torch.isnan(context_embeddings).any() or torch.isinf(context_embeddings).any():
                            logging.error("NaN or Inf in context_embeddings. Stopping training.")
                            break

                    energies = ebm(article_embedding.unsqueeze(0).repeat(len(contexts), 1), context_embeddings)

                    # Check energies for NaN/Inf
                    if torch.isnan(energies).any() or torch.isinf(energies).any():
                        logging.error("NaN or Inf in energies before softmax. Stopping training.")
                        break

                    ebm_loss += energies.mean() / len(all_contexts)

                    # Clamp energies to avoid overflow
                    energies = torch.clamp(energies, -1e6, 1e6)
                    energies_neg = -energies
                    probabilities = torch.softmax(energies_neg, dim=0)

                    # Check probabilities
                    if torch.isnan(probabilities).any() or torch.isinf(probabilities).any() or (probabilities < 0).any():
                        logging.error("NaN, Inf, or negative values in probabilities. Stopping training.")
                        break

                    sampled_idx = torch.multinomial(probabilities, num_samples=1).item()
                    sampled_context = contexts[sampled_idx]

                    sampled_context_encoding = tokenizer(
                        sampled_context,
                        truncation=True,
                        padding='max_length',
                        max_length=config.BLOCK_SIZE,
                        return_tensors='pt'
                    )
                    sampled_context_input_ids_single = sampled_context_encoding['input_ids'].squeeze(0).to(dtype=torch.long)
                    best_contexts_input_ids.append(sampled_context_input_ids_single)

                best_contexts_input_ids = torch.stack(best_contexts_input_ids).to(device, dtype=torch.long)

                combined_input_ids = []
                for i in range(input_ids.size(0)):
                    art_tokens = input_ids[i].long()
                    ctx_tokens = best_contexts_input_ids[i].long()
                    combined = torch.cat([art_tokens, ctx_tokens], dim=0)
                    if combined.size(0) > config.BLOCK_SIZE:
                        combined = combined[:config.BLOCK_SIZE]
                    combined_input_ids.append(combined.unsqueeze(0))

                selected_input_ids = torch.cat(combined_input_ids, dim=0).to(device, dtype=torch.long)

                pad_token_id = tokenizer.pad_token_id
                num_tokens_per_sample = (selected_input_ids != pad_token_id).sum(dim=1)
                logging.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Tokens per sample: {num_tokens_per_sample.tolist()}")

                if torch.isnan(selected_input_ids).any() or torch.isinf(selected_input_ids).any():
                    logging.error(f"NaN/Inf in selected_input_ids at Epoch {epoch+1}, Batch {batch_idx+1}. Stopping training.")
                    break

            with amp.autocast('cuda'):
                outputs, loss = model(
                    input_ids=selected_input_ids,
                    targets=labels.float(),
                    use_entropy_reg=args.use_entropy_reg,
                    lambda_entropy=args.lambda_entropy
                )

            if si:
                loss += si.penalty()
            if ewc:
                for ewc_instance in ewc:
                    loss += args.lambda_ewc * ewc_instance.penalty(model)
            if getattr(args, 'use_l2', False):
                loss += args.lambda_l2 * compute_l2_loss(model)

            total_batch_loss = loss
            if use_ebm and ebm is not None and batch['all_contexts'] is not None:
                total_batch_loss += ebm_loss

            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                logging.error(f"NaN/Inf in total_batch_loss at Epoch {epoch+1}, Batch {batch_idx+1}. Stopping training.")
                break

            scaler.scale(total_batch_loss).backward()

            # Clip EBM gradients if EBM is used
            if use_ebm and ebm is not None:
                torch.nn.utils.clip_grad_norm_(ebm.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            if use_ebm and ebm is not None and ebm_optimizer is not None and batch['all_contexts'] is not None:
                scaler.step(ebm_optimizer)
            scaler.update()

            total_loss += total_batch_loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            actuals.extend(labels.cpu().numpy())

            if replay_buffer:
                replay_samples = [{
                    'input_ids': input_ids[i].detach().cpu(),
                    'labels': labels[i].detach().cpu(),
                    'sector': batch['sector'][i]
                } for i in range(len(labels))]
                replay_buffer.add_examples(replay_samples, [0] * len(labels))

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        if len(predictions) > 0:
            mse = mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
        else:
            mse, r2 = 0.0, 0.0

        if rank == 0:
            logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")

            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pt')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if use_ebm and ebm is not None:
                checkpoint['ebm_state_dict'] = ebm.state_dict()
                if ebm_optimizer is not None:
                    checkpoint['ebm_optimizer_state_dict'] = ebm_optimizer.state_dict()
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"No improvement in loss for {patience} consecutive epochs. Stopping early.")
                    break

        if replay_buffer:
            replay_batch_size = getattr(args, 'replay_batch_size', labels.size(0))
            replay_samples = replay_buffer.sample(replay_batch_size)
            if len(replay_samples) > 0:
                replay_input_ids = torch.stack([s['input_ids'] for s in replay_samples]).to(device, dtype=torch.long)
                replay_labels = torch.stack([s['labels'] for s in replay_samples]).to(device)

                with amp.autocast('cuda'):
                    replay_outputs, replay_loss = model(
                        input_ids=replay_input_ids,
                        targets=replay_labels.float(),
                        use_entropy_reg=args.use_entropy_reg,
                        lambda_entropy=args.lambda_entropy
                    )

                replay_loss = replay_loss * getattr(args, 'replay_buffer_weight', 1.0)
                scaler.scale(replay_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
