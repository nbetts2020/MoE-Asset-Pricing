# utils/train.py

import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import amp
from torch.cuda.amp import GradScaler
import os
from tqdm import tqdm
import logging
import torch.distributed as dist

from utils.config import config
from utils.utils import (
    compute_l2_loss,
    evaluate_model,
    load_checkpoint
)

logging.basicConfig(level=logging.INFO)

def train_model(model, optimizer, epochs, device, dataloader, args, si=None, ewc=None,
                replay_buffer=None, df=None, ebm=None, ebm_optimizer=None):
    """
    Train the transformer model with optional EBM integration.
    EBM and EBM optimizer are now passed as arguments from main.py.
    """
    model.train()
    scaler = GradScaler()
    logging.info("Starting training loop.")

    # Early Stopping parameters
    patience = args.early_stopping_patience
    best_loss = float('inf')
    epochs_no_improve = 0

    # Determine if using DDP
    use_ddp = getattr(args, 'use_ddp', False) and torch.cuda.device_count() > 1
    rank = dist.get_rank() if use_ddp else 0

    use_ebm = getattr(args, 'use_ebm', False)

    # Create checkpoint directory if needed
    checkpoint_dir = args.checkpoint_dir if hasattr(args, 'checkpoint_dir') else './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load from checkpoint if specified
    start_epoch = 0
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, ebm if use_ebm else None,
                                      ebm_optimizer if use_ebm else None,
                                      checkpoint_path=args.checkpoint_path)

    for epoch in range(start_epoch, epochs):
        logging.info(f"Rank {rank}: Start of Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        predictions = []
        actuals = []

        # If using DDP with DistributedSampler, set epoch
        if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        for batch in tqdm(dataloader, desc=f"Rank {rank} - Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            if use_ebm and ebm_optimizer is not None:
                ebm_optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if use_ebm and ebm is not None and 'context_input_ids' in batch:
                context_input_ids = batch['context_input_ids'].to(device)
                # Concatenate input_ids and context_input_ids
                selected_input_ids = torch.cat([input_ids, context_input_ids], dim=1)
                # If sequence exceeds BLOCK_SIZE, truncate
                max_seq_len = config.BLOCK_SIZE
                if selected_input_ids.size(1) > max_seq_len:
                    selected_input_ids = selected_input_ids[:, :max_seq_len]

                # Get embeddings for EBM
                with torch.no_grad():
                    article_embeddings = model.get_embeddings(input_ids)          # (batch, embed_dim)
                    context_embeddings = model.get_embeddings(context_input_ids)  # (batch, embed_dim)

                # Compute energies for EBM
                energies = ebm(article_embeddings, context_embeddings)
                ebm_loss = energies.mean()
            else:
                selected_input_ids = input_ids
                ebm_loss = 0.0

            with amp.autocast('cuda'):
                outputs, loss = model(
                    input_ids=selected_input_ids,
                    targets=labels.float(),
                    use_entropy_reg=args.use_entropy_reg,
                    lambda_entropy=args.lambda_entropy
                )

            # Add regularizations
            if si:
                loss += si.penalty()
            if ewc:
                for ewc_instance in ewc:
                    loss += args.lambda_ewc * ewc_instance.penalty(model)
            if getattr(args, 'use_l2', False):
                loss += args.lambda_l2 * compute_l2_loss(model)

            # Combine losses
            if use_ebm and ebm is not None and 'context_input_ids' in batch:
                total_batch_loss = loss + ebm_loss
            else:
                total_batch_loss = loss

            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            if use_ebm and ebm is not None and 'context_input_ids' in batch and ebm_optimizer is not None:
                scaler.step(ebm_optimizer)
            scaler.update()

            total_loss += total_batch_loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            actuals.extend(labels.cpu().numpy())

            # Add to replay buffer if necessary
            if replay_buffer:
                replay_samples = [{
                    'input_ids': input_ids[i].detach().cpu(),
                    'labels': labels[i].detach().cpu(),
                    'sector': batch['sector'][i]
                } for i in range(len(labels))]
                replay_buffer.add_examples(replay_samples, [0] * len(labels))

        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        if rank == 0:
            logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")

            # Save checkpoint
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

            # Early Stopping Logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"No improvement in loss for {patience} consecutive epochs. Stopping early.")
                    break

        # Sample from the replay buffer and train on replayed samples
        if replay_buffer:
            replay_batch_size = getattr(args, 'replay_batch_size', labels.size(0))
            replay_samples = replay_buffer.sample(replay_batch_size)
            if len(replay_samples) > 0:
                # Prepare replay batch
                replay_input_ids = torch.stack([s['input_ids'] for s in replay_samples]).to(device)
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
