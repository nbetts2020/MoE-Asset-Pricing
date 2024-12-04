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
    load_checkpoint,
    save_model_and_states
)
from utils.ebm import EnergyBasedModel
from utils.data import ArticlePriceDataset

def train_model(model, optimizer, epochs, device, dataloader, args, si=None, ewc=None,
                replay_buffer=None, df=None):
    """
    Train the transformer model with optional EBM integration.

    Args:
        model (nn.Module): The transformer model.
        optimizer (torch.optim.Optimizer): Optimizer for the transformer model.
        epochs (int): Number of training epochs.
        device (torch.device): Device to perform computations on.
        dataloader (DataLoader): DataLoader for the training set.
        args (Namespace): Parsed command-line arguments.
        si (optional): Synaptic Intelligence (SI) regularizer.
        ewc (optional): Elastic Weight Consolidation (EWC) regularizer.
        replay_buffer (optional): Replay buffer for experience replay.
        df (optional): DataFrame containing the dataset.
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

    dataset = dataloader.dataset

    # Initialize EBM if specified
    use_ebm = getattr(args, 'use_ebm', False)
    if use_ebm:
        ebm = EnergyBasedModel(embedding_dim=config.N_EMBED).to(device)
        ebm_optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.ebm_learning_rate)

    # Directory to save checkpoints
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

        # Prepare epoch data
        dataset.prepare_epoch(current_epoch=epoch)

        if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        for batch in tqdm(dataloader, desc=f"Rank {rank} - Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            if use_ebm:
                ebm_optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)                 # Shape: (batch_size, seq_len)
            labels = batch['labels'].to(device)                       # Shape: (batch_size,)
            sectors = batch['sector']

            if use_ebm:
                context_input_ids = batch.get('context_input_ids', None)
                if context_input_ids is not None:
                    context_input_ids = context_input_ids.to(device)  # Shape: (batch_size, seq_len_context)

                    # Concatenate input_ids and context_input_ids
                    selected_input_ids = torch.cat([input_ids, context_input_ids], dim=1)  # Shape: (batch_size, total_seq_len)

                    # Truncate to max_seq_len if necessary
                    max_seq_len = config.BLOCK_SIZE
                    if selected_input_ids.size(1) > max_seq_len:
                        selected_input_ids = selected_input_ids[:, :max_seq_len]

                    # Get embeddings
                    with torch.no_grad():
                        article_embeddings = model.get_embeddings(input_ids)          # Shape: (batch_size, embedding_dim)
                        context_embeddings = model.get_embeddings(context_input_ids)  # Shape: (batch_size, embedding_dim)

                    # Compute energies using EBM
                    energies = ebm(article_embeddings, context_embeddings)  # Shape: (batch_size,)

                    # Compute EBM loss (e.g., minimize energies)
                    ebm_loss = energies.mean()
                else:
                    selected_input_ids = input_ids  # No context provided
                    ebm_loss = 0.0
            else:
                selected_input_ids = input_ids  # Standard training without EBM
                ebm_loss = 0.0

            # Forward pass through the model
            with amp.autocast('cuda'):
                outputs, loss = model(
                    input_ids=selected_input_ids,
                    targets=labels.float(),
                    use_entropy_reg=args.use_entropy_reg,
                    lambda_entropy=args.lambda_entropy
                )

            # Regularization
            if si:
                loss += si.penalty()
            if ewc:
                for ewc_instance in ewc:
                    loss += args.lambda_ewc * ewc_instance.penalty(model)
            if getattr(args, 'use_l2', False):
                loss += args.lambda_l2 * compute_l2_loss(model)

            # Combine losses
            if use_ebm and context_input_ids is not None:
                total_batch_loss = loss + 1 * ebm_loss
            else:
                total_batch_loss = loss

            # Backward and optimization
            scaler.scale(total_batch_loss).backward()

            # Step optimizers
            scaler.step(optimizer)
            if use_ebm and context_input_ids is not None:
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
                    'sector': sectors[i]
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
            if use_ebm:
                checkpoint['ebm_state_dict'] = ebm.state_dict()
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

                # Forward pass on replayed samples
                with amp.autocast('cuda'):
                    replay_outputs, replay_loss = model(
                        input_ids=replay_input_ids,
                        targets=replay_labels.float(),
                        use_entropy_reg=args.use_entropy_reg,
                        lambda_entropy=args.lambda_entropy
                    )

                # Backward pass and optimization on replayed samples
                replay_loss = replay_loss * getattr(args, 'replay_buffer_weight', 1.0)
                scaler.scale(replay_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
