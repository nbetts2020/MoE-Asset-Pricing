# utils/train.py

import torch
from torch.cuda.amp import autocast, GradScaler
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
from utils.ebm import EnergyBasedModel, scale_energy, compute_sampling_probabilities
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
        ebm_loss_fn = torch.nn.MSELoss()

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

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            sectors = batch['sector']

            if use_ebm:
                # EBM and Monte Carlo Sampling logic
                context_input_ids = batch.get('context_input_ids', None)
                if context_input_ids is not None:
                    context_input_ids = context_input_ids.to(device)  # shape: (batch_size, num_contexts, seq_len)
                    num_contexts = context_input_ids.shape[1]
                    batch_size = input_ids.shape[0]

                    # Flatten context_input_ids for embedding
                    flat_context_ids = context_input_ids.view(-1, context_input_ids.size(-1))  # (batch_size * num_contexts, seq_len)

                    # Generate embeddings
                    with torch.no_grad():
                        article_embeddings = model.get_embeddings(input_ids)  # (batch_size, embedding_dim)
                        context_embeddings = model.get_embeddings(flat_context_ids)  # (batch_size * num_contexts, embedding_dim)
                        context_embeddings = context_embeddings.view(batch_size, num_contexts, -1)  # (batch_size, num_contexts, embedding_dim)

                    # Compute energies
                    energies = ebm(
                        article_embeddings.unsqueeze(1).expand(-1, num_contexts, -1),  # (batch_size, num_contexts, embedding_dim)
                        context_embeddings  # (batch_size, num_contexts, embedding_dim)
                    )  # (batch_size, num_contexts)

                    # Scale energies
                    scaled_energies = scale_energy(energies)  # (batch_size, num_contexts)

                    # Compute sampling probabilities
                    probabilities = compute_sampling_probabilities(scaled_energies, temperature=args.temperature)  # (batch_size, num_contexts)

                    # Sample contexts
                    sampled_indices = torch.multinomial(probabilities, num_samples=1).squeeze(-1)  # (batch_size,)
                    selected_contexts = context_input_ids[torch.arange(batch_size), sampled_indices, :]  # (batch_size, seq_len)

                    # Concatenate selected contexts with input_ids
                    selected_input_ids = torch.cat([input_ids, selected_contexts], dim=1)  # (batch_size, total_seq_len)
                else:
                    selected_input_ids = input_ids  # Fallback if no context provided
            else:
                selected_input_ids = input_ids  # if ebm arg not set, use input_ids as is

            with autocast():
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

            # Backward and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if si:
                si.update_omega()

            if use_ebm:
                # Define EBM loss as predicting the scaled MSE
                with torch.no_grad():
                    # Compute MSE for selected contexts
                    selected_outputs, _ = model(
                        input_ids=selected_input_ids,
                        targets=labels.float(),
                        use_entropy_reg=args.use_entropy_reg,
                        lambda_entropy=args.lambda_entropy
                    )
                    mse_loss = torch.nn.functional.mse_loss(selected_outputs, labels.float(), reduction='none')  # (batch_size,)

                # Scale the MSE loss
                scaled_mse = scale_energy(mse_loss.unsqueeze(1))  # (batch_size, 1)

                # EBM predictions for the selected contexts
                selected_context_embeddings = context_embeddings[torch.arange(batch_size), sampled_indices]  # (batch_size, embedding_dim)
                ebm_predictions = ebm(article_embeddings, selected_context_embeddings)  # (batch_size, 1)

                # Compute EBM loss
                ebm_loss = torch.nn.functional.mse_loss(ebm_predictions.squeeze(1), scaled_mse.squeeze(1))  # (batch_size,)

                # Backward and optimize EBM
                scaler.scale(ebm_loss).backward()
                scaler.step(ebm_optimizer)
                scaler.update()

            total_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            actuals.extend(labels.cpu().numpy())

            # Add to replay buffer if necessary
            if replay_buffer:
                replay_samples = [{
                    'input_ids': input_ids[i].detach().cpu(),
                    'labels': labels[i].detach().cpu(),
                    'sector': sectors[i]
                } for i in range(len(labels))]
                replay_buffer.add_examples(replay_samples, [0]*len(labels))

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
                with autocast():
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
