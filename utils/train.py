# utils/train.py

import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import numpy as np
import torch.distributed as dist

from utils.config import config

def compute_l2_loss(model):
    """
    Compute the L2 penalty (squared L2 norm) of the model parameters.

    Args:
        model (nn.Module): The model.

    Returns:
        l2_loss (torch.Tensor): L2 penalty term.
    """
    l2_loss = torch.tensor(0., device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, 2) ** 2
    return l2_loss

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    actuals = []
    sector_metrics = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            sectors = batch['sector']
            with autocast():
                outputs, _ = model(input_ids=input_ids)
            outputs = outputs.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            predictions.extend(outputs)
            actuals.extend(labels)
            # Compute per-sector metrics
            for i, sector in enumerate(sectors):
                if sector not in sector_metrics:
                    sector_metrics[sector] = {'predictions': [], 'actuals': []}
                sector_metrics[sector]['predictions'].append(outputs[i])
                sector_metrics[sector]['actuals'].append(labels[i])

    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    # Calculate per-sector metrics
    for sector in sector_metrics:
        sector_predictions = sector_metrics[sector]['predictions']
        sector_actuals = sector_metrics[sector]['actuals']
        sector_mse = mean_squared_error(sector_actuals, sector_predictions)
        sector_r2 = r2_score(sector_actuals, sector_predictions)
        sector_metrics[sector] = {'mse': sector_mse, 'r2': sector_r2}

    model.train()
    return mse, r2, sector_metrics

def train_model(model, optimizer, epochs, device, dataloader, args, si=None, ewc=None,
                replay_buffer=None, df=None):
    model.train()
    scaler = GradScaler()
    logging.info("Starting training loop.")

    # Early Stopping parameters
    patience = args.early_stopping_patience
    best_loss = float('inf')
    epochs_no_improve = 0

    # Determine if using DDP
    use_ddp = args.use_ddp and torch.cuda.device_count() > 1
    rank = dist.get_rank() if use_ddp else 0

    dataset = dataloader.dataset

    # Initialize EBM given arg
    use_ebm = getattr(args, 'use_ebm', False)
    if use_ebm:
        ebm = EnergyBasedModel(embedding_dim=config.N_EMBED).to(device)
        ebm_optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.ebm_learning_rate)

    for epoch in range(epochs):
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

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            sectors = batch['sector']

            if use_ebm:
                # EBM and Monte Carlo Sampling logic
                context_input_ids = batch['context_input_ids'].to(device)  # shape: (batch_size, num_contexts, seq_len)
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
                scaled_energies = scale_energy(energies)

                # Compute sampling probabilities
                probabilities = compute_sampling_probabilities(scaled_energies, temperature=args.temperature)  # (batch_size, num_contexts)

                # Sample contexts
                sampled_indices = torch.multinomial(probabilities, num_samples=1).squeeze(-1)  # (batch_size,)
                selected_contexts = context_input_ids[torch.arange(batch_size), sampled_indices, :]  # (batch_size, seq_len)

                # Concatenate selected contexts with input_ids
                selected_input_ids = torch.cat([input_ids, selected_contexts], dim=1)  # (batch_size, total_seq_len)
            else:
                selected_input_ids = input_ids # if ebm arg not set, use input_ids as is

            with autocast():
                outputs, loss = model(
                    input_ids=input_ids,
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
            if args.use_l2:
                loss += args.lambda_l2 * compute_l2_loss(model)

            # Backward and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if si:
                si.update_omega()

            if use_ebm:
                # Define EBM loss as predicting the MSE
                with torch.no_grad():
                    # Compute MSE for selected contexts
                    selected_outputs, _ = model(
                        input_ids=selected_input_ids,
                        targets=labels.float(),
                        use_entropy_reg=args.use_entropy_reg,
                        lambda_entropy=args.lambda_entropy
                    )
                    mse_loss = F.mse_loss(selected_outputs, labels.float(), reduction='none')  # (batch_size,)

                # EBM should predict scaled MSE
                ebm_predictions = ebm(article_embeddings, context_embeddings[torch.arange(batch_size), sampled_indices])  # (batch_size,)
                ebm_scaled_mse = scale_energy(mse_loss.unsqueeze(1))  # (batch_size, 1)
                ebm_loss = F.mse_loss(ebm_predictions, ebm_scaled_mse.squeeze(1))  # (batch_size,)

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
            replay_batch_size = args.replay_batch_size if hasattr(args, 'replay_batch_size') else labels.size(0)
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
                replay_loss = replay_loss * args.replay_buffer_weight if hasattr(args, 'replay_buffer_weight') else replay_loss
                scaler.scale(replay_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    logging.info("Training loop completed.")
