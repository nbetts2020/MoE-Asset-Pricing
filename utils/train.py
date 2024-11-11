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
