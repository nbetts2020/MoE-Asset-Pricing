# utils/train.py

import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import numpy as np
import torch.distributed as dist

from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.config import config
from utils.ebm import generate_context, select_best_context

def train_model(model, optimizer, epochs, device, dataloader, args, si=None, ewc=None, replay_buffer=None, test_dataloader=None):
    model.train()
    scaler = GradScaler()
    logging.info("Starting training loop.")

    # Early Stopping parameters
    patience = args.early_stopping_patience if hasattr(args, 'early_stopping_patience') else 5
    best_loss = float('inf')
    epochs_no_improve = 0

    # Determine if we're using DDP
    use_ddp = args.use_ddp and torch.cuda.device_count() > 1

    # Get the rank (process ID) and world size (total number of processes)
    if use_ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    for epoch in range(epochs):
        logging.info(f"Rank {rank}: Start of Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        total_samples = 0
        predictions = []
        actuals = []

        current_epoch = epoch
        num_samples = epochs - current_epoch if args.use_ebm else 1

        # Initialize dictionaries to track errors per sector
        sector_errors = {}
        sector_counts = {}

        # Set the epoch for the sampler if using DDP
        if use_ddp:
            dataloader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Rank {rank} - Epoch {epoch + 1}/{epochs}")):
            optimizer.zero_grad()  # Zero gradients for each batch

            if args.use_ebm:
                expanded_input_ids = []
                expanded_labels = []
                expanded_sectors = []

                # Generate multiple contexts per sample
                for i in range(len(batch['input_ids'])):
                    current_input_ids = batch['input_ids'][i]
                    current_labels = batch['labels'][i]
                    current_sector = batch['sector'][i]

                    contexts = generate_context(current_input_ids, num_samples)

                    expanded_input_ids.extend(contexts)
                    expanded_labels.extend([current_labels.clone()] * num_samples)
                    expanded_sectors.extend([current_sector] * num_samples)

                # Create tensors
                input_ids = torch.stack(expanded_input_ids).to(device)      # Shape: (batch_size * num_samples, seq_length)
                labels = torch.stack(expanded_labels).to(device)          # Shape: (batch_size * num_samples, ...)
                sectors = expanded_sectors                               # List of sectors

                # Select the best context per sample
                input_ids, labels, selected_sectors = select_best_context(input_ids, labels, model, device, args)
                sectors = selected_sectors  # Update sectors to match selected contexts

            else:
                # Proceed as usual
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                sectors = batch['sector']

            # Prepare data
            if device.type == 'cuda':
                with torch.cuda.amp.autocast('cuda'):
                    outputs, model_loss = model(
                        input_ids=input_ids,
                        targets=labels.float(),
                        use_entropy_reg=args.use_entropy_reg,
                        lambda_entropy=args.lambda_entropy
                    )
            else:
                outputs, model_loss = model(
                    input_ids=input_ids,
                    targets=labels.float(),
                    use_entropy_reg=args.use_entropy_reg,
                    lambda_entropy=args.lambda_entropy
                )

            loss = model_loss  # start with the loss returned by the model

            # Add SI penalty if applicable
            if si is not None:
                si_loss = si.penalty()
                loss += si_loss

            # Add EWC penalty if applicable
            if ewc is not None and len(ewc) > 0:
                ewc_loss = 0.0
                for ewc_instance in ewc:
                    ewc_loss += ewc_instance.penalty(model)
                loss += args.lambda_ewc * ewc_loss

            # Add L2 regularization if applicable
            if args.use_l2:
                l2_loss = compute_l2_loss(model)
                loss += args.lambda_l2 * l2_loss

            if si is not None:
                si.total_loss += loss.item()

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if si is not None:
                si.update_omega()  # update SI after each optimization step

            total_loss += loss.item()
            total_samples += input_ids.size(0)
            predictions.extend(outputs.detach().cpu().numpy())
            actuals.extend(labels.cpu().numpy())

            # Compute prediction errors and track errors per sector
            errors = torch.abs(outputs.detach().cpu() - labels.cpu()).numpy()
            for i in range(len(sectors)):
                sector = sectors[i]
                error = errors[i]
                sector_errors[sector] = sector_errors.get(sector, 0.0) + error
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            # Prepare samples and errors to add to the replay buffer
            if replay_buffer is not None:
                batch_samples = []
                batch_errors = []
                for i in range(labels.size(0)):
                    sample = {
                        'input_ids': input_ids[i].detach().cpu(),
                        'labels': labels[i].detach().cpu(),
                        'sector': sectors[i]
                    }
                    error = errors[i]
                    batch_samples.append(sample)
                    batch_errors.append(error)
                # Add samples to the replay buffer
                replay_buffer.add_examples(batch_samples, batch_errors)

        # End of epoch actions
        if si is not None:
            si.consolidate_omega()  # consolidate SI omega at the end of the epoch
            logging.info("Consolidated omega after epoch.")

        # Synchronize sector_errors and sector_counts across processes
        if replay_buffer is not None and use_ddp:
            # Gather sector_errors and sector_counts from all processes
            gathered_sector_errors = [None for _ in range(world_size)]
            gathered_sector_counts = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_sector_errors, sector_errors)
            dist.all_gather_object(gathered_sector_counts, sector_counts)

            # Merge sector_errors and sector_counts
            total_sector_errors = {}
            total_sector_counts = {}
            for se, sc in zip(gathered_sector_errors, gathered_sector_counts):
                for sector in se:
                    total_sector_errors[sector] = total_sector_errors.get(sector, 0.0) + se[sector]
                for sector in sc:
                    total_sector_counts[sector] = total_sector_counts.get(sector, 0) + sc[sector]
            sector_errors = total_sector_errors
            sector_counts = total_sector_counts

        # Compute average sector errors
        average_sector_errors = {sector: sector_errors[sector] / sector_counts[sector] for sector in sector_errors}

        # Synchronize the replay buffer across processes
        if replay_buffer is not None and use_ddp:
            replay_buffer.sync_buffer()

        # Reduce total_loss across all processes
        if use_ddp:
            total_loss_tensor = torch.tensor(total_loss).to(device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = total_loss_tensor.item() / (total_samples * world_size)
        else:
            avg_loss = total_loss / total_samples

        # Gather predictions and actuals from all processes for metrics
        if use_ddp:
            # Convert lists to tensors
            predictions_tensor = torch.tensor(predictions).to(device)
            actuals_tensor = torch.tensor(actuals).to(device)

            # Gather all predictions and actuals
            gathered_predictions = [torch.zeros_like(predictions_tensor) for _ in range(world_size)]
            gathered_actuals = [torch.zeros_like(actuals_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_predictions, predictions_tensor)
            dist.all_gather(gathered_actuals, actuals_tensor)

            # Concatenate
            all_predictions = torch.cat(gathered_predictions)
            all_actuals = torch.cat(gathered_actuals)

            # Compute metrics
            mse = mean_squared_error(all_actuals.cpu().numpy(), all_predictions.cpu().numpy())
            r2 = r2_score(all_actuals.cpu().numpy(), all_predictions.cpu().numpy())
        else:
            # Compute metrics
            mse = mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)

        if rank == 0:
            logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"No improvement in loss for {patience} consecutive epochs. Stopping early.")
                    break

        # Sample from the replay buffer and train on replayed samples
        if replay_buffer is not None:
            replay_batch_size = args.replay_batch_size if hasattr(args, 'replay_batch_size') else labels.size(0)
            replay_samples = replay_buffer.sample(replay_batch_size, average_sector_errors)
            if len(replay_samples) > 0:
                # Prepare replay batch
                replay_input_ids = torch.stack([s['input_ids'] for s in replay_samples]).to(device)
                replay_labels = torch.stack([s['labels'] for s in replay_samples]).to(device)

                # Forward pass on replayed samples
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast(device_type='cuda'):
                        replay_outputs, replay_loss = model(
                            input_ids=replay_input_ids,
                            targets=replay_labels.float(),
                            use_entropy_reg=args.use_entropy_reg,
                            lambda_entropy=args.lambda_entropy
                        )
                else:
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
