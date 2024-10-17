# utils/train.py

import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import numpy as np
import torch.distributed as dist

from utils.config import *

def train_model(model, optimizer, epochs, device, dataloader, args, si=None, ewc=None, accumulation_steps=1, replay_buffer=None, test_dataloader=None):
    model.train()
    scaler = GradScaler()
    logging.info("Starting training loop.")

    # Determine if using DDP
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
        predictions = []
        actuals = []
        batch_count = 0

        # Initialize dictionaries to track errors per sector
        sector_errors = {}
        sector_counts = {}

        # Set the epoch for the sampler if using DDP
        if use_ddp:
            dataloader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Rank {rank} - Epoch {epoch + 1}/{epochs}")):
            optimizer.zero_grad()  # Zero gradients for each batch

            # Determine new data limit and replay sample size based on replay buffer usage
            if replay_buffer is not None:
                new_data_limit = BATCH_SIZE // 2  # use half the batch for new data
                replay_sample_size = BATCH_SIZE - new_data_limit
            else:
                new_data_limit = BATCH_SIZE  # use entire batch for new data
                replay_sample_size = 0

            # Prepare new data
            new_input_ids = batch['input_ids'][:new_data_limit].to(device)
            new_labels = batch['labels'][:new_data_limit].to(device)
            new_sectors = batch['sector'][:new_data_limit]

            # Sample from replay buffer to fill the rest of the batch
            if replay_sample_size > 0 and replay_buffer is not None:
                # Compute average errors per sector
                average_sector_errors = {}
                total_error = sum(sector_errors.values())
                total_count = sum(sector_counts.values())
                mean_sector_error = total_error / total_count if total_count > 0 else 1.0

                for sector in sector_errors:
                    average_error = sector_errors[sector] / sector_counts[sector]
                    average_sector_errors[sector] = average_error

                replay_samples = replay_buffer.sample(replay_sample_size, average_sector_errors)
                if replay_samples:
                    replay_input_ids = torch.stack([s['input_ids'] for s in replay_samples]).to(device)
                    replay_labels = torch.stack([s['labels'] for s in replay_samples]).to(device)
                    replay_sectors = [s['sector'] for s in replay_samples]

                    # Combine new data with replay samples
                    input_ids = torch.cat([new_input_ids, replay_input_ids], dim=0)
                    labels = torch.cat([new_labels, replay_labels], dim=0)
                    sectors = list(new_sectors) + replay_sectors
                else:
                    # If replay buffer is empty, use only new data
                    input_ids = new_input_ids
                    labels = new_labels
                    sectors = new_sectors
            else:
                # If replay buffer is not used or no samples to replay, use only new data
                input_ids = new_input_ids
                labels = new_labels
                sectors = new_sectors

            if device.type == 'cuda':
                with torch.cuda.amp.autocast(device_type='cuda'):
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

            loss = model_loss  # start with loss returned by the model

            # Add SI penalty if applicable
            if si is not None:
                si_loss = si.penalty()
                loss += si_loss

            # Add EWC penalty if applicable
            if ewc is not None:
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

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()

                if si is not None:
                    si.update_omega()  # update SI after each batch

                optimizer.zero_grad()  # reset gradients after each optimization step

            total_loss += loss.item()
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
            if replay_buffer is not None and replay_sample_size > 0:
                replay_buffer.add_examples(batch_samples, batch_errors)

        # End of epoch actions
        if si is not None:
            si.consolidate_omega()  # consolidate SI omega at the end of the epoch
            logging.info("Consolidated omega after epoch.")

        # Reduce total_loss across all processes
        if use_ddp:
            total_loss_tensor = torch.tensor(total_loss).to(device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = total_loss_tensor.item() / (len(dataloader) * world_size)
        else:
            avg_loss = total_loss / len(dataloader)

        # Gather predictions and actuals from all processes for metrics
        if use_ddp:
            # Convert lists to tensors
            predictions_tensor = torch.tensor(predictions).to(device)
            actuals_tensor = torch.tensor(actuals).to(device)

            # Determine the maximum length across all processes
            pred_size = torch.tensor([predictions_tensor.size(0)], device=device)
            actual_size = torch.tensor([actuals_tensor.size(0)], device=device)
            sizes = torch.stack([pred_size, actual_size])
            dist.all_reduce(sizes, op=dist.ReduceOp.MAX)
            max_pred_size = sizes[0].item()
            max_actual_size = sizes[1].item()

            # Pad tensors to max size
            padded_predictions = torch.zeros(max_pred_size, device=device)
            padded_predictions[:predictions_tensor.size(0)] = predictions_tensor
            padded_actuals = torch.zeros(max_actual_size, device=device)
            padded_actuals[:actuals_tensor.size(0)] = actuals_tensor

            # Gather all predictions and actuals
            gathered_predictions = [torch.zeros_like(padded_predictions) for _ in range(world_size)]
            gathered_actuals = [torch.zeros_like(padded_actuals) for _ in range(world_size)]
            dist.all_gather(gathered_predictions, padded_predictions)
            dist.all_gather(gathered_actuals, padded_actuals)

            # Concatenate and trim to original sizes
            all_predictions = torch.cat([gp[:pred_size.item()].cpu() for gp in gathered_predictions])
            all_actuals = torch.cat([ga[:actual_size.item()].cpu() for ga in gathered_actuals])

            # Compute metrics
            mse = mean_squared_error(all_actuals.numpy(), all_predictions.numpy())
            r2 = r2_score(all_actuals.numpy(), all_predictions.numpy())
        else:
            # Compute metrics
            mse = mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)

        if rank == 0:
            logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}")

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
