# train.py

import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

from utils.config import *

def train_model(model, optimizer, epochs, device, dataloader, si=None, accumulation_steps=1, replay_buffer=None):
    model.train()
    scaler = GradScaler()
    logging.info("Starting training loop.")

    # Initialize dictionaries to track sector errors
    sector_errors = {}  # Cumulative errors per sector
    sector_counts = {}  # Counts per sector

    for epoch in range(epochs):
        logging.info(f"Start of Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        predictions = []
        actuals = []
        batch_count = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):

            # Determine new data limit and replay sample size based on replay buffer usage
            if replay_buffer is not None:
                new_data_limit = 8  # Fixed limit on new data samples per batch
                replay_sample_size = BATCH_SIZE - new_data_limit
            else:
                new_data_limit = BATCH_SIZE  # Use entire batch for new data
                replay_sample_size = 0

            # Prepare new data
            new_input_ids = batch['input_ids'][:new_data_limit].to(device)
            new_labels = batch['labels'][:new_data_limit].to(device)
            new_sectors = batch['sector'][:new_data_limit]

            # Sample from replay buffer to fill the rest of the batch
            if replay_sample_size > 0 and replay_buffer is not None:
                # Calculate average sector errors
                if sector_counts:
                    average_sector_errors = {
                        sector: sector_errors[sector] / sector_counts[sector]
                        for sector in sector_errors
                    }
                else:
                    average_sector_errors = {}

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

            with autocast():
                outputs, _ = model(input_ids=input_ids)
                loss = torch.nn.functional.mse_loss(outputs.squeeze(), labels.float())

                if si is not None:
                    si_loss = si.penalty()
                    loss += si_loss

            if si is not None:
                si.total_loss += loss.item() * input_ids.size(0)

            scaler.scale(loss).backward()

            batch_count += 1

            if batch_count % accumulation_steps == 0 or (batch_idx == len(dataloader) - 1):
                scaler.step(optimizer)
                scaler.update()

                if si is not None:
                    si.update_omega()

                optimizer.zero_grad()

            total_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            actuals.extend(labels.cpu().numpy())

            # Compute errors for the batch
            batch_errors = torch.abs(outputs.squeeze() - labels).detach().cpu().numpy()

            # Update sector errors and counts
            for i in range(len(sectors)):
                sector = sectors[i]
                error = batch_errors[i]

                if sector not in sector_errors:
                    sector_errors[sector] = 0.0
                    sector_counts[sector] = 0

                sector_errors[sector] += error
                sector_counts[sector] += 1

            # Prepare samples to add to the replay buffer
            batch_samples = []
            errors = []

            for i in range(labels.size(0)):
                sample = {
                    'input_ids': input_ids[i].detach().cpu(),
                    'labels': labels[i].detach().cpu(),
                    'sector': sectors[i]
                }
                batch_samples.append(sample)
                errors.append(batch_errors[i])

            # Add samples and errors to the replay buffer
            if replay_buffer is not None and replay_sample_size > 0:
                replay_buffer.add_examples(batch_samples, errors)

        if si is not None:
            si.consolidate_omega()
            logging.info("Consolidated omega after epoch.")

        avg_loss = total_loss / len(dataloader)
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    logging.info("Training loop completed.")
