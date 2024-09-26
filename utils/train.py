import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

def train_model(model, optimizer, epochs, device, dataloader, si=None, accumulation_steps=1):
    """
    Trains the model with optional Synaptic Intelligence (SI) for mitigating catastrophic forgetting.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        epochs (int): Number of training epochs.
        device (torch.device): The device to train on.
        dataloader (DataLoader): DataLoader for training data.
        si (SynapticIntelligence, optional): SI instance for regularization. Defaults to None.
        accumulation_steps (int): Number of steps to accumulate gradients before updating.
    """
    model.train()
    scaler = GradScaler()  # Initialize GradScaler for mixed precision
    logging.info("Starting training loop.")

    for epoch in range(epochs):
        logging.info(f"Start of Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        predictions = []
        actuals = []
        batch_count = 0  # To track number of batches processed

        optimizer.zero_grad()  # Ensure optimizer starts with zero gradients

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.mse_loss(outputs.squeeze(), labels.float())

                # If using Synaptic Intelligence, add SI penalty to the loss
                if si is not None:
                    si_loss = si.penalty()
                    loss += si_loss

            # Accumulate gradients over several smaller batches
            scaler.scale(loss).backward()

            # Update SI with current loss and batch size
            if si is not None:
                si.update_omega(loss, batch_size=input_ids.size(0))

            batch_count += 1

            # Perform the optimizer step every `accumulation_steps` batches
            if batch_count % accumulation_steps == 0 or (batch_idx == len(dataloader) - 1):
                # Step the optimizer and update the scaler
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()  # Reset gradients after each accumulation

            total_loss += loss.item()

            # Collect predictions and actuals for metrics
            predictions.extend(outputs.detach().cpu().numpy())
            actuals.extend(labels.cpu().numpy())

        # Consolidate omega after each epoch (or after a task)
        if si is not None:
            si.consolidate_omega()
            logging.info("Consolidated omega after epoch.")

        avg_loss = total_loss / len(dataloader)
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    logging.info("Training loop completed.")
