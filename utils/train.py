import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

def train_model(model, optimizer, epochs, device, dataloader, si=None, accumulation_steps=1):
    model.train()
    scaler = GradScaler()
    logging.info("Starting training loop.")

    for epoch in range(epochs):
        logging.info(f"Start of Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        predictions = []
        actuals = []
        batch_count = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                # Remove attention_mask from the model call
                outputs, _ = model(input_ids=input_ids)
                loss = torch.nn.functional.mse_loss(outputs.squeeze(), labels.float())

                if si is not None:
                    si_loss = si.penalty()
                    loss += si_loss

            scaler.scale(loss).backward()

            if si is not None:
                si.update_omega(loss, batch_size=input_ids.size(0))

            batch_count += 1

            if batch_count % accumulation_steps == 0 or (batch_idx == len(dataloader) - 1):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            actuals.extend(labels.cpu().numpy())

        if si is not None:
            si.consolidate_omega()
            logging.info("Consolidated omega after epoch.")

        avg_loss = total_loss / len(dataloader)
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    logging.info("Training loop completed.")
