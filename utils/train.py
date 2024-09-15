import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.cuda.amp import autocast, GradScaler

def train_model(model, optimizer, epochs, device, dataloader):
    model.train()
    scaler = GradScaler()  # Initialize GradScaler for mixed precision

    for epoch in range(epochs):
        print(f"Start of Epoch {epoch}")
        total_loss = 0
        predictions = []
        actuals = []

        # Get the total number of batches
        total_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1} of {total_batches}")

            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass with autocast for mixed precision
            with autocast():
                outputs, loss = model(input_ids=input_ids, targets=labels)

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()

            # Step the optimizer and update the scaler
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Collect predictions and actuals for metrics
            predictions.extend(outputs.detach().cpu().numpy())
            actuals.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")

    # Optional: Save the final model state if needed
    # torch.save(model.state_dict(), 'model/model_weights_final.pth')
