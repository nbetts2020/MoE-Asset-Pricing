import torch
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(model, optimizer, epochs, device, dataloader):
    model.train()
    for epoch in range(epochs):
        print(f"Start of Epoch {epoch}")
        total_loss = 0
        predictions = []
        actuals = []
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs, loss = model(input_ids=input_ids, targets=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Collect predictions and actuals for metrics
            predictions.extend(outputs.detach().cpu().numpy())
            actuals.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")

    # Save the final model (optional)
    # torch.save(model.state_dict(), 'model/model_weights_final.pth')
