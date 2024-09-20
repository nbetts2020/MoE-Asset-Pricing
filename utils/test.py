# utils/test.py

import torch
from sklearn.metrics import mean_squared_error, r2_score

def test_forgetting(model, tasks, optimizer, epochs, device, si=None):
    """
    Test the model for catastrophic forgetting across multiple tasks.

    Args:
        model (nn.Module): The model to test.
        tasks (list of DataLoader): List of tasks (DataLoaders for each task).
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of epochs for training each task.
        device (torch.device): Device (CPU or GPU).
        si (SynapticIntelligence): Synaptic Intelligence object for regularization (optional).

    Returns:
        results (dict): Dictionary of performance metrics for each task across all stages of training.
    """
    model.train()
    results = {}

    for i, task_dataloader in enumerate(tasks):
        # Train on current task
        print(f"Training on Task {i + 1}")
        train_model(model, optimizer, epochs, device, task_dataloader, si=si)

        # After training on each task, evaluate on all tasks seen so far
        for j, previous_task_dataloader in enumerate(tasks[:i + 1]):
            print(f"Evaluating on Task {j + 1} after training on Task {i + 1}")
            predictions, actuals = evaluate_task(model, previous_task_dataloader, device)
            mse = mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            print(f"Task {j + 1} - MSE: {mse:.4f}, R2: {r2:.4f}")

            # Store results
            if f"Task_{j + 1}" not in results:
                results[f"Task_{j + 1}"] = []
            results[f"Task_{j + 1}"].append((mse, r2))

    return results

def evaluate_task(model, dataloader, device):
    """
    Evaluate the model on a given task (dataloader).

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Dataloader for the task to evaluate.
        device (torch.device): Device (CPU or GPU).

    Returns:
        predictions (list): Model predictions.
        actuals (list): Ground truth labels.
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs, _ = model(input_ids=input_ids, targets=labels)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    return predictions, actuals
