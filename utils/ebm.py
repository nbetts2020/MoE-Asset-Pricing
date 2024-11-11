# utils/ebm.py

import torch
import logging
from utils.config import config

def generate_context(current_context, num_samples): # placeholder for now
    logging.debug(f"Generating {num_samples} contexts for the current sample.")
    return [current_context.clone() for _ in range(num_samples)]

def select_best_context(contexts, labels, model, device, args):
    """
    Selects the context with the lowest MSE for each sample.
    
    Args:
        contexts (torch.Tensor): Tensor of shape (batch_size * num_samples, seq_length).
        labels (torch.Tensor): Tensor of shape (batch_size * num_samples, ...).
        model (nn.Module): The model being trained.
        device (torch.device): The device to perform computations on.
        args (Namespace): Parsed command-line arguments.
    
    Returns:
        torch.Tensor: Selected input_ids tensor of shape (batch_size, seq_length).
        torch.Tensor: Corresponding labels tensor of shape (batch_size, ...).
    """
    batch_size = config.BATCH_SIZE
    num_samples = len(contexts) // batch_size

    # Move contexts and labels to device
    contexts = contexts.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs, _ = model(input_ids=contexts, targets=labels.float(), use_entropy_reg=args.use_entropy_reg, lambda_entropy=args.lambda_entropy)
        else:
            outputs, _ = model(input_ids=contexts, targets=labels.float(), use_entropy_reg=args.use_entropy_reg, lambda_entropy=args.lambda_entropy)

    # Compute MSE for each context
    mse_loss = torch.nn.MSELoss(reduction='none')
    losses = mse_loss(outputs, labels.float())  # Shape: (batch_size * num_samples, ...)

    losses = losses.mean(dim=1)  # Shape: (batch_size * num_samples,)

    # Reshape losses to (batch_size, num_samples)
    losses = losses.view(batch_size, num_samples)

    # Find the index of the minimum loss for each sample
    _, min_indices = torch.min(losses, dim=1)  # Shape: (batch_size,)

    # Select the best contexts and corresponding labels
    selected_contexts = []
    selected_labels = []
    for i in range(batch_size):
        idx = min_indices[i].item()
        selected_contexts.append(contexts[i * num_samples + idx].unsqueeze(0))
        selected_labels.append(labels[i * num_samples + idx].unsqueeze(0))

    selected_contexts = torch.cat(selected_contexts, dim=0)  # Shape: (batch_size, seq_length)
    selected_labels = torch.cat(selected_labels, dim=0)      # Shape: (batch_size, ...)

    return selected_contexts, selected_labels
