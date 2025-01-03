# utils/ebm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils.config import config

class EnergyBasedModel(nn.Module):
    def __init__(self, embedding_dim, temperature=0.7):
        super(EnergyBasedModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.temperature = temperature

        # Initialize weights to small values for stability
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, article_embedding, context_embedding):
        # Both embeddings have shape: (batch_size, embedding_dim)
        x = torch.cat((article_embedding, context_embedding), dim=-1)  # (batch_size, 2 * embedding_dim)
        energy = self.fc(x).squeeze(-1)  # (batch_size,)

        # Divide by temperature to control scale
        energy = energy / self.temperature

        # Debug checks
        if torch.isnan(energy).any():
            logging.error("EBM forward produced NaN energies.")
        if torch.isinf(energy).any():
            logging.error("EBM forward produced Inf energies.")

        return energy  # (batch_size,)

def scale_energy(energy_values, epsilon=1e-8):
    # energy_values: (batch_size, num_contexts)
    min_energy = energy_values.min(dim=1, keepdim=True)[0]
    max_energy = energy_values.max(dim=1, keepdim=True)[0]
    scaled_energy = (energy_values - min_energy) / (max_energy - min_energy + epsilon)
    return scaled_energy  # (batch_size, num_contexts)

def compute_sampling_probabilities(scaled_energies, temperature):
    probabilities = F.softmax(-scaled_energies / temperature, dim=1)  # softmax over contexts
    return probabilities  # (batch_size, num_contexts)

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

    mse_loss = torch.nn.MSELoss(reduction='none')
    losses = mse_loss(outputs, labels.float())  # (batch_size * num_samples, ...)
    losses = losses.mean(dim=1)  # (batch_size * num_samples,)

    losses = losses.view(batch_size, num_samples)
    _, min_indices = torch.min(losses, dim=1)  # (batch_size,)

    selected_contexts = []
    selected_labels = []
    for i in range(batch_size):
        idx = min_indices[i].item()
        selected_contexts.append(contexts[i * num_samples + idx].unsqueeze(0))
        selected_labels.append(labels[i * num_samples + idx].unsqueeze(0))

    selected_contexts = torch.cat(selected_contexts, dim=0)  # (batch_size, seq_length)
    selected_labels = torch.cat(selected_labels, dim=0)      # (batch_size, ...)

    return selected_contexts, selected_labels
