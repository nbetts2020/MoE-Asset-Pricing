import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils.config import config

class EnergyBasedModel(nn.Module):
    def __init__(self, embedding_dim, temperature=0.7):
        super(EnergyBasedModel, self).__init__()
        # Now the input dimension is just embedding_dim instead of 2*embedding_dim.
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.temperature = temperature

        # Initialize weights with small gains for stability.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Only apply Xavier if weight is at least 2D
                if m.weight.dim() >= 2:
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, full_embedding):
        """
        Args:
            full_embedding (torch.Tensor): shape (batch_size, embedding_dim)
        Returns:
            energy (torch.Tensor): shape (batch_size,)
        """
        energy = self.fc(full_embedding).squeeze(-1)  # (batch_size,)
        energy = energy / self.temperature

        if torch.isnan(energy).any():
            logging.error("EBM forward produced NaN energies.")
        if torch.isinf(energy).any():
            logging.error("EBM forward produced Inf energies.")

        return energy

def scale_energy(energy_values, epsilon=1e-8):
    # energy_values: (batch_size, num_contexts)
    min_energy = energy_values.min(dim=1, keepdim=True)[0]
    max_energy = energy_values.max(dim=1, keepdim=True)[0]
    scaled_energy = (energy_values - min_energy) / (max_energy - min_energy + epsilon)
    return scaled_energy  # (batch_size, num_contexts)

def compute_sampling_probabilities(scaled_energies, temperature):
    probabilities = F.softmax(-scaled_energies / temperature, dim=1)  # softmax over contexts
    return probabilities  # (batch_size, num_contexts)
