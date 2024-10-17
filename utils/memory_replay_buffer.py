# utils/memory_replay_buffer.py

import random
import torch

class MemoryReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []  # list to store samples
        self.errors = []  # list to store errors (used for sampling probabilities)
        self.total_error = 0.0  # sum of all errors, used for normalization

    def add_examples(self, examples, errors):
        """
        Add new examples to the buffer along with their prediction errors.

        Args:
            examples (list): List of data samples (dicts with 'input_ids', 'labels', 'sector', etc.)
            errors (list): List of errors corresponding to each example
        """
        for example, error in zip(examples, errors):
            # If buffer is full, remove least important sample
            if len(self.buffer) >= self.capacity:
                # Find index of sample with the lowest error
                min_error = min(self.errors)
                min_index = self.errors.index(min_error)
                # Remove the least important sample
                removed_error = self.errors.pop(min_index)
                self.buffer.pop(min_index)
                self.total_error -= removed_error

            # Add the new example
            self.buffer.append(example)
            self.errors.append(error)
            self.total_error += error

    def sample(self, sample_size, average_sector_errors):
        """
        Sample data from the buffer based on their errors and sector performance.

        Args:
            sample_size (int): Number of samples to draw
            average_sector_errors (dict): Average error per sector

        Returns:
            List of sampled data
        """
        if len(self.buffer) == 0:
            return []

        # Compute mean sector error
        if average_sector_errors:
            mean_sector_error = sum(average_sector_errors.values()) / len(average_sector_errors)
        else:
            mean_sector_error = 1.0  # fallback if dictionary is empty

        # Compute sampling probabilities
        sampling_probs = []
        for i in range(len(self.buffer)):
            error = self.errors[i]
            sector = self.buffer[i]['sector']
            sector_error = average_sector_errors.get(sector, mean_sector_error)

            # Combine sample error and sector error
            combined_error = error * sector_error
            sampling_probs.append(combined_error)

        # Normalize sampling probabilities
        total = sum(sampling_probs)
        if total == 0:
            sampling_probs = [1.0 / len(self.buffer)] * len(self.buffer)
        else:
            sampling_probs = [p / total for p in sampling_probs]

        # Sample indices based on sampling probabilities
        sampled_indices = random.choices(
            population=range(len(self.buffer)),
            weights=sampling_probs,
            k=sample_size
        )

        sampled_examples = [self.buffer[i] for i in sampled_indices]

        return sampled_examples

    def save(self, filepath):
        """Save the replay buffer to a file."""
        state = {
            'capacity': self.capacity,
            'buffer': self.buffer,
            'errors': self.errors,
            'total_error': self.total_error
        }
        torch.save(state, filepath)

    def load(self, filepath):
        """Load the replay buffer from a file."""
        state = torch.load(filepath)
        self.capacity = state['capacity']
        self.buffer = state['buffer']
        self.errors = state['errors']
        self.total_error = state['total_error']
