# utils/memory_replay_buffer.py

import random

class MemoryReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []        # List to store samples
        self.errors = []        # List to store errors (used for sampling probabilities)
        self.total_error = 0.0  # Sum of all errors, used for normalization

    def add_examples(self, examples, errors):
        """
        Add new examples to the buffer along with their prediction errors.

        Args:
            examples (list): List of data samples (dicts with 'input_ids', 'labels', 'sector', etc.)
            errors (list): List of errors corresponding to each example
        """
        for example, error in zip(examples, errors):
            self.buffer.append(example)
            self.errors.append(error)
            self.total_error += error

            # Remove oldest data if capacity is exceeded
            if len(self.buffer) > self.capacity:
                removed_error = self.errors.pop(0)
                self.buffer.pop(0)
                self.total_error -= removed_error

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

        # Compute sampling probabilities
        sampling_probs = []
        for i in range(len(self.buffer)):
            error = self.errors[i]
            sector = self.buffer[i]['sector']
            sector_error = average_sector_errors.get(sector, 1.0)  # Default to 1.0 if not found

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
