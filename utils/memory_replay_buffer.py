import random
import torch
import torch.nn.functional as F
import torch.distributed as dist

class MemoryReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []  # list to store samples (dicts)
        self.errors = []  # list to store per-sample errors
        self.total_error = 0.0  # sum of all errors, for convenience

    def add_examples(self, examples, errors):
        """
        Add new examples to the buffer along with their prediction errors.

        Args:
            examples (list): List of data samples (dicts with 'input_ids', 'labels', 'sector', etc.)
            errors (list): List of float errors corresponding to each example
        """
        for example, error in zip(examples, errors):
            # If buffer is full, remove the sample with the lowest error
            if len(self.buffer) >= self.capacity:
                min_error = min(self.errors)
                min_index = self.errors.index(min_error)
                removed_error = self.errors.pop(min_index)
                self.buffer.pop(min_index)
                self.total_error -= removed_error

            # Add the new example
            self.buffer.append(example)
            self.errors.append(error)
            self.total_error += error

    def add_batch(self, batch, errors=None):
        """
        Convenience method to add an entire batch of samples.
        If 'errors' is not provided, assumes zero errors for each sample.

        Args:
            batch (dict): The training batch containing 'input_ids', 'labels', etc.
            errors (list of float, optional): Per-sample errors. If None, defaults to 0.0.
        """
        batch_size = len(batch['input_ids'])
        if errors is None or len(errors) != batch_size:
            # default to zero if no valid errors provided
            errors = [0.0] * batch_size

        examples = []
        for i in range(batch_size):
            sample = {
                'input_ids': batch['input_ids'][i].cpu(),
                'labels': batch['labels'][i].cpu(),
                'sector': batch.get('sector', ['Unknown'])[i],
                'symbol': batch.get('symbol', ['Unknown'])[i],
            }
            examples.append(sample)

        self.add_examples(examples, errors)

    def sample(self, sample_size, average_sector_errors):
        """
        Sample data from the buffer based on the product of their errors
        and sector-level errors.

        Args:
            sample_size (int): Number of samples to draw
            average_sector_errors (dict): e.g., {sector_name: sector_error}

        Returns:
            List of sampled data (dicts)
        """
        if len(self.buffer) == 0:
            return []

        # Compute mean sector error to fall back on
        if average_sector_errors:
            mean_sector_error = sum(average_sector_errors.values()) / len(average_sector_errors)
        else:
            mean_sector_error = 1.0

        # Build sampling probabilities
        sampling_probs = []
        for i in range(len(self.buffer)):
            sample_error = self.errors[i]
            sample_sector = self.buffer[i].get('sector', 'Unknown')
            sector_error = average_sector_errors.get(sample_sector, mean_sector_error)
            combined_error = sample_error * sector_error
            sampling_probs.append(combined_error)

        total = sum(sampling_probs)
        if total == 0:
            sampling_probs = [1.0 / len(self.buffer)] * len(self.buffer)
        else:
            sampling_probs = [p / total for p in sampling_probs]

        sampled_indices = random.choices(
            population=range(len(self.buffer)),
            weights=sampling_probs,
            k=sample_size
        )
        return [self.buffer[i] for i in sampled_indices]

    def replay_and_calculate_loss(self, model, tokenizer, replay_batch_size, device, alpha=1.0):
        """
        Samples data from the buffer, computes a forward pass,
        and returns the MSE loss on these replay samples.

        Args:
            model: The main model (can be Deepspeed-wrapped or standard PyTorch).
            tokenizer: The tokenizer, if needed for reconstruction (not always needed).
            replay_batch_size (int): Number of samples to retrieve from the buffer.
            device: CPU or CUDA device.
            alpha (float): Optional weighting factor for the replay loss.

        Returns:
            torch.Tensor (scalar): Weighted MSE loss for replay samples. 0 if buffer empty.
        """
        if len(self.buffer) == 0 or replay_batch_size <= 0:
            return 0.0

        # Simple random sample from the buffer:
        if replay_batch_size > len(self.buffer):
            replay_batch_size = len(self.buffer)
        # We'll just randomly sample indices (without weighting by sector).
        sampled_indices = random.sample(range(len(self.buffer)), replay_batch_size)
        sampled_data = [self.buffer[i] for i in sampled_indices]

        # Convert the sampled data into a batch
        input_ids_list = []
        labels_list = []
        for sample in sampled_data:
            input_ids_list.append(sample['input_ids'])
            labels_list.append(sample['labels'])

        # Create a mini-batch
        input_ids = torch.stack(input_ids_list).to(device)
        labels = torch.stack(labels_list).float().to(device)

        # Forward pass for replay
        with torch.cuda.amp.autocast(enabled=True):
            preds, _ = model(input_ids=input_ids)
            replay_loss = F.mse_loss(preds.squeeze(-1), labels)

        return alpha * replay_loss

    def sync_buffer(self):
        """
        Synchronize the replay buffer across all processes in DDP.
        Concatenates buffers from all ranks, then trims to capacity by highest errors.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        buffer_list = [None for _ in range(world_size)]
        errors_list = [None for _ in range(world_size)]

        dist.all_gather_object(buffer_list, self.buffer)
        dist.all_gather_object(errors_list, self.errors)

        merged_buffer = []
        merged_errors = []
        for buf, err in zip(buffer_list, errors_list):
            merged_buffer.extend(buf)
            merged_errors.extend(err)

        # Trim if above capacity
        if len(merged_buffer) > self.capacity:
            combined = list(zip(merged_buffer, merged_errors))
            # Sort descending by error => keep largest errors
            combined.sort(key=lambda x: x[1], reverse=True)
            combined = combined[:self.capacity]
            merged_buffer, merged_errors = zip(*combined)
            merged_buffer = list(merged_buffer)
            merged_errors = list(merged_errors)

        self.buffer = merged_buffer
        self.errors = merged_errors
        self.total_error = sum(self.errors)

    def save(self, filepath):
        """
        Save the replay buffer to a file.
        """
        state = {
            'capacity': self.capacity,
            'buffer': self.buffer,
            'errors': self.errors,
            'total_error': self.total_error
        }
        torch.save(state, filepath)

    def load(self, filepath):
        """
        Load the replay buffer from a file.
        """
        state = torch.load(filepath)
        self.capacity = state['capacity']
        self.buffer = state['buffer']
        self.errors = state['errors']
        self.total_error = state['total_error']
