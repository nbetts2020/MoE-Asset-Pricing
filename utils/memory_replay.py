# utils/memory_replay.py

import random
import torch

class MemoryReplayBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, input_ids, labels):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((input_ids.cpu(), labels.cpu()))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
