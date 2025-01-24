"""
data/data_loader.py

Implements lightweight data loading of tokenised shards for training/validation.
"""

import os
import numpy as np
import torch

master_process = True  # This is typically overridden by DDP logic in ddp_training.py

def load_tokens(filename: str) -> torch.Tensor:
    """
    Loads a NumPy token file and converts it into a torch.LongTensor.
    """
    npt = np.load(filename)
    return torch.tensor(npt, dtype=torch.long)

class DataLoaderLite:
    """
    Simple data loader that cycles through shards of tokens.
    Each shard is stored in a .npy file and is segmented into mini-batches.
    """

    def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: str):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}, f"Split must be 'train' or 'val', got {split}."

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split '{split}'."

        # Only the master process prints the shard count
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")

        self.reset()

    def reset(self):
        """Resets the data loader to the first shard and correct initial position."""
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        Returns the next (x, y) pair from the current shard. 
        If we exceed shard boundaries, it moves to the next shard.
        """
        buf = self.tokens[self.current_position : self.current_position + self.B*self.T + 1]
        x = buf[:-1].view(self.B, self.T)  # inputs
        y = buf[1:].view(self.B, self.T)   # targets

        self.current_position += self.B * self.T * self.num_processes

        # If we are out of tokens in this shard, move to the next.
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y