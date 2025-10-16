import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import re
import torch.nn.functional as F
import random
import numpy as np

def collate_fn(batch):
    """
    简单的collate函数，处理包含branch_sequence和select_sequence的batch
    Args:
        batch: list of (state, select_sequence, branch_sequence, select_cand, branch_cand, label) from dataset
    Returns:
        batched_data: dictionary containing batched tensors with masks
    """
    #for sequence in batch:

    return batch

class BnBSequentialDataset(Dataset):
    def __init__(self, data_dir, max_samples=None):
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.trajectories = self.collect_trajectories()

    def collect_trajectories(self):
        # Each folder is a trajectory (e.g., a MILP instance run)
        all_dirs = [d for d in Path(self.data_dir).iterdir() if d.is_dir()]
        
        print(f"Total directories: {len(all_dirs)}")

        return all_dirs
    

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        dir_path = self.trajectories[idx]

        sequence = torch.load(dir_path / 'sequence.pt')

        return sequence
