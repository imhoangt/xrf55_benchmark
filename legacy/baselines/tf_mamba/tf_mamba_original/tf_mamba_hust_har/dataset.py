"""HUST-HAR dataset with 2D Haar DWT preprocessing for TF-Mamba.

Raw (3600, 270, 1000) float32 at 200 Hz (paper-published, already 5x downsampled
from the 1000 Hz collection rate).
  -> normalize per-(channel, timepoint)   # mean(dim=0) on (3600,270,1000) -> (1,270,1000)
  -> 2D Haar DWT per sample (270, 1000) -> XH (135, 500), XV (135, 500)
  -> Time stream: XH.T = (500, 135)
  -> Freq stream: XV   = (135, 500)
  -> 80/20 random split

DWT is precomputed once in __init__ (vectorized over the batch axis), avoiding
redundant per-epoch CPU work. Trades ~1.9 GB extra RAM for full DWT cache, but
the raw 3.9 GB tensor is dropped once DWT outputs are stored.
"""
from pathlib import Path

import numpy as np
import pywt
import torch
from torch.utils.data import DataLoader, Dataset, random_split

CLASSES = ['lying_down', 'picking_up', 'sitting_down', 'standing', 'standing_up', 'walking']
NUM_CLASSES = 6


class HUSTHARDWTDataset(Dataset):
    """Precomputes Haar DWT once at construction; __getitem__ becomes a slice."""

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        # data: (N, 270, 1000) float32, already normalized
        # Vectorized DWT over batch axis (axes=(-2,-1) operates on each (270,1000) plane).
        data_np = data.numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
        _, (xh, xv, _) = pywt.dwt2(
            data_np, 'haar', mode='periodization', axes=(-2, -1))
        # xh, xv: (N, 135, 500)
        # Both transposed → (N, 500, 135); L=500, M=135.
        self.XH = torch.from_numpy(np.ascontiguousarray(xh.transpose(0, 2, 1)))  # (N, 500, 135)
        self.XV = torch.from_numpy(np.ascontiguousarray(xv.transpose(0, 2, 1)))  # (N, 500, 135)
        self.labels = labels  # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.XH[idx], self.XV[idx], int(self.labels[idx])


def build_loaders(data_root: str, batch_size: int = 32,
                  num_workers: int = 4, seed: int = 42):
    root = Path(data_root)
    raw    = torch.load(root / 'HUST_HAR_dataset-001.pt', weights_only=False)  # (3600, 270, 1000)
    labels = torch.load(root / 'HUST_HAR_labels.pt',        weights_only=False)  # (3600,)

    # Per-(channel, timepoint) normalization — matches Mamba_HUST-HAR.py data_norm().
    # mean(dim=0) on (3600,270,1000) → (1,270,1000): one stat per (channel, timepoint) pair.
    # Stats fitted on full dataset (all 3600 samples) intentionally, matching the original paper.
    mean = raw.mean(dim=0, keepdim=True)   # (1, 270, 1000)
    std  = raw.std(dim=0, keepdim=True)    # (1, 270, 1000)
    raw  = (raw - mean) / (std + 1e-6)

    full_ds = HUSTHARDWTDataset(raw, labels)
    del raw     # raw no longer needed after DWT is cached
    n_train = int(0.8 * len(full_ds))
    n_test  = len(full_ds) - n_train
    gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              pin_memory=True, num_workers=num_workers)
    return train_loader, test_loader
