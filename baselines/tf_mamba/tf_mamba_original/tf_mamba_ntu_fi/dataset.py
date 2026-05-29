"""NTU-Fi dataset with 2D Haar DWT preprocessing for TF-Mamba.

Preprocessing follows CSI_Dataset from the original paper exactly (per sample):
  -> scipy.io.loadmat(path)['CSIamp']
  -> (x - 42.3199) / 4.9802               # fixed normalization constants
  -> x[:, ::4]                             # downsample 2000 -> 500
  -> x.reshape(3, 114, 500)               # (3 links, 114 subcarriers, 500 time)

TF-Mamba DWT step added on top:
  -> reshape(342, 500)                     # flatten: 3x114 features, 500 time
  -> 2D Haar DWT (342, 500) -> cH, cV: (N, 171, 250)
  -> Time stream: cH.T = (N, 250, 171)
  -> Freq stream: cV   = (N, 171, 250)

Split: pre-existing train_amp/ and test_amp/ directories.
Classes (alphabetical order): box=0, circle=1, clean=2, fall=3, run=4, walk=5.
"""
import glob as _glob
from pathlib import Path

import numpy as np
import pywt
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset

CLASSES = ['box', 'circle', 'clean', 'fall', 'run', 'walk']
NUM_CLASSES = 6


class NTUFiDWTDataset(Dataset):
    """Precomputes Haar DWT once at construction; __getitem__ becomes a slice."""

    def __init__(self, data_np: np.ndarray, labels: np.ndarray):
        # data_np: (N, 342, 500) float32, already preprocessed
        # 2D DWT on (342, 500) per sample.
        _, (xh, xv, _) = pywt.dwt2(data_np, 'haar', mode='periodization', axes=(-2, -1))
        # xh, xv: (N, 171, 250)
        # Both transposed → (N, 250, 171); L=250, M=171.
        self.XH = torch.from_numpy(np.ascontiguousarray(xh.transpose(0, 2, 1)))  # (N, 250, 171)
        self.XV = torch.from_numpy(np.ascontiguousarray(xv.transpose(0, 2, 1)))  # (N, 250, 171)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.XH[idx], self.XV[idx], int(self.labels[idx])


def _load_split_dir(split_dir: Path):
    """Load all .mat files from split_dir/*/*.mat using CSI_Dataset preprocessing."""
    folders = sorted(_glob.glob(str(split_dir / '*/') ))
    category = {Path(f).name: i for i, f in enumerate(folders)}

    samples, labels = [], []
    for mat_path in sorted(_glob.glob(str(split_dir / '*' / '*.mat'))):
        cls_name = Path(mat_path).parent.name
        x = scipy.io.loadmat(mat_path)['CSIamp']
        x = (x - 42.3199) / 4.9802          # paper's fixed normalization
        x = x[:, ::4]                        # downsample 2000 -> 500
        x = x.reshape(3, 114, 500)           # (3 links, 114 subcarriers, 500 time)
        x = x.reshape(342, 500)              # flatten for DWT
        samples.append(x.astype(np.float32))
        labels.append(category[cls_name])

    return np.stack(samples, axis=0), np.array(labels, dtype=np.int64)


def build_loaders(data_root: str, batch_size: int = 32, num_workers: int = 4):
    root = Path(data_root)

    train_data, train_labels = _load_split_dir(root / 'train_amp')
    test_data,  test_labels  = _load_split_dir(root / 'test_amp')

    train_ds = NTUFiDWTDataset(train_data, train_labels)
    test_ds  = NTUFiDWTDataset(test_data,  test_labels)
    del train_data, test_data

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              pin_memory=True, num_workers=num_workers)
    return train_loader, test_loader
