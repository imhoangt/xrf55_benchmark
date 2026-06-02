"""UT-HAR dataset with 2D Haar DWT preprocessing for TF-Mamba.

Preprocessing follows UT_HAR_dataset() from the original paper exactly:
  -> np.load each CSV in binary mode
  -> reshape to (N, 1, 250, 90)
  -> min-max normalize per split: (data - min) / (max - min)

TF-Mamba DWT step added on top:
  -> squeeze channel dim + transpose -> (N, 90, 250)  # (features, time) convention
  -> 2D Haar DWT (90, 250) -> cH, cV: (N, 45, 125)
  -> Time stream: cH.T = (N, 125, 45)
  -> Freq stream: cV.T = (N, 125, 45)

Split follows benchmark convention (xyanchen/wifi-csi-sensing-benchmark):
  Train: X_train only
  Test:  X_val + X_test concatenated (each normalized independently before concat)
"""
from pathlib import Path

import numpy as np
import pywt
import torch
from torch.utils.data import DataLoader, Dataset


class UTHARDWTDataset(Dataset):
    """Precomputes Haar DWT once at construction; __getitem__ becomes a slice."""

    def __init__(self, data_norm: np.ndarray, labels: np.ndarray):
        # data_norm: (N, 1, 250, 90) float32, already min-max normalized
        # Squeeze channel dim, transpose to (features, time), then apply 2D DWT.
        x = data_norm.squeeze(1)                                    # (N, 250, 90)
        x = x.transpose(0, 2, 1).astype(np.float32)                # (N, 90, 250)
        _, (xh, xv, _) = pywt.dwt2(x, 'haar', mode='periodization', axes=(-2, -1))
        # xh, xv: (N, 45, 125)
        # Both transposed → (N, 125, 45); L=125, M=45.
        self.XH = torch.from_numpy(np.ascontiguousarray(xh.transpose(0, 2, 1)))  # (N, 125, 45)
        self.XV = torch.from_numpy(np.ascontiguousarray(xv.transpose(0, 2, 1)))  # (N, 125, 45)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.XH[idx], self.XV[idx], int(self.labels[idx])


def _load_split(data_path, label_path):
    """Load and normalize one split using paper's UT_HAR_dataset() preprocessing."""
    with open(data_path, 'rb') as f:
        data = np.load(f)
    data = data.reshape(len(data), 1, 250, 90)
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    with open(label_path, 'rb') as f:
        labels = np.load(f)
    return data_norm.astype(np.float32), labels.astype(np.int64)


def build_loaders(data_root: str, batch_size: int = 32, num_workers: int = 4):
    root = Path(data_root)
    data_dir  = root / 'data'
    label_dir = root / 'label'

    train_data,  train_labels  = _load_split(data_dir / 'X_train.csv', label_dir / 'y_train.csv')
    val_data,    val_labels    = _load_split(data_dir / 'X_val.csv',   label_dir / 'y_val.csv')
    test_data,   test_labels   = _load_split(data_dir / 'X_test.csv',  label_dir / 'y_test.csv')

    # Test = val + test (benchmark convention)
    eval_data   = np.concatenate([val_data,   test_data],   axis=0)
    eval_labels = np.concatenate([val_labels, test_labels], axis=0)
    del val_data, test_data

    train_ds = UTHARDWTDataset(train_data, train_labels)
    test_ds  = UTHARDWTDataset(eval_data,  eval_labels)
    del train_data, eval_data

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=num_workers, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              pin_memory=True, num_workers=num_workers)
    return train_loader, test_loader
