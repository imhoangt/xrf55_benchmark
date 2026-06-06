"""Dataset loaders for the XRF55 benchmark.

Loads pre-built bench arrays from bench/raw_nosc/ or bench/processed_nosc/.
Build these with 01_build_dataset_raw.py (raw) or 02_build_dataset_processed.py (processed).

Split: train=reps 1-14 (4620 samples), test=reps 15-20 (1980 samples). No val.

Model input shapes after normalization:
  resnet    → (270, 1000)          per-channel z-score (270,)
  tfmamba   → XH (500, 135)        per-channel z-score on cH.T
               XV (500, 135)        per-channel z-score on cV.T
  wavmamba  → (27, 500, 15)        per-freq z-score (27, 15)
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_stats(bench_dir) -> dict:
    with open(Path(bench_dir) / 'stats.json') as f:
        return json.load(f)


def infer_data_mode(stats: dict) -> str:
    """Infer 'proc' | 'raw' from stats.json meta.

    Processed stats (02_build_dataset_processed.py) carry a 'filter' key and
    meta.source='raw_npy_270_hampel_lpf'.  Raw stats (01_build_dataset_raw.py)
    use meta.source='raw_npy_270' with no filter key.
    """
    meta = stats.get('meta', {})
    if 'filter' in meta or 'hampel' in str(meta.get('source', '')).lower():
        return 'proc'
    return 'raw'


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _kw(num_workers):
    kw = dict(pin_memory=True, num_workers=num_workers,
              persistent_workers=(num_workers > 0))
    if num_workers > 0:
        kw['worker_init_fn'] = _worker_init_fn
    return kw


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset classes — load from pre-built bench arrays
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocResNetDataset(Dataset):
    """Loads resnet/X_{split}.npy, applies per-channel z-score. Returns (X, label)."""

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.X   = np.load(bench_dir / 'resnet' / f'X_{split}.npy', mmap_mode='r')
        self.y   = np.load(bench_dir / f'y_{split}.npy')
        self.mu  = np.array(stats['resnet']['mean'], dtype=np.float32)  # (270,)
        self.sig = np.array(stats['resnet']['std'],  dtype=np.float32)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mu[:, None]) / self.sig[:, None]   # (270, 1000)
        return torch.from_numpy(x), int(self.y[idx])


class PreprocTFMambaDataset(Dataset):
    """Loads tfmamba XH/XV arrays (N,500,135), normalizes. Returns (XH, XV, label)."""

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.XH = np.load(bench_dir / 'tfmamba' / f'X_{split}_xh.npy', mmap_mode='r')  # (N, 500, 135)
        self.XV = np.load(bench_dir / 'tfmamba' / f'X_{split}_xv.npy', mmap_mode='r')  # (N, 500, 135)
        self.y  = np.load(bench_dir / f'y_{split}.npy')
        s = stats['tfmamba']
        self.xh_mean = np.array(s['xh_mean'], dtype=np.float32)  # (135,)
        self.xh_std  = np.array(s['xh_std'],  dtype=np.float32)
        self.xv_mean = np.array(s['xv_mean'], dtype=np.float32)  # (135,)
        self.xv_std  = np.array(s['xv_std'],  dtype=np.float32)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        xh = (self.XH[idx] - self.xh_mean[None, :]) / self.xh_std[None, :]  # (500, 135)
        xv = (self.XV[idx] - self.xv_mean[None, :]) / self.xv_std[None, :]  # (500, 135)
        return torch.from_numpy(xh), torch.from_numpy(xv), int(self.y[idx])


class PreprocWavMambaDataset(Dataset):
    """Loads wavmamba/X_{split}.npy, applies per-channel z-score. Returns (X, label)."""

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.X   = np.load(bench_dir / 'wavmamba' / f'X_{split}.npy', mmap_mode='r')
        self.y   = np.load(bench_dir / f'y_{split}.npy')
        s = stats['wavmamba']
        self.mu  = np.array(s['mean'], dtype=np.float32)  # (27, 15)
        self.sig = np.array(s['std'],  dtype=np.float32)  # (27, 15)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mu[:, None, :]) / self.sig[:, None, :]  # (27, 500, 15)
        return torch.from_numpy(x), int(self.y[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# Loader factory
# ═══════════════════════════════════════════════════════════════════════════════

_PREPROC_DS = {
    'resnet':        PreprocResNetDataset,
    'tfmamba':       PreprocTFMambaDataset,
    'wavmamba':      PreprocWavMambaDataset,
    'wavdualmamba':   PreprocWavMambaDataset,  # same data format as wavmamba
    'wavdualmamba_v2': PreprocWavMambaDataset,  # same data format as wavmamba
}
_PREPROC_SENTINEL = {
    'resnet':           'resnet/X_train.npy',
    'tfmamba':          'tfmamba/X_train_xh.npy',
    'wavmamba':         'wavmamba/X_train.npy',
    'wavdualmamba':     'wavmamba/X_train.npy',  # same files
    'wavdualmamba_v2':  'wavmamba/X_train.npy',  # same files
}


def build_loaders(model_name: str, stats: dict, bench_dir,
                  batch_size: int = 32, num_workers: int = 4):
    """Build (train_loader, test_loader) from pre-built bench arrays.

    Args:
        model_name : 'resnet', 'tfmamba', or 'wavmamba'
        stats      : dict from load_stats(bench_dir)
        bench_dir  : path to bench/raw_nosc/ or bench/processed_nosc/
    """
    if model_name not in _PREPROC_DS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_PREPROC_DS)}")

    bench_dir = Path(bench_dir)
    sentinel  = bench_dir / _PREPROC_SENTINEL[model_name]
    if not sentinel.exists():
        raise FileNotFoundError(
            f'Bench arrays not found: {sentinel}\n'
            'Run 01_build_dataset_raw.py or 02_build_dataset_processed.py first.')

    data_label = ('Processed CSI Amplitude (Hampel + Butterworth LPF)'
                  if infer_data_mode(stats) == 'proc' else 'Raw CSI Amplitude')
    print(f'  Data  : {data_label}')
    print(f'  Loaded: {bench_dir}')

    DS = _PREPROC_DS[model_name]
    kw = _kw(num_workers)
    train_ds = DS(bench_dir, 'train', stats)
    test_ds  = DS(bench_dir, 'test',  stats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)
    return train_loader, test_loader
