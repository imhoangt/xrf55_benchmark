"""Dataset loader for WavMamba (packed Haar wavmamba format).

Loads pre-built bench arrays from
    <root>/<DIR>/bench/<mode>_<prenorm>_<z_gran>/wavmamba/X_<split>.npy
built by build_dataset.py.

z-norm AFTER DWT is ALWAYS applied; its granularity (perpos / pcb) is read from
the stats.json shape and handled by _bcast. Pre-norm (if any) was baked into
X_*.npy at build time — this is the single z layer at load time.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_stats(bench_dir) -> dict:
    with open(Path(bench_dir) / 'stats.json') as f:
        return json.load(f)


def infer_data_mode(stats: dict) -> str:
    """Infer 'proc' | 'raw' from stats.json meta."""
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


# ── Dataset class — load from pre-built bench arrays ──────────────────────────

class PreprocWavMambaDataset(Dataset):
    """Loads wavmamba/X_<split>.npy, applies z-norm, returns (X, label).

    z-norm is ALWAYS applied. The mean/std shape is either:
      (C, F2)   per-channel-bin  [z_gran=pcb]   -> broadcast to (C, 1, F2)
      (C, T2,F2) per-position    [z_gran=perpos] -> used as-is
    """

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.X   = np.load(bench_dir / 'wavmamba' / f'X_{split}.npy', mmap_mode='r')
        self.y   = np.load(bench_dir / f'y_{split}.npy')
        s = stats['wavmamba']
        def _bcast(a):
            a = np.array(a, dtype=np.float32)
            return a[:, None, :] if a.ndim == 2 else a
        self.mu = _bcast(s['mean']); self.sig = _bcast(s['std'])

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        x = np.array(self.X[idx], dtype=np.float32)                 # copy -> writable
        x = (x - self.mu) / self.sig          # mu/sig broadcast (C,1,F2) | (C,T2,F2)
        return torch.from_numpy(x), int(self.y[idx])


# ── Loader builder ────────────────────────────────────────────────────────────

def build_loaders(stats: dict, bench_dir, batch_size: int = 32, num_workers: int = 4):
    """Build (train_loader, test_loader) from pre-built bench arrays.

    Args:
        stats      : dict from load_stats(bench_dir)
        bench_dir  : path to bench/<mode>_<prenorm>_<z_gran>/
    """
    bench_dir = Path(bench_dir)
    sentinel  = bench_dir / 'wavmamba' / 'X_train.npy'
    if not sentinel.exists():
        raise FileNotFoundError(
            f'Bench arrays not found: {sentinel}\n'
            'Run build_dataset.py first.')

    data_label = ('Processed CSI Amplitude (Hampel + Butterworth LPF)'
                  if infer_data_mode(stats) == 'proc' else 'Raw CSI Amplitude')
    print(f'  Data  : {data_label}')
    print(f'  Loaded: {bench_dir}')

    kw = _kw(num_workers)
    train_ds = PreprocWavMambaDataset(bench_dir, 'train', stats)
    test_ds  = PreprocWavMambaDataset(bench_dir, 'test',  stats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)
    return train_loader, test_loader
