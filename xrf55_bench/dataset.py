"""Dataset loaders for the XRF55 benchmark — two source modes.

source='preproc'  : load pre-transformed arrays from bench_processed/
                    (fast; requires 02a + 02b to have been run, ~16 GB disk)
source='raw'      : load amplitude_npy_4d/ files and compute transforms on-the-fly
                    (slow per batch; no preprocessing required)
source='auto'     : use preproc if arrays exist, otherwise fall back to raw

Split: train=reps 1-14 (4620 samples), test=reps 15-20 (1980 samples). No val.

Model input shapes after normalization:
  resnet    → (270, 1000)          per-channel z-score (270,)
  tfmamba   → XH (500, 135)        per-channel z-score on cH.T
               XV (500, 135)        per-channel z-score on cV, then transposed
  wavmamba  → (27, 500, 15)        per-channel z-score
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pywt
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.dwt    import apply_dwt2_stack
from src.data.preprocessing.parser import ACTION_ID_TO_LABEL, ACTION_IDS_USED

TRAIN_REPS = list(range(1, 15))
TEST_REPS  = list(range(15, 21))


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_stats(bench_dir) -> dict:
    with open(Path(bench_dir) / 'stats.json') as f:
        return json.load(f)


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


def _collect_split(amp4d_dir: Path, rep_list: list) -> list:
    """Sorted list of (fpath, label) for all (vol, action, rep) in rep_list."""
    items = []
    for vol_id in range(1, 31):
        for action_id in ACTION_IDS_USED:
            label = ACTION_ID_TO_LABEL[action_id]
            for rep_id in rep_list:
                p = (amp4d_dir / f'{vol_id:02d}'
                     / f'{vol_id:02d}_{action_id:02d}_{rep_id:02d}.npy')
                items.append((p, label))
    return items


def _to_flat270(raw4d: np.ndarray) -> np.ndarray:
    """(1000, 3, 3, 30) float64 → (270, 1000) float32."""
    return (raw4d
            .transpose(1, 2, 0, 3)
            .reshape(9, 1000, 30)
            .transpose(0, 2, 1)
            .reshape(270, 1000)
            .astype(np.float32))


def _to_x9(raw4d: np.ndarray) -> np.ndarray:
    """(1000, 3, 3, 30) float64 → (9, 1000, 30) float32."""
    return (raw4d
            .transpose(1, 2, 0, 3)
            .reshape(9, 1000, 30)
            .astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSED mode — loads bench_processed/ arrays
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocResNetDataset(Dataset):
    """Loads resnet/X_{split}.npy, applies per-channel z-score. Returns (X, label)."""

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.X   = np.load(bench_dir / 'resnet' / f'X_{split}.npy', mmap_mode='r')
        self.y   = np.load(bench_dir / f'y_{split}.npy')
        self.mu  = np.array(stats['resnet']['mean'], dtype=np.float32)  # (270,)
        self.sig = np.array(stats['resnet']['std'],  dtype=np.float32)  # (270,)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mu[:, None]) / self.sig[:, None]
        return torch.from_numpy(x), int(self.y[idx])


class PreprocTFMambaDataset(Dataset):
    """Loads tfmamba XH/XV arrays, normalizes, transposes XV. Returns (XH, XV, label)."""

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.XH = np.load(bench_dir / 'tfmamba' / f'X_{split}_xh.npy', mmap_mode='r')  # (N, 500, 135)
        self.XV = np.load(bench_dir / 'tfmamba' / f'X_{split}_xv.npy', mmap_mode='r')  # (N, 135, 500)
        self.y  = np.load(bench_dir / f'y_{split}.npy')
        s = stats['tfmamba']
        self.xh_mean = np.array(s['xh_mean'], dtype=np.float32)  # (135,)
        self.xh_std  = np.array(s['xh_std'],  dtype=np.float32)
        self.xv_mean = np.array(s['xv_mean'], dtype=np.float32)  # (135,)
        self.xv_std  = np.array(s['xv_std'],  dtype=np.float32)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        xh = (self.XH[idx] - self.xh_mean[None, :]) / self.xh_std[None, :]  # (500,135)
        xv = (self.XV[idx] - self.xv_mean[:, None]) / self.xv_std[:, None]  # (135,500)
        xv = xv.T                                                              # (500,135)
        return torch.from_numpy(xh), torch.from_numpy(xv), int(self.y[idx])


class PreprocWavMambaDataset(Dataset):
    """Loads wavmamba/X_{split}.npy, applies per-channel z-score. Returns (X, label)."""

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.X   = np.load(bench_dir / 'wavmamba' / f'X_{split}.npy', mmap_mode='r')
        self.y   = np.load(bench_dir / f'y_{split}.npy')
        s = stats['wavmamba']
        self.mu  = np.array(s['mean'], dtype=np.float32)  # (27,)
        self.sig = np.array(s['std'],  dtype=np.float32)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mu[:, None, None]) / self.sig[:, None, None]
        return torch.from_numpy(x), int(self.y[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# RAW mode — loads amplitude_npy_4d/, computes transforms on-the-fly
# ═══════════════════════════════════════════════════════════════════════════════

class RawResNetDataset(Dataset):
    """Loads raw 4D files, applies flat transform + per-channel z-score. Returns (X, label)."""

    def __init__(self, amp4d_dir: Path, split: str, stats: dict):
        rep_list = TRAIN_REPS if split == 'train' else TEST_REPS
        self.samples = _collect_split(Path(amp4d_dir), rep_list)
        self.mu  = np.array(stats['resnet']['mean'], dtype=np.float32)  # (270,)
        self.sig = np.array(stats['resnet']['std'],  dtype=np.float32)  # (270,)

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        raw  = np.load(fpath)
        flat = _to_flat270(raw)
        x    = (flat - self.mu[:, None]) / self.sig[:, None]
        return torch.from_numpy(x), label


class RawTFMambaDataset(Dataset):
    """Loads raw 4D files, applies Haar DWT + per-channel z-score. Returns (XH, XV, label)."""

    def __init__(self, amp4d_dir: Path, split: str, stats: dict):
        rep_list = TRAIN_REPS if split == 'train' else TEST_REPS
        self.samples = _collect_split(Path(amp4d_dir), rep_list)
        s = stats['tfmamba']
        self.xh_mean = np.array(s['xh_mean'], dtype=np.float32)
        self.xh_std  = np.array(s['xh_std'],  dtype=np.float32)
        self.xv_mean = np.array(s['xv_mean'], dtype=np.float32)
        self.xv_std  = np.array(s['xv_std'],  dtype=np.float32)

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        raw  = np.load(fpath)
        flat = _to_flat270(raw)                                         # (270, 1000)
        _, (cH, cV, _) = pywt.dwt2(flat, 'haar', mode='periodization')
        # cH, cV: (135, 500)
        xh = cH.T                                                       # (500, 135)
        xv = cV                                                         # (135, 500)
        xh = (xh - self.xh_mean[None, :]) / self.xh_std[None, :]
        xv = (xv - self.xv_mean[:, None]) / self.xv_std[:, None]
        xv = xv.T                                                       # (500, 135)
        return torch.from_numpy(xh), torch.from_numpy(xv), label


class RawWavMambaDataset(Dataset):
    """Loads raw 4D files, applies db4 DWT + per-channel z-score. Returns (X, label)."""

    def __init__(self, amp4d_dir: Path, split: str, stats: dict):
        rep_list = TRAIN_REPS if split == 'train' else TEST_REPS
        self.samples = _collect_split(Path(amp4d_dir), rep_list)
        s = stats['wavmamba']
        self.mu  = np.array(s['mean'], dtype=np.float32)
        self.sig = np.array(s['std'],  dtype=np.float32)

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        raw  = np.load(fpath)
        x9   = _to_x9(raw)                                # (9, 1000, 30)
        xdwt = apply_dwt2_stack(x9[None])[0]              # (27, 500, 15)
        x    = (xdwt - self.mu[:, None, None]) / self.sig[:, None, None]
        return torch.from_numpy(x), label


# ═══════════════════════════════════════════════════════════════════════════════
# Loader factory — auto-detect or explicit source
# ═══════════════════════════════════════════════════════════════════════════════

_PREPROC_DS = {
    'resnet':   PreprocResNetDataset,
    'tfmamba':  PreprocTFMambaDataset,
    'wavmamba': PreprocWavMambaDataset,
}
_RAW_DS = {
    'resnet':   RawResNetDataset,
    'tfmamba':  RawTFMambaDataset,
    'wavmamba': RawWavMambaDataset,
}
_PREPROC_SENTINEL = {
    'resnet':   'resnet/X_train.npy',
    'tfmamba':  'tfmamba/X_train_xh.npy',
    'wavmamba': 'wavmamba/X_train.npy',
}


def _resolve_source(model_name: str, bench_dir, amp4d_dir, source: str) -> str:
    if source == 'preproc':
        if bench_dir is None:
            raise ValueError('bench_dir required for source="preproc"')
        sentinel = Path(bench_dir) / _PREPROC_SENTINEL[model_name]
        if not sentinel.exists():
            raise ValueError(
                f'source="preproc": array not found at {sentinel}\n'
                'Run the preprocessing scripts first.')
        return 'preproc'
    if source == 'raw':
        if amp4d_dir is None:
            raise ValueError('amp4d_dir required for source="raw"')
        return 'raw'
    # auto
    if bench_dir is not None:
        sentinel = Path(bench_dir) / _PREPROC_SENTINEL[model_name]
        if sentinel.exists():
            return 'preproc'
    if amp4d_dir is not None:
        return 'raw'
    raise ValueError(
        'source="auto": bench_dir arrays not found and amp4d_dir not provided.')


def build_loaders(model_name: str, stats: dict,
                  bench_dir=None, amp4d_dir=None,
                  source: str = 'auto',
                  batch_size: int = 32, num_workers: int = 4):
    """Build (train_loader, test_loader) for model_name.

    Args:
        model_name  : 'resnet', 'tfmamba', or 'wavmamba'
        stats       : dict from load_stats(bench_dir)
        bench_dir   : path to bench_processed/ (preproc mode)
        amp4d_dir   : path to amplitude_npy_4d/ (raw mode)
        source      : 'preproc' | 'raw' | 'auto' (default)
    """
    if model_name not in _PREPROC_DS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_PREPROC_DS)}")

    src = _resolve_source(model_name, bench_dir, amp4d_dir, source)

    if src == 'preproc':
        DS   = _PREPROC_DS[model_name]
        root = Path(bench_dir)
        print(f'  Source: preproc  ({root})')
    else:
        DS   = _RAW_DS[model_name]
        root = Path(amp4d_dir)
        print(f'  Source: raw  ({root})')

    kw = _kw(num_workers)
    train_ds = DS(root, 'train', stats)
    test_ds  = DS(root, 'test',  stats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)
    return train_loader, test_loader
