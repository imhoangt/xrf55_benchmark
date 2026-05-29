"""XRF55 raw amplitude dataset for ResNet1D-Amp baseline.

Loads pre-computed normalized raw amplitude files saved by scripts/local/01_preprocess.py:
  X_amp_raw_{split}.npy  shape (N, 270, 1000) — per-channel z-score (fit reps 1-12)

Splits: train (reps 1-12), val (reps 13-14), test (reps 15-20) — pre-split at preprocessing time.
__getitem__ returns (X, label) — single-stream, no DWT.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.loso_splits import get_vol_ids, get_loso_val_subjects, LOSO_FOLD_SUBJECTS


def _load_amp_raw(processed_dir: Path, split: str):
    p = Path(processed_dir)
    X = np.load(p / f'X_amp_raw_{split}.npy').astype(np.float32, copy=False)  # (N, 270, 1000)
    y = np.load(p / f'y_{split}.npy')
    return X, y


class XRF55AmpRawDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X   # (N, 270, 1000)
        self.y = y   # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _kw(num_workers=4):
    kwargs = dict(pin_memory=True, num_workers=num_workers,
                  persistent_workers=(num_workers > 0))
    if num_workers > 0:
        kwargs['worker_init_fn'] = _worker_init_fn
    return kwargs


# ── Split-protocol loaders (pre-split files) ─────────────────────────────────

def build_noval_loaders_with_val(processed_dir, batch_size=32, num_workers=4):
    """Train=reps 1-12, Val=reps 13-14, Test=reps 15-20 (from pre-split files).

    Returns (train_loader, val_loader, test_loader).
    """
    X_tr,  y_tr  = _load_amp_raw(processed_dir, 'train')
    X_val, y_val = _load_amp_raw(processed_dir, 'val')
    X_te,  y_te  = _load_amp_raw(processed_dir, 'test')
    kw = _kw(num_workers)
    train_loader = DataLoader(
        XRF55AmpRawDataset(X_tr,  y_tr),
        batch_size=batch_size, shuffle=True, **kw)
    val_loader   = DataLoader(
        XRF55AmpRawDataset(X_val, y_val),
        batch_size=64, shuffle=False, **kw)
    test_loader  = DataLoader(
        XRF55AmpRawDataset(X_te,  y_te),
        batch_size=64, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ── LOSO-5fold loaders ────────────────────────────────────────────────────────

def build_loso_loaders_with_val(fold_idx: int, processed_dir,
                                batch_size: int = 32, num_workers: int = 4):
    """LOSO rotation: test=G[i], val=G[(i+1)%5], train=3 remaining groups.

    Iterates all three pre-split files (train/val/test) to collect correct subjects.
    Returns (train_loader, val_loader, test_loader).
    """
    if not 0 <= fold_idx < len(LOSO_FOLD_SUBJECTS):
        raise ValueError(f"fold_idx={fold_idx} out of range [0, {len(LOSO_FOLD_SUBJECTS)-1}]")
    test_subjects  = set(LOSO_FOLD_SUBJECTS[fold_idx])
    val_subjects   = set(get_loso_val_subjects(fold_idx))
    train_subjects = set(range(1, 31)) - test_subjects - val_subjects

    tr_X, tr_y = [], []
    va_X, va_y = [], []
    te_X, te_y = [], []

    for split in ('train', 'val', 'test'):
        X, y    = _load_amp_raw(processed_dir, split)
        vol_ids = get_vol_ids(split)
        for subj_set, Xl, yl in [
            (train_subjects, tr_X, tr_y),
            (val_subjects,   va_X, va_y),
            (test_subjects,  te_X, te_y),
        ]:
            m = np.isin(vol_ids, list(subj_set))
            if m.any():
                Xl.append(X[m])
                yl.append(y[m])
        del X, y

    kw = _kw(num_workers)
    train_loader = DataLoader(
        XRF55AmpRawDataset(np.concatenate(tr_X), np.concatenate(tr_y)),
        batch_size=batch_size, shuffle=True, **kw)
    val_loader   = DataLoader(
        XRF55AmpRawDataset(np.concatenate(va_X), np.concatenate(va_y)),
        batch_size=64, shuffle=False, **kw)
    test_loader  = DataLoader(
        XRF55AmpRawDataset(np.concatenate(te_X), np.concatenate(te_y)),
        batch_size=64, shuffle=False, **kw)
    return train_loader, val_loader, test_loader
