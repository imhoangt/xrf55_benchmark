"""XRF55 amplitude dataset for TF-Mamba-XRF55-Amp baseline.

Loads pre-computed Haar DWT files saved by scripts/local/01_preprocess.py:
  X_amp_xh_{split}.npy  shape (N, 500, 135) — Time-Mamba stream (per-channel z-score)
  X_amp_xv_{split}.npy  shape (N, 135, 500) — Freq-Mamba stream (per-channel z-score)

Splits: train (reps 1-12), val (reps 13-14), test (reps 15-20) — pre-split at preprocessing time.
__getitem__ returns (XH, XV, label) directly without any on-the-fly computation.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.loso_splits import get_vol_ids, get_loso_val_subjects, LOSO_FOLD_SUBJECTS


def _load_amp(processed_dir: Path, split: str):
    """Load pre-computed Haar DWT amp split."""
    p = Path(processed_dir)
    XH = np.load(p / f'X_amp_xh_{split}.npy').astype(np.float32, copy=False)  # (N, 500, 135)
    XV = np.load(p / f'X_amp_xv_{split}.npy').astype(np.float32, copy=False)  # (N, 135, 500)
    y  = np.load(p / f'y_{split}.npy')
    return XH, XV, y


class XRF55AmpHaarDatasetFromArrays(Dataset):
    """Dataset built from pre-sliced numpy arrays (for noval/LOSO splits)."""

    def __init__(self, XH: np.ndarray, XV: np.ndarray, y: np.ndarray):
        self.XH = XH   # (N, 500, 135)
        self.XV = XV   # (N, 135, 500)
        self.y  = y    # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.XH[idx]),
                torch.from_numpy(self.XV[idx]),
                int(self.y[idx]))


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
    """Train=reps 1-12, Val=reps 13-14, Test=reps 15-20 (loaded from pre-split files).

    Returns (train_loader, val_loader, test_loader).
    """
    XH_tr,  XV_tr,  y_tr  = _load_amp(processed_dir, 'train')
    XH_val, XV_val, y_val = _load_amp(processed_dir, 'val')
    XH_te,  XV_te,  y_te  = _load_amp(processed_dir, 'test')
    kw = _kw(num_workers)
    train_loader = DataLoader(
        XRF55AmpHaarDatasetFromArrays(XH_tr,  XV_tr,  y_tr),
        batch_size=batch_size, shuffle=True, **kw)
    val_loader   = DataLoader(
        XRF55AmpHaarDatasetFromArrays(XH_val, XV_val, y_val),
        batch_size=64, shuffle=False, **kw)
    test_loader  = DataLoader(
        XRF55AmpHaarDatasetFromArrays(XH_te,  XV_te,  y_te),
        batch_size=64, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ── LOSO-5fold loaders ────────────────────────────────────────────────────────

def build_loso_loaders_with_val(fold_idx: int, processed_dir,
                                batch_size: int = 32, num_workers: int = 4):
    """LOSO rotation val split: test=G[i], val=G[(i+1)%5], train=3 remaining groups.

    All subjects use ALL 20 reps — iterates train/val/test split files.
    Returns (train_loader, val_loader, test_loader).
    """
    if not 0 <= fold_idx < len(LOSO_FOLD_SUBJECTS):
        raise ValueError(
            f"fold_idx={fold_idx} out of range [0, {len(LOSO_FOLD_SUBJECTS) - 1}]")
    test_subjects  = set(LOSO_FOLD_SUBJECTS[fold_idx])
    val_subjects   = set(get_loso_val_subjects(fold_idx))
    train_subjects = set(range(1, 31)) - test_subjects - val_subjects

    train_XH, train_XV, train_y = [], [], []
    val_XH,   val_XV,   val_y   = [], [], []
    test_XH,  test_XV,  test_y  = [], [], []

    for split in ('train', 'val', 'test'):
        XH, XV, y = _load_amp(processed_dir, split)
        vol_ids   = get_vol_ids(split)
        tr_mask  = np.isin(vol_ids, list(train_subjects))
        v_mask   = np.isin(vol_ids, list(val_subjects))
        te_mask  = np.isin(vol_ids, list(test_subjects))
        if tr_mask.any():
            train_XH.append(XH[tr_mask]); train_XV.append(XV[tr_mask]); train_y.append(y[tr_mask])
        if v_mask.any():
            val_XH.append(XH[v_mask]);    val_XV.append(XV[v_mask]);    val_y.append(y[v_mask])
        if te_mask.any():
            test_XH.append(XH[te_mask]);  test_XV.append(XV[te_mask]);  test_y.append(y[te_mask])
        del XH, XV, y

    kw = _kw(num_workers)
    train_loader = DataLoader(
        XRF55AmpHaarDatasetFromArrays(
            np.concatenate(train_XH), np.concatenate(train_XV), np.concatenate(train_y)),
        batch_size=batch_size, shuffle=True, **kw)
    val_loader   = DataLoader(
        XRF55AmpHaarDatasetFromArrays(
            np.concatenate(val_XH), np.concatenate(val_XV), np.concatenate(val_y)),
        batch_size=64, shuffle=False, **kw)
    test_loader  = DataLoader(
        XRF55AmpHaarDatasetFromArrays(
            np.concatenate(test_XH), np.concatenate(test_XV), np.concatenate(test_y)),
        batch_size=64, shuffle=False, **kw)
    return train_loader, val_loader, test_loader
