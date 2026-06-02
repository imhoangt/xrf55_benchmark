import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from baselines.apwmamba.config import (
    NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS, PREFETCH_FACTOR,
    DATA_ROOT,
)
from src.data.loso_splits import (
    get_vol_ids, get_loso_val_subjects, LOSO_FOLD_SUBJECTS,
)


class XRF55DatasetFromArrays(Dataset):
    """Dataset constructed from pre-sliced numpy arrays (for noval/LOSO splits)."""

    def __init__(self, X_amp, X_phase, y):
        assert X_amp.shape[1:] == (27, 500, 15), f'Bad amp shape: {X_amp.shape}'
        assert X_phase.shape[1:] == (18, 500, 15), f'Bad phase shape: {X_phase.shape}'
        assert X_amp.dtype   == np.float32, f'X_amp must be float32, got {X_amp.dtype}'
        assert X_phase.dtype == np.float32, f'X_phase must be float32, got {X_phase.dtype}'
        self.X_amp   = X_amp
        self.X_phase = X_phase
        self.y       = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        Xa = torch.from_numpy(self.X_amp[idx])
        Xp = torch.from_numpy(self.X_phase[idx])
        y  = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return Xa, Xp, y


def _loader_kwargs(shuffle=False):
    kwargs = dict(
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0 and PERSISTENT_WORKERS),
    )
    if NUM_WORKERS > 0:
        kwargs['prefetch_factor'] = PREFETCH_FACTOR
    return kwargs


def _load_split(data_root, split):
    """Load cached split arrays. X arrays are forced to float32 to avoid silent
    float64 upcasts on GPU (2× memory waste)."""
    d = Path(data_root)
    return (
        np.load(d / f'X_amp_dwt_{split}.npy').astype(np.float32, copy=False),
        np.load(d / f'X_phase_dwt_{split}.npy').astype(np.float32, copy=False),
        np.load(d / f'y_{split}.npy'),
    )


# ── Split-protocol loaders (pre-split files) ─────────────────────────────────

def build_noval_loaders_with_val(batch_size, data_root=None):
    """Train=reps 1-12, Val=reps 13-14, Test=reps 15-20 (loaded from pre-split files).

    Returns (train_loader, val_loader, test_loader).
    """
    if data_root is None:
        data_root = DATA_ROOT

    Xa_tr,  Xp_tr,  y_tr  = _load_split(data_root, 'train')
    Xa_val, Xp_val, y_val = _load_split(data_root, 'val')
    Xa_te,  Xp_te,  y_te  = _load_split(data_root, 'test')

    train_ds = XRF55DatasetFromArrays(Xa_tr,  Xp_tr,  y_tr)
    val_ds   = XRF55DatasetFromArrays(Xa_val, Xp_val, y_val)
    test_ds  = XRF55DatasetFromArrays(Xa_te,  Xp_te,  y_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              drop_last=False, **_loader_kwargs(shuffle=True))
    val_loader   = DataLoader(val_ds,   batch_size=64,
                              drop_last=False, **_loader_kwargs(shuffle=False))
    test_loader  = DataLoader(test_ds,  batch_size=64,
                              drop_last=False, **_loader_kwargs(shuffle=False))
    return train_loader, val_loader, test_loader


# ── LOSO-5fold loaders ────────────────────────────────────────────────────────

def build_loso_loaders_with_val(fold_idx, batch_size, data_root=None):
    """LOSO rotation val split: test=G[i], val=G[(i+1)%5], train=3 remaining groups.

    All subjects use ALL 20 reps — iterates train/val/test split files.
    Returns (train_loader, val_loader, test_loader).
    """
    if data_root is None:
        data_root = DATA_ROOT

    if not 0 <= fold_idx < len(LOSO_FOLD_SUBJECTS):
        raise ValueError(
            f"fold_idx={fold_idx} out of range [0, {len(LOSO_FOLD_SUBJECTS) - 1}]")

    test_subjects  = set(LOSO_FOLD_SUBJECTS[fold_idx])
    val_subjects   = set(get_loso_val_subjects(fold_idx))
    train_subjects = set(range(1, 31)) - test_subjects - val_subjects

    train_Xa, train_Xp, train_y = [], [], []
    val_Xa,   val_Xp,   val_y   = [], [], []
    test_Xa,  test_Xp,  test_y  = [], [], []

    for split in ('train', 'val', 'test'):
        Xa, Xp, y = _load_split(data_root, split)
        vol_ids   = get_vol_ids(split)

        tr_mask  = np.isin(vol_ids, list(train_subjects))
        v_mask   = np.isin(vol_ids, list(val_subjects))
        te_mask  = np.isin(vol_ids, list(test_subjects))

        if tr_mask.any():
            train_Xa.append(Xa[tr_mask]); train_Xp.append(Xp[tr_mask]); train_y.append(y[tr_mask])
        if v_mask.any():
            val_Xa.append(Xa[v_mask]);    val_Xp.append(Xp[v_mask]);    val_y.append(y[v_mask])
        if te_mask.any():
            test_Xa.append(Xa[te_mask]);  test_Xp.append(Xp[te_mask]);  test_y.append(y[te_mask])

        del Xa, Xp, y

    train_ds = XRF55DatasetFromArrays(
        np.concatenate(train_Xa), np.concatenate(train_Xp), np.concatenate(train_y))
    val_ds   = XRF55DatasetFromArrays(
        np.concatenate(val_Xa),   np.concatenate(val_Xp),   np.concatenate(val_y))
    test_ds  = XRF55DatasetFromArrays(
        np.concatenate(test_Xa),  np.concatenate(test_Xp),  np.concatenate(test_y))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              drop_last=False, **_loader_kwargs(shuffle=True))
    val_loader   = DataLoader(val_ds,   batch_size=64,
                              drop_last=False, **_loader_kwargs(shuffle=False))
    test_loader  = DataLoader(test_ds,  batch_size=64,
                              drop_last=False, **_loader_kwargs(shuffle=False))
    return train_loader, val_loader, test_loader
