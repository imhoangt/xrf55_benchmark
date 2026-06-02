"""Preprocessing script: raw .dat/.mat -> cached .npy + normalization_stats.json

Splits (pre-split at preprocessing time):
  train = reps  1-12 (3960 samples)
  val   = reps 13-14  ( 660 samples)
  test  = reps 15-20 (1980 samples)

Normalization strategy: per-channel z-score, fit on reps 1-12 only
(val reps 13-14 excluded to prevent leakage into normalization stats).
  APWMamba (db4):        fit on reps 1-12, C=27 amp / C=18 phase
  TF-Mamba XRF55 (Haar): fit on reps 1-12, C=135 for XH/XV amp, C=90 for XH/XV phase
  ResNet1D (raw):        fit on reps 1-12, C=270 amp / C=180 phase (pre-DWT, same transpose+reshape as Haar)
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pywt
from tqdm import tqdm

from src.data.preprocessing.parser import (
    load_xrf55_sample, ACTION_ID_TO_LABEL, ACTION_IDS_USED,
)
from src.data.preprocessing.amplitude import (
    extract_amplitude_raw, fit_amplitude_stats, apply_amplitude_stats,
)
from src.data.preprocessing.phase import (
    extract_phase_raw, fit_phase_stats, apply_phase_stats,
)
from src.data.preprocessing.dwt import apply_dwt2_stack
from src.data.preprocessing.normalizer import (
    save_normalization_stats_apwmamba, save_normalization_stats_tfmamba,
)
from src.data.splits import TRAINVAL_REPS, VAL_REPS, TEST_REPS
from src.data.verify_ids import scan_action_ids

RAW_DIR       = PROJECT_ROOT / 'dataset' / 'xrf55' / 'raw' / 'scene_01'
PROCESSED_DIR = PROJECT_ROOT / 'dataset' / 'xrf55' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

_OLD_FILES = [
    'X_amp_pre_haar_train.npy',
    'normalization_stats.json',
]
_OLD_GLOBS = [
    'X_amp_raw_*.npy',    # stale ResNet1D raw amp from older schema
    'X_phase_raw_*.npy',  # stale ResNet1D raw phase from older schema
]


def split_id(rep):
    return 0 if rep in TRAINVAL_REPS else 1


def _fit_haar_stats(X_train):
    """Fit per-channel z-score on Haar subband (N, C, T). Returns (mean, std) each (C,)."""
    mean = X_train.mean(axis=(0, 2))                       # (C,)
    std  = np.maximum(X_train.std(axis=(0, 2)), 1e-6)     # (C,)
    return mean, std


def _apply_haar_stats(X, mean, std):
    """Apply per-channel z-score. X: (N, C, T)."""
    return ((X - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


if __name__ == '__main__':
    # Remove stale files from previous preprocessing runs
    for _fname in _OLD_FILES:
        _p = PROCESSED_DIR / _fname
        if _p.exists():
            _p.unlink()
            print(f'  Removed stale file: {_fname}')
    for _pat in _OLD_GLOBS:
        for _p in PROCESSED_DIR.glob(_pat):
            _p.unlink()
            print(f'  Removed stale file: {_p.name}')

    # Step 0: Verify dataset structure
    scan_action_ids(RAW_DIR)

    # Step 1: Load all raw + extract amplitude/phase features
    X_amp_all   = []
    X_phase_all = []
    labels_all  = []
    splits_all  = []
    ids_all     = []

    n_total = 30 * 11 * 20
    pbar = tqdm(total=n_total, desc='Loading + extracting')

    for vol_id in range(1, 31):
        for action_id in ACTION_IDS_USED:
            label = ACTION_ID_TO_LABEL[action_id]
            for rep_id in range(1, 21):
                H = load_xrf55_sample(RAW_DIR, vol_id, action_id, rep_id)
                X_amp_all.append(extract_amplitude_raw(H))
                X_phase_all.append(extract_phase_raw(H))
                labels_all.append(label)
                splits_all.append(split_id(rep_id))
                ids_all.append((vol_id, action_id, rep_id))
                pbar.update(1)
    pbar.close()

    X_amp_all   = np.stack(X_amp_all)    # (6600, 9, 1000, 30)
    X_phase_all = np.stack(X_phase_all)  # (6600, 6, 1000, 30)
    labels_all  = np.array(labels_all)
    splits_all  = np.array(splits_all)
    ids_all     = np.array(ids_all, dtype=np.int16)

    assert X_amp_all.shape   == (6600, 9, 1000, 30)
    assert X_phase_all.shape == (6600, 6, 1000, 30)
    assert (splits_all == 0).sum() == 4620
    assert (splits_all == 1).sum() == 1980

    N = X_amp_all.shape[0]
    train_mask = (splits_all == 0)

    # Split masks: train=reps 1-12, val=reps 13-14
    # norm_train_mask covers reps 1-12 of ALL 30 subjects — normalization stats are global,
    # not refit per LOSO fold. This matches common practice in CSI-HAR literature.
    norm_train_mask = train_mask & ~np.isin(ids_all[:, 2], VAL_REPS)
    val_mask        = train_mask &  np.isin(ids_all[:, 2], VAL_REPS)
    test_mask       = (splits_all == 1)
    assert norm_train_mask.sum() == 30 * 11 * 12, \
        f"Expected 3960 train samples, got {norm_train_mask.sum()}"
    assert val_mask.sum()  == 30 * 11 * 2,  f"Expected 660 val samples, got {val_mask.sum()}"
    assert test_mask.sum() == 30 * 11 * 6, f"Expected 1980 test samples, got {test_mask.sum()}"

    # ── Step 2: Haar DWT for TF-Mamba XRF55 baselines ────────────────────────
    # (N,9,1000,30) → transpose(0,1,3,2) → (N,9,30,1000) → reshape → (N,270,1000)
    # Row c*30+f = time series for (channel c, subcarrier f) — clean (C×F, T) layout.
    # pywt.dwt2 on (270,1000) plane → cH,cV: (N,135,500)
    # Phase: same pattern with C=6 → 180.
    print("Applying Haar DWT to amplitude for TF-Mamba XRF55...")
    _, (amp_cH, amp_cV, _) = pywt.dwt2(
        X_amp_all.transpose(0, 1, 3, 2).reshape(N, 9 * 30, 1000), 'haar', mode='periodization')
    # amp_cH, amp_cV: (N, 135, 500)

    print("Applying Haar DWT to phase for TF-Mamba XRF55...")
    _, (phase_cH, phase_cV, _) = pywt.dwt2(
        X_phase_all.transpose(0, 1, 3, 2).reshape(N, 6 * 30, 1000), 'haar', mode='periodization')
    # phase_cH, phase_cV: (N, 90, 500)

    # Step 3: Fit per-channel Haar stats on reps 1-12 only (val reps 13-14 excluded)
    amp_xh_mean,   amp_xh_std   = _fit_haar_stats(amp_cH[norm_train_mask])
    amp_xv_mean,   amp_xv_std   = _fit_haar_stats(amp_cV[norm_train_mask])
    phase_xh_mean, phase_xh_std = _fit_haar_stats(phase_cH[norm_train_mask])
    phase_xv_mean, phase_xv_std = _fit_haar_stats(phase_cV[norm_train_mask])

    amp_xh_stats   = {'mean': amp_xh_mean.tolist(),   'std': amp_xh_std.tolist()}
    amp_xv_stats   = {'mean': amp_xv_mean.tolist(),   'std': amp_xv_std.tolist()}
    phase_xh_stats = {'mean': phase_xh_mean.tolist(), 'std': phase_xh_std.tolist()}
    phase_xv_stats = {'mean': phase_xv_mean.tolist(), 'std': phase_xv_std.tolist()}

    # Step 4: Apply Haar normalization
    amp_cH   = _apply_haar_stats(amp_cH,   amp_xh_mean,   amp_xh_std)
    amp_cV   = _apply_haar_stats(amp_cV,   amp_xv_mean,   amp_xv_std)
    phase_cH = _apply_haar_stats(phase_cH, phase_xh_mean, phase_xh_std)
    phase_cV = _apply_haar_stats(phase_cV, phase_xv_mean, phase_xv_std)

    # Step 5: Save visualization sample (vol=01, action=35 Running, rep=11)
    H_viz    = load_xrf55_sample(RAW_DIR, 1, 35, 11)
    raw_amp   = np.abs(H_viz[:, :, 0, 0]).astype(np.float32)    # (1000, 30)
    raw_phase = np.angle(H_viz[:, :, 0, 0]).astype(np.float32)  # (1000, 30)
    filt_amp  = extract_amplitude_raw(H_viz)[0].astype(np.float32)  # (1000, 30)
    filt_phase = extract_phase_raw(H_viz)[0].astype(np.float32)     # (1000, 30)
    amp_cA_viz,   (amp_cH_viz,   amp_cV_viz,   _) = pywt.dwt2(
        filt_amp,   wavelet='db4', mode='periodization')
    phase_cA_viz, (phase_cH_viz, phase_cV_viz, _) = pywt.dwt2(
        filt_phase, wavelet='db4', mode='periodization')
    np.savez(
        PROCESSED_DIR / 'visualization_sample_running_vol01_rep11.npz',
        raw_amplitude=raw_amp,
        filtered_amplitude=filt_amp,
        processed_amplitude=filt_amp,   # pre-DWT = filtered (norm applied after DWT)
        raw_phase=raw_phase,
        filtered_phase=filt_phase,
        processed_phase=filt_phase,
        amp_dwt_cA=amp_cA_viz.astype(np.float32),
        amp_dwt_cH=amp_cH_viz.astype(np.float32),
        amp_dwt_cV=amp_cV_viz.astype(np.float32),
        phase_dwt_cA=phase_cA_viz.astype(np.float32),
        phase_dwt_cH=phase_cH_viz.astype(np.float32),
        phase_dwt_cV=phase_cV_viz.astype(np.float32),
        vol_id=np.array(1, dtype=np.int16),
        action_id=np.array(35, dtype=np.int16),
        rep_id=np.array(11, dtype=np.int16),
    )
    print("Visualization sample saved")

    # Step 6: Save TF-Mamba XRF55 Haar split files (train / val / test)
    # XH: (N, 135/90, 500) → save transposed as (N, 500, 135/90) to match model input
    # XV: (N, 135/90, 500) → save as-is
    print("Saving TF-Mamba XRF55 Haar split files...")
    for mask, split_name in [
        (norm_train_mask, 'train'),
        (val_mask,        'val'),
        (test_mask,       'test'),
    ]:
        np.save(PROCESSED_DIR / f'X_amp_xh_{split_name}.npy',
                amp_cH[mask].transpose(0, 2, 1))       # (N, 500, 135)
        np.save(PROCESSED_DIR / f'X_amp_xv_{split_name}.npy',
                amp_cV[mask])                            # (N, 135, 500)
        np.save(PROCESSED_DIR / f'X_phase_xh_{split_name}.npy',
                phase_cH[mask].transpose(0, 2, 1))      # (N, 500, 90)
        np.save(PROCESSED_DIR / f'X_phase_xv_{split_name}.npy',
                phase_cV[mask])                          # (N, 90, 500)
        print(f"  Saved Haar {split_name}: {mask.sum()} samples")

    del amp_cH, amp_cV, phase_cH, phase_cV

    # ── Step 7: Raw (pre-DWT) normalized data for ResNet1D ───────────────────
    # (N,9,1000,30) → transpose(0,1,3,2) → (N,9,30,1000) → reshape → (N,270,1000)
    # Same layout as Haar DWT above: row c*30+f = time series for (channel c, subcarrier f).
    # Per-channel z-score fitted on reps 1-12 only (norm_train_mask).
    print("Fitting + saving raw pre-DWT data for ResNet1D...")

    amp_tr_flat   = X_amp_all[norm_train_mask].transpose(0, 1, 3, 2).reshape(int(norm_train_mask.sum()), 270, 1000)
    amp_raw_mean  = amp_tr_flat.mean(axis=(0, 2))                          # (270,)
    amp_raw_std   = np.maximum(amp_tr_flat.std(axis=(0, 2)), 1e-6)
    del amp_tr_flat

    phase_tr_flat  = X_phase_all[norm_train_mask].transpose(0, 1, 3, 2).reshape(int(norm_train_mask.sum()), 180, 1000)
    phase_raw_mean = phase_tr_flat.mean(axis=(0, 2))                       # (180,)
    phase_raw_std  = np.maximum(phase_tr_flat.std(axis=(0, 2)), 1e-6)
    del phase_tr_flat

    for mask, split_name in [
        (norm_train_mask, 'train'),
        (val_mask,        'val'),
        (test_mask,       'test'),
    ]:
        n = int(mask.sum())
        amp_chunk = ((X_amp_all[mask].transpose(0, 1, 3, 2).reshape(n, 270, 1000)
                      - amp_raw_mean[None, :, None]) / amp_raw_std[None, :, None]).astype(np.float32)
        np.save(PROCESSED_DIR / f'X_amp_raw_{split_name}.npy',   amp_chunk)
        del amp_chunk

        phase_chunk = ((X_phase_all[mask].transpose(0, 1, 3, 2).reshape(n, 180, 1000)
                        - phase_raw_mean[None, :, None]) / phase_raw_std[None, :, None]).astype(np.float32)
        np.save(PROCESSED_DIR / f'X_phase_raw_{split_name}.npy', phase_chunk)
        del phase_chunk
        print(f"  Saved raw {split_name}: {n} samples")

    _resnet1d_stats = {
        'amp_raw':   {'mean': amp_raw_mean.tolist(),   'std': amp_raw_std.tolist()},
        'phase_raw': {'mean': phase_raw_mean.tolist(), 'std': phase_raw_std.tolist()},
    }
    with open(PROCESSED_DIR / 'normalization_stats_resnet1d.json', 'w') as _f:
        json.dump(_resnet1d_stats, _f, indent=2)
    del amp_raw_mean, amp_raw_std, phase_raw_mean, phase_raw_std

    # ── Step 8: db4 DWT for APWMamba ─────────────────────────────────────────
    print("Applying db4 DWT to amplitude for APWMamba...")
    X_amp_dwt   = apply_dwt2_stack(X_amp_all)    # (N, 27, 500, 15)
    del X_amp_all

    print("Applying db4 DWT to phase for APWMamba...")
    X_phase_dwt = apply_dwt2_stack(X_phase_all)  # (N, 18, 500, 15)
    del X_phase_all

    # Step 9: Fit per-channel stats on reps 1-12 only (27 amp, 18 phase)
    amp_stats   = fit_amplitude_stats(X_amp_dwt[norm_train_mask])
    phase_stats = fit_phase_stats(X_phase_dwt[norm_train_mask])

    # Step 10: Apply per-channel normalization
    X_amp_dwt   = apply_amplitude_stats(X_amp_dwt,   amp_stats)
    X_phase_dwt = apply_phase_stats(X_phase_dwt, phase_stats)

    # Step 11: Save APWMamba split files + labels (train / val / test)
    print("Saving APWMamba db4-DWT split files...")
    for mask, split_name in [
        (norm_train_mask, 'train'),
        (val_mask,        'val'),
        (test_mask,       'test'),
    ]:
        np.save(PROCESSED_DIR / f'X_amp_dwt_{split_name}.npy',   X_amp_dwt[mask])
        np.save(PROCESSED_DIR / f'X_phase_dwt_{split_name}.npy', X_phase_dwt[mask])
        np.save(PROCESSED_DIR / f'y_{split_name}.npy',           labels_all[mask])
        print(f"  Saved db4 {split_name}: {mask.sum()} samples")

    del X_amp_dwt, X_phase_dwt

    # Step 12: Save normalization stats (three files: one per model family)
    save_normalization_stats_apwmamba(
        amp_stats, phase_stats,
        PROCESSED_DIR / 'normalization_stats_apwmamba.json',
    )
    save_normalization_stats_tfmamba(
        amp_xh_stats, amp_xv_stats,
        phase_xh_stats, phase_xv_stats,
        PROCESSED_DIR / 'normalization_stats_tfmamba.json',
    )

    print(f"\nPreprocessing complete. Cache in: {PROCESSED_DIR}")
    print("Splits: train=3960 (reps 1-12), val=660 (reps 13-14), test=1980 (reps 15-20)")
    print("APWMamba:             X_amp_dwt_{{train,val,test}}.npy (N,27,500,15), X_phase_dwt_*.npy (N,18,500,15)")
    print("TF-Mamba XRF55 amp:   X_amp_xh_*.npy (N,500,135), X_amp_xv_*.npy (N,135,500)")
    print("TF-Mamba XRF55 phase: X_phase_xh_*.npy (N,500,90), X_phase_xv_*.npy (N,90,500)")
    print("ResNet1D amp:         X_amp_raw_*.npy (N,270,1000)")
    print("ResNet1D phase:       X_phase_raw_*.npy (N,180,1000)")
    print("Labels:               y_{{train,val,test}}.npy (shared across all model families)")
