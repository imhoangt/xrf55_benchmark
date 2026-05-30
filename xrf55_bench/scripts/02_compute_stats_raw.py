"""Compute normalization statistics for XRF55 benchmark.

Streams all 6600 amplitude_npy_4d files one-by-one (peak RAM < 15 MB).
Stats are fitted on ALL reps 1-20 (train + test combined).

Stats computed per model family:
  resnet   → mean, std                  : (270,)   (per channel, over N×T=1000)
  tfmamba  → xh_mean/std, xv_mean/std  : (135,)   (per Haar-DWT channel, over N×T=500)
  wavmamba → mean, std                  : (27,)    (per db4-DWT channel, over N×T×F2=500×15)

DWT inputs are float32 (matches arrays saved by 02b_save_arrays.py).
Accumulators are float64 to avoid numerical drift over 6600×large_products.

Output: dataset/XRF55/bench/raw/stats.json

Run this before 03_save_arrays_raw.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pywt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.parser import ACTION_ID_TO_LABEL, ACTION_IDS_USED
from src.data.preprocessing.dwt import apply_dwt2_stack

AMP4D_DIR  = PROJECT_ROOT / 'dataset' / 'XRF55' / 'amplitude_npy_4d'
BENCH_DIR  = PROJECT_ROOT / 'dataset' / 'XRF55' / 'bench' / 'raw'
STATS_PATH = BENCH_DIR / 'stats.json'

TRAIN_REPS = list(range(1, 15))    # reps 1-14  → 4620 samples
TEST_REPS  = list(range(15, 21))   # reps 15-20 → 1980 samples


def _to_flat270(raw4d: np.ndarray) -> np.ndarray:
    """(1000, 3, 3, 30) float64 → (270, 1000) float32.

    Layout: row i encodes (dev*ant pair i//30, subcarrier i%30).
    Matches 01_preprocess.py: (N,9,1000,30).transpose(0,1,3,2).reshape(N,270,1000).
    """
    return (raw4d
            .transpose(1, 2, 0, 3)   # (3, 3, 1000, 30)
            .reshape(9, 1000, 30)     # (9, 1000, 30)  — copy (non-contiguous input)
            .transpose(0, 2, 1)       # (9, 30, 1000)
            .reshape(270, 1000)       # (270, 1000)     — copy
            .astype(np.float32))


def _to_x9(raw4d: np.ndarray) -> np.ndarray:
    """(1000, 3, 3, 30) float64 → (9, 1000, 30) float32."""
    return (raw4d
            .transpose(1, 2, 0, 3)   # (3, 3, 1000, 30)
            .reshape(9, 1000, 30)     # (9, 1000, 30)
            .astype(np.float32))


def _collect_all_files() -> list:
    """Sorted list of all 6600 expected file paths (vol × action × rep)."""
    files = []
    for vol_id in range(1, 31):
        for action_id in ACTION_IDS_USED:
            for rep_id in range(1, 21):
                p = (AMP4D_DIR / f'{vol_id:02d}'
                     / f'{vol_id:02d}_{action_id:02d}_{rep_id:02d}.npy')
                files.append(p)
    return files


if __name__ == '__main__':
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--bench-dir', type=str, default=None,
                         help='Override bench_processed output dir')
    _args = _parser.parse_args()
    if _args.bench_dir:
        BENCH_DIR  = Path(_args.bench_dir)
        STATS_PATH = BENCH_DIR / 'stats.json'

    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    # ── Verify all files present before starting ───────────────────────────
    print('Scanning files...')
    all_files = _collect_all_files()
    missing   = [p for p in all_files if not p.exists()]
    if missing:
        for p in missing[:5]:
            print(f'  MISSING: {p}')
        if len(missing) > 5:
            print(f'  ... and {len(missing) - 5} more')
        raise FileNotFoundError(
            f'{len(missing)} files missing in {AMP4D_DIR}')
    print(f'  All {len(all_files)} files found.\n')

    # ── Accumulators (float64 throughout) ─────────────────────────────────
    # ResNet: per-channel (270,) — each channel accumulates N×1000 values
    res_n      = np.int64(0)
    res_sum    = np.zeros(270, dtype=np.float64)
    res_sum_sq = np.zeros(270, dtype=np.float64)

    # TF-Mamba Haar DWT: per-channel (135,)
    # Each file contributes 500 values per channel → haar_n = 6600 × 500
    C_HAAR     = 135
    xh_sum     = np.zeros(C_HAAR, dtype=np.float64)
    xh_sum_sq  = np.zeros(C_HAAR, dtype=np.float64)
    xv_sum     = np.zeros(C_HAAR, dtype=np.float64)
    xv_sum_sq  = np.zeros(C_HAAR, dtype=np.float64)
    haar_n     = np.int64(0)

    # WavMamba db4 DWT: per-channel (27,)
    # Each file contributes 500×15=7500 values per channel → wav_n = 6600 × 7500
    C_WAV      = 27
    wav_sum    = np.zeros(C_WAV, dtype=np.float64)
    wav_sum_sq = np.zeros(C_WAV, dtype=np.float64)
    wav_n      = np.int64(0)

    # ── Stream all 6600 files ─────────────────────────────────────────────
    for fpath in tqdm(all_files, desc='Computing stats', unit='file'):
        raw = np.load(fpath)   # (1000, 3, 3, 30) float64

        # ── ResNet: per-channel (270,) ────────────────────────────────────
        flat = _to_flat270(raw)                          # (270, 1000) float32
        v    = flat.astype(np.float64)                   # (270, 1000) float64
        res_n      += np.int64(1000)
        res_sum    += v.sum(axis=1)                      # (270,)
        res_sum_sq += (v * v).sum(axis=1)                # (270,)

        # ── TF-Mamba: Haar DWT → per-channel (135,) ──────────────────────
        _, (cH, cV, _) = pywt.dwt2(flat, 'haar', mode='periodization')
        # cH, cV: (135, 500) float32  (pywt preserves input dtype)
        cH64 = cH.astype(np.float64)
        cV64 = cV.astype(np.float64)
        xh_sum    += cH64.sum(axis=1)           # sum over T=500  → (135,)
        xh_sum_sq += (cH64 * cH64).sum(axis=1)
        xv_sum    += cV64.sum(axis=1)
        xv_sum_sq += (cV64 * cV64).sum(axis=1)
        haar_n    += np.int64(cH.shape[1])      # += 500

        # ── WavMamba: db4 DWT → per-channel (27,) ────────────────────────
        x9    = _to_x9(raw)                              # (9, 1000, 30) float32
        x_dwt = apply_dwt2_stack(x9[None])               # (1, 27, 500, 15) float32
        d64   = x_dwt[0].astype(np.float64)              # (27, 500, 15) float64
        wav_sum    += d64.sum(axis=(1, 2))      # sum over T×F2=7500  → (27,)
        wav_sum_sq += (d64 * d64).sum(axis=(1, 2))
        wav_n      += np.int64(d64.shape[1] * d64.shape[2])  # += 7500

    # ── Finalize: E[X²] - E[X]² ──────────────────────────────────────────
    def _vector(n, s, s2):
        mean = s / n
        var  = np.maximum(s2 / n - mean * mean, 0.0)
        std  = np.maximum(np.sqrt(var), 1e-6)
        return mean.tolist(), std.tolist()

    resnet_mean, resnet_std = _vector(res_n, res_sum, res_sum_sq)
    xh_mean,  xh_std        = _vector(haar_n, xh_sum,  xh_sum_sq)
    xv_mean,  xv_std        = _vector(haar_n, xv_sum,  xv_sum_sq)
    wav_mean, wav_std       = _vector(wav_n,  wav_sum, wav_sum_sq)

    stats = {
        'meta': {
            'source':     'amplitude_npy_4d',
            'n_files':    len(all_files),
            'fitted_on':  'all_reps_1_to_20',
            'train_reps': TRAIN_REPS,
            'test_reps':  TEST_REPS,
        },
        'resnet': {
            'mean': resnet_mean,
            'std':  resnet_std,
        },
        'tfmamba': {
            'xh_mean': xh_mean,
            'xh_std':  xh_std,
            'xv_mean': xv_mean,
            'xv_std':  xv_std,
        },
        'wavmamba': {
            'mean': wav_mean,
            'std':  wav_std,
        },
    }

    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\nStats saved → {STATS_PATH}')
    print(f'  resnet  : mean[0]={resnet_mean[0]:.5f}  std[0]={resnet_std[0]:.5f}  (270 channels)')
    print(f'  tfmamba : xh_mean[0]={xh_mean[0]:.5f}  xh_std[0]={xh_std[0]:.5f}  (135 channels)')
    print(f'  wavmamba: mean[0]={wav_mean[0]:.5f}  std[0]={wav_std[0]:.5f}  (27 channels)')
    print(f'  counts/ch: n_resnet={int(res_n):,}  n_haar={int(haar_n):,}  n_wav={int(wav_n):,}')
    print('\nNext step: python xrf55_bench/scripts/03_save_arrays_raw.py')
