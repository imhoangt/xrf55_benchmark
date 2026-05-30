"""Preprocessing + stats for XRF55 benchmark — processed (Hampel+LPF) source.

Applies Hampel filter + Low-Pass Filter to raw amplitude (from amplitude_npy_4d/),
then computes model-specific transforms and saves UN-NORMALIZED arrays.

Filtering pipeline (per file, same params as extract_amplitude_raw in amplitude.py):
  amplitude_npy_4d file : (1000,3,3,30) float64
  → .transpose(1,2,0,3).reshape(9,1000,30).astype(float32)   → (9,1000,30)
  → hampel_vectorized(window=11, n_sigma=3.0)  along axis=1   → (9,1000,30)
  → sosfiltfilt(butter(4,20Hz,fs=200Hz))       along axis=1   → (9,1000,30)

Transforms per model (from filtered amp (9,1000,30)):
  resnet   : .transpose(0,2,1).reshape(270,1000)            → (270,1000)
  tfmamba  : flat270 → pywt.dwt2('haar') → cH.T, cV         → XH(500,135), XV(135,500)
  wavmamba : apply_dwt2_stack(amp[None])[0]                  → (27,500,15)

Pass 1 — stats (all 6600 files, accumulators float64):
  fit on all reps 1-20 (same as 02_compute_stats_raw.py)
  output: bench/processed/stats.json  [same format as bench/raw/stats.json]

Pass 2 — arrays (train reps 1-14, test reps 15-20):
  save UN-NORMALIZED float32 arrays to bench/processed/

Output layout:
  bench/processed/
    stats.json
    y_train.npy              (4620,)           int64
    y_test.npy               (1980,)           int64
    resnet/
      X_train.npy            (4620, 270, 1000) float32
      X_test.npy             (1980, 270, 1000) float32
    tfmamba/
      X_train_xh.npy         (4620, 500, 135)  float32
      X_train_xv.npy         (4620, 135, 500)  float32
      X_test_xh.npy          (1980, 500, 135)  float32
      X_test_xv.npy          (1980, 135, 500)  float32
    wavmamba/
      X_train.npy            (4620, 27, 500, 15) float32
      X_test.npy             (1980, 27, 500, 15) float32

Peak RAM pass 2: ~11.2 GB (train split pre-allocated).

Usage:
    cd har_csi
    python xrf55_bench/scripts/04_preprocess_processed.py
    python xrf55_bench/scripts/04_preprocess_processed.py --amp4d-dir E:/amplitude_npy_4d
    python xrf55_bench/scripts/04_preprocess_processed.py --bench-dir E:/bench/processed
"""
import json
import sys
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.amplitude import hampel_vectorized
from src.data.preprocessing.dwt       import apply_dwt2_stack
from src.data.preprocessing.parser    import ACTION_ID_TO_LABEL, ACTION_IDS_USED

AMP4D_DIR  = PROJECT_ROOT / 'dataset' / 'XRF55' / 'amplitude_npy_4d'
BENCH_DIR  = PROJECT_ROOT / 'dataset' / 'XRF55' / 'bench' / 'processed'

TRAIN_REPS = list(range(1, 15))    # reps 1-14  → 4620 samples
TEST_REPS  = list(range(15, 21))   # reps 15-20 → 1980 samples

# LPF coefficients created once at module load (butter is deterministic)
_SOS = butter(4, 20.0, btype='low', fs=200.0, output='sos')


# ── Core filter ───────────────────────────────────────────────────────────────

def _filter_amp(raw4d: np.ndarray) -> np.ndarray:
    """(1000,3,3,30) float64 → (9,1000,30) float32 — Hampel+LPF filtered.

    Axis layout after reshape:
      axis 0: 9 channel pairs  (3 devices × 3 antennas)
      axis 1: 1000 time steps  ← filter axis
      axis 2: 30 subcarriers
    """
    amp = raw4d.transpose(1, 2, 0, 3).reshape(9, 1000, 30).astype(np.float32)
    amp = hampel_vectorized(amp, window=11, n_sigma=3.0)
    amp = sosfiltfilt(_SOS, amp, axis=1).astype(np.float32)
    return amp


def _to_flat270(amp9: np.ndarray) -> np.ndarray:
    """(9,1000,30) float32 → (270,1000) float32.

    Row i = time series for channel pair (i//30) × subcarrier (i%30).
    Matches layout produced by _to_flat270 in 03_save_arrays_raw.py.
    """
    return amp9.transpose(0, 2, 1).reshape(270, 1000)


# ── File collection helpers ───────────────────────────────────────────────────

def _collect_all_files() -> list:
    """All 6600 file paths (vol × action × rep), sorted."""
    files = []
    for vol_id in range(1, 31):
        for action_id in ACTION_IDS_USED:
            for rep_id in range(1, 21):
                p = (AMP4D_DIR / f'{vol_id:02d}'
                     / f'{vol_id:02d}_{action_id:02d}_{rep_id:02d}.npy')
                files.append(p)
    return files


def _collect_split(rep_list: list) -> list:
    """(fpath, label) pairs for one split."""
    items = []
    for vol_id in range(1, 31):
        for action_id in ACTION_IDS_USED:
            label = ACTION_ID_TO_LABEL[action_id]
            for rep_id in rep_list:
                p = (AMP4D_DIR / f'{vol_id:02d}'
                     / f'{vol_id:02d}_{action_id:02d}_{rep_id:02d}.npy')
                items.append((p, label))
    return items


# ── Pass 2 helper ─────────────────────────────────────────────────────────────

def _process_split(split_name: str, rep_list: list, bench_dir: Path) -> None:
    samples = _collect_split(rep_list)
    n       = len(samples)
    print(f'\n[{split_name}]  {n} samples  (reps {rep_list[0]}-{rep_list[-1]})')

    missing = [p for p, _ in samples if not p.exists()]
    if missing:
        for p in missing[:5]:
            print(f'  MISSING: {p}')
        if len(missing) > 5:
            print(f'  ... and {len(missing) - 5} more')
        raise FileNotFoundError(f'{len(missing)} files missing in {AMP4D_DIR}')

    d_resnet  = bench_dir / 'resnet'
    d_tfmamba = bench_dir / 'tfmamba'
    d_wav     = bench_dir / 'wavmamba'

    # Memory-mapped output files: written to disk per sample, no large RAM allocation.
    y        = np.empty(n, dtype=np.int64)
    X_resnet = np.lib.format.open_memmap(
        str(d_resnet  / f'X_{split_name}.npy'),
        mode='w+', dtype=np.float32, shape=(n, 270, 1000))
    X_xh     = np.lib.format.open_memmap(
        str(d_tfmamba / f'X_{split_name}_xh.npy'),
        mode='w+', dtype=np.float32, shape=(n, 500, 135))
    X_xv     = np.lib.format.open_memmap(
        str(d_tfmamba / f'X_{split_name}_xv.npy'),
        mode='w+', dtype=np.float32, shape=(n, 135, 500))
    X_wav    = np.lib.format.open_memmap(
        str(d_wav     / f'X_{split_name}.npy'),
        mode='w+', dtype=np.float32, shape=(n, 27, 500, 15))

    total_gb = (X_resnet.nbytes + X_xh.nbytes + X_xv.nbytes + X_wav.nbytes) / 1e9
    print(f'  Output: {total_gb:.2f} GB (memory-mapped, written to disk per sample)')

    for i, (fpath, label) in enumerate(
            tqdm(samples, desc='  Filter+transform', unit='file')):
        raw4d = np.load(fpath)       # (1000, 3, 3, 30) float64
        amp   = _filter_amp(raw4d)   # (9, 1000, 30) float32

        # ResNet
        flat        = _to_flat270(amp)           # (270, 1000)
        X_resnet[i] = flat

        # TF-Mamba: Haar DWT on flat270
        _, (cH, cV, _) = pywt.dwt2(flat, 'haar', mode='periodization')
        # cH: (135, 500) → cH.T: (500, 135) = XH
        # cV: (135, 500) = XV
        X_xh[i] = cH.T
        X_xv[i] = cV

        # WavMamba: db4 DWT — apply_dwt2_stack expects (B,C,T=1000,F=30)
        X_wav[i] = apply_dwt2_stack(amp[None])[0]   # (27, 500, 15)

        y[i] = label

    np.save(bench_dir / f'y_{split_name}.npy', y)
    print(f'  y_{split_name}:      {y.shape}  labels {sorted(set(y.tolist()))}')
    print(f'  resnet:      ({n}, 270, 1000)  float32')
    print(f'  tfmamba xh:  ({n}, 500, 135)   float32')
    print(f'  tfmamba xv:  ({n}, 135, 500)   float32')
    print(f'  wavmamba:    ({n}, 27, 500, 15) float32')
    del X_resnet, X_xh, X_xv, X_wav   # flush memmaps


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    _parser = argparse.ArgumentParser(
        description='Preprocess XRF55 amplitude with Hampel+LPF for bench/processed/')
    _parser.add_argument('--amp4d-dir', type=str, default=None,
                         help='Path to amplitude_npy_4d/ (default: dataset/XRF55/amplitude_npy_4d)')
    _parser.add_argument('--bench-dir', type=str, default=None,
                         help='Output path for bench/processed/ (default: dataset/XRF55/bench/processed)')
    _args = _parser.parse_args()
    if _args.amp4d_dir:
        AMP4D_DIR = Path(_args.amp4d_dir)
    if _args.bench_dir:
        BENCH_DIR = Path(_args.bench_dir)

    for subdir in ['resnet', 'tfmamba', 'wavmamba']:
        (BENCH_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # ── Verify all files ──────────────────────────────────────────────────────
    print('Scanning amplitude_npy_4d files...')
    all_files = _collect_all_files()
    missing   = [p for p in all_files if not p.exists()]
    if missing:
        for p in missing[:5]:
            print(f'  MISSING: {p}')
        if len(missing) > 5:
            print(f'  ... and {len(missing) - 5} more')
        raise FileNotFoundError(f'{len(missing)} files missing in {AMP4D_DIR}')
    print(f'  All {len(all_files)} files found.\n')

    # ─────────────────────────────────────────────────────────────────────────
    # PASS 1 — Normalization stats (all 6600 files, stream one-by-one)
    # ─────────────────────────────────────────────────────────────────────────
    print('Pass 1: computing stats on Hampel+LPF filtered amplitude (all reps 1-20)...')

    res_n      = np.int64(0)
    res_sum    = np.zeros(270, dtype=np.float64)
    res_sum_sq = np.zeros(270, dtype=np.float64)

    C_HAAR    = 135
    xh_sum    = np.zeros(C_HAAR, dtype=np.float64)
    xh_sum_sq = np.zeros(C_HAAR, dtype=np.float64)
    xv_sum    = np.zeros(C_HAAR, dtype=np.float64)
    xv_sum_sq = np.zeros(C_HAAR, dtype=np.float64)
    haar_n    = np.int64(0)

    C_WAV      = 27
    wav_sum    = np.zeros(C_WAV, dtype=np.float64)
    wav_sum_sq = np.zeros(C_WAV, dtype=np.float64)
    wav_n      = np.int64(0)

    for fpath in tqdm(all_files, desc='Stats', unit='file'):
        raw4d = np.load(fpath)       # (1000, 3, 3, 30) float64
        amp   = _filter_amp(raw4d)   # (9, 1000, 30) float32

        # ResNet: per-channel (270,)
        flat = _to_flat270(amp)                      # (270, 1000)
        v    = flat.astype(np.float64)               # (270, 1000)
        res_n      += np.int64(1000)
        res_sum    += v.sum(axis=1)                  # (270,)
        res_sum_sq += (v * v).sum(axis=1)            # (270,)

        # TF-Mamba: Haar DWT on flat270 → per-channel (135,)
        _, (cH, cV, _) = pywt.dwt2(flat, 'haar', mode='periodization')
        cH64 = cH.astype(np.float64)    # (135, 500)
        cV64 = cV.astype(np.float64)    # (135, 500)
        xh_sum    += cH64.sum(axis=1)
        xh_sum_sq += (cH64 * cH64).sum(axis=1)
        xv_sum    += cV64.sum(axis=1)
        xv_sum_sq += (cV64 * cV64).sum(axis=1)
        haar_n    += np.int64(cH.shape[1])    # += 500

        # WavMamba: db4 DWT on (9,1000,30) → per-channel (27,)
        d64 = apply_dwt2_stack(amp[None])[0].astype(np.float64)   # (27,500,15)
        wav_sum    += d64.sum(axis=(1, 2))
        wav_sum_sq += (d64 * d64).sum(axis=(1, 2))
        wav_n      += np.int64(d64.shape[1] * d64.shape[2])       # += 7500

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
            'source':   'amplitude_npy_4d_hampel_lpf',
            'filter':   'hampel(window=11,n_sigma=3.0) + butter(4,20Hz,fs=200Hz)',
            'n_files':  len(all_files),
            'fitted_on': 'all_reps_1_to_20',
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

    stats_path = BENCH_DIR / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\nStats saved → {stats_path}')
    print(f'  resnet  : mean[0]={resnet_mean[0]:.5f}  std[0]={resnet_std[0]:.5f}  (270 channels)')
    print(f'  tfmamba : xh_mean[0]={xh_mean[0]:.5f}  xh_std[0]={xh_std[0]:.5f}')
    print(f'  wavmamba: mean[0]={wav_mean[0]:.5f}  std[0]={wav_std[0]:.5f}')
    print(f'  counts  : n_resnet={int(res_n):,}  n_haar={int(haar_n):,}  n_wav={int(wav_n):,}')

    # ─────────────────────────────────────────────────────────────────────────
    # PASS 2 — Save UN-normalized arrays (train + test)
    # ─────────────────────────────────────────────────────────────────────────
    print('\nPass 2: saving filtered UN-normalized arrays...')
    _process_split('train', TRAIN_REPS, BENCH_DIR)
    _process_split('test',  TEST_REPS,  BENCH_DIR)

    print(f'\nDone. Output: {BENCH_DIR}')
    print('Arrays are UN-NORMALIZED (Hampel+LPF filtered, same params as extract_amplitude_raw).')
    print('Apply stats.json normalization at training time via dataset loader.')
    print('\nTo train with processed data:')
    print('  python xrf55_bench/trainer.py --model resnet   --bench-dir dataset/XRF55/bench/processed')
    print('  python xrf55_bench/trainer.py --model tfmamba  --bench-dir dataset/XRF55/bench/processed')
    print('  python xrf55_bench/trainer.py --model wavmamba --bench-dir dataset/XRF55/bench/processed')
