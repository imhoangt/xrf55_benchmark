"""Build XRF55 bench arrays (raw) from per-sample (270, 1000) amplitude files.

No additional filtering — files are used as-is (raw amplitude, no RSSI normalization).

Input:  dataset/XRF55/raw_npy_nosc/{vol:02d}_{action:02d}_{rep:02d}.npy  (270, 1000) float64

Channel layout of the 270-axis (from read_dat.m):
    row i = antenna (i // 30) × subcarrier (i % 30)
    9 antennas = 3 RX devices × 3 antennas/device  (antenna-major, subcarrier-minor)

Output: dataset/XRF55/bench/raw_nosc/
    stats.json
    y_train.npy              (4620,)            int64
    y_test.npy               (1980,)            int64
    resnet/
        X_train.npy          (4620, 270, 1000)  float32
        X_test.npy           (1980, 270, 1000)  float32
    tfmamba/
        X_train_xh.npy       (4620, 500, 135)   float32
        X_train_xv.npy       (4620, 500, 135)   float32
        X_test_xh.npy        (1980, 500, 135)   float32
        X_test_xv.npy        (1980, 500, 135)   float32
    wavmamba/
        X_train.npy          (4620, 27, 500, 15) float32
        X_test.npy           (1980, 27, 500, 15) float32

Split: train = reps 1-14 (4620 samples), test = reps 15-20 (1980 samples).
Stats fitted on all reps 1-20 (6600 files).

Usage:
    python xrf55_bench/scripts/01_build_dataset_raw.py
    python xrf55_bench/scripts/01_build_dataset_raw.py --npy-dir D:/XRF55/raw_npy_nosc
    python xrf55_bench/scripts/01_build_dataset_raw.py --bench-dir E:/bench/raw_nosc
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pywt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.dwt    import apply_dwt2_stack
from src.data.preprocessing.parser import ACTION_ID_TO_LABEL, ACTION_IDS_USED

NPY_DIR   = PROJECT_ROOT / 'dataset' / 'XRF55' / 'raw_npy_nosc'
BENCH_DIR = PROJECT_ROOT / 'dataset' / 'XRF55' / 'bench' / 'raw_nosc'

TRAIN_REPS = list(range(1, 15))    # reps 1-14  → 4620 samples
TEST_REPS  = list(range(15, 21))   # reps 15-20 → 1980 samples


# ── File helpers ──────────────────────────────────────────────────────────────

def _fpath(npy_dir: Path, vol_id: int, action_id: int, rep_id: int) -> Path:
    return npy_dir / f'{vol_id:02d}_{action_id:02d}_{rep_id:02d}.npy'


def _collect_all(npy_dir: Path) -> list:
    files = []
    for vol_id in range(1, 31):
        for action_id in ACTION_IDS_USED:
            for rep_id in range(1, 21):
                files.append(_fpath(npy_dir, vol_id, action_id, rep_id))
    return files


def _collect_split(npy_dir: Path, rep_list: list) -> list:
    items = []
    for vol_id in range(1, 31):
        for action_id in ACTION_IDS_USED:
            label = ACTION_ID_TO_LABEL[action_id]
            for rep_id in rep_list:
                items.append((_fpath(npy_dir, vol_id, action_id, rep_id), label))
    return items


def _check_files(paths: list) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing[:5]:
            print(f'  MISSING: {p}')
        if len(missing) > 5:
            print(f'  ... and {len(missing) - 5} more')
        raise FileNotFoundError(f'{len(missing)} files missing')


# ── Per-sample transforms ─────────────────────────────────────────────────────

def _transforms(arr: np.ndarray):
    """(270, 1000) float64 → (flat, xh, xv, wav).

    flat : (270, 1000)   float32 — ResNet input
    xh   : (500, 135)    float32 — TFMamba XH  (Haar cH.T)
    xv   : (500, 135)    float32 — TFMamba XV  (Haar cV.T)
    wav  : (27, 500, 15) float32 — WavMamba    (db4 DWT)
    """
    flat = arr.astype(np.float32)

    _, (cH, cV, _) = pywt.dwt2(flat, 'haar', mode='periodization')
    xh = cH.T   # (500, 135)
    xv = cV.T   # (500, 135)

    # (270, 1000) → (9, 1000, 30) for db4 DWT
    x9  = flat.reshape(9, 30, 1000).transpose(0, 2, 1)
    wav = apply_dwt2_stack(x9[None])[0]   # (27, 500, 15)

    return flat, xh, xv, wav


# ── Pass 1: stats ─────────────────────────────────────────────────────────────

def _compute_stats(all_files: list, npy_dir: Path) -> dict:
    print(f'Pass 1: computing stats  ({len(all_files)} files)')

    res_n      = np.int64(0)
    res_sum    = np.zeros(270, dtype=np.float64)
    res_sum_sq = np.zeros(270, dtype=np.float64)

    xh_sum    = np.zeros(135, dtype=np.float64)
    xh_sum_sq = np.zeros(135, dtype=np.float64)
    xv_sum    = np.zeros(135, dtype=np.float64)
    xv_sum_sq = np.zeros(135, dtype=np.float64)
    haar_n    = np.int64(0)

    wav_sum    = np.zeros((27, 15), dtype=np.float64)
    wav_sum_sq = np.zeros((27, 15), dtype=np.float64)
    wav_n      = np.int64(0)

    for fpath in tqdm(all_files, desc='  Stats', unit='file'):
        arr              = np.load(fpath)
        flat, xh, xv, wav = _transforms(arr)

        v = flat.astype(np.float64)
        res_n      += np.int64(1000)
        res_sum    += v.sum(axis=1)
        res_sum_sq += (v * v).sum(axis=1)

        xh64 = xh.astype(np.float64)    # (500, 135) — sum axis=0 (T)
        xv64 = xv.astype(np.float64)    # (500, 135) — sum axis=0 (T)
        xh_sum    += xh64.sum(axis=0)
        xh_sum_sq += (xh64 * xh64).sum(axis=0)
        xv_sum    += xv64.sum(axis=0)
        xv_sum_sq += (xv64 * xv64).sum(axis=0)
        haar_n    += np.int64(500)

        w64 = wav.astype(np.float64)
        wav_sum    += w64.sum(axis=1)          # sum over time (500) → (27, 15)
        wav_sum_sq += (w64 * w64).sum(axis=1)
        wav_n      += np.int64(500)

    def _vector(n, s, s2):
        mean = s / n
        std  = np.maximum(np.sqrt(np.maximum(s2 / n - mean * mean, 0.0)), 1e-6)
        return mean.tolist(), std.tolist()

    resnet_mean, resnet_std = _vector(res_n,  res_sum,  res_sum_sq)
    xh_mean,     xh_std    = _vector(haar_n, xh_sum,   xh_sum_sq)
    xv_mean,     xv_std    = _vector(haar_n, xv_sum,   xv_sum_sq)
    wav_mean,    wav_std   = _vector(wav_n,  wav_sum,  wav_sum_sq)

    stats = {
        'meta': {
            'source':     'raw_npy_nosc_270',
            'n_files':    len(all_files),
            'fitted_on':  'all_reps_1_to_20',
            'train_reps': TRAIN_REPS,
            'test_reps':  TEST_REPS,
        },
        'resnet':   {'mean': resnet_mean, 'std': resnet_std},
        'tfmamba':  {'xh_mean': xh_mean, 'xh_std': xh_std,
                     'xv_mean': xv_mean, 'xv_std': xv_std},
        'wavmamba': {'mean': wav_mean, 'std': wav_std},
    }

    print(f'  resnet  : mean[0]={resnet_mean[0]:.5f}  std[0]={resnet_std[0]:.5f}')
    print(f'  tfmamba : xh_mean[0]={xh_mean[0]:.5f}  xh_std[0]={xh_std[0]:.5f}')
    print(f'  wavmamba: mean[0,0]={wav_mean[0][0]:.5f}  std[0,0]={wav_std[0][0]:.5f}')
    return stats


# ── Pass 2: save arrays ───────────────────────────────────────────────────────

def _save_split(split_name: str, rep_list: list,
                npy_dir: Path, bench_dir: Path) -> None:
    samples = _collect_split(npy_dir, rep_list)
    n       = len(samples)
    print(f'\n[{split_name}]  {n} samples  (reps {rep_list[0]}-{rep_list[-1]})')

    missing = [p for p, _ in samples if not p.exists()]
    if missing:
        for p in missing[:5]:
            print(f'  MISSING: {p}')
        raise FileNotFoundError(f'{len(missing)} missing files')

    y        = np.empty(n, dtype=np.int64)
    X_resnet = np.lib.format.open_memmap(
        str(bench_dir / 'resnet'   / f'X_{split_name}.npy'),
        mode='w+', dtype=np.float32, shape=(n, 270, 1000))
    X_xh     = np.lib.format.open_memmap(
        str(bench_dir / 'tfmamba' / f'X_{split_name}_xh.npy'),
        mode='w+', dtype=np.float32, shape=(n, 500, 135))
    X_xv     = np.lib.format.open_memmap(
        str(bench_dir / 'tfmamba' / f'X_{split_name}_xv.npy'),
        mode='w+', dtype=np.float32, shape=(n, 500, 135))
    X_wav    = np.lib.format.open_memmap(
        str(bench_dir / 'wavmamba' / f'X_{split_name}.npy'),
        mode='w+', dtype=np.float32, shape=(n, 27, 500, 15))

    total_gb = (X_resnet.nbytes + X_xh.nbytes + X_xv.nbytes + X_wav.nbytes) / 1e9
    print(f'  Output: {total_gb:.2f} GB (memory-mapped)')

    for i, (fpath, label) in enumerate(
            tqdm(samples, desc='  Transform', unit='file')):
        flat, xh, xv, wav = _transforms(np.load(fpath))
        X_resnet[i] = flat
        X_xh[i]     = xh
        X_xv[i]     = xv
        X_wav[i]     = wav
        y[i]         = label

    np.save(bench_dir / f'y_{split_name}.npy', y)
    del X_resnet, X_xh, X_xv, X_wav

    print(f'  resnet:      ({n}, 270, 1000)    float32')
    print(f'  tfmamba xh:  ({n}, 500, 135)     float32')
    print(f'  tfmamba xv:  ({n}, 500, 135)     float32')
    print(f'  wavmamba:    ({n}, 27, 500, 15)  float32')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Build bench/raw_nosc arrays from (270,1000) raw_npy_nosc files')
    ap.add_argument('--npy-dir',   type=str, default=None,
                    help='Input dir  (default: dataset/XRF55/raw_npy_nosc)')
    ap.add_argument('--bench-dir', type=str, default=None,
                    help='Output dir  (default: dataset/XRF55/bench/raw_nosc)')
    args = ap.parse_args()

    npy_dir   = Path(args.npy_dir)   if args.npy_dir   else NPY_DIR
    bench_dir = Path(args.bench_dir) if args.bench_dir else BENCH_DIR

    for subdir in ['resnet', 'tfmamba', 'wavmamba']:
        (bench_dir / subdir).mkdir(parents=True, exist_ok=True)

    print(f'Input:  {npy_dir}')
    print(f'Output: {bench_dir}\n')

    all_files = _collect_all(npy_dir)
    _check_files(all_files)
    sample = np.load(all_files[0])
    assert sample.shape == (270, 1000), \
        f'Expected (270, 1000), got {sample.shape}'
    print(f'All {len(all_files)} files found.  shape={sample.shape}  dtype={sample.dtype}\n')
    del sample

    stats = _compute_stats(all_files, npy_dir)
    stats_path = bench_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\nStats saved: {stats_path}')

    print('\nPass 2: saving arrays...')
    _save_split('train', TRAIN_REPS, npy_dir, bench_dir)
    _save_split('test',  TEST_REPS,  npy_dir, bench_dir)

    print(f'\nDone.  Output: {bench_dir}')
    print('Arrays are UN-NORMALIZED — normalization applied at training time via stats.json.')
