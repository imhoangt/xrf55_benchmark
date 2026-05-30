"""Save UN-NORMALIZED benchmark arrays for ResNet18-1D, TF-Mamba, WavMamba.

Reads amplitude_npy_4d files, applies model-specific transforms, saves float32
arrays WITHOUT normalization applied. Normalization stats (from 02a_compute_stats.py)
are applied at training time by the dataset loader.

Split:
  train: reps 1-14  (30 × 11 × 14 = 4620 samples)
  test:  reps 15-20 (30 × 11 ×  6 = 1980 samples)

Transforms per model:
  resnet   : (1000,3,3,30) → (270,1000)           flat amplitude
  tfmamba  : (1000,3,3,30) → (270,1000) → Haar DWT → XH(500,135), XV(135,500)
  wavmamba : (1000,3,3,30) → (9,1000,30) → db4 DWT → (27,500,15)

Output layout:
  bench/raw/
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

Peak RAM: ~11.2 GB (train split — 4 pre-allocated arrays + per-file intermediates ~10 MB).

Requires: 02_compute_stats_raw.py must have been run first (stats.json must exist).
"""
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

TRAIN_REPS = list(range(1, 15))    # 14 reps → 4620 samples
TEST_REPS  = list(range(15, 21))   #  6 reps → 1980 samples


def _to_flat270(raw4d: np.ndarray) -> np.ndarray:
    """(1000, 3, 3, 30) float64 → (270, 1000) float32.

    Layout: row i encodes (dev*ant pair i//30, subcarrier i%30).
    Matches 01_preprocess.py: (N,9,1000,30).transpose(0,1,3,2).reshape(N,270,1000).
    """
    return (raw4d
            .transpose(1, 2, 0, 3)   # (3, 3, 1000, 30)
            .reshape(9, 1000, 30)     # (9, 1000, 30)  — copy
            .transpose(0, 2, 1)       # (9, 30, 1000)
            .reshape(270, 1000)       # (270, 1000)    — copy
            .astype(np.float32))


def _to_x9(raw4d: np.ndarray) -> np.ndarray:
    """(1000, 3, 3, 30) float64 → (9, 1000, 30) float32."""
    return (raw4d
            .transpose(1, 2, 0, 3)   # (3, 3, 1000, 30)
            .reshape(9, 1000, 30)     # (9, 1000, 30)  — copy
            .astype(np.float32))


def _collect_split(rep_list: list) -> list:
    """Sorted list of (fpath, label) for all (vol, action, rep) in rep_list."""
    items = []
    for vol_id in range(1, 31):
        for action_id in ACTION_IDS_USED:
            label = ACTION_ID_TO_LABEL[action_id]
            for rep_id in rep_list:
                p = (AMP4D_DIR / f'{vol_id:02d}'
                     / f'{vol_id:02d}_{action_id:02d}_{rep_id:02d}.npy')
                items.append((p, label))
    return items


def _process_split(split_name: str, rep_list: list) -> None:
    samples = _collect_split(rep_list)
    n       = len(samples)
    print(f'\n[{split_name}]  {n} samples  (reps {rep_list[0]}-{rep_list[-1]})')

    # Verify all files before allocating RAM
    missing = [p for p, _ in samples if not p.exists()]
    if missing:
        for p in missing[:5]:
            print(f'  MISSING: {p}')
        if len(missing) > 5:
            print(f'  ... and {len(missing) - 5} more')
        raise FileNotFoundError(f'{len(missing)} files missing in {AMP4D_DIR}')

    d_resnet  = BENCH_DIR / 'resnet'
    d_tfmamba = BENCH_DIR / 'tfmamba'
    d_wav     = BENCH_DIR / 'wavmamba'

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

    # Fill sample-by-sample
    for i, (fpath, label) in enumerate(
            tqdm(samples, desc=f'  Transforming', unit='file')):
        raw = np.load(fpath)   # (1000, 3, 3, 30) float64

        # ResNet: flat (270, 1000)
        flat        = _to_flat270(raw)
        X_resnet[i] = flat

        # TF-Mamba: Haar DWT on flat
        # cH.T: (135,500).T → (500,135) = XH   |   cV: (135,500) = XV
        _, (cH, cV, _) = pywt.dwt2(flat, 'haar', mode='periodization')
        X_xh[i] = cH.T
        X_xv[i] = cV

        # WavMamba: db4 DWT on (9,1000,30)
        x9      = _to_x9(raw)
        X_wav[i] = apply_dwt2_stack(x9[None])[0]   # (27, 500, 15)

        y[i] = label

    np.save(BENCH_DIR / f'y_{split_name}.npy', y)
    print(f'  y_{split_name}:      {y.shape}  labels {sorted(set(y.tolist()))}')
    print(f'  resnet:      ({n}, 270, 1000)  float32')
    print(f'  tfmamba xh:  ({n}, 500, 135)   float32')
    print(f'  tfmamba xv:  ({n}, 135, 500)   float32')
    print(f'  wavmamba:    ({n}, 27, 500, 15) float32')
    del X_resnet, X_xh, X_xv, X_wav   # flush memmaps


if __name__ == '__main__':
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--bench-dir', type=str, default=None,
                         help='Override bench_processed output dir (must contain stats.json)')
    _args = _parser.parse_args()
    if _args.bench_dir:
        BENCH_DIR  = Path(_args.bench_dir)
        STATS_PATH = BENCH_DIR / 'stats.json'

    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f'stats.json not found at {STATS_PATH}\n'
            'Run 02a_compute_stats.py first.')

    for subdir in ['resnet', 'tfmamba', 'wavmamba']:
        (BENCH_DIR / subdir).mkdir(parents=True, exist_ok=True)

    _process_split('train', TRAIN_REPS)
    _process_split('test',  TEST_REPS)

    print(f'\nDone. Output: {BENCH_DIR}')
    print('Arrays are UN-NORMALIZED.')
    print('Apply stats.json normalization at training time (dataset loader).')
