"""Verify a built multi-dataset bench (wavmamba layout) end-to-end.

Checks: shapes, dtype, label counts/range, stats shape, and that loading through
PreprocWavMambaDataset yields mean~0 / std~1 over a real sample of the data.

Usage: python xrf55_bench/scripts/_verify_multi.py --dataset hust --mode proc
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xrf55_bench.dataset import PreprocWavMambaDataset, load_stats, infer_data_mode

DIRMAP = {'hust': 'HUST-HAR', 'uthar': 'UT_HAR', 'ntufi': 'NTU-Fi_HAR'}

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=list(DIRMAP))
ap.add_argument('--mode', required=True, choices=['raw', 'proc'])
ap.add_argument('--classes', type=int, default=6)
a = ap.parse_args()

bench = PROJECT_ROOT / 'dataset' / DIRMAP[a.dataset] / 'bench' / a.mode
print(f'== verify {a.dataset}/{a.mode}  ({bench})')

stats = load_stats(bench)
mode_inf = infer_data_mode(stats)
mean = np.array(stats['wavmamba']['mean']); std = np.array(stats['wavmamba']['std'])
print(f'  infer_data_mode={mode_inf}  (expect {a.mode})')
print(f'  stats mean/std shape={mean.shape}  std.min={std.min():.4g} (>=1e-6)')
assert mode_inf == a.mode, 'mode mismatch'
assert std.min() >= 1e-6

for sp in ('train', 'test'):
    X = np.load(bench / 'wavmamba' / f'X_{sp}.npy', mmap_mode='r')
    y = np.load(bench / f'y_{sp}.npy')
    cls, cnt = np.unique(y, return_counts=True)
    print(f'  [{sp}] X={X.shape} {X.dtype}  y={y.shape}  '
          f'labels={cls.tolist()}  counts={cnt.tolist()}')
    assert X.dtype == np.float32
    assert X.shape[0] == y.shape[0]
    assert cls.min() == 0 and cls.max() == a.classes - 1, 'label range bad'
    assert mean.shape == (X.shape[1], X.shape[3]), 'stats shape != (C,F2)'

# Normalization check over a real sample (up to 256 train items) through the loader.
ds = PreprocWavMambaDataset(bench, 'train', stats)
n = min(256, len(ds))
xb = torch.stack([ds[i][0] for i in range(n)]).numpy()
print(f'  normalized sample n={n}: mean={xb.mean():.4f} (~0)  std={xb.std():.4f} (~1)  '
      f'finite={np.isfinite(xb).all()}')
assert np.isfinite(xb).all()
assert abs(xb.mean()) < 0.1 and abs(xb.std() - 1.0) < 0.1, 'normalization off'
print('VERIFY OK')
