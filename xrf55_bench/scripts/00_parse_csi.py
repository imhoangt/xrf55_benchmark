"""Parse XRF55 raw .dat/.mat files to per-sample amplitude .npy files.

No RSSI normalization — raw complex CSI amplitude only.

Input:  dataset/XRF55/raw/scene_01/rx_{01,02,03}/{vol:02d}/{stem}.dat|.mat

Output: dataset/XRF55/raw_npy_nosc/{vol:02d}_{action:02d}_{rep:02d}.npy  (270, 1000) float64

Channel layout of the 270-axis:
    Row   0– 29 : antenna 0  (RX1, ant 0), subcarrier 0–29
    Row  30– 59 : antenna 1  (RX1, ant 1), subcarrier 0–29
    Row  60– 89 : antenna 2  (RX1, ant 2), subcarrier 0–29
    Row  90–119 : antenna 3  (RX2, ant 0), subcarrier 0–29
    Row 120–149 : antenna 4  (RX2, ant 1), subcarrier 0–29
    Row 150–179 : antenna 5  (RX2, ant 2), subcarrier 0–29
    Row 180–209 : antenna 6  (RX3, ant 0), subcarrier 0–29
    Row 210–239 : antenna 7  (RX3, ant 1), subcarrier 0–29
    Row 240–269 : antenna 8  (RX3, ant 2), subcarrier 0–29

    row i = antenna (i // 30), subcarrier (i % 30)
    antenna k = RX device (k // 3), antenna within device (k % 3)

Usage:
    python xrf55_bench/scripts/00_parse_csi.py
    python xrf55_bench/scripts/00_parse_csi.py --scene-dir E:/XRF55/raw/scene_01
    python xrf55_bench/scripts/00_parse_csi.py --out-dir D:/XRF55/raw_npy_nosc
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xrf55_bench.preprocessing.parser import (
    ACTION_IDS_USED,
    load_xrf55_sample,
)

SCENE_DIR_DEFAULT = PROJECT_ROOT / 'dataset' / 'XRF55' / 'raw'  / 'scene_01'
OUT_DIR_DEFAULT   = PROJECT_ROOT / 'dataset' / 'XRF55' / 'raw_npy_nosc'

VOLS    = list(range(1, 31))
REPS    = list(range(1, 21))
N_TOTAL = len(VOLS) * len(ACTION_IDS_USED) * len(REPS)   # 6600


def _build(scene_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = errors = 0

    with tqdm(total=N_TOTAL, unit='file') as pbar:
        for vol in VOLS:
            for action in ACTION_IDS_USED:
                for rep in REPS:
                    out_path = out_dir / f'{vol:02d}_{action:02d}_{rep:02d}.npy'

                    if out_path.exists():
                        skipped += 1
                        pbar.update(1)
                        continue

                    try:
                        # (1000, 30, 3_rx, 3_ant) complex128
                        H = load_xrf55_sample(scene_dir, vol, action, rep)
                    except FileNotFoundError as e:
                        tqdm.write(f'  MISSING: {e}')
                        errors += 1
                        pbar.update(1)
                        continue

                    # (T, F, rx, ant) → (rx, ant, F, T) → (270, 1000)
                    # row i: antenna = i // 30  (= rx*3 + ant)
                    #        subcarrier = i % 30
                    amp = (np.abs(H)
                           .transpose(2, 3, 1, 0)   # (rx, ant, F, T) = (3, 3, 30, 1000)
                           .reshape(270, 1000)
                           .astype(np.float64))

                    np.save(out_path, amp)
                    pbar.update(1)

    saved = N_TOTAL - skipped - errors
    print(f'\nDone.')
    print(f'  Saved  : {saved}')
    print(f'  Skipped: {skipped}  (already existed)')
    print(f'  Missing: {errors}')
    print(f'  Output : {out_dir}')


def _verify(out_dir: Path) -> None:
    files = list(out_dir.glob('*.npy'))
    print(f'\nVerification: {len(files)} / {N_TOTAL} files')
    if files:
        sample = np.load(files[0])
        print(f'  shape={sample.shape}  dtype={sample.dtype}'
              f'  min={sample.min():.4f}  max={sample.max():.4f}')
        assert sample.shape == (270, 1000), f'Unexpected shape: {sample.shape}'
        assert sample.dtype == np.float64
        assert np.isfinite(sample).all(), 'Non-finite values found'
        print('  Shape, dtype, finiteness OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Extract XRF55 raw amplitude .npy — no RSSI normalization')
    ap.add_argument('--scene-dir', type=str, default=None,
                    help='Path to raw/scene_01/  (default: dataset/XRF55/raw/scene_01)')
    ap.add_argument('--out-dir', type=str, default=None,
                    help='Output directory  (default: dataset/XRF55/raw_npy_nosc)')
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir) if args.scene_dir else SCENE_DIR_DEFAULT
    out_dir   = Path(args.out_dir)   if args.out_dir   else OUT_DIR_DEFAULT

    if not scene_dir.exists():
        raise FileNotFoundError(f'scene_01 not found: {scene_dir}')

    print(f'Input : {scene_dir}')
    print(f'Output: {out_dir}')
    print(f'Total : {N_TOTAL} samples\n')

    _build(scene_dir, out_dir)
    _verify(out_dir)
