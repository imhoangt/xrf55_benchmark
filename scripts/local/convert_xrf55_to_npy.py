"""Convert XRF55 raw .dat/.mat files to per-sample amplitude .npy files.

Each output file: (270, 1000) float64
  Row layout: row = dev*90 + sub*3 + ant
  (3 devices × 30 subcarriers × 3 antennas = 270 channels, 1000 time steps)

No RSSI normalization — raw complex CSI amplitude (no get_scaled_csi).

Usage:
    python scripts/local/convert_xrf55_to_npy.py
    python scripts/local/convert_xrf55_to_npy.py --raw dataset/XRF55/raw/scene_01 --out dataset/XRF55/amplitude_npy
    python scripts/local/convert_xrf55_to_npy.py --vols 1 2 3   # only selected volunteers
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.parser import ACTION_IDS_USED, load_xrf55_sample


def convert_scene(raw_scene_dir: Path, output_dir: Path, vol_ids: list[int]):
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(vol_ids) * len(ACTION_IDS_USED) * 20
    done = skipped = errors = 0

    for vol in vol_ids:
        vol_out = output_dir / f"{vol:02d}"
        vol_out.mkdir(exist_ok=True)
        for action in ACTION_IDS_USED:
            for rep in range(1, 21):
                out_path = vol_out / f"{vol:02d}_{action:02d}_{rep:02d}.npy"
                if out_path.exists():
                    skipped += 1
                    done += 1
                    continue

                try:
                    # (1000, 30, 3_dev, 3_ant) complex128
                    csi = load_xrf55_sample(raw_scene_dir, vol, action, rep)
                except FileNotFoundError as e:
                    print(f"  MISSING: {e}")
                    errors += 1
                    done += 1
                    continue

                # Amplitude → (3_dev, 30_sub, 3_ant, 1000) → (270, 1000)
                amp = np.abs(csi).transpose(2, 1, 3, 0).reshape(270, 1000)
                np.save(out_path, amp.astype(np.float64))

                done += 1
                if done % 200 == 0:
                    pct = 100 * done / total
                    print(f"  {done}/{total} ({pct:.1f}%) — last: {out_path.name}")

    saved = done - skipped - errors
    print(f"\nDone.")
    print(f"  Saved:   {saved}")
    print(f"  Skipped: {skipped} (already existed)")
    print(f"  Missing: {errors}")
    print(f"  Output:  {output_dir}")


def detect_vol_ids(raw_scene_dir: Path) -> list[int]:
    rx1 = raw_scene_dir / "rx_01"
    return sorted(int(d.name) for d in rx1.iterdir() if d.is_dir())


def main():
    parser = argparse.ArgumentParser(description="Convert XRF55 raw to amplitude .npy")
    parser.add_argument("--raw", default="dataset/XRF55/raw/scene_01",
                        help="Path to scene_01 dir (relative to project root or absolute)")
    parser.add_argument("--out", default="dataset/XRF55/amplitude_npy",
                        help="Output dir (relative to project root or absolute)")
    parser.add_argument("--vols", nargs="+", type=int, default=None,
                        help="Volunteer IDs to process (default: all detected)")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    vol_ids = args.vols if args.vols else detect_vol_ids(raw_dir)
    print(f"Raw:        {raw_dir}")
    print(f"Output:     {out_dir}")
    print(f"Volunteers: {vol_ids}")
    print(f"Total:      {len(vol_ids) * len(ACTION_IDS_USED) * 20} samples\n")

    convert_scene(raw_dir, out_dir, vol_ids)


if __name__ == "__main__":
    main()
