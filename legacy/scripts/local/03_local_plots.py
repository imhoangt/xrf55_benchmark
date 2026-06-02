"""Self-contained local preprocessing plots (F5-F9).

Loads 1 raw .dat file (vol=01, action=35 Running, rep=11) and generates
6 visualization plots of the preprocessing pipeline.

INDEPENDENT of 01_preprocess.py — no normalization or DWT needed because
plot_dwt_triplet recomputes DWT internally from the filtered signal.

Output:
  outputs/plots/preprocessing/
    amplitude_raw_vs_proc.png      (F5)
    phase_raw_vs_proc.png          (F6)
    dwt_amplitude.png              (F7a)
    dwt_phase.png                  (F7b)
    amplitude_1d_subcarrier.png    (F8)
    phase_1d_subcarrier.png        (F9)

Usage:
    python scripts/local/03_local_plots.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.preprocessing.parser import load_xrf55_sample
from src.data.preprocessing.amplitude import extract_amplitude_raw
from src.data.preprocessing.phase import extract_phase_raw
from src.evaluation.plots import (
    plot_raw_vs_preprocessed_amplitude,
    plot_raw_vs_preprocessed_phase,
    plot_dwt_triplet,
    plot_1d_subcarrier_amplitude,
    plot_1d_subcarrier_phase,
)

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR   = PROJECT_ROOT / 'dataset' / 'xrf55' / 'raw' / 'scene_01'
PLOTS_DIR = PROJECT_ROOT / 'outputs' / 'plots' / 'preprocessing'

VOL_ID    = 1
ACTION_ID = 35
VIZ_REP   = 11


def _build_sample():
    """Load one .dat file and extract raw/filtered signals for visualization."""
    print(f"Loading vol={VOL_ID:02d}, action={ACTION_ID}, rep={VIZ_REP}...")
    H_viz = load_xrf55_sample(RAW_DIR, VOL_ID, ACTION_ID, VIZ_REP)
    return {
        'raw_amplitude':      np.abs(H_viz[:, :, 0, 0]).astype(np.float32),
        'filtered_amplitude': extract_amplitude_raw(H_viz)[0].astype(np.float32),
        'raw_phase':          np.angle(H_viz[:, :, 0, 0]).astype(np.float32),
        'filtered_phase':     extract_phase_raw(H_viz)[0].astype(np.float32),
        'vol_id':             np.array(VOL_ID,    dtype=np.int16),
        'action_id':          np.array(ACTION_ID, dtype=np.int16),
        'rep_id':             np.array(VIZ_REP,   dtype=np.int16),
    }


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f'Raw dataset not found: {RAW_DIR}')

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Plots dir: {PLOTS_DIR}')

    sample = _build_sample()

    plot_raw_vs_preprocessed_amplitude(
        sample, PLOTS_DIR / 'amplitude_raw_vs_proc.png')
    print("F5 amplitude comparison saved")

    plot_raw_vs_preprocessed_phase(
        sample, PLOTS_DIR / 'phase_raw_vs_proc.png')
    print("F6 phase comparison saved")

    plot_dwt_triplet(sample, 'amplitude',
                     PLOTS_DIR / 'dwt_amplitude.png', 'viridis')
    print("F7a DWT amplitude triplet saved")

    plot_dwt_triplet(sample, 'phase',
                     PLOTS_DIR / 'dwt_phase.png', 'RdBu_r')
    print("F7b DWT phase triplet saved")

    plot_1d_subcarrier_amplitude(
        sample, PLOTS_DIR / 'amplitude_1d_subcarrier.png')
    print("F8 amplitude 1D saved")

    plot_1d_subcarrier_phase(
        sample, PLOTS_DIR / 'phase_1d_subcarrier.png')
    print("F9 phase 1D saved")

    print(f"\nAll 6 plots saved to {PLOTS_DIR}")


if __name__ == '__main__':
    main()
