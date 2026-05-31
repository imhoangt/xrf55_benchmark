"""Amplitude pipeline visualization — raw, preprocessed, and DWT stages.

Sample:
  vol=01  action=35 (Running)  rep=11
  File: <amp4d_dir>/01/01_35_11.npy
  Channel displayed: receiver 0, antenna 0  (channel 0 of 9 RX×ANT pairs)

Generates 6 plots:
  plot1_single_subcarrier_raw_vs_proc.png  1D: raw vs Hampel+Butterworth LPF, subcarrier 10/30
  plot2_all_subcarriers_raw_vs_proc.png    2D: raw vs Hampel+Butterworth LPF, all 30 subcarriers
  plot3_raw_vs_raw_dwt_haar.png            top: raw  /  bottom: 4 DWT-Haar subbands
  plot4_raw_vs_raw_dwt_db4.png             top: raw  /  bottom: 4 DWT-db4  subbands
  plot5_proc_vs_proc_dwt_haar.png          top: proc /  bottom: 4 DWT-Haar subbands
  plot6_proc_vs_proc_dwt_db4.png           top: proc /  bottom: 4 DWT-db4  subbands

Output: xrf55_bench/outputs/local_plots/

Usage:
    cd har_csi
    python xrf55_bench/scripts/05_amplitude_dwt_plots.py
    python xrf55_bench/scripts/05_amplitude_dwt_plots.py --amp4d-dir E:/amplitude_npy_4d
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent    # har_csi/
_BENCH_ROOT  = Path(__file__).parent.parent           # har_csi/xrf55_bench/
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

from src.data.preprocessing.amplitude import hampel_vectorized
from src.data.preprocessing.parser    import ACTION_NAMES, ACTION_ID_TO_LABEL

# ── Config ─────────────────────────────────────────────────────────────────────
_AMP4D_DIR_DEFAULT = PROJECT_ROOT / 'dataset' / 'XRF55' / 'amplitude_npy_4d'
PLOTS_DIR          = _BENCH_ROOT  / 'outputs' / 'local_plots'

VOL_ID         = 1
ACTION_ID      = 35    # Running  (filename ID, NOT class label 0-10)
VIZ_REP        = 11
VIZ_SUBCARRIER = 9     # 0-indexed  →  Subcarrier 10/30

_DUR_S = 5.0           # 1000 frames @ 200 Hz
_N_SC  = 30

# Low-pass filter — Butterworth order-4, cutoff 20 Hz, fs=200 Hz
_SOS = butter(4, 20.0, btype='low', fs=200.0, output='sos')


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_sample(amp4d_dir: Path):
    """Load one .npy file; return (raw_ch0, filt_ch0).

    raw_ch0  : (1000, 30) float32 — raw amplitude, receiver 0 / antenna 0
    filt_ch0 : (1000, 30) float32 — same channel after Hampel + Butterworth LPF
    """
    fpath = amp4d_dir / f'{VOL_ID:02d}' / f'{VOL_ID:02d}_{ACTION_ID:02d}_{VIZ_REP:02d}.npy'
    if not fpath.exists():
        raise FileNotFoundError(f'Sample not found: {fpath}')

    raw4d = np.load(fpath)   # (1000, 3, 3, 30) float64

    # (1000,3,3,30) → (9,1000,30): axis0 = receiver×antenna pairs
    amp9 = raw4d.transpose(1, 2, 0, 3).reshape(9, 1000, 30).astype(np.float32)

    raw_ch0 = amp9[0].copy()   # (1000, 30)  receiver 0, antenna 0

    amp9_filt = hampel_vectorized(amp9, window=11, n_sigma=3.0)
    amp9_filt = sosfiltfilt(_SOS, amp9_filt, axis=1).astype(np.float32)
    filt_ch0  = amp9_filt[0].copy()   # (1000, 30)

    return raw_ch0, filt_ch0


def _suptitle(subcarrier=None, all_subcarriers=False) -> str:
    action_name = ACTION_NAMES[ACTION_ID_TO_LABEL[ACTION_ID]]
    parts = [action_name, f'Subject {VOL_ID:02d}', 'Rx 01']
    if all_subcarriers:
        parts.append('All 30 Subcarriers')
    elif subcarrier is not None:
        parts.append(f'Subcarrier {subcarrier + 1}/30')
    parts.append(f'Rep {VIZ_REP:02d}')
    return '  |  '.join(parts)


# ── Shared heatmap helper ──────────────────────────────────────────────────────

def _heatmap(ax, data: np.ndarray, title: str,
             x_end: float, y_end: float, cmap: str = 'viridis'):
    """Render (T, F) as a heatmap. Per-panel percentile color range."""
    vmin = float(np.percentile(data, 1))
    vmax = float(np.percentile(data, 99))
    im = ax.imshow(
        data.T, aspect='auto', origin='lower', cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[0, x_end, 0, y_end],
    )
    ax.set_title(title)
    return im


# ── Plot 1: 1D, single subcarrier ─────────────────────────────────────────────

def plot1_single_subcarrier(raw, filt, out_path):
    sc = VIZ_SUBCARRIER
    t  = np.linspace(0, _DUR_S, raw.shape[0], endpoint=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, raw[:, sc],  color='steelblue',  lw=0.8, alpha=0.8,
            label='Raw amplitude')
    ax.plot(t, filt[:, sc], color='darkorange', lw=1.2,
            label='After Hampel + Butterworth LPF')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('CSI Amplitude raw and after preprocessing')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.suptitle(_suptitle(subcarrier=sc), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Plot 2: 2D, all subcarriers ───────────────────────────────────────────────

def plot2_all_subcarriers(raw, filt, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, data, title in [
        (axes[0], raw,  'Raw CSI Amplitude'),
        (axes[1], filt, 'After Hampel + Butterworth LPF'),
    ]:
        im = _heatmap(ax, data, title, x_end=_DUR_S, y_end=_N_SC - 1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Subcarrier')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(_suptitle(all_subcarriers=True), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Plots 3-6: GridSpec — top=original, bottom=4 DWT subbands ─────────────────

_WAV_DISPLAY = {'haar': 'Haar', 'db4': 'db4'}


def _plot_vs_dwt(amp: np.ndarray, amp_title: str, wavelet: str, out_path):
    """GridSpec layout:
      top row  (full width) : input signal before DWT
      middle                : row label "... after DWT"
      bottom row (4 panels) : LL (cA), HL (cH), LH (cV), HH (cD)
    """
    wav_display = _WAV_DISPLAY.get(wavelet, wavelet)
    cA, (cH, cV, cD) = pywt.dwt2(amp, wavelet=wavelet, mode='periodization')

    fig = plt.figure(figsize=(20, 9))
    gs  = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1],
                   hspace=0.45, wspace=0.35)

    # ── Top: input signal ──────────────────────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])
    vmax   = float(np.percentile(np.abs(amp), 99))
    im_top = ax_top.imshow(
        amp.T, aspect='auto', origin='lower', cmap='viridis',
        vmin=None, vmax=vmax,
    )
    ax_top.set_title(f'{amp_title} before {wav_display} DWT', fontsize=14)
    plt.colorbar(im_top, ax=ax_top, fraction=0.023)

    # ── Bottom: 4 DWT subbands ─────────────────────────────────────────────────
    bottom_axes = []
    for col, (subband, title) in enumerate([
        (cA, 'LL (cA)'), (cH, 'HL (cH)'), (cV, 'LH (cV)'), (cD, 'HH (cD)'),
    ]):
        ax   = fig.add_subplot(gs[1, col])
        vmax = float(np.percentile(np.abs(subband), 99))
        im   = ax.imshow(
            subband.T, aspect='auto', origin='lower', cmap='viridis',
            vmin=None, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel('Timestep (DWT)')
        ax.set_ylabel('Subcarrier Bin (DWT)')
        plt.colorbar(im, ax=ax, fraction=0.046)
        bottom_axes.append(ax)

    # ── Middle label: "{amp_title} after {wavelet} DWT" ───────────────────────
    fig.canvas.draw()
    top_y0    = ax_top.get_position().y0
    bottom_y1 = bottom_axes[0].get_position().y1
    mid_y     = (top_y0 + bottom_y1) / 2
    fig.text(0.5, mid_y, f'{amp_title} after {wav_display} DWT',
             ha='center', va='center', fontsize=14)

    fig.suptitle(_suptitle(all_subcarriers=True), fontsize=14)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot3_raw_vs_raw_haar(raw, out_path):
    _plot_vs_dwt(raw, 'Raw CSI Amplitude', 'haar', out_path)


def plot4_raw_vs_raw_db4(raw, out_path):
    _plot_vs_dwt(raw, 'Raw CSI Amplitude', 'db4', out_path)


def plot5_proc_vs_proc_haar(filt, out_path):
    _plot_vs_dwt(filt, 'Processed CSI Amplitude', 'haar', out_path)


def plot6_proc_vs_proc_db4(filt, out_path):
    _plot_vs_dwt(filt, 'Processed CSI Amplitude', 'db4', out_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(amp4d_dir: Path):
    fpath       = amp4d_dir / f'{VOL_ID:02d}' / f'{VOL_ID:02d}_{ACTION_ID:02d}_{VIZ_REP:02d}.npy'
    action_name = ACTION_NAMES[ACTION_ID_TO_LABEL[ACTION_ID]]

    print('─' * 60)
    print(f'Sample  : vol={VOL_ID:02d}  action={ACTION_ID} ({action_name})  rep={VIZ_REP:02d}')
    print(f'File    : {fpath}')
    print(f'Channel : receiver 0, antenna 0  (index 0 of 9 RX×ANT pairs)')
    print(f'Output  : {PLOTS_DIR}')
    print('─' * 60)

    raw, filt = _load_sample(amp4d_dir)
    print(f'Loaded  : shape {raw.shape}  dtype {raw.dtype}')
    print(f'Raw     : min={raw.min():.3f}  max={raw.max():.3f}  mean={raw.mean():.3f}')
    print(f'Filtered: min={filt.min():.3f}  max={filt.max():.3f}  mean={filt.mean():.3f}\n')

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot1_single_subcarrier(raw, filt,
                            PLOTS_DIR / 'plot1_single_subcarrier_raw_vs_proc.png')
    print('Plot 1: 1D subcarrier 10/30 — raw vs preprocessed')

    plot2_all_subcarriers(raw, filt,
                          PLOTS_DIR / 'plot2_all_subcarriers_raw_vs_proc.png')
    print('Plot 2: 2D all 30 subcarriers — raw vs preprocessed')

    plot3_raw_vs_raw_haar(raw,
                          PLOTS_DIR / 'plot3_raw_vs_raw_dwt_haar.png')
    print('Plot 3: raw + DWT-Haar (4 subbands)')

    plot4_raw_vs_raw_db4(raw,
                         PLOTS_DIR / 'plot4_raw_vs_raw_dwt_db4.png')
    print('Plot 4: raw + DWT-db4  (4 subbands)')

    plot5_proc_vs_proc_haar(filt,
                            PLOTS_DIR / 'plot5_proc_vs_proc_dwt_haar.png')
    print('Plot 5: preprocessed + DWT-Haar (4 subbands)')

    plot6_proc_vs_proc_db4(filt,
                           PLOTS_DIR / 'plot6_proc_vs_proc_dwt_db4.png')
    print('Plot 6: preprocessed + DWT-db4  (4 subbands)')

    print(f'\nAll 6 plots saved to {PLOTS_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Amplitude visualization: raw, preprocessed, DWT (6 plots)')
    parser.add_argument(
        '--amp4d-dir', type=str, default=None,
        help='Path to amplitude_npy_4d/ (default: dataset/XRF55/amplitude_npy_4d)')
    args = parser.parse_args()

    d = Path(args.amp4d_dir) if args.amp4d_dir else _AMP4D_DIR_DEFAULT
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    if not d.exists():
        raise FileNotFoundError(f'amplitude_npy_4d not found: {d}\n'
                                f'Run 01_convert_to_npy_4d.py first.')
    main(d)
