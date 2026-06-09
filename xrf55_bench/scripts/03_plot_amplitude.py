"""Amplitude pipeline visualization — raw, preprocessed, and DWT stages.

Reads from raw_npy_nosc/ (270, 1000) format. Displays channel 0
(device 1, antenna 0 — rows 0-29 of the 270-axis) across all 30 subcarriers.

Sample:
  vol=01  action=35 (Running)  rep=11

Generates 6 plots:
  plot1_single_subcarrier_raw_vs_proc.png   1D: raw vs Hampel+Butterworth LPF, subcarrier 10/30
  plot2_all_subcarriers_raw_vs_proc.png     2D: raw vs Hampel+Butterworth LPF, all 30 subcarriers
  plot3_raw_vs_raw_dwt_haar.png             top: raw  /  bottom: 4 DWT-Haar subbands
  plot4_proc_vs_proc_dwt_haar.png           top: proc /  bottom: 4 DWT-Haar subbands
  plot5_raw_vs_raw_dwt_db4.png              top: raw  /  bottom: 4 DWT-db4  subbands
  plot6_proc_vs_proc_dwt_db4.png            top: proc /  bottom: 4 DWT-db4  subbands

Output: assets/figures/  (tracked — these are the figures used in the paper)

Usage:
    cd har_csi
    python xrf55_bench/scripts/03_plot_amplitude.py
    python xrf55_bench/scripts/03_plot_amplitude.py --npy-dir D:/XRF55/raw_npy_nosc
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
_BENCH_ROOT  = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

from xrf55_bench.preprocessing.amplitude import hampel_vectorized
from xrf55_bench.preprocessing.parser    import ACTION_NAMES, ACTION_ID_TO_LABEL

# ── Config ─────────────────────────────────────────────────────────────────────
_NPY_DIR_DEFAULT = PROJECT_ROOT / 'dataset' / 'XRF55' / 'raw_npy_nosc'
PLOTS_DIR        = PROJECT_ROOT / 'assets' / 'figures'

VOL_ID         = 1
ACTION_ID      = 35    # Running
VIZ_REP        = 11
VIZ_SUBCARRIER = 9     # 0-indexed → Subcarrier 10/30

_DUR_S = 5.0           # 1000 frames @ 200 Hz
_N_SC  = 30

_SOS = butter(4, 20.0, btype='low', fs=200.0, output='sos')


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_sample(npy_dir: Path):
    """Load one raw_npy file; return (raw_ch0, filt_ch0) both (1000, 30) float32.

    Channel 0 = device 1, antenna 0 = rows 0-29 of the (270, 1000) array.
    """
    fpath = npy_dir / f'{VOL_ID:02d}_{ACTION_ID:02d}_{VIZ_REP:02d}.npy'
    if not fpath.exists():
        raise FileNotFoundError(f'Sample not found: {fpath}')

    arr = np.load(fpath)   # (270, 1000) float64

    # Channel 0: rows 0-29 (device 1, antenna 0), all 30 subcarriers
    # arr[0:30, :] = (30, 1000) → .T = (1000, 30)
    raw_ch0 = arr[0:30, :].T.astype(np.float32)   # (1000, 30)

    # Filtered version: reshape to (9, 1000, 30), apply Hampel+LPF, take channel 0
    x9 = arr.reshape(9, 30, 1000).transpose(0, 2, 1).astype(np.float32)
    x9 = hampel_vectorized(x9, window=8, n_sigma=3.0)
    x9 = sosfiltfilt(_SOS, x9, axis=1).astype(np.float32)
    filt_ch0 = x9[0].copy()   # (1000, 30)

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
    vmin = float(np.percentile(data, 1))
    vmax = float(np.percentile(data, 99))
    im = ax.imshow(
        data.T, aspect='auto', origin='lower', cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[0, x_end, 0, y_end],
    )
    ax.set_title(title)
    return im


# ── Plot 1: 1D, single subcarrier ──────────────────────────────────────────────

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
    wav_display = _WAV_DISPLAY.get(wavelet, wavelet)
    cA, (cH, cV, cD) = pywt.dwt2(amp, wavelet=wavelet, mode='periodization')

    fig = plt.figure(figsize=(20, 9))
    gs  = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1],
                   hspace=0.45, wspace=0.35)

    ax_top = fig.add_subplot(gs[0, :])
    vmax   = float(np.percentile(np.abs(amp), 99))
    im_top = ax_top.imshow(
        amp.T, aspect='auto', origin='lower', cmap='viridis',
        vmin=None, vmax=vmax,
    )
    ax_top.set_title(f'{amp_title} before {wav_display} DWT', fontsize=14)
    ax_top.set_xlabel('Time (frame)')
    ax_top.set_ylabel('Subcarrier')
    plt.colorbar(im_top, ax=ax_top, fraction=0.023)

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
    _plot_vs_dwt(raw,  'Raw CSI Amplitude',       'haar', out_path)

def plot4_proc_vs_proc_haar(filt, out_path):
    _plot_vs_dwt(filt, 'Processed CSI Amplitude', 'haar', out_path)

def plot5_raw_vs_raw_db4(raw, out_path):
    _plot_vs_dwt(raw,  'Raw CSI Amplitude',       'db4',  out_path)

def plot6_proc_vs_proc_db4(filt, out_path):
    _plot_vs_dwt(filt, 'Processed CSI Amplitude', 'db4',  out_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(npy_dir: Path):
    fpath       = npy_dir / f'{VOL_ID:02d}_{ACTION_ID:02d}_{VIZ_REP:02d}.npy'
    action_name = ACTION_NAMES[ACTION_ID_TO_LABEL[ACTION_ID]]

    print('-' * 60)
    print(f'Sample : vol={VOL_ID:02d}  action={ACTION_ID} ({action_name})  rep={VIZ_REP:02d}')
    print(f'File   : {fpath}')
    print(f'Channel: device 1, antenna 0  (rows 0-29 of 270-axis)')
    print(f'Output : {PLOTS_DIR}')
    print('-' * 60)

    raw, filt = _load_sample(npy_dir)
    print(f'Loaded : shape {raw.shape}  dtype {raw.dtype}')
    print(f'Raw    : min={raw.min():.3f}  max={raw.max():.3f}  mean={raw.mean():.3f}')
    print(f'Filt   : min={filt.min():.3f}  max={filt.max():.3f}  mean={filt.mean():.3f}\n')

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot1_single_subcarrier(raw, filt,
                            PLOTS_DIR / 'plot1_single_subcarrier_raw_vs_proc.png')
    print('Plot 1: 1D subcarrier 10/30 - raw vs preprocessed')

    plot2_all_subcarriers(raw, filt,
                          PLOTS_DIR / 'plot2_all_subcarriers_raw_vs_proc.png')
    print('Plot 2: 2D all 30 subcarriers - raw vs preprocessed')

    plot3_raw_vs_raw_haar(raw, PLOTS_DIR / 'plot3_raw_vs_raw_dwt_haar.png')
    print('Plot 3: raw + DWT-Haar (4 subbands)')

    plot4_proc_vs_proc_haar(filt, PLOTS_DIR / 'plot4_proc_vs_proc_dwt_haar.png')
    print('Plot 4: preprocessed + DWT-Haar (4 subbands)')

    plot5_raw_vs_raw_db4(raw, PLOTS_DIR / 'plot5_raw_vs_raw_dwt_db4.png')
    print('Plot 5: raw + DWT-db4  (4 subbands)')

    plot6_proc_vs_proc_db4(filt, PLOTS_DIR / 'plot6_proc_vs_proc_dwt_db4.png')
    print('Plot 6: preprocessed + DWT-db4  (4 subbands)')

    print(f'\nAll 6 plots saved to {PLOTS_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Amplitude visualization: raw, preprocessed, DWT (6 plots)')
    parser.add_argument('--npy-dir', type=str, default=None,
                        help='Path to raw_npy_nosc/ (default: dataset/XRF55/raw_npy_nosc)')
    args = parser.parse_args()

    d = Path(args.npy_dir) if args.npy_dir else _NPY_DIR_DEFAULT
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    if not d.exists():
        raise FileNotFoundError(f'raw_npy_nosc not found: {d}')
    main(d)
