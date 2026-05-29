from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

from src.data.preprocessing.parser import ACTION_NAMES, ACTION_ID_TO_LABEL

VIZ_SAMPLE_FILE = 'visualization_sample_running_vol01_rep11.npz'


def load_visualization_sample(processed_dir):
    path = Path(processed_dir) / VIZ_SAMPLE_FILE
    if not path.exists():
        raise FileNotFoundError(f'Missing visualization sample: {path}')
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _sample_info(sample, subcarrier=None, all_subcarriers=False):
    """Return metadata string for plot titles.
    subcarrier: 0-indexed int — adds 'Subcarrier N' (for 1D plots).
    all_subcarriers: True — adds 'All 30 Subcarriers' (for heatmap/DWT plots).
    """
    vol_id      = int(sample['vol_id'])
    action_id   = int(sample['action_id'])
    rep_id      = int(sample['rep_id'])
    action_name = ACTION_NAMES[ACTION_ID_TO_LABEL[action_id]]
    parts = [action_name, f"Subject {vol_id:02d}", "Rx 01"]
    if all_subcarriers:
        parts.append("All 30 Subcarriers")
    elif subcarrier is not None:
        parts.append(f"Subcarrier {subcarrier + 1}")
    parts.append(f"Rep {rep_id:02d}")
    return "  |  ".join(parts)


def plot_confusion_matrix(y_true, y_pred, output_path):
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(11)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=ACTION_NAMES, yticklabels=ACTION_NAMES,
                square=True, cbar_kws={'label': 'Proportion'}, ax=ax)
    ax.set_title('Confusion Matrix (Normalized)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_class_f1(per_class_f1, output_path):
    colors = ['#2ecc71' if f >= 0.9 else '#f1c40f' if f >= 0.7 else '#e74c3c'
              for f in per_class_f1]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(11), per_class_f1, color=colors,
                  edgecolor='black', linewidth=0.5)
    for bar, f1 in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(11))
    ax.set_xticklabels(ACTION_NAMES, rotation=45, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0, 1.05])
    ax.set_title('Per-class F1 Score (Test Set)')
    ax.axhline(y=float(np.mean(per_class_f1)), color='black',
               linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_raw_vs_preprocessed_amplitude(sample, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, data, title in [
        (axes[0], sample['raw_amplitude'],      'Raw CSI Amplitude'),
        (axes[1], sample['filtered_amplitude'], 'CSI Amplitude after Hampel + LPF'),
    ]:
        vmin, vmax = np.percentile(data, [1, 99])
        im = ax.imshow(data.T, aspect='auto', origin='lower',
                       cmap='viridis', vmin=vmin, vmax=vmax,
                       extent=[0, 5, 0, 29])
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Subcarrier')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(_sample_info(sample, all_subcarriers=True), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_raw_vs_preprocessed_phase(sample, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, data, title in [
        (axes[0], sample['raw_phase'],      'Raw CSI Phase'),
        (axes[1], sample['filtered_phase'], 'CSI Phase after Conj. Diff + Unwrap + Detrend'),
    ]:
        vmax = float(np.percentile(np.abs(data), 99))
        im = ax.imshow(data.T, aspect='auto', origin='lower',
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       extent=[0, 5, 0, 29])
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Subcarrier')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(_sample_info(sample, all_subcarriers=True), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dwt_triplet(sample, prefix, output_path, cmap):
    """prefix: 'amplitude' or 'phase'"""
    data = sample[f'filtered_{prefix}']   # (1000, 30) after filter, before norm
    cA, (cH, cV, cD) = pywt.dwt2(data, wavelet='db4', mode='periodization')

    fig  = plt.figure(figsize=(20, 9))
    gs   = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1],
                    hspace=0.45, wspace=0.35)

    ax_top = fig.add_subplot(gs[0, :])
    vmax   = float(np.percentile(np.abs(data), 99))
    vmin_  = -vmax if cmap == 'RdBu_r' else None
    im = ax_top.imshow(data.T, aspect='auto', origin='lower',
                       cmap=cmap, vmin=vmin_, vmax=vmax)
    ax_top.set_title(f'Filtered {prefix.capitalize()} before DWT', fontsize=14)
    plt.colorbar(im, ax=ax_top, fraction=0.023)

    for col, (subband, title) in enumerate(
            [(cA, 'LL (cA)'), (cH, 'HL (cH)'), (cV, 'LH (cV)'), (cD, 'HH (cD)')]):
        ax    = fig.add_subplot(gs[1, col])
        vmax  = float(np.percentile(np.abs(subband), 99))
        vmin_ = -vmax if cmap == 'RdBu_r' else None
        im = ax.imshow(subband.T, aspect='auto', origin='lower',
                       cmap=cmap, vmin=vmin_, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Timestep (DWT)')
        ax.set_ylabel('Subcarrier Bin (DWT)')
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(_sample_info(sample, all_subcarriers=True), fontsize=14)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_1d_subcarrier_amplitude(sample, output_path, subcarrier=9):
    """1D line chart: raw vs filtered amplitude for one subcarrier (rx_01)."""
    raw  = sample['raw_amplitude'][:, subcarrier]
    filt = sample['filtered_amplitude'][:, subcarrier]
    t    = np.arange(len(raw))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, raw,  color='steelblue',  linewidth=0.8, alpha=0.8, label='Before filtering')
    ax.plot(t, filt, color='darkorange', linewidth=1.2,             label='After filtering')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Amplitude')
    ax.set_title('CSI Amplitude before and after preprocessing')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.suptitle(_sample_info(sample, subcarrier=subcarrier), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_1d_subcarrier_phase(sample, output_path, subcarrier=9):
    """1D line chart: raw vs filtered phase for one subcarrier (rx_01)."""
    raw  = sample['raw_phase'][:, subcarrier]
    filt = sample['filtered_phase'][:, subcarrier]
    t    = np.arange(len(raw))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, raw,  color='steelblue',  linewidth=0.8, alpha=0.8, label='Raw CSI Phase (wrapped)')
    ax.plot(t, filt, color='darkorange', linewidth=1.2,             label='CSI Phase after Conj. Diff + Unwrap + Detrend')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('CSI Phase before and after preprocessing')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.suptitle(_sample_info(sample, subcarrier=subcarrier), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
