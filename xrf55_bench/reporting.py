"""Plotting, metrics serialization, and result archiving for the XRF55 benchmark.

Functions are standalone and can be called independently of trainer.py,
e.g. to regenerate plots from saved training_log.csv without re-running training.
"""
import dataclasses
import json
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


_COLORS = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD']


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_training_curve(log_per_seed: dict, plots_dir: Path, title: str):
    from matplotlib.lines import Line2D

    multi      = len(log_per_seed) > 1
    _LOSS_COLOR = '#E74C3C'   # red   — loss always red dashed
    _ACC_COLOR  = '#2ECC71'   # green — acc always green solid (single-seed)
    fig, ax1    = plt.subplots(figsize=(10, 5))
    ax2         = ax1.twinx()

    loss_handles = []
    acc_handles  = []

    for i, (seed, rows) in enumerate(log_per_seed.items()):
        c      = _COLORS[i % len(_COLORS)]
        loss_c = _LOSS_COLOR if not multi else c
        acc_c  = _ACC_COLOR  if not multi else c
        epochs = [r['epoch']         for r in rows]
        losses = [r['train_loss']     for r in rows]
        accs   = [r['test_accuracy'] * 100 for r in rows]
        alpha  = 0.85 if multi else 1.0
        lw     = 1.5  if multi else 2.0
        lbl    = f's={seed}'
        ax1.plot(epochs, losses, color=loss_c, lw=lw, alpha=alpha, ls='--')
        ax2.plot(epochs, accs,   color=acc_c,  lw=lw, alpha=alpha, ls='-')
        loss_handles.append(Line2D([0], [0], color=loss_c, lw=lw, ls='--', label=lbl))
        acc_handles.append( Line2D([0], [0], color=acc_c,  lw=lw, ls='-',  label=lbl))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss',         color=_LOSS_COLOR)
    ax2.set_ylabel('Test Acc (%)', color=_ACC_COLOR)
    ax1.tick_params(axis='y', colors=_LOSS_COLOR)
    ax2.tick_params(axis='y', colors=_ACC_COLOR)
    ax2.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{title} — Training Curve')

    if not multi:
        loss_handles[0].set_label('Loss')
        acc_handles[0].set_label('Test Acc (%)')
        ax1.legend(handles=[loss_handles[0], acc_handles[0]],
                   loc='center right', fontsize=9)
        fig.tight_layout()
    else:
        # Legend layout (column-major fill):
        #   col 0      col 1    col 2    col 3    col 4
        #   Loss       s=4      s=8      s=17     s=42   (row 0 — dashed)
        #   Acc (%)    s=4      s=8      s=17     s=42   (row 1 — solid)
        # Interleave: [Loss, Acc(%), l0, a0, l1, a1, l2, a2, l3, a3]
        loss_hdr = Line2D([], [], color='none', label='Loss')
        acc_hdr  = Line2D([], [], color='none', label='Acc (%)')
        interleaved = [loss_hdr, acc_hdr]
        for lh, ah in zip(loss_handles, acc_handles):
            interleaved.extend([lh, ah])
        fig.legend(handles=interleaved,
                   ncol=len(log_per_seed) + 1,
                   loc='lower center', bbox_to_anchor=(0.47, 0.01),
                   fontsize=8, framealpha=0.95,
                   handlelength=2.5, columnspacing=1.0, handletextpad=0.5)
        fig.subplots_adjust(bottom=0.20, top=0.93, left=0.08, right=0.95)

    fig.savefig(plots_dir / 'training_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_confusion_matrix(cms_per_seed: dict, class_names: list,
                            plots_dir: Path, title: str):
    n_cls  = len(class_names)
    cm_avg = np.mean([np.array(c) for c in cms_per_seed.values()], axis=0)
    cm_n   = cm_avg / (cm_avg.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(class_names, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    n = len(cms_per_seed)
    suffix = f' (avg {n} seeds)' if n > 1 else ''
    ax.set_title(f'{title} — Confusion Matrix (normalized){suffix}')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, f'{cm_n[i, j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if cm_n[i, j] > 0.5 else 'black')
    fig.tight_layout()
    fig.savefig(plots_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_seed_comparison(per_seed_results: dict, plots_dir: Path, title: str):
    seeds = list(per_seed_results.keys())
    accs  = [per_seed_results[s]['test_accuracy'] * 100 for s in seeds]
    f1s   = [per_seed_results[s]['test_f1_macro']  * 100 for s in seeds]
    x     = np.arange(len(seeds))
    w     = 0.35
    acc_mean = float(np.mean(accs))
    f1_mean  = float(np.mean(f1s))
    fig, ax = plt.subplots(figsize=(max(6, len(seeds) * 1.5 + 2), 5))
    ax.bar(x - w / 2, accs, w, label='Accuracy (%)', color='#3498DB', alpha=0.85)
    ax.bar(x + w / 2, f1s,  w, label='F1 Macro (%)', color='#2ECC71', alpha=0.85)
    ax.axhline(acc_mean, color='#3498DB', ls='--', lw=1.5,
               label=f'Acc mean = {acc_mean:.2f}%')
    ax.axhline(f1_mean,  color='#2ECC71', ls='--', lw=1.5,
               label=f'F1 mean  = {f1_mean:.2f}%')
    ax.set_xticks(x)
    ax.set_xticklabels([f'seed={s}' for s in seeds])
    ax.set_ylabel('%')
    y_lo = max(0.0, min(accs + f1s) - 5)
    ax.set_ylim(y_lo, 105)
    ax.set_title(
        f'{title} — Seed Comparison\n'
        f'acc = {acc_mean:.2f}% ± {np.std(accs):.2f}%  '
        f'macro_f1 = {f1_mean:.2f}% ± {np.std(f1s):.2f}%'
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(plots_dir / 'seed_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── ZIP ───────────────────────────────────────────────────────────────────────

def save_combined_zip(output_dir: Path, seeds) -> Path:
    """Single zip named {run_tag}_{MMDD_HHMMSS}.zip, run_tag = output_dir.name.

    Folder layout:
      run_config.json   — compact run manifest (top level, first thing visible)
      results_summary/  — metrics, plots, logs, predictions (no weights)
      model/            — last_model.pt + best_model.pt per seed (stored uncompressed)

    Naming uses output_dir.name (the notebook's auto-tagged run id) so each
    ablation's zip is self-identifying at the file-browser level; the seconds
    field (%S) avoids same-minute overwrites on quick reruns.

    Retention: keeps the 2 most-recent zips of the same run_tag so that at
    most 3 zips (2 old + 1 new) coexist on disk at any time.
    """
    zip_base = output_dir.name

    # Retain only the 2 newest existing zips; delete the rest
    old_zips = sorted(output_dir.glob(f'{zip_base}_*.zip'),
                      key=lambda p: p.stat().st_mtime)
    for old in old_zips[:-2]:
        old.unlink()

    ts       = datetime.now().strftime('%m%d_%H%M%S')
    zip_path = output_dir / f'{zip_base}_{ts}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # run_config.json — top-level manifest (first entry)
        p = output_dir / 'run_config.json'
        if p.exists():
            zf.write(p, 'run_config.json')
        # results_summary/
        p = output_dir / 'metrics.json'
        if p.exists():
            zf.write(p, 'results_summary/metrics.json')
        for fname in ['training_curve.png', 'confusion_matrix.png', 'seed_comparison.png']:
            p = output_dir / 'plots' / fname
            if p.exists():
                zf.write(p, f'results_summary/plots/{fname}')
        for seed in seeds:
            sd = output_dir / 'seeds' / f'{seed:03d}'
            for fname in ['training_log.csv', 'test_predictions.npz']:
                p = sd / fname
                if p.exists():
                    zf.write(p, f'results_summary/seeds/{seed:03d}/{fname}')
        # model/ — weights stored uncompressed (already binary)
        for seed in seeds:
            sd = output_dir / 'seeds' / f'{seed:03d}'
            for fname in ['last_model.pt', 'best_model.pt']:
                p = sd / fname
                if p.exists():
                    zf.write(p, f'model/seeds/{seed:03d}/{fname}',
                             compress_type=zipfile.ZIP_STORED)
    return zip_path


# ── Metrics ───────────────────────────────────────────────────────────────────

def build_metrics(model_name: str, bench_dir, cfg,
                  per_seed_results: dict, summary: dict,
                  model_kwargs: dict = None,
                  dataset: str = None, split: str = None) -> dict:
    """Assemble the full metrics dict from training results.

    cfg: TrainCfg instance (duck-typed to avoid circular imports).
    model_kwargs: extra model-constructor kwargs (e.g. ablation subbands),
        recorded under 'model_config' for traceability.
    dataset/split: dataset name + split description for metadata. Default to
        XRF55 values when not given (backward compatible); HUST/UT-HAR/NTU-Fi
        pass their own so metrics.json is labelled correctly.
    """
    dataset = dataset or 'xrf55'
    split   = split   or 'train=reps1-14  test=reps15-20'
    cfg_dict = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in dataclasses.asdict(cfg).items()
    }
    model_config = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in (model_kwargs or {}).items()
    }
    return {
        'model':        f'xrf55_bench_{model_name}',
        'dataset':      dataset,
        'split':        split,
        'eval':         ('Reported metrics (per_seed.test_* and summary.test_*) come '
                         'from last_model.pt = final epoch. The per_seed.best_epoch / '
                         'best_test_acc fields are train-time diagnostics selected by '
                         'peeking at test accuracy and MUST NOT be used as headline results.'),
        'bench_dir':    str(bench_dir) if bench_dir else None,
        'config':       cfg_dict,
        'model_config': model_config,
        'per_seed':     {str(s): v for s, v in per_seed_results.items()},
        'summary':      summary,
    }


def save_metrics(output_dir: Path, metrics: dict):
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


# ── Run manifest (compact, top-level, aggregatable) ────────────────────────────

def _build_results_block(per_seed: dict, summary: dict) -> dict:
    """Headline (final-epoch) + diagnostic (best-epoch) + efficiency.

    final_epoch is the HEADLINE (last_model.pt). best_epoch peeks at test
    accuracy during training, so it is a diagnostic only and is flagged as such
    — never report it. Shape adapts to single- vs multi-seed.
    """
    note = ('final_epoch = last_model.pt (final epoch) is the HEADLINE result. '
            'best_epoch is selected by peeking at test accuracy during training, '
            'so it is a DIAGNOSTIC only and MUST NOT be reported as a result.')
    eff = {
        'model_size_mb':   summary.get('model_size_mb'),
        'macs_G':          summary.get('macs_G'),
        'latency_mean_ms': summary.get('latency_mean_ms'),
        'latency_std_ms':  summary.get('latency_std_ms'),
    }
    seeds = list(per_seed.keys())

    if len(seeds) == 1:
        v = per_seed[seeds[0]]
        return {
            'headline':       'final_epoch',
            'note':           note,
            'epochs_trained': v.get('epochs_trained'),
            'final_epoch': {
                'test_accuracy': v.get('test_accuracy'),
                'test_f1_macro': v.get('test_f1_macro'),
            },
            'best_epoch': {
                'epoch':         v.get('best_epoch'),
                'test_accuracy': v.get('best_test_acc'),
            },
            'efficiency': eff,
        }

    return {
        'headline': 'final_epoch',
        'note':     note,
        'final_epoch': {
            'test_accuracy_mean': summary.get('test_accuracy_mean'),
            'test_accuracy_std':  summary.get('test_accuracy_std'),
            'test_f1_macro_mean': summary.get('test_f1_macro_mean'),
            'test_f1_macro_std':  summary.get('test_f1_macro_std'),
        },
        'best_epoch': {
            'test_accuracy_mean': summary.get('best_test_acc_mean'),
            'test_accuracy_std':  summary.get('best_test_acc_std'),
            'epochs':             summary.get('best_epochs'),
        },
        'per_seed': {
            s: {
                'final_acc':      per_seed[s].get('test_accuracy'),
                'best_acc':       per_seed[s].get('best_test_acc'),
                'best_epoch':     per_seed[s].get('best_epoch'),
                'epochs_trained': per_seed[s].get('epochs_trained'),
            } for s in seeds
        },
        'efficiency': eff,
    }


def build_run_config(model_name: str, metrics: dict, output_dir: Path,
                     stats: dict, input_shape, env: dict) -> dict:
    """Compact, top-level run manifest — inputs + headline results in one file.

    A projection of metrics.json reorganised so a run is self-describing at a
    glance and aggregatable across ablations into a comparison table. The full,
    authoritative record stays in metrics.json; this never replaces it.

    Captures the five axes that determine a run: data, training (resolved cfg
    INCLUDING any overrides — not the protocol label), model (name + kwargs),
    seeds (via training.seeds), and results. input_shape/env are passed in by
    the trainer (which holds torch + device).
    """
    cfg          = metrics['config']         # resolved TrainCfg (tuples → lists)
    model_config = metrics['model_config']   # model_kwargs (tuples → lists)
    summary      = metrics['summary']
    per_seed     = metrics['per_seed']        # str(seed) → dict
    meta         = stats.get('meta', {}) if isinstance(stats, dict) else {}

    return {
        'run_tag': output_dir.name,
        'model': {
            'name':        model_name,
            'params_M':    summary.get('params_M'),
            'input_shape': input_shape,
            'kwargs':      model_config,
        },
        'data': {
            'mode':      cfg.get('data_mode'),
            'source':    meta.get('source'),
            'bench_dir': metrics.get('bench_dir'),
        },
        'training': cfg,                       # verbatim resolved config
        'results':  _build_results_block(per_seed, summary),
        'env':      env,
    }


def save_run_config(output_dir: Path, run_config: dict):
    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)
