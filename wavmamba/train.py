"""Training loop, evaluation, metrics, and plots for WavMamba.

Single-model, single-stream. Drives the full train -> eval -> metrics -> plots
pipeline across the configured seeds. Also provides a CLI entry point.

    output_dir/
        metrics.json            (config + per_seed + summary)
        plots/{training_curve.png, confusion_matrix.png}
        seeds/{seed:03d}/{training_log.csv, last_model.pt, best_model.pt,
                         test_predictions.npz}
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from config import TrainCfg, default_cfg, cfg_asdict
from dataset import build_loaders, load_stats, infer_data_mode
from model import WavMamba


# ── Seeding + speed ───────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def configure_speed_mode():
    """Faster, not bit-level deterministic: cuDNN auto-tuning + TF32 matmul."""
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


# ── Evaluation + efficiency ───────────────────────────────────────────────────

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate(model, loader, device):
    """Quick eval. Returns (acc, f1_macro)."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X.to(device)).argmax(1).cpu().tolist()
            gts   += y.tolist()
    return accuracy_score(gts, preds), f1_score(gts, preds, average='macro')


def evaluate_full(model, loader, device, num_classes):
    """Full eval. Returns (acc, f1, f1_per_cls, cm, preds, probs, gts)."""
    model.eval()
    preds, probs, gts = [], [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            probs  += torch.softmax(logits, 1).cpu().numpy().tolist()
            preds  += logits.argmax(1).cpu().tolist()
            gts    += y.tolist()
    acc        = accuracy_score(gts, preds)
    f1         = f1_score(gts, preds, average='macro')
    f1_per_cls = f1_score(gts, preds, average=None, labels=list(range(num_classes))).tolist()
    cm         = confusion_matrix(gts, preds, labels=list(range(num_classes))).tolist()
    return acc, f1, f1_per_cls, cm, preds, probs, gts


def measure_efficiency(model, device, input_shapes):
    """Params, model size, MACs, GPU latency.

    Args:
        input_shapes: tuple of per-input shapes WITHOUT batch dim. For WavMamba:
            ((C, T2, F2),) -> model(X).
    """
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_f:
        tmp = Path(tmp_f.name)
    torch.save(model.state_dict(), tmp)
    model_size_mb = tmp.stat().st_size / 1e6
    tmp.unlink(missing_ok=True)

    inputs = tuple(torch.randn(1, *s).to(device) for s in input_shapes)

    try:
        from fvcore.nn import FlopCountAnalysis
        _flops = FlopCountAnalysis(model, inputs)
        _flops.unsupported_ops_warnings(False)
        macs_g    = round(_flops.total() / 1e9, 3)
        macs_note = 'GMACs via fvcore (Mamba selective_scan_cuda excluded if present)'
    except Exception:
        macs_g    = None
        macs_note = 'N/A — fvcore not installed'

    if device.type == 'cuda':
        model.eval()
        with torch.no_grad():
            for _ in range(50):
                model(*inputs)
        timings = []
        with torch.no_grad():
            for _ in range(200):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); model(*inputs); e.record()
                torch.cuda.synchronize()
                timings.append(s.elapsed_time(e))
        lat_mean = round(float(np.mean(timings)), 2)
        lat_std  = round(float(np.std(timings)), 2)
    else:
        lat_mean = lat_std = None

    return params_m, model_size_mb, macs_g, macs_note, lat_mean, lat_std


# ── Plots ─────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_COLORS = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD']


def _plot_training_curve(log_per_seed: dict, plots_dir: Path, title: str):
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


# ── Metrics serialization ─────────────────────────────────────────────────────

def build_metrics(bench_dir, cfg, per_seed_results: dict, summary: dict,
                  model_kwargs: dict = None,
                  dataset: str = None, split: str = None) -> dict:
    """Assemble the full metrics dict from training results."""
    dataset = dataset or 'unknown'
    split   = split   or 'unknown'
    cfg_dict = cfg_asdict(cfg)
    model_config = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in (model_kwargs or {}).items()
    }
    return {
        'model':        'wavmamba',
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


# ── Weight-decay exclusion ────────────────────────────────────────────────────

_NO_DECAY_KEYS  = {'bias', 'A_log', 'D', 'pos_emb'}
_NORM_MODULES   = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                   nn.BatchNorm3d, nn.GroupNorm)


def _build_no_decay_set(model: nn.Module) -> set:
    """Param names that must NOT receive weight decay: norm-layer params,
    Mamba SSM A_log/D, pos_emb, and all biases. Matched by leaf name."""
    no_decay: set = set()
    for mn, m in model.named_modules():
        is_norm = isinstance(m, _NORM_MODULES) or type(m).__name__ == 'RMSNorm'
        if is_norm:
            for pn, _ in m.named_parameters(recurse=False):
                no_decay.add(f'{mn}.{pn}' if mn else pn)
    for pn, _ in model.named_parameters():
        if pn.split('.')[-1] in _NO_DECAY_KEYS:
            no_decay.add(pn)
    return no_decay


def _make_optimizer(model: nn.Module, cfg: TrainCfg):
    if cfg.wd_exclude_norm_bias:
        no_decay   = _build_no_decay_set(model)
        decay_p    = [p for n, p in model.named_parameters()
                      if p.requires_grad and n not in no_decay]
        no_decay_p = [p for n, p in model.named_parameters()
                      if p.requires_grad and n in no_decay]
        params = [
            {'params': decay_p,    'weight_decay': cfg.weight_decay},
            {'params': no_decay_p, 'weight_decay': 0.0},
        ]
        wd_kw = {}
    else:
        params = model.parameters()
        wd_kw  = {'weight_decay': cfg.weight_decay}

    opt_kw = dict(lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, **wd_kw)
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(params, **opt_kw)
    if cfg.optimizer == 'adam':
        return torch.optim.Adam(params, **opt_kw)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


def _make_scheduler(optimizer, cfg: TrainCfg):
    if cfg.scheduler is None:
        return None
    if cfg.scheduler == 'warmup_cosine':
        W           = cfg.warmup_epochs
        T           = cfg.num_epochs
        floor_ratio = cfg.floor_lr / cfg.lr

        def _lr_lambda(epoch):
            if epoch < W:
                return (epoch + 1) / max(W, 1)          # linear warmup
            progress = min((epoch - W + 1) / max(T - W, 1), 1.0)
            cos_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cos_val

        return LambdaLR(optimizer, _lr_lambda)
    raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}")


def _make_criterion(cfg: TrainCfg):
    if cfg.criterion == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    raise ValueError(f"Unknown criterion: {cfg.criterion!r}")


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip):
    """Run one epoch. Returns (avg_loss, avg_grad_norm). grad_clip=None: no clip."""
    model.train()
    total_loss = 0.0
    grad_norms = []
    max_norm   = grad_clip if grad_clip is not None else float('inf')
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        grad_norms.append(
            nn.utils.clip_grad_norm_(model.parameters(), max_norm).item())
        optimizer.step()
        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(loader), float(np.mean(grad_norms))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(output_dir,
         bench_dir,
         cfg: TrainCfg = None,
         num_workers: int = 4,
         model_kwargs: dict = None,
         num_classes: int = 7,
         class_names: list = None,
         dataset_name: str = None,
         split_desc: str = None):
    """Train WavMamba across cfg.seeds and save metrics + plots.

    num_classes / class_names / dataset_name / split_desc label metrics.json
    and plot titles for the dataset (UT-HAR=7, NTU-Fi=6).
    """
    if class_names is None:
        raise ValueError('class_names is required (UT-HAR/NTU-Fi labels).')
    if cfg is None:
        cfg = default_cfg()
    if not cfg.seeds:
        raise ValueError('cfg.seeds is empty — provide at least one seed.')
    model_kwargs = model_kwargs or {}

    bench_label = f'{dataset_name}' if dataset_name else 'wavmamba'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    configure_speed_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stats = load_stats(bench_dir)

    cfg = copy.copy(cfg)
    if cfg.data_mode is None:
        cfg.data_mode = infer_data_mode(stats)

    per_seed_results  = {}
    per_seed_log_rows = {}
    model             = None
    t_total_start     = time.time()

    for si, seed in enumerate(cfg.seeds):
        print(f'\n==== Seed {si + 1}/{len(cfg.seeds)} [seed={seed}] ' + '=' * 38)

        seed_dir = output_dir / 'seeds' / f'{seed:03d}'
        seed_dir.mkdir(parents=True, exist_ok=True)

        set_seed(seed)

        train_loader, test_loader = build_loaders(
            stats, bench_dir,
            batch_size=cfg.batch_size, num_workers=num_workers,
        )

        model     = WavMamba(num_classes=num_classes, **model_kwargs).to(device)
        n_params  = sum(p.numel() for p in model.parameters())
        criterion = _make_criterion(cfg)
        optimizer = _make_optimizer(model, cfg)
        scheduler = _make_scheduler(optimizer, cfg)

        clip_str = str(cfg.grad_clip) if cfg.grad_clip is not None else 'None'
        print(f'Model    : WavMamba  Device: {device}')
        print(f'Train    : {len(train_loader.dataset)}  Test: {len(test_loader.dataset)}')
        print(f'Params   : {n_params:,} ({n_params / 1e6:.3f}M)')
        print(f'Hyper    : lr={cfg.lr}  bs={cfg.batch_size}  epochs={cfg.num_epochs}  '
              f'wd={cfg.weight_decay}  clip={clip_str}  seeds={list(cfg.seeds)}')
        print('-' * 65)

        log_rows      = []
        t_seed_start  = time.time()
        best_test_acc = -1.0   # < 0 so epoch 1 always saves best_model.pt
        best_epoch    = 1
        _interrupted  = False

        try:
            for epoch in range(1, cfg.num_epochs + 1):
                t_ep   = time.time()
                cur_lr = optimizer.param_groups[0]['lr']

                avg_loss, grad_norm = _train_epoch(
                    model, train_loader, criterion, optimizer, scheduler,
                    device, cfg.grad_clip)

                ep_time  = time.time() - t_ep
                test_acc, test_f1 = evaluate(model, test_loader, device)
                elapsed  = time.time() - t_seed_start

                is_best = test_acc > best_test_acc
                if is_best:
                    best_test_acc = test_acc
                    best_epoch    = epoch
                    torch.save(model.state_dict(), seed_dir / 'best_model.pt')

                marker    = '*' if is_best else ' '
                gnorm_tag = '*' if cfg.grad_clip is not None else ''
                print(f'Epoch {epoch:3d}/{cfg.num_epochs}  '
                      f'lr={cur_lr:.3e}  loss={avg_loss:.4f}  gnorm={grad_norm:.3f}{gnorm_tag}  |  '
                      f'acc={test_acc * 100:.2f}%{marker}  macro_f1={test_f1 * 100:.2f}%  |  '
                      f'{ep_time:.1f}s  [{elapsed:.0f}s]')

                log_rows.append({
                    'epoch':         epoch,
                    'lr':            cur_lr,
                    'train_loss':    avg_loss,
                    'grad_norm':     round(grad_norm, 6),
                    'test_accuracy': test_acc,
                    'test_f1_macro': test_f1,
                    'epoch_time_s':  round(ep_time, 2),
                    'total_time_s':  round(elapsed, 1),
                })

                torch.save(model.state_dict(), seed_dir / 'last_model.pt')

        except KeyboardInterrupt:
            _interrupted = True

        if not log_rows:
            print(f'\n  Seed {seed}: interrupted before epoch 1 completed — skipping.')
            continue

        # Full eval on the final model (last_model.pt = headline).
        model.load_state_dict(torch.load(seed_dir / 'last_model.pt', map_location=device))
        acc, f1, f1_per_cls, cm, preds, probs, gts = evaluate_full(
            model, test_loader, device, num_classes)
        np.savez(seed_dir / 'test_predictions.npz',
                 predictions=np.array(preds, dtype=np.int64),
                 probabilities=np.array(probs, dtype=np.float32),
                 labels=np.array(gts, dtype=np.int64))

        seed_time = time.time() - t_seed_start
        print(f'Seed {seed} — acc={acc * 100:.2f}%  macro_f1={f1 * 100:.2f}%  '
              f'(best ep={best_epoch} acc={best_test_acc * 100:.2f}%, {seed_time:.0f}s)')

        per_seed_results[seed] = {
            'test_accuracy':         round(acc, 6),
            'test_f1_macro':         round(f1, 6),
            'test_f1_per_class':     [round(v, 6) for v in f1_per_cls],
            'test_confusion_matrix': cm,
            'best_epoch':            best_epoch,
            'best_test_acc':         round(best_test_acc, 6),
            'epochs_trained':        len(log_rows),
            'total_time_s':          round(seed_time),
        }
        per_seed_log_rows[seed] = log_rows

        with open(seed_dir / 'training_log.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

        if _interrupted:
            break  # don't start next seed

    if not per_seed_results:
        print('\n  No seeds completed — nothing to save.')
        return

    # ── Efficiency (once, last seed's model) ──────────────────────────────────
    _sample     = next(iter(test_loader))
    meas_shapes = tuple(tuple(int(d) for d in t.shape[1:]) for t in _sample[:-1])
    params_m, model_size_mb, macs_g, macs_note, lat_mean, lat_std = \
        measure_efficiency(model, device, meas_shapes)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    accs       = [v['test_accuracy'] for v in per_seed_results.values()]
    f1s        = [v['test_f1_macro']  for v in per_seed_results.values()]
    best_accs  = [v['best_test_acc']  for v in per_seed_results.values()]
    total_time = round(time.time() - t_total_start)

    summary = {
        'test_accuracy_mean':  round(float(np.mean(accs)), 6),
        'test_accuracy_std':   round(float(np.std(accs)),  6),
        'test_f1_macro_mean':  round(float(np.mean(f1s)),  6),
        'test_f1_macro_std':   round(float(np.std(f1s)),   6),
        'best_test_acc_mean':  round(float(np.mean(best_accs)), 6),
        'best_test_acc_std':   round(float(np.std(best_accs)),  6),
        'best_epochs':         [v['best_epoch'] for v in per_seed_results.values()],
        'params_M':            round(params_m, 3),
        'model_size_mb':       round(model_size_mb, 2),
        'macs_G':              macs_g,
        'macs_note':           macs_note,
        'latency_mean_ms':     lat_mean,
        'latency_std_ms':      lat_std,
        'total_time_s':        total_time,
    }

    if len(cfg.seeds) > 1:
        print(f'\n==== Summary [seeds: {list(cfg.seeds)}] ' + '=' * 33)
        print(f'  acc      = {summary["test_accuracy_mean"] * 100:.2f}%'
              f' +/- {summary["test_accuracy_std"] * 100:.2f}%')
        print(f'  macro_f1 = {summary["test_f1_macro_mean"] * 100:.2f}%'
              f' +/- {summary["test_f1_macro_std"] * 100:.2f}%')
        print(f'  Best epochs : {summary["best_epochs"]}   Total: {total_time}s')
        print('=' * 65)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_title = f'{bench_label} WavMamba'
    _plot_training_curve(per_seed_log_rows, plots_dir, plot_title)
    _plot_confusion_matrix(
        {s: v['test_confusion_matrix'] for s, v in per_seed_results.items()},
        class_names, plots_dir, plot_title)

    # ── metrics.json ──────────────────────────────────────────────────────────
    metrics = build_metrics(bench_dir, cfg, per_seed_results, summary,
                            model_kwargs=model_kwargs,
                            dataset=dataset_name, split=split_desc)
    save_metrics(output_dir, metrics)

    print(f'\nSaved : {output_dir}')


# ── Entry point ───────────────────────────────────────────────────────────────

def run(bench_dir, output_dir, train_cfg=None,
        num_workers: int = 4, model_kwargs: dict = None,
        num_classes: int = 7, class_names: list = None,
        dataset_name: str = None, split_desc: str = None):
    """Callable entry point for notebooks.

    model_kwargs: WavMamba constructor kwargs (e.g.
        {'n_links': 1, 'n_antennas': 3, 'f2': 15}). The architecture flags
        (subbands/pool/stem_norm/fusion) are fixed inside the model.
    """
    main(output_dir,
         bench_dir=bench_dir,
         cfg=train_cfg,
         num_workers=num_workers,
         model_kwargs=model_kwargs,
         num_classes=num_classes,
         class_names=class_names,
         dataset_name=dataset_name,
         split_desc=split_desc)


def _parse_seeds(s):
    return tuple(int(x.strip()) for x in s.split(',') if x.strip())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Train WavMamba on UT-HAR / NTU-Fi.')
    ap.add_argument('--dataset', required=True, choices=['uthar', 'ntufi'])
    ap.add_argument('--mode', default='raw', choices=['raw', 'proc'])
    ap.add_argument('--raw-root', default=None,
                    help='Raw dataset root (default ../dataset/<DIR>; on Kaggle: mount path)')
    ap.add_argument('--out-root', default=None,
                    help='Where bench/ + outputs/ are written (default ../dataset/<DIR>; '
                         'on Kaggle: /kaggle/working)')
    ap.add_argument('--prenorm', default='sensefi', choices=['none', 'sensefi'],
                    help="pre-norm RAW (before DWT): 'sensefi' | 'none' (no raw pre-normalization)")
    ap.add_argument('--z-gran', default='perpos', choices=['perpos', 'pcb'],
                    help="z-norm AFTER DWT: 'perpos' (per-position) | 'pcb' (per-channel-bin)")
    ap.add_argument('--merge-val', action='store_true',
                    help='ONLY UT-HAR: merge X_val into test')
    ap.add_argument('--seeds', default='0,4,8,17,42', help='comma-separated seeds')
    ap.add_argument('--num-epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--no-build', action='store_true',
                    help='Skip the build step (use an existing bench dir)')
    ap.add_argument('--bench-dir', default=None,
                    help='Existing bench dir (implies --no-build). '
                         'e.g. ../dataset/UT_HAR/bench/raw_none_pcb')
    args = ap.parse_args()

    from build_dataset import build, DIRMAP

    DATA_ROOT = Path(__file__).parent.parent / 'dataset'

    if args.bench_dir:
        bench_dir = Path(args.bench_dir)
        out_root  = args.out_root or str(bench_dir.parent.parent.parent)
    else:
        out_root = args.out_root or str(DATA_ROOT)
        bench_dir = (Path(out_root) / DIRMAP[args.dataset]
                     / 'bench' / f'{args.mode}_{args.prenorm}_{args.z_gran}')
        if not args.no_build:
            build(args.dataset, args.mode,
                  raw_root=args.raw_root, out_root=out_root,
                  merge_val=args.merge_val,
                  prenorm=args.prenorm, z_gran=args.z_gran)

    meta = json.load(open(bench_dir / 'stats.json'))['meta']
    if args.bench_dir:
        # --bench-dir reuses a prebuilt bench: verify the CLI labels used for the
        # output name match the bench's own metadata, so results never get filed
        # under a mismatched dataset/mode/normalization tag.
        expected = {'dataset': args.dataset, 'mode': args.mode,
                    'prenorm': args.prenorm, 'z_gran': args.z_gran}
        mism = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
        if meta.get('merge_val') not in (None, args.merge_val):
            mism['merge_val'] = (meta.get('merge_val'), args.merge_val)
        if mism:
            detail = '; '.join(f'{k}: bench={b!r} cli={c!r}' for k, (b, c) in mism.items())
            raise ValueError(
                f'--bench-dir {bench_dir} metadata does not match CLI args ({detail}). '
                'Pass CLI flags that match the prebuilt bench, or rebuild.')
    cfg = default_cfg(seeds=_parse_seeds(args.seeds),
                      num_epochs=args.num_epochs,
                      batch_size=args.batch_size, lr=args.lr)
    model_kwargs = {'n_links': 1, 'n_antennas': meta['n_per_sub'], 'f2': meta['F2']}
    run_name = (f'wavmamba_{args.dataset}_{args.mode}_{args.prenorm}_{args.z_gran}'
                + ('_mv' if args.merge_val else ''))
    output_dir = Path(out_root) / 'outputs' / run_name
    run(bench_dir=bench_dir, output_dir=output_dir, train_cfg=cfg,
        num_workers=args.num_workers, model_kwargs=model_kwargs,
        num_classes=meta['classes'], class_names=meta['class_names'],
        dataset_name=args.dataset, split_desc=meta['split'])
