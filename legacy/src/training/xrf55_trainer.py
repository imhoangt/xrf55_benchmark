"""Shared training pipeline for the 4 XRF55 baselines.

Used by: tf_mamba_xrf55_amp, tf_mamba_xrf55_phase, resnet1d_xrf55_amp, resnet1d_xrf55_phase.

Each baseline supplies its specifics (model factory, loader builders, efficiency probe)
via an `ExperimentConfig`; this module owns the train/eval loop, resume logic, CSV
logging, run aggregation and AMP wrappers.

Features
--------
* `train_step` works for both single-stream (X, y) and dual-stream (XH, XV, y) loaders
  via positional unpacking — no per-baseline forward callback needed.
* Patience counter for early stop only starts AFTER warmup, so warmup LR ramp does
  not exhaust patience before the model has had a chance to learn.
* AMP (mixed precision) is enabled by default when CUDA is available; GradScaler
  state is checkpointed and restored on resume.
* In addition to the heavy resume checkpoints (`checkpoint.pt`, `checkpoint_best.pt`)
  this module also writes clean, optimizer-free `best_model.pt` and `final_model.pt`
  for downstream inference / distribution.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.training.amp_utils import torch_load_checkpoint
from src.training.train_utils import (
    build_lr_scheduler,
    configure_speed_mode,
    disable_windows_sleep,
    load_checkpoint,
    make_optimizer,
    restore_windows_sleep,
    save_checkpoint,
    set_seed,
)

CSV_FIELDS = [
    'epoch', 'lr', 'train_loss', 'grad_norm', 'epoch_time_s',
    'val_acc', 'val_f1_macro', 'test_acc', 'test_f1_macro',
]

SEEDS_DEFAULT = [4, 8, 17]


@dataclass
class ExperimentConfig:
    """Per-baseline configuration consumed by `run_all_experiments`."""
    model_name: str
    output_dir: Path
    data_root: Path

    # Build train/val/test loaders for the chosen protocol.
    # build_loaders_split(data_root, batch_size, num_workers) -> (train, val, test)
    # build_loaders_loso(fold_idx, data_root, batch_size, num_workers) -> (train, val, test)
    build_loaders_split: Callable
    build_loaders_loso: Callable

    # Factory: () -> nn.Module. A fresh model is built per (fold, seed).
    model_factory: Callable[[], nn.Module]

    # Efficiency probe: measure_efficiency_fn(model, device) -> 6-tuple.
    # Each baseline binds its own input shape(s) before passing here.
    measure_efficiency_fn: Callable

    num_classes: int

    # Hyperparameters (defaults match the original per-baseline configs).
    seeds: list = field(default_factory=lambda: list(SEEDS_DEFAULT))
    max_epochs: int = 100
    batch_size: int = 32
    base_lr: float = 5e-4
    base_batch_size: int = 16
    lr_cap: float = 1e-3
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.95)
    label_smoothing: float = 0.10
    num_workers: int = 4
    warmup_epochs: int = 5
    patience: int = 10
    floor_ratio: float = 0.1
    use_amp: bool = True   # auto-disabled on CPU


# ─────────────────────────────────────────────────────────────────────────────
# Run-marker helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_run_complete(output_dir: Path, run_key: str) -> bool:
    return (output_dir / 'results' / f'{run_key}.DONE').exists()


def _mark_run_complete(output_dir: Path, run_key: str):
    marker = output_dir / 'results' / f'{run_key}.DONE'
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_suffix('.DONE.tmp')
    tmp.touch()
    os.replace(tmp, marker)


# ─────────────────────────────────────────────────────────────────────────────
# Stream-agnostic forward + evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _split_batch(batch, device):
    """Generic (X1, X2, ..., y) → (inputs_list, labels). Supports any positional arity."""
    *inputs, labels = batch
    inputs = [t.to(device, non_blocking=True) for t in inputs]
    labels = labels.to(device, non_blocking=True)
    return inputs, labels


@torch.no_grad()
def _eval_quick(model, loader, device):
    """Returns (acc, f1_macro). Stream-agnostic via positional unpack."""
    model.eval()
    preds, gts = [], []
    for batch in loader:
        inputs, labels = _split_batch(batch, device)
        logits = model(*inputs)
        preds += logits.argmax(1).cpu().tolist()
        gts   += labels.cpu().tolist()
    return accuracy_score(gts, preds), f1_score(gts, preds, average='macro')


@torch.no_grad()
def _eval_full(model, loader, device, num_classes):
    """Returns (acc, f1, f1_per_cls, cm, preds, probs, gts). Stream-agnostic."""
    model.eval()
    preds, probs, gts = [], [], []
    for batch in loader:
        inputs, labels = _split_batch(batch, device)
        logits = model(*inputs)
        probs += torch.softmax(logits, 1).cpu().numpy().tolist()
        preds += logits.argmax(1).cpu().tolist()
        gts   += labels.cpu().tolist()
    acc        = accuracy_score(gts, preds)
    f1         = f1_score(gts, preds, average='macro')
    f1_per_cls = f1_score(gts, preds, average=None,
                          labels=list(range(num_classes))).tolist()
    cm         = confusion_matrix(gts, preds, labels=list(range(num_classes))).tolist()
    return acc, f1, f1_per_cls, cm, preds, probs, gts


# ─────────────────────────────────────────────────────────────────────────────
# Inner training loop
# ─────────────────────────────────────────────────────────────────────────────

def _train_epochs(cfg: ExperimentConfig,
                  model, optimizer, scheduler, criterion,
                  train_loader, val_loader, test_loader, device,
                  start_epoch, ckpt_path, ckpt_best_path,
                  csv_writer, csv_file,
                  start_best_acc=0.0,
                  scaler: Optional[torch.amp.GradScaler] = None):
    best_acc   = start_best_acc
    no_improve = 0
    epoch      = start_epoch - 1   # guard
    use_amp    = scaler is not None

    for epoch in range(start_epoch, cfg.max_epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']

        model.train()
        epoch_loss = 0.0
        grad_norms = []

        for batch in train_loader:
            inputs, labels = _split_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(*inputs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(*inputs)
                loss   = criterion(logits, labels)
                loss.backward()
                gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            grad_norms.append(gn.item())
            epoch_loss += loss.item()

        avg_loss      = epoch_loss / len(train_loader)
        grad_norm_avg = float(np.mean(grad_norms))
        elapsed       = time.time() - t0
        val_acc,  val_f1  = _eval_quick(model, val_loader,  device)
        test_acc, test_f1 = _eval_quick(model, test_loader, device)

        print(f'  Ep {epoch+1:3d}/{cfg.max_epochs}  lr={lr:.2e}  '
              f'loss={avg_loss:.4f}  val={val_acc*100:.2f}%  test={test_acc*100:.2f}%  '
              f'({elapsed:.0f}s)')

        csv_writer.writerow({
            'epoch':         epoch + 1,
            'lr':            lr,
            'train_loss':    round(avg_loss, 6),
            'grad_norm':     round(grad_norm_avg, 6),
            'epoch_time_s':  round(elapsed, 2),
            'val_acc':       round(val_acc,  6),
            'val_f1_macro':  round(val_f1,   6),
            'test_acc':      round(test_acc, 6),
            'test_f1_macro': round(test_f1,  6),
        })
        csv_file.flush()

        scaler_state = scaler.state_dict() if use_amp else None

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            save_checkpoint(ckpt_best_path, epoch, model, optimizer,
                            scheduler=scheduler, best_acc=best_acc,
                            scaler_state=scaler_state)
        else:
            # Patience only counts post-warmup; otherwise the linear LR ramp
            # can exhaust the patience window before real learning starts.
            if epoch + 1 > cfg.warmup_epochs:
                no_improve += 1

        scheduler.step()
        save_checkpoint(ckpt_path, epoch, model, optimizer,
                        scheduler=scheduler, train_loss=avg_loss,
                        scaler_state=scaler_state)

        if no_improve >= cfg.patience:
            print(f'  Early stopping at epoch {epoch+1} '
                  f'(val_acc did not improve for {cfg.patience} epochs after warmup)')
            break

    return epoch + 1


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_one_experiment(cfg: ExperimentConfig,
                       protocol: str, fold_idx: int, seed: int,
                       output_dir: Path, data_root: Path) -> dict:
    """Train and evaluate one (protocol, fold, seed) combo."""
    run_key        = f'fold{fold_idx}_seed{seed:02d}'
    ckpt_dir       = output_dir / 'checkpoints' / run_key
    ckpt_path      = ckpt_dir / 'checkpoint.pt'
    ckpt_best_path = ckpt_dir / 'checkpoint_best.pt'
    log_dir        = output_dir / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n[{run_key}] protocol={protocol} fold={fold_idx} seed={seed}')

    set_seed(seed)
    configure_speed_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loaders
    if protocol == 'split':
        train_loader, val_loader, test_loader = cfg.build_loaders_split(
            data_root, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    else:
        train_loader, val_loader, test_loader = cfg.build_loaders_loso(
            fold_idx, data_root, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # Model + LR (sqrt-scaled, capped)
    import math
    lr = min(cfg.base_lr * math.sqrt(cfg.batch_size / cfg.base_batch_size), cfg.lr_cap)
    model     = cfg.model_factory().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = make_optimizer(model, lr=lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    scheduler = build_lr_scheduler(optimizer, warmup_epochs=cfg.warmup_epochs,
                                   total_epochs=cfg.max_epochs,
                                   floor_ratio=cfg.floor_ratio)

    # AMP scaler — only on CUDA.
    use_amp = cfg.use_amp and device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    # Resume
    start_epoch    = 0
    start_best_acc = 0.0
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, model, optimizer, scheduler=scheduler)
        start_epoch = int(ckpt['epoch']) + 1
        if scaler is not None and ckpt.get('scaler_state') is not None:
            scaler.load_state_dict(ckpt['scaler_state'])
        if ckpt_best_path.exists():
            try:
                _best = torch_load_checkpoint(ckpt_best_path, map_location=device)
                start_best_acc = float(_best.get('best_acc', 0.0))
            except Exception:
                pass
        print(f'  Resuming from epoch {start_epoch} | best_val_acc={start_best_acc:.4f}')

    # CSV
    csv_path   = log_dir / f'{run_key}.csv'
    append_csv = start_epoch > 0 and csv_path.exists()
    csv_file   = csv_path.open('a' if append_csv else 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not append_csv:
        csv_writer.writeheader()

    # Training loop
    disable_windows_sleep()
    t_start = time.time()
    try:
        epochs_trained = _train_epochs(
            cfg, model, optimizer, scheduler, criterion,
            train_loader, val_loader, test_loader, device,
            start_epoch, ckpt_path, ckpt_best_path,
            csv_writer, csv_file,
            start_best_acc=start_best_acc, scaler=scaler)
    finally:
        csv_file.close()
        restore_windows_sleep()

    # Save clean final_model.pt (state_dict only, no optimizer/scheduler).
    torch.save(model.state_dict(), output_dir / 'checkpoints' / run_key / 'final_model.pt')

    # Load best checkpoint for final test eval + write clean best_model.pt.
    if ckpt_best_path.exists():
        _best = torch_load_checkpoint(ckpt_best_path, map_location=device)
        model.load_state_dict(_best['model_state_dict'])
        torch.save(_best['model_state_dict'],
                   output_dir / 'checkpoints' / run_key / 'best_model.pt')
        print('  Loaded checkpoint_best (best val_acc) for final test evaluation')

    acc, f1, f1_per_cls, cm, preds, probs, labels = \
        _eval_full(model, test_loader, device, cfg.num_classes)

    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f'{run_key}.npz',
             predictions=np.array(preds),
             probabilities=np.array(probs),
             labels=np.array(labels))

    total_time = round(time.time() - t_start)
    print(f'  [{run_key}] Acc={acc*100:.2f}%  F1={f1*100:.2f}%  ({total_time}s)')

    return {
        'protocol':              protocol,
        'fold':                  fold_idx,
        'seed':                  seed,
        'test_accuracy':         round(acc, 6),
        'test_f1_macro':         round(f1,  6),
        'test_f1_per_class':     [round(v, 6) for v in f1_per_cls],
        'test_confusion_matrix': cm,
        'epochs_trained':        epochs_trained,
        'total_time_s':          total_time,
    }


def run_all_experiments(cfg: ExperimentConfig,
                        protocol: str = 'split', n_seeds: int = 1,
                        fold_range=None,
                        output_dir: Optional[Path] = None,
                        data_root: Optional[Path] = None) -> dict:
    """Outer loop — skip completed runs, aggregate at the end."""
    output_dir = Path(output_dir) if output_dir is not None else Path(cfg.output_dir)
    data_root  = Path(data_root)  if data_root  is not None else Path(cfg.data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds           = cfg.seeds[:n_seeds]
    n_folds         = 1 if protocol == 'split' else 5
    effective_folds = fold_range if fold_range is not None else list(range(n_folds))

    print(f'\n{"="*65}')
    print(f'  {cfg.model_name} | protocol={protocol} | seeds={seeds}')
    print(f'{"="*65}')

    all_results: dict = {}

    for fold_idx in effective_folds:
        for seed in seeds:
            run_key     = f'fold{fold_idx}_seed{seed:02d}'
            result_path = output_dir / 'results' / f'{run_key}.json'

            if _is_run_complete(output_dir, run_key):
                with open(result_path) as f:
                    all_results[run_key] = json.load(f)
                r = all_results[run_key]
                print(f'  [skip] {run_key} — acc={r["test_accuracy"]*100:.2f}%')
                continue

            result = run_one_experiment(cfg, protocol, fold_idx, seed, output_dir, data_root)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            _mark_run_complete(output_dir, run_key)
            all_results[run_key] = result

    if not all_results:
        print('No completed runs.')
        return {}

    accs = [v['test_accuracy'] for v in all_results.values()]
    f1s  = [v['test_f1_macro']  for v in all_results.values()]

    summary_path = output_dir / 'summary.json'

    # Reuse cached efficiency metrics across re-aggregations.
    params_m = model_size_mb = macs_g = macs_note = lat_mean = lat_std = None
    if summary_path.exists():
        try:
            _prev = json.loads(summary_path.read_text())
            if all(k in _prev for k in ('params_M', 'model_size_mb', 'macs_G')):
                params_m      = _prev['params_M']
                model_size_mb = _prev['model_size_mb']
                macs_g        = _prev['macs_G']
                macs_note     = _prev['macs_note']
                lat_mean      = _prev['latency_mean_ms']
                lat_std       = _prev['latency_std_ms']
        except Exception:
            pass
    if params_m is None:
        _eff_dev   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _eff_model = cfg.model_factory().to(_eff_dev)
        params_m, model_size_mb, macs_g, macs_note, lat_mean, lat_std = \
            cfg.measure_efficiency_fn(_eff_model, _eff_dev)
        del _eff_model

    summary = {
        'model':           cfg.model_name,
        'protocol':        protocol,
        'n_seeds':         n_seeds,
        'n_runs':          len(all_results),
        'acc_mean':        round(float(np.mean(accs)), 6),
        'acc_std':         round(float(np.std(accs)),  6),
        'f1_mean':         round(float(np.mean(f1s)),  6),
        'f1_std':          round(float(np.std(f1s)),   6),
        'params_M':        round(params_m, 3),
        'model_size_mb':   round(model_size_mb, 2),
        'macs_G':          macs_g,
        'macs_note':       macs_note,
        'latency_mean_ms': lat_mean,
        'latency_std_ms':  lat_std,
        'runs':            all_results,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{"="*65}')
    print(f'  SUMMARY | {len(all_results)} runs')
    print(f'  Accuracy : {summary["acc_mean"]*100:.2f} +- {summary["acc_std"]*100:.2f}%')
    print(f'  F1 Macro : {summary["f1_mean"]*100:.2f} +- {summary["f1_std"]*100:.2f}%')
    print(f'  Params   : {params_m:.3f}M  |  Size: {model_size_mb:.2f} MB')
    if macs_g:
        print(f'  MACs     : {macs_g:.3f}G  [{macs_note}]')
    if lat_mean is not None:
        print(f'  Latency  : {lat_mean:.2f} +- {lat_std:.2f} ms')
    print(f'{"="*65}\n')

    return summary
