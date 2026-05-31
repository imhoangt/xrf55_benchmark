"""XRF55 benchmark trainer — unified, 3-protocol, configurable.

Single trainer for all three benchmark models. Configure via TrainCfg or
use TrainCfg_for_protocol() presets to select a training protocol.

Split: train=reps 1-14 (4620), test=reps 15-20 (1980). No val.

Protocols
---------
  01  AdamW lr=1e-4  wd=0.01, no scheduler, 40ep  (tf_mamba paper)
  02  Adam  lr=1e-3, MultiStepLR,   200ep  (XRF55 paper)
  03  AdamW lr=4e-4, warmup+cosine, 120ep  (APWMamba paper)

All protocols: no early stop, FP32.
  last_model.pt  — epoch cuối (model chính, dùng cho final eval)
  best_model.pt  — epoch có test acc cao nhất trong lúc train

Usage:
    cd har_csi
    python xrf55_bench/trainer.py --model resnet --protocol 01
    python xrf55_bench/trainer.py --model resnet --protocol 02
    python xrf55_bench/trainer.py --model wavmamba --protocol 03 --seeds 4 8 17 42

Output: output_dir/
    metrics.json            (config + per_seed + summary)
    plots/                  (training_curve, confusion_matrix, [seed_comparison])
    seeds/{seed:03d}/       (training_log.csv, last_model.pt, best_model.pt,
                             test_predictions.npz)
    results_summary.zip     (metrics + plots + logs + predictions; no model weights)
    model.zip               (last_model.pt + best_model.pt for all seeds)
"""
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LambdaLR, MultiStepLR, StepLR,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.parser import ACTION_NAMES
from xrf55_bench.config    import TrainCfg, TrainCfg_for_protocol, _PROTOCOL_DEFAULTS
from xrf55_bench.dataset   import build_loaders, load_stats
from xrf55_bench.reporting import (
    _plot_training_curve, _plot_confusion_matrix, _plot_seed_comparison,
    _save_zip, _save_model_zip, build_metrics, save_metrics,
)
from src.training.amp_utils   import torch_load_checkpoint
from src.training.train_utils import configure_speed_mode, set_seed


# ── Model configs ─────────────────────────────────────────────────────────────

NUM_CLASSES  = 11
_MODEL_NAMES = ['resnet', 'tfmamba', 'wavmamba']


def _get_model_cfg(model_name: str) -> dict:
    """Return model config dict with lazy-loaded imports."""
    if model_name == 'resnet':
        from baselines.base_models.resnet1d_base.model import resnet18
        from baselines.base_models.resnet1d_base.train_utils import (
            evaluate, evaluate_full, measure_efficiency)
        return dict(
            factory      = lambda: resnet18(inchannel=270, num_classes=NUM_CLASSES),
            title        = 'ResNet18-1D',
            is_2stream   = False,
            eval_fn      = evaluate,
            eval_full_fn = evaluate_full,
            meas_fn      = lambda m, d: measure_efficiency(m, d, x_shape=(270, 1000)),
        )
    if model_name == 'tfmamba':
        from baselines.base_models.tf_mamba_base.model import TFMamba
        from baselines.base_models.tf_mamba_base.train_utils import (
            evaluate, evaluate_full, measure_efficiency)
        return dict(
            factory      = lambda: TFMamba(
                num_features=135, d_model=64, num_layers=3,
                num_classes=NUM_CLASSES, max_len=500,
            ),
            title        = 'TF-Mamba',
            is_2stream   = True,
            eval_fn      = evaluate,
            eval_full_fn = evaluate_full,
            meas_fn      = lambda m, d: measure_efficiency(
                m, d, xh_shape=(500, 135), xv_shape=(500, 135)),
        )
    if model_name == 'wavmamba':
        from baselines.base_models.wavcnnmamba.model import WavMambaHAR
        from baselines.base_models.resnet1d_base.train_utils import (
            evaluate, evaluate_full, measure_efficiency)
        return dict(
            factory      = lambda: WavMambaHAR(num_classes=NUM_CLASSES),
            title        = 'WavMambaHAR',
            is_2stream   = False,
            eval_fn      = evaluate,
            eval_full_fn = evaluate_full,
            meas_fn      = lambda m, d: measure_efficiency(m, d, x_shape=(27, 500, 15)),
        )
    raise ValueError(f"Unknown model '{model_name}'. Choose from: {_MODEL_NAMES}")


# ── Factory functions ─────────────────────────────────────────────────────────

# Params excluded from weight_decay in apwmamba protocol
_NO_DECAY_KEYS = {'bias', 'A_log', 'D', 'pos'}


def _make_optimizer(model: nn.Module, cfg: TrainCfg):
    if cfg.protocol == '03':
        # Selective weight decay — exclude bias/norm/A_log/D/pos
        decay_p    = [p for n, p in model.named_parameters()
                      if p.requires_grad and not any(k in n for k in _NO_DECAY_KEYS)]
        no_decay_p = [p for n, p in model.named_parameters()
                      if p.requires_grad and any(k in n for k in _NO_DECAY_KEYS)]
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
        return optim.AdamW(params, **opt_kw)
    if cfg.optimizer == 'adam':
        return optim.Adam(params, **opt_kw)
    if cfg.optimizer == 'sgd':
        return optim.SGD(params, lr=cfg.lr, momentum=0.9, **wd_kw)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


def _make_scheduler(optimizer, cfg: TrainCfg):
    if cfg.scheduler is None:
        return None
    kw = cfg.scheduler_kwargs or {}
    if cfg.scheduler == 'cosine':
        return CosineAnnealingLR(optimizer,
                                  T_max=kw.get('T_max', cfg.num_epochs),
                                  eta_min=kw.get('eta_min', cfg.floor_lr))
    if cfg.scheduler == 'step':
        return StepLR(optimizer,
                      step_size=kw.get('step_size', 10),
                      gamma=kw.get('gamma', 0.1))
    if cfg.scheduler == 'multistep':
        return MultiStepLR(optimizer,
                           milestones=kw.get('milestones', [40, 80, 120, 160]),
                           gamma=kw.get('gamma', 0.5))
    if cfg.scheduler == 'warmup_cosine':
        W           = cfg.warmup_epochs
        T           = cfg.num_epochs
        floor_ratio = cfg.floor_lr / cfg.lr

        def _lr_lambda(epoch):
            if epoch < W:
                # Linear warmup: epoch 0 → 1/W, ..., epoch W-1 → 1.0
                return (epoch + 1) / max(W, 1)
            # Cosine: starts just below 1.0 (no plateau), reaches floor_ratio at last epoch
            progress = min((epoch - W + 1) / max(T - W, 1), 1.0)
            cos_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cos_val

        return LambdaLR(optimizer, _lr_lambda)
    raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}")


def _make_criterion(cfg: TrainCfg):
    if cfg.criterion in ('ce', 'label_smooth'):
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    raise ValueError(f"Unknown criterion: {cfg.criterion!r}")


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, scheduler,
                 device, is_2stream, grad_clip):
    """Run one epoch. Returns (avg_loss, avg_grad_norm).

    grad_clip=None: compute gradient norm but do not clip.
    """
    model.train()
    total_loss = 0.0
    grad_norms = []
    max_norm   = grad_clip if grad_clip is not None else float('inf')

    if is_2stream:
        for XH, XV, y in loader:
            XH, XV, y = XH.to(device), XV.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(XH, XV), y)
            loss.backward()
            grad_norms.append(
                nn.utils.clip_grad_norm_(model.parameters(), max_norm).item())
            optimizer.step()
            total_loss += loss.item()
    else:
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

def main(model_name: str, output_dir,
         bench_dir=None, amp4d_dir=None,
         cfg: TrainCfg = None,
         source: str = 'auto', num_workers: int = 4):
    if cfg is None:
        cfg = TrainCfg()
    if not cfg.seeds:
        raise ValueError('cfg.seeds is empty — provide at least one seed.')
    if model_name not in _MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {_MODEL_NAMES}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    configure_speed_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mc           = _get_model_cfg(model_name)
    model_title  = mc['title']
    is_2stream   = mc['is_2stream']
    eval_fn      = mc['eval_fn']
    eval_full_fn = mc['eval_full_fn']
    meas_fn      = mc['meas_fn']

    if bench_dir is None:
        raise ValueError(
            'bench_dir is required (must contain stats.json). '
            'Run 02_compute_stats_raw.py first, even when source="raw".')
    stats = load_stats(bench_dir)

    per_seed_results  = {}
    per_seed_log_rows = {}
    model             = None
    t_total_start     = time.time()

    for si, seed in enumerate(cfg.seeds):
        print(f'\n══ Seed {si + 1}/{len(cfg.seeds)} [seed={seed}] ' + '═' * 38)

        seed_dir = output_dir / 'seeds' / f'{seed:03d}'
        seed_dir.mkdir(parents=True, exist_ok=True)

        set_seed(seed)

        train_loader, test_loader = build_loaders(
            model_name, stats,
            bench_dir=bench_dir, amp4d_dir=amp4d_dir,
            source=source,
            batch_size=cfg.batch_size, num_workers=num_workers,
        )

        model     = mc['factory']().to(device)
        n_params  = sum(p.numel() for p in model.parameters())
        criterion = _make_criterion(cfg)
        optimizer = _make_optimizer(model, cfg)
        scheduler = _make_scheduler(optimizer, cfg)

        sched_str = cfg.scheduler or 'None'
        clip_str  = str(cfg.grad_clip) if cfg.grad_clip is not None else 'None'
        print(f'Model    : {model_name:<10}  Device   : {device}')
        print(f'Train    : {len(train_loader.dataset):<10}  Test     : {len(test_loader.dataset)}')
        print(f'Params   : {n_params:,} ({n_params / 1e6:.3f}M)')
        print(f'Protocol : {cfg.protocol}  |  '
              f'opt={cfg.optimizer}  lr={cfg.lr}  bs={cfg.batch_size}  '
              f'wd={cfg.weight_decay}  clip={clip_str}')
        sched_detail = ''
        if cfg.scheduler == 'warmup_cosine':
            sched_detail = f'  warmup={cfg.warmup_epochs}ep  floor={cfg.floor_lr}'
        elif cfg.scheduler == 'multistep':
            kw = cfg.scheduler_kwargs or {}
            sched_detail = f"  milestones={kw.get('milestones',[40,80,120,160])}  gamma={kw.get('gamma',0.5)}"
        print(f'Sched    : {sched_str}{sched_detail}')
        print('─' * 65)

        log_rows      = []
        t_seed_start  = time.time()
        best_test_acc = 0.0
        best_epoch    = 1
        _interrupted  = False

        try:
            for epoch in range(1, cfg.num_epochs + 1):
                t_ep   = time.time()
                cur_lr = optimizer.param_groups[0]['lr']   # LR for this epoch

                avg_loss, grad_norm = _train_epoch(
                    model, train_loader, criterion, optimizer, scheduler,
                    device, is_2stream, cfg.grad_clip)

                ep_time  = time.time() - t_ep
                test_acc, test_f1 = eval_fn(model, test_loader, device)
                elapsed  = time.time() - t_seed_start

                is_best = test_acc > best_test_acc
                if is_best:
                    best_test_acc = test_acc
                    best_epoch    = epoch
                    torch.save(model.state_dict(), seed_dir / 'best_model.pt')

                marker    = '★' if is_best else ' '
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
                    'test_acc':      test_acc,
                    'test_f1_macro': test_f1,
                    'epoch_time_s':  round(ep_time, 2),
                    'total_time_s':  round(elapsed, 1),
                })

                # Overwrite every epoch — last_model.pt always = last completed epoch
                torch.save(model.state_dict(), seed_dir / 'last_model.pt')

        except KeyboardInterrupt:
            _interrupted = True

        if not log_rows:
            print(f'\n⚠  Seed {seed}: interrupted before epoch 1 completed — skipping.')
            continue

        if _interrupted:
            print(f'\n⚠  Seed {seed}: interrupted at epoch {len(log_rows)}/{cfg.num_epochs}. '
                  f'Saving partial results...')

        with open(seed_dir / 'training_log.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'epoch', 'lr', 'train_loss', 'grad_norm',
                'test_acc', 'test_f1_macro', 'epoch_time_s', 'total_time_s'])
            writer.writeheader()
            writer.writerows(log_rows)

        # Final eval — từ last_model.pt (model chính)
        model.load_state_dict(
            torch_load_checkpoint(seed_dir / 'last_model.pt', map_location=device))
        acc, f1, f1_per_cls, cm, all_preds, all_probs, all_labels = \
            eval_full_fn(model, test_loader, device, NUM_CLASSES)

        np.savez(seed_dir / 'test_predictions.npz',
                 predictions=np.array(all_preds),
                 probabilities=np.array(all_probs, dtype=np.float32),
                 labels=np.array(all_labels))

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
            'total_time_s':          round(seed_time),
        }
        per_seed_log_rows[seed] = log_rows

        if _interrupted:
            break  # don't start next seed

    if not per_seed_results:
        print('\n⚠  No seeds completed — nothing to save.')
        return

    # ── Efficiency (once, last seed's model) ──────────────────────────────────
    params_m, model_size_mb, macs_g, macs_note, lat_mean, lat_std = \
        meas_fn(model, device)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    accs       = [v['test_accuracy'] for v in per_seed_results.values()]
    f1s        = [v['test_f1_macro']  for v in per_seed_results.values()]
    total_time = round(time.time() - t_total_start)

    summary = {
        'test_accuracy_mean':  round(float(np.mean(accs)), 6),
        'test_accuracy_std':   round(float(np.std(accs)),  6),
        'test_f1_macro_mean':  round(float(np.mean(f1s)),  6),
        'test_f1_macro_std':   round(float(np.std(f1s)),   6),
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
        print(f'\n══ Summary [seeds: {list(cfg.seeds)}] ' + '═' * 33)
        print(f'  acc      = {summary["test_accuracy_mean"] * 100:.2f}%'
              f' ± {summary["test_accuracy_std"] * 100:.2f}%')
        print(f'  macro_f1 = {summary["test_f1_macro_mean"] * 100:.2f}%'
              f' ± {summary["test_f1_macro_std"] * 100:.2f}%')
        print(f'  Best epochs : {summary["best_epochs"]}   Total: {total_time}s')
        print('═' * 65)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_training_curve(per_seed_log_rows, plots_dir, model_title)
    _plot_confusion_matrix(
        {s: v['test_confusion_matrix'] for s, v in per_seed_results.items()},
        ACTION_NAMES, plots_dir, model_title)
    if len(cfg.seeds) > 1:
        _plot_seed_comparison(per_seed_results, plots_dir, model_title)

    # ── metrics.json ──────────────────────────────────────────────────────────
    metrics = build_metrics(model_name, bench_dir, cfg, per_seed_results, summary)
    save_metrics(output_dir, metrics)

    # ── ZIP ───────────────────────────────────────────────────────────────────
    zip_path   = _save_zip(output_dir, model_name, cfg.seeds)
    model_zip  = _save_model_zip(output_dir, model_name, cfg.seeds)
    print(f'\nSaved     : {output_dir}')
    print(f'ZIP       : {zip_path}')
    print(f'Model ZIP : {model_zip}')


# ── Public API ────────────────────────────────────────────────────────────────

def run(model_name: str, bench_dir=None, amp4d_dir=None,
        output_dir=None, train_cfg=None,
        source: str = 'auto', num_workers: int = 4):
    """Callable entry point for Kaggle notebooks."""
    _bench = Path(bench_dir)  if bench_dir  else _default_bench_dir()
    _out   = Path(output_dir) if output_dir else _default_output_dir(model_name)
    main(model_name, _out,
         bench_dir=_bench, amp4d_dir=amp4d_dir,
         cfg=train_cfg, source=source, num_workers=num_workers)


def _default_bench_dir():
    return PROJECT_ROOT / 'dataset' / 'XRF55' / 'bench' / 'raw'


def _default_output_dir(model_name: str):
    return PROJECT_ROOT / 'outputs' / 'runs' / 'xrf55_bench' / model_name


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='XRF55 benchmark trainer')
    parser.add_argument('--model',          required=True, choices=_MODEL_NAMES,
                        help='Model: resnet | tfmamba | wavmamba')
    parser.add_argument('--protocol',       default='01',
                        choices=list(_PROTOCOL_DEFAULTS),
                        help='Protocol preset (default: 01)')
    parser.add_argument('--bench-dir',      default=None)
    parser.add_argument('--amp4d-dir',      default=None)
    parser.add_argument('--source',         default='auto',
                        choices=['auto', 'preproc', 'raw'])
    parser.add_argument('--output-dir',     default=None)
    parser.add_argument('--seeds',          nargs='+', type=int, default=None,
                        help='Seeds, e.g. --seeds 4 8 17 42  (default: [42])')
    # Hyperparameter overrides — all optional, override protocol defaults
    parser.add_argument('--lr',             type=float, default=None)
    parser.add_argument('--batch-size',     type=int,   default=None)
    parser.add_argument('--num-epochs',     type=int,   default=None)
    parser.add_argument('--optimizer',      default=None, choices=['adamw', 'adam', 'sgd'])
    parser.add_argument('--scheduler',      default=None,
                        choices=['cosine', 'step', 'multistep', 'warmup_cosine'])
    parser.add_argument('--warmup-epochs',  type=int,   default=None)
    parser.add_argument('--floor-lr',       type=float, default=None)
    parser.add_argument('--weight-decay',   type=float, default=None)
    parser.add_argument('--grad-clip',      type=float, default=None)
    parser.add_argument('--criterion',      default=None, choices=['ce', 'label_smooth'])
    parser.add_argument('--num-workers',    type=int,   default=4)
    args = parser.parse_args()

    overrides = {}
    if args.seeds         is not None: overrides['seeds']         = tuple(args.seeds)
    if args.lr            is not None: overrides['lr']            = args.lr
    if args.batch_size    is not None: overrides['batch_size']    = args.batch_size
    if args.num_epochs    is not None: overrides['num_epochs']    = args.num_epochs
    if args.optimizer     is not None: overrides['optimizer']     = args.optimizer
    if args.scheduler     is not None: overrides['scheduler']     = args.scheduler
    if args.warmup_epochs is not None: overrides['warmup_epochs'] = args.warmup_epochs
    if args.floor_lr      is not None: overrides['floor_lr']      = args.floor_lr
    if args.weight_decay  is not None: overrides['weight_decay']  = args.weight_decay
    if args.grad_clip     is not None: overrides['grad_clip']     = args.grad_clip
    if args.criterion     is not None: overrides['criterion']     = args.criterion

    _cfg   = TrainCfg_for_protocol(args.protocol, **overrides)
    _bench = Path(args.bench_dir)  if args.bench_dir  else _default_bench_dir()
    _out   = Path(args.output_dir) if args.output_dir else _default_output_dir(args.model)
    _amp4d = Path(args.amp4d_dir)  if args.amp4d_dir  else None
    main(args.model, _out,
         bench_dir=_bench, amp4d_dir=_amp4d,
         cfg=_cfg, source=args.source, num_workers=args.num_workers)
