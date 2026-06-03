"""XRF55 benchmark trainer — unified, 3-protocol, configurable.

Single trainer for all three benchmark models. Configure via TrainCfg or
use TrainCfg_for_protocol() presets to select a training protocol.

Split: train=reps 1-14 (4620), test=reps 15-20 (1980). No val.

Protocols
---------
  01  AdamW lr=1e-4  wd=0.01, no scheduler, 40ep  (tf_mamba paper)
  02  Adam  lr=1e-3, MultiStepLR,   200ep  (XRF55 paper)
  03  AdamW lr=5e-4, warmup(10ep)+cosine, 200ep  (APWMamba paper)

All protocols: no early stop, FP32.
  last_model.pt  — epoch cuối (model chính, dùng cho final eval)
  best_model.pt  — epoch có test acc cao nhất trong lúc train

Usage (from notebook / Python):
    from xrf55_bench.config  import TrainCfg_for_protocol
    from xrf55_bench.trainer import run
    cfg = TrainCfg_for_protocol('01', seeds=(42,))
    run(model_name='resnet', bench_dir=BENCH_DIR, output_dir=OUTPUT_DIR, train_cfg=cfg)

Output: output_dir/
    metrics.json            (config + per_seed + summary)
    plots/                  (training_curve, confusion_matrix, [seed_comparison])
    seeds/{seed:03d}/       (training_log.csv, last_model.pt, best_model.pt,
                             test_predictions.npz)
    {model}_{data_mode}_{protocol}.zip  (two folders: results_summary/ + model/)
"""
import copy
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
    CosineAnnealingLR, LambdaLR, MultiStepLR,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xrf55_bench.preprocessing.parser import ACTION_NAMES
from xrf55_bench.config    import TrainCfg, TrainCfg_for_protocol
from xrf55_bench.dataset   import build_loaders, load_stats, infer_data_mode
from xrf55_bench.reporting import (
    _plot_training_curve, _plot_confusion_matrix, _plot_seed_comparison,
    save_combined_zip, build_metrics, save_metrics,
)
from xrf55_bench.utils.amp_utils   import torch_load_checkpoint
from xrf55_bench.utils.train_utils import configure_speed_mode, set_seed


# ── Model configs ─────────────────────────────────────────────────────────────

NUM_CLASSES  = 11
_MODEL_NAMES = ['resnet', 'tfmamba', 'wavmamba', 'wavdualmamba']


def _get_model_cfg(model_name: str, model_kwargs: dict = None) -> dict:
    """Return model config dict with lazy-loaded model imports.

    eval/efficiency are the unified helpers in utils.eval (1- and 2-stream aware).
    input_shapes lists each forward argument's shape (no batch dim).

    model_kwargs: extra constructor kwargs forwarded to the model factory.
        Currently only 'wavdualmamba' accepts them (e.g. subbands, share_branches,
        use_pos_emb, bidirectional, d_model, ...) — used to run ablations.
    """
    from xrf55_bench.utils.eval import evaluate, evaluate_full, measure_efficiency

    model_kwargs = model_kwargs or {}
    if model_kwargs and model_name != 'wavdualmamba':
        raise ValueError(
            f"model_kwargs is only supported for 'wavdualmamba', got {model_name!r}")

    if model_name == 'resnet':
        from xrf55_bench.models.resnet1d.model import resnet18
        return dict(
            factory      = lambda: resnet18(inchannel=270, num_classes=NUM_CLASSES),
            title        = 'ResNet18-1D',
            is_2stream   = False,
            eval_fn      = evaluate,
            eval_full_fn = evaluate_full,
            meas_fn      = lambda m, d: measure_efficiency(m, d, ((270, 1000),)),
        )
    if model_name == 'tfmamba':
        from xrf55_bench.models.tf_mamba.model import TFMamba
        return dict(
            factory      = lambda: TFMamba(
                num_features=135, d_model=64, num_layers=3,
                num_classes=NUM_CLASSES, max_len=500,
            ),
            title        = 'TF-Mamba',
            is_2stream   = True,
            eval_fn      = evaluate,
            eval_full_fn = evaluate_full,
            meas_fn      = lambda m, d: measure_efficiency(m, d, ((500, 135), (500, 135))),
        )
    if model_name == 'wavmamba':
        from xrf55_bench.models.wavcnnmamba.model import WavMambaHAR
        return dict(
            factory      = lambda: WavMambaHAR(num_classes=NUM_CLASSES),
            title        = 'WavMambaHAR',
            is_2stream   = False,
            eval_fn      = evaluate,
            eval_full_fn = evaluate_full,
            meas_fn      = lambda m, d: measure_efficiency(m, d, ((27, 500, 15),)),
        )
    if model_name == 'wavdualmamba':
        from xrf55_bench.models.wavdualmamba.model import WavDualMamba
        return dict(
            factory      = lambda: WavDualMamba(num_classes=NUM_CLASSES, **model_kwargs),
            title        = 'WavDualMamba',
            is_2stream   = False,
            eval_fn      = evaluate,
            eval_full_fn = evaluate_full,
            meas_fn      = lambda m, d: measure_efficiency(m, d, ((27, 500, 15),)),
        )
    raise ValueError(f"Unknown model '{model_name}'. Choose from: {_MODEL_NAMES}")


# ── Factory functions ─────────────────────────────────────────────────────────

# Params excluded from weight_decay in protocol 03
_NO_DECAY_KEYS  = {'bias', 'A_log', 'D', 'pos_emb'}
_NORM_MODULES   = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                   nn.BatchNorm3d, nn.GroupNorm)


def _build_no_decay_set(model: nn.Module) -> set:
    """Return the set of parameter names that must NOT receive weight decay.

    Three categories:
      - All params of norm layers (weight/bias): LayerNorm, BatchNorm, GroupNorm,
        and custom RMSNorm (WavMamba) — detected by module type, not by name.
      - Mamba SSM structural params: A_log, D.
      - Learnable positional embedding: pos_emb (WavMamba).
      - All bias params.

    Matched by leaf name (last dotted component), not substring: 'D' as a
    substring would spuriously match any param name containing a capital D;
    leaf-name equality matches only the actual Mamba `.D` parameter.
    """
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
    if cfg.protocol == '03':
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
    if cfg.criterion == 'ce':
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
         bench_dir=None,
         cfg: TrainCfg = None,
         num_workers: int = 4,
         model_kwargs: dict = None):
    if cfg is None:
        cfg = TrainCfg_for_protocol('03')
    if not cfg.seeds:
        raise ValueError('cfg.seeds is empty — provide at least one seed.')
    if model_name not in _MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {_MODEL_NAMES}")
    model_kwargs = model_kwargs or {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    configure_speed_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mc           = _get_model_cfg(model_name, model_kwargs)
    model_title  = mc['title']
    is_2stream   = mc['is_2stream']
    eval_fn      = mc['eval_fn']
    eval_full_fn = mc['eval_full_fn']
    meas_fn      = mc['meas_fn']

    if bench_dir is None:
        raise ValueError(
            'bench_dir is required (must contain stats.json). '
            'Run 01_build_dataset_raw.py or 02_build_dataset_processed.py first.')
    stats = load_stats(bench_dir)

    # Work on a local copy so we never mutate the caller's cfg object.
    cfg = copy.copy(cfg)
    if cfg.data_mode is None:
        cfg.data_mode = infer_data_mode(stats)

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
            bench_dir=bench_dir,
            batch_size=cfg.batch_size, num_workers=num_workers,
        )

        model     = mc['factory']().to(device)
        n_params  = sum(p.numel() for p in model.parameters())
        criterion = _make_criterion(cfg)
        optimizer = _make_optimizer(model, cfg)
        scheduler = _make_scheduler(optimizer, cfg)

        clip_str  = str(cfg.grad_clip) if cfg.grad_clip is not None else 'None'
        betas_str = f'({cfg.betas[0]},{cfg.betas[1]})'
        sched_str = cfg.scheduler or 'None'
        sched_detail = ''
        if cfg.scheduler == 'warmup_cosine':
            sched_detail = f'  warmup={cfg.warmup_epochs}ep  floor={cfg.floor_lr}'
        elif cfg.scheduler == 'multistep':
            kw = cfg.scheduler_kwargs or {}
            sched_detail = (f"  milestones={kw.get('milestones',[40,80,120,160])}"
                            f"  gamma={kw.get('gamma',0.5)}")
        print(f'Model    : {model_name:<10}  Device   : {device}')
        print(f'Train    : {len(train_loader.dataset):<10}  Test     : {len(test_loader.dataset)}')
        print(f'Params   : {n_params:,} ({n_params / 1e6:.3f}M)')
        print(f'Protocol : {cfg.protocol}  |  data={cfg.data_mode}  seeds={list(cfg.seeds)}')
        if model_kwargs:
            print(f'ModelCfg : {model_kwargs}')
        print(f'Opt      : {cfg.optimizer}  betas={betas_str}  eps={cfg.eps}')
        print(f'Hyper    : lr={cfg.lr}  bs={cfg.batch_size}  epochs={cfg.num_epochs}  '
              f'wd={cfg.weight_decay}  clip={clip_str}')
        print(f'Sched    : {sched_str}{sched_detail}')
        print(f'Loss     : {cfg.criterion}  label_smooth={cfg.label_smoothing}')
        print('─' * 65)

        log_rows      = []
        t_seed_start  = time.time()
        best_test_acc = -1.0   # < 0 so epoch 1 always saves best_model.pt
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
                    'test_accuracy': test_acc,
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
            # Kaggle Stop kills DataLoader worker PIDs — rebuild with num_workers=0
            _, test_loader = build_loaders(
                model_name, stats,
                bench_dir=bench_dir,
                batch_size=cfg.batch_size, num_workers=0,
            )

        with open(seed_dir / 'training_log.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'epoch', 'lr', 'train_loss', 'grad_norm',
                'test_accuracy', 'test_f1_macro', 'epoch_time_s', 'total_time_s'])
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
    metrics = build_metrics(model_name, bench_dir, cfg, per_seed_results, summary,
                            model_kwargs=model_kwargs)
    save_metrics(output_dir, metrics)

    # ── ZIP ───────────────────────────────────────────────────────────────────
    zip_path = save_combined_zip(
        output_dir, model_name, cfg.data_mode, cfg.protocol, cfg.seeds)
    print(f'\nSaved : {output_dir}')
    print(f'ZIP   : {zip_path}')


# ── Public API ────────────────────────────────────────────────────────────────

def run(model_name: str, bench_dir=None,
        output_dir=None, train_cfg=None,
        num_workers: int = 4, model_kwargs: dict = None):
    """Callable entry point for Kaggle notebooks.

    model_kwargs: extra model constructor kwargs (wavdualmamba only), e.g.
        run('wavdualmamba', ..., model_kwargs={'subbands': ('HL', 'LH')})
        to run a subband ablation.
    """
    _bench = Path(bench_dir)  if bench_dir  else _default_bench_dir()
    _out   = Path(output_dir) if output_dir else _default_output_dir(model_name)
    main(model_name, _out,
         bench_dir=_bench,
         cfg=train_cfg, num_workers=num_workers, model_kwargs=model_kwargs)


def _default_bench_dir():
    return PROJECT_ROOT / 'dataset' / 'XRF55' / 'bench' / 'raw_nosc'


def _default_output_dir(model_name: str):
    return PROJECT_ROOT / 'outputs' / 'runs' / 'xrf55_bench' / model_name
