"""APWMamba trainer — multi-protocol, multi-seed, resume-safe.

Protocols
---------
  "split" : rep-split with val — train=reps 1-12, val=reps 13-14, test=reps 15-20  (1 fold)
  "loso"  : LOSO-5fold rotation — test=G[i], val=G[(i+1)%5], train=3 remaining groups

Seeds
-----
  n_seeds=1 → only seeds[0]=4
  n_seeds=3 → seeds [4, 8, 17]

Each (fold, seed) combo is one independent run:
  • checkpoint saved every epoch  → full resume-safety
  • result JSON saved after run   → skipped on re-run

Output layout under output_dir/
  checkpoints/fold{i}_seed{s}/checkpoint.pt   (overwritten every epoch)
  results/fold{i}_seed{s}.json                (written once after completion)
  logs/fold{i}_seed{s}.csv                    (appended per epoch)
  summary.json                                 (updated after every completed run)
"""
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from baselines.apwmamba.dataset import build_noval_loaders_with_val, build_loso_loaders_with_val
from baselines.base_models.tf_mamba_base.train_utils import measure_efficiency
from baselines.apwmamba.model import APWMamba
from baselines.apwmamba.config import (
    BATCH_SIZE, BETAS, GRAD_CLIP_NORM, LABEL_SMOOTHING,
    LR, MAX_EPOCHS, PATIENCE, OUTPUT_DIR, SEEDS,
    WARMUP_EPOCHS, WEIGHT_DECAY,
)
from src.training.amp_utils import torch_load_checkpoint
from src.training.train_utils import (
    build_lr_scheduler,
    configure_speed_mode,
    disable_windows_sleep,
    load_checkpoint,
    make_optimizer,
    restore_windows_sleep,
    sanity_check_apwmamba,
    save_checkpoint,
    save_environment_info,
    set_seed,
)

CSV_FIELDS = [
    'epoch', 'lr', 'train_loss', 'grad_norm', 'epoch_time_s',
    'val_acc', 'val_f1_macro',
    'test_acc', 'test_f1_macro',
]


def _is_run_complete(output_dir: Path, run_key: str) -> bool:
    return (output_dir / 'results' / f'{run_key}.DONE').exists()


def _mark_run_complete(output_dir: Path, run_key: str):
    marker = output_dir / 'results' / f'{run_key}.DONE'
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_suffix('.DONE.tmp')
    tmp.touch()
    os.replace(tmp, marker)


def _setup_logger(prefix: str) -> logging.Logger:
    name   = f'apwmamba.{prefix}'
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def _train_epochs(model, optimizer, scheduler, criterion,
                  train_loader, val_loader, test_loader,
                  start_epoch, ckpt_path, ckpt_best_path, csv_writer, csv_file, logger,
                  start_best_acc: float = 0.0, patience: int = 10,
                  warmup_epochs: int = 0, scaler=None, device=None):
    """Inner training loop — up to MAX_EPOCHS with early stopping on val_acc.

    Patience counter only starts incrementing AFTER warmup_epochs to avoid
    triggering early stop during the linear warmup phase. When `scaler` is
    provided, gradients flow through `torch.amp.autocast` for FP16
    mixed-precision training.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc   = start_best_acc
    no_improve = 0
    epoch      = start_epoch - 1   # guard: if loop body never runs, return start_epoch
    use_amp    = scaler is not None
    for epoch in range(start_epoch, MAX_EPOCHS):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]['lr']   # set by scheduler before this epoch

        model.train()
        train_losses, grad_norms = [], []

        for Xa, Xp, y in tqdm(train_loader,
                               desc=f'Epoch {epoch+1}/{MAX_EPOCHS}',
                               leave=False):
            Xa = Xa.to(device, non_blocking=True)
            Xp = Xp.to(device, non_blocking=True)
            y  = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(Xa, Xp)
                    loss   = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(Xa, Xp)
                loss   = criterion(logits, y)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            train_losses.append(loss.item())
            grad_norms.append(grad_norm.item())

        train_loss    = float(np.mean(train_losses))
        grad_norm_avg = float(np.mean(grad_norms)) if grad_norms else 0.0
        epoch_time    = time.time() - epoch_start

        val_acc,  val_f1_ep,  *_ = _eval_test(model, val_loader)
        test_acc, test_f1_ep, *_ = _eval_test(model, test_loader)

        logger.info(
            f'Epoch {epoch+1}/{MAX_EPOCHS} | lr={lr:.2e} | '
            f'loss={train_loss:.4f} | val={val_acc:.4f} | test={test_acc:.4f}')

        csv_writer.writerow({
            'epoch':         epoch + 1,
            'lr':            lr,
            'train_loss':    train_loss,
            'grad_norm':     grad_norm_avg,
            'epoch_time_s':  epoch_time,
            'val_acc':       round(val_acc,  6),
            'val_f1_macro':  round(val_f1_ep, 6),
            'test_acc':      round(test_acc, 6),
            'test_f1_macro': round(test_f1_ep, 6),
        })
        csv_file.flush()

        scaler_state = scaler.state_dict() if use_amp else None

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            save_checkpoint(ckpt_best_path, epoch, model, optimizer,
                            scheduler=scheduler, train_loss=train_loss,
                            best_acc=best_acc, scaler_state=scaler_state)
        else:
            # Only count no-improve epochs AFTER warmup phase finishes.
            # Without this guard, patience can be exhausted during warmup
            # when val_acc legitimately fluctuates while LR ramps up.
            if epoch + 1 > warmup_epochs:
                no_improve += 1

        scheduler.step()   # update LR for next epoch BEFORE saving so
        save_checkpoint(   # resume reads the correct LR at epoch+1
            ckpt_path, epoch, model, optimizer,
            scheduler=scheduler, train_loss=train_loss,
            scaler_state=scaler_state,
        )

        if no_improve >= patience:
            logger.info(
                f'Early stopping at epoch {epoch+1} '
                f'(val_acc did not improve for {patience} epochs after warmup)')
            break

    return epoch + 1   # actual epochs trained (≤ MAX_EPOCHS)


@torch.no_grad()
def _eval_test(model, test_loader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds, probs, labels = [], [], []
    for Xa, Xp, y in test_loader:
        Xa    = Xa.to(device, non_blocking=True)
        Xp    = Xp.to(device, non_blocking=True)
        logits = model(Xa, Xp)
        probs.extend(torch.softmax(logits, 1).cpu().numpy().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(y.tolist())
    acc        = float(accuracy_score(labels, preds))
    f1         = float(f1_score(labels, preds, average='macro'))
    f1_per_cls = f1_score(labels, preds, average=None,
                          labels=list(range(11))).tolist()
    cm         = confusion_matrix(labels, preds,
                                  labels=list(range(11))).tolist()
    return acc, f1, f1_per_cls, cm, preds, probs, labels


def run_one(protocol: str, fold_idx: int, seed: int,
            output_dir: Path, data_root=None) -> dict:
    """Train and evaluate one (protocol, fold, seed) combo.

    Resume-safe: if checkpoint.pt exists, training resumes from it.
    Skipping is handled by the caller (run_all).
    """
    run_key   = f'fold{fold_idx}_seed{seed:02d}'
    ckpt_dir  = output_dir / 'checkpoints' / run_key
    log_dir   = output_dir / 'logs'
    ckpt_path      = ckpt_dir / 'checkpoint.pt'
    ckpt_best_path = ckpt_dir / 'checkpoint_best.pt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(run_key)
    logger.info(f'run_one | protocol={protocol} fold={fold_idx} seed={seed}')

    set_seed(seed)
    configure_speed_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model     = APWMamba().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = make_optimizer(model, lr=LR,
                               weight_decay=WEIGHT_DECAY, betas=BETAS)
    scheduler = build_lr_scheduler(optimizer,
                                   warmup_epochs=WARMUP_EPOCHS,
                                   total_epochs=MAX_EPOCHS,
                                   floor_ratio=0.1)

    # Mixed precision — enabled by default on CUDA. The GradScaler state is
    # checkpointed alongside optimizer/scheduler for safe resume.
    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    def _make_loaders():
        if protocol == 'split':
            return build_noval_loaders_with_val(BATCH_SIZE, data_root=data_root)
        return build_loso_loaders_with_val(fold_idx, BATCH_SIZE, data_root=data_root)

    train_loader, val_loader, test_loader = _make_loaders()

    # ── Resume or fresh start ─────────────────────────────────────────────────
    start_epoch = 0
    start_best_acc = 0.0
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, model, optimizer, scheduler=scheduler)
        start_epoch = int(ckpt['epoch']) + 1
        if scaler is not None and ckpt.get('scaler_state') is not None:
            scaler.load_state_dict(ckpt['scaler_state'])
        if ckpt_best_path.exists():
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                _best = torch_load_checkpoint(ckpt_best_path, map_location=device)
                start_best_acc = float(_best.get('best_acc', 0.0))
            except Exception:
                pass
        logger.info(f'Resuming from epoch {start_epoch} | best_acc={start_best_acc:.4f}')
    else:
        sanity_check_apwmamba(model, train_loader, criterion)
        save_environment_info(log_dir / f'{run_key}_env.txt', seed=seed)
        logger.info(f'Fresh start | batch={BATCH_SIZE} | lr={LR:.2e} | amp={use_amp}')

    # ── CSV log ───────────────────────────────────────────────────────────────
    csv_path   = log_dir / f'{run_key}.csv'
    append_csv = start_epoch > 0 and csv_path.exists()
    csv_file   = csv_path.open('a' if append_csv else 'w',
                               newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not append_csv:
        csv_writer.writeheader()

    # ── Training ──────────────────────────────────────────────────────────────
    t_start = time.time()
    disable_windows_sleep()
    try:
        epochs_trained = _train_epochs(
            model, optimizer, scheduler, criterion,
            train_loader, val_loader, test_loader,
            start_epoch, ckpt_path, ckpt_best_path, csv_writer, csv_file, logger,
            start_best_acc=start_best_acc, patience=PATIENCE,
            warmup_epochs=WARMUP_EPOCHS, scaler=scaler, device=device)
    finally:
        csv_file.close()
        restore_windows_sleep()

    # ── Save clean final state_dict (just weights, no optimizer/scheduler) ───
    torch.save(model.state_dict(), ckpt_dir / 'final_model.pt')

    # ── Final test eval: load best val checkpoint (C1) ───────────────────────
    if ckpt_best_path.exists():
        load_checkpoint(ckpt_best_path, model)
        # Also write a clean best_model.pt (state_dict only).
        torch.save(model.state_dict(), ckpt_dir / 'best_model.pt')
        logger.info('Loaded checkpoint_best (best val_acc) for final test evaluation')

    acc, f1, f1_per_cls, cm, preds, probs, gt_labels = _eval_test(model, test_loader, device=device)
    total_time = round(time.time() - t_start)

    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f'{run_key}.npz',
             predictions=np.array(preds),
             probabilities=np.array(probs),
             labels=np.array(gt_labels))

    logger.info(f'Test | acc={acc:.4f} | f1={f1:.4f} | time={total_time}s')
    print(f'\n  [{run_key}] Acc={acc*100:.2f}%  F1={f1*100:.2f}%  ({total_time}s)')

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


def run_all(protocol: str = 'split', n_seeds: int = 1,
            fold_range=None, output_dir: Path = None,
            data_root=None) -> dict:
    """Outer loop over all (fold, seed) combos for a given protocol.

    Completed runs (result JSON exists) are skipped automatically — safe to
    call across multiple Kaggle sessions to finish a long LOSO run.

    Args:
        protocol:      "split" | "loso"
        n_seeds:       1 (seed=4 only) | 3 (seeds 4, 8, 17)
        fold_range:    list of fold indices to run; None = all folds
        output_dir:       output root; defaults to OUTPUT_DIR from config
        data_root: preprocessed .npy directory; defaults to DATA_ROOT

    Returns:
        summary dict with mean+-std across all completed runs
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds           = SEEDS[:n_seeds]
    n_folds         = 1 if protocol == 'split' else 5
    effective_folds = fold_range if fold_range is not None else list(range(n_folds))

    print(f'\n{"="*65}')
    print(f'  APWMamba | protocol={protocol} | seeds={seeds} | folds={effective_folds}')
    print(f'{"="*65}')

    all_results: dict = {}

    for fold_idx in effective_folds:
        for seed in seeds:
            run_key     = f'fold{fold_idx}_seed{seed:02d}'
            result_path = output_dir / 'results' / f'{run_key}.json'

            if _is_run_complete(output_dir, run_key):
                with open(result_path) as f:
                    all_results[run_key] = json.load(f)
                print(f'  [skip] {run_key} — already done '
                      f'(acc={all_results[run_key]["test_accuracy"]*100:.2f}%)')
                continue

            result = run_one(protocol, fold_idx, seed, output_dir, data_root)

            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            _mark_run_complete(output_dir, run_key)
            all_results[run_key] = result

    if not all_results:
        print('No completed runs to aggregate.')
        return {}

    accs = [v['test_accuracy'] for v in all_results.values()]
    f1s  = [v['test_f1_macro']  for v in all_results.values()]

    summary_path = output_dir / 'summary.json'

    # Reuse cached efficiency metrics — avoid re-instantiating model on every aggregate call
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
        _eff_model = APWMamba().to(_eff_dev)
        params_m, model_size_mb, macs_g, macs_note, lat_mean, lat_std = measure_efficiency(
            _eff_model, _eff_dev,
            xh_shape=(27, 500, 15),
            xv_shape=(18, 500, 15),
        )
        del _eff_model

    summary = {
        'model':           'apwmamba',
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
    print(f'  SUMMARY — {protocol.upper()} | {len(all_results)} runs')
    print(f'  Accuracy : {summary["acc_mean"]*100:.2f} +- {summary["acc_std"]*100:.2f}%')
    print(f'  F1 Macro : {summary["f1_mean"]*100:.2f} +- {summary["f1_std"]*100:.2f}%')
    print(f'  Params   : {params_m:.3f}M  |  Size: {model_size_mb:.2f} MB')
    if macs_g:
        print(f'  MACs     : {macs_g:.3f}G  [{macs_note}]')
    if lat_mean is not None:
        print(f'  Latency  : {lat_mean:.2f} +- {lat_std:.2f} ms')
    print(f'  Saved    : {summary_path}')
    print(f'{"="*65}\n')

    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='APWMamba multi-run trainer')
    parser.add_argument('--protocol',   default='split', choices=['split', 'loso'])
    parser.add_argument('--n_seeds',    type=int, default=1, choices=[1, 3])
    parser.add_argument('--fold_range', type=int, nargs='*', default=None,
                        help='Fold indices to run (default: all)')
    args = parser.parse_args()
    run_all(args.protocol, args.n_seeds, args.fold_range)
