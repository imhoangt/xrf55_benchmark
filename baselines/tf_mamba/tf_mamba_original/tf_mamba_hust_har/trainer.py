"""Train and evaluate TF-Mamba on HUST-HAR.

Reproduces paper results: ~99.72% accuracy, ~0.104M params.

Selection protocol: no validation set — checkpoint saved when early stop triggers
(avg_loss < 0.01) or at final epoch. Per-epoch test evaluation is monitoring only.

Usage:
    cd har_csi
    python baselines/tf_mamba_hust_har/trainer.py

Outputs saved to runs/tf_mamba_husthar/:
    final_model.pt         checkpoint at early stop (or last epoch)
    metrics.json           final metrics in 0-1 range
    training_log.csv       per-epoch loss, test_acc (monitoring only)
    test_predictions.npz   predictions, probabilities, labels
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.base_models.tf_mamba_base.train_utils import evaluate, evaluate_full, measure_efficiency
from baselines.tf_mamba.tf_mamba_original.tf_mamba_hust_har.dataset import build_loaders
from baselines.tf_mamba.tf_mamba_original.tf_mamba_hust_har.model import TFMamba
from src.training.amp_utils import torch_load_checkpoint
from src.training.train_utils import set_seed

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT       = PROJECT_ROOT / 'dataset' / 'HUST-HAR'
OUTPUT_DIR      = PROJECT_ROOT / 'outputs' / 'runs' / 'tf_mamba_husthar'

BATCH_SIZE      = 32
LR              = 1e-4   # paper Section V-B: lr = 1×10⁻⁴
NUM_EPOCHS      = 40
EARLY_STOP_LOSS = 0.01   # paper protocol: stop when train loss drops below this
GRAD_CLIP       = 1.0

D_MODEL      = 64
NUM_LAYERS   = 3
NUM_CLASSES  = 6
NUM_FEATURES = 135    # M = 270/2 = 135 after DWT
MAX_LEN      = 500    # L = 1000/2 = 500 after DWT


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    # Paper protocol: deterministic cuDNN for exact reproducibility.
    # XRF55 baselines override this via configure_speed_mode(); HUST-HAR does not.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_loader, test_loader = build_loaders(
        DATA_ROOT, batch_size=BATCH_SIZE, num_workers=4
    )
    print(f'Train: {len(train_loader.dataset)} samples  '
          f'Test: {len(test_loader.dataset)} samples')

    model = TFMamba(
        num_features=NUM_FEATURES,
        d_model=D_MODEL, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
        max_len=MAX_LEN,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_params:,} ({total_params/1e6:.3f}M)')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    log_rows   = []
    t_start    = time.time()
    saved      = False
    stopped_ep = NUM_EPOCHS

    for epoch in range(1, NUM_EPOCHS + 1):
        t_epoch = time.time()
        model.train()
        epoch_loss = 0.0
        grad_norms = []
        for XH, XV, labels in train_loader:
            XH, XV, labels = XH.to(device), XV.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(XH, XV)
            loss   = criterion(logits, labels)
            loss.backward()
            grad_norms.append(nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item())
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss      = epoch_loss / len(train_loader)
        grad_norm_avg = float(np.mean(grad_norms))
        epoch_time    = time.time() - t_epoch
        test_acc, test_f1 = evaluate(model, test_loader, device)
        elapsed       = time.time() - t_start

        print(f'Epoch {epoch:3d}/{NUM_EPOCHS}  '
              f'Loss: {avg_loss:.4f}  '
              f'Test: {test_acc*100:.2f}%  F1: {test_f1*100:.2f}%  '
              f'({elapsed:.0f}s)')

        log_rows.append({
            'epoch': epoch, 'lr': LR, 'train_loss': avg_loss,
            'grad_norm': round(grad_norm_avg, 6),
            'epoch_time_s': round(epoch_time, 2),
            'test_acc': test_acc, 'test_f1_macro': test_f1,
        })

        if avg_loss < EARLY_STOP_LOSS:
            print(f'Early stop: loss {avg_loss:.4f} < {EARLY_STOP_LOSS}')
            torch.save(model.state_dict(), OUTPUT_DIR / 'final_model.pt')
            saved      = True
            stopped_ep = epoch
            break

    if not saved:
        torch.save(model.state_dict(), OUTPUT_DIR / 'final_model.pt')

    # Log CSV
    with open(OUTPUT_DIR / 'training_log.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'lr', 'train_loss', 'grad_norm', 'epoch_time_s', 'test_acc', 'test_f1_macro'])
        writer.writeheader()
        writer.writerows(log_rows)

    # Final eval — test set evaluated exactly once with the saved checkpoint
    model.load_state_dict(torch_load_checkpoint(OUTPUT_DIR / 'final_model.pt', map_location=device))
    acc, f1, f1_per_cls, cm, all_preds, all_probs, all_labels = \
        evaluate_full(model, test_loader, device, NUM_CLASSES)

    np.savez(OUTPUT_DIR / 'test_predictions.npz',
             predictions=np.array(all_preds),
             probabilities=np.array(all_probs),
             labels=np.array(all_labels))

    params_m, model_size_mb, macs_g, macs_note, lat_mean, lat_std = \
        measure_efficiency(model, device, xh_shape=(MAX_LEN, NUM_FEATURES), xv_shape=(MAX_LEN, NUM_FEATURES))

    metrics = {
        'model':                  'tf_mamba_husthar',
        'dataset':                'hust_har',
        'selection_method':       'early_stop_loss_lt_0.01_or_last_epoch',
        'test_accuracy':          round(acc, 6),
        'test_f1_macro':          round(f1, 6),
        'test_f1_per_class':      [round(v, 6) for v in f1_per_cls],
        'test_confusion_matrix':  cm,
        'stopped_epoch':          stopped_ep,
        'params_M':               round(params_m, 3),
        'model_size_mb':          round(model_size_mb, 2),
        'macs_G':                 macs_g,
        'macs_note':              macs_note,
        'latency_mean_ms':        lat_mean,
        'latency_std_ms':         lat_std,
        'total_time_s':           round(time.time() - t_start),
    }
    with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('\n=== Final Results (TF-Mamba HUST-HAR) ===')
    print(f'  Accuracy : {acc*100:.2f}%  (paper: 99.72%)')
    print(f'  F1 Macro : {f1*100:.2f}%')
    print(f'  Params   : {params_m:.3f}M  (paper: 0.104M)  |  Size: {model_size_mb:.2f} MB')
    if macs_g:
        print(f'  MACs     : {macs_g:.3f}G  (paper: 0.829G)  [{macs_note}]')
    if lat_mean is not None:
        print(f'  Latency  : {lat_mean:.2f} +/- {lat_std:.2f} ms')
    print(f'  Saved to : {OUTPUT_DIR}')


def run(data_root=None, output_dir=None):
    """Callable wrapper for notebook/script usage with custom paths."""
    global DATA_ROOT, OUTPUT_DIR
    if data_root is not None:
        DATA_ROOT = Path(data_root)
    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir)
    main()


if __name__ == '__main__':
    main()
