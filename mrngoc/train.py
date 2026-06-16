"""Huấn luyện + đánh giá TF-Mamba gốc theo PROTOCOL của tác giả.

Protocol (theirs) — khớp HUST_HAR/Mamba_HUST-HAR.py + paper:
    AdamW  lr=1e-4 (paper; code release dùng 1e-3 — ta theo paper vì là số công bố)
    weight_decay=0.01, betas=(0.9,0.999), eps=1e-8
    CrossEntropyLoss, 40 epoch, batch_size=32
    clip_grad_norm_(max_norm=1.0)            (code dòng 140)
    early-stop: dừng khi train-loss trung bình < 0.01
    KHÔNG scheduler
1 seed cố định = 42 (chặt hơn tác giả: họ random_split không seed, chạy 1 lần).
Đánh giá model CUỐI (last) — giống evaluate_model của họ.
"""
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from data import make_datasets
from model import TFMamba

# ── PROTOCOL (theirs) ────────────────────────────────────────────────────────
PROTO = dict(lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8,
             num_epochs=40, batch_size=32, grad_clip=1.0, early_stop_loss=0.01)
SEED = 42
NORM_MODE = 'author'   # 'author' (SenseFi norm, dung tac gia) | 'double' (+ z-norm)
MERGE_VAL = False      # CHI UT-HAR: False = git SenseFi (test=X_test) | True = gop val vao test


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train(model, loader, device):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=PROTO['lr'],
                            weight_decay=PROTO['weight_decay'],
                            betas=PROTO['betas'], eps=PROTO['eps'])
    model.train()
    for epoch in range(PROTO['num_epochs']):
        t0 = time.time()
        tot = 0.0
        for xh, xv, y in loader:
            xh, xv, y = xh.to(device), xv.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(xh, xv), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), PROTO['grad_clip'])
            opt.step()
            tot += loss.item()
        avg = tot / len(loader)
        print(f"    epoch {epoch+1:2d}/{PROTO['num_epochs']}  "
              f"loss={avg:.4f}  {time.time()-t0:.1f}s")
        if avg < PROTO['early_stop_loss']:
            print(f"    early stop @ epoch {epoch+1} (loss {avg:.4f} < "
                  f"{PROTO['early_stop_loss']})")
            break


@torch.no_grad()
def _evaluate(model, loader, device, num_classes):
    model.eval()
    preds, labels, probs = [], [], []
    for xh, xv, y in loader:
        xh, xv = xh.to(device), xv.to(device)
        out = model(xh, xv)
        probs.extend(out.softmax(dim=1).cpu().numpy())
        preds.extend(out.argmax(dim=1).cpu().numpy())
        labels.extend(y.numpy())
    labels, preds, probs = np.array(labels), np.array(preds), np.array(probs)
    m = dict(
        accuracy=accuracy_score(labels, preds),
        precision=precision_score(labels, preds, average='macro', zero_division=0),
        recall=recall_score(labels, preds, average='macro', zero_division=0),
        f1=f1_score(labels, preds, average='macro', zero_division=0),
        confusion=confusion_matrix(labels, preds).tolist(),
    )
    try:
        m['auc'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        m['auc'] = float('nan')
    return m


def run_dataset(name, raw_root, device='cuda', seed=SEED, norm_mode=NORM_MODE,
                merge_val=MERGE_VAL):
    """Chạy 1 dataset: load -> DWT -> train -> eval. Trả dict metrics."""
    print(f"\n===== {name.upper()} (seed {seed}, norm={norm_mode}, merge_val={merge_val}) =====")
    set_seed(seed)
    train_ds, test_ds, meta = make_datasets(name, raw_root, norm_mode=norm_mode,
                                            merge_val=merge_val)
    train_loader = DataLoader(train_ds, batch_size=PROTO['batch_size'],
                              shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=PROTO['batch_size'],
                             shuffle=False, pin_memory=True)
    model = TFMamba(num_features=meta['M'], num_classes=meta['num_classes'],
                    max_len=meta['T2']).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  TFMamba params={n_param/1e6:.3f}M  max_len={meta['T2']}  M={meta['M']}")
    t0 = time.time()
    _train(model, train_loader, device)
    metrics = _evaluate(model, test_loader, device, meta['num_classes'])
    metrics.update(dataset=name, seed=seed, norm_mode=norm_mode,
                   merge_val=meta['merge_val'], params=n_param,
                   train_time_s=round(time.time() - t0, 1),
                   n_train=meta['n_train'], n_test=meta['n_test'])
    print(f"  -> acc={metrics['accuracy']*100:.2f}  f1={metrics['f1']*100:.2f}  "
          f"auc={metrics['auc']*100:.2f}  ({metrics['train_time_s']}s)")
    return metrics


def run_all(specs, device='cuda', seed=SEED, norm_mode=NORM_MODE, merge_val=MERGE_VAL):
    """specs: dict {name: raw_root}. Chạy tuần tự, in bảng tổng hợp, trả list."""
    results = []
    for name, root in specs.items():
        results.append(run_dataset(name, root, device=device, seed=seed,
                                   norm_mode=norm_mode, merge_val=merge_val))
    print("\n===== TONG HOP =====")
    print(f"{'dataset':8s} {'acc':>7s} {'precision':>10s} {'recall':>7s} "
          f"{'f1':>7s} {'auc':>7s} {'params':>8s} {'time':>7s}")
    for r in results:
        print(f"{r['dataset']:8s} {r['accuracy']*100:7.2f} {r['precision']*100:10.2f} "
              f"{r['recall']*100:7.2f} {r['f1']*100:7.2f} {r['auc']*100:7.2f} "
              f"{r['params']/1e6:7.3f}M {r['train_time_s']:6.0f}s")
    return results
