"""Shared training utilities for ResNet1D baselines (single-stream input)."""
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate(model, loader, device):
    """Quick eval — single-stream (X, y) loader. Returns (acc, f1_macro)."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X.to(device)).argmax(1).cpu().tolist()
            gts   += y.tolist()
    return accuracy_score(gts, preds), f1_score(gts, preds, average='macro')


def evaluate_full(model, loader, device, num_classes):
    """Full eval — single-stream (X, y) loader. Returns all metrics + raw arrays."""
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


def measure_efficiency(model, device, x_shape):
    """Params, model size, MACs, latency for single-stream model.

    Args:
        x_shape: (channels, time) — e.g. (270, 1000) for amp, (180, 1000) for phase
    """
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_f:
        tmp = Path(tmp_f.name)
    torch.save(model.state_dict(), tmp)
    model_size_mb = tmp.stat().st_size / 1e6
    tmp.unlink(missing_ok=True)

    X_d = torch.randn(1, *x_shape).to(device)

    try:
        from fvcore.nn import FlopCountAnalysis
        _flops = FlopCountAnalysis(model, X_d)
        _flops.unsupported_ops_warnings(False)
        macs_g    = round(_flops.total() / 2e9, 3)   # FLOPs / 2 = MACs
        macs_note = 'fvcore'
    except Exception:
        macs_g    = None
        macs_note = 'N/A — fvcore not installed'

    if device.type == 'cuda':
        model.eval()
        with torch.no_grad():
            for _ in range(50): model(X_d)
        timings = []
        with torch.no_grad():
            for _ in range(200):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); model(X_d); e.record()
                torch.cuda.synchronize()
                timings.append(s.elapsed_time(e))
        lat_mean = round(float(np.mean(timings)), 2)
        lat_std  = round(float(np.std(timings)),  2)
    else:
        lat_mean = lat_std = None

    return params_m, model_size_mb, macs_g, macs_note, lat_mean, lat_std
