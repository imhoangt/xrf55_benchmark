"""Unified evaluation + efficiency probe for all XRF55-bench models.

Works for single-stream models (loader yields (X, y)) and dual-stream models
(loader yields (XH, XV, y)) via generic `*inputs, y` unpacking — no per-model
variant needed. Used by trainer.py for resnet / tfmamba / wavmamba alike.
"""
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate(model, loader, device):
    """Quick eval. Returns (acc, f1_macro). Handles 1- or 2-stream loaders."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for *inputs, y in loader:
            inputs = [t.to(device) for t in inputs]
            preds += model(*inputs).argmax(1).cpu().tolist()
            gts   += y.tolist()
    return accuracy_score(gts, preds), f1_score(gts, preds, average='macro')


def evaluate_full(model, loader, device, num_classes):
    """Full eval. Returns (acc, f1, f1_per_cls, cm, preds, probs, gts)."""
    model.eval()
    preds, probs, gts = [], [], []
    with torch.no_grad():
        for *inputs, y in loader:
            inputs = [t.to(device) for t in inputs]
            logits = model(*inputs)
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
        input_shapes: tuple of per-input shapes WITHOUT batch dim. One entry per
            forward argument:
                single-stream : ((270, 1000),)             → model(X)
                dual-stream    : ((500, 135), (500, 135))  → model(XH, XV)
                wavmamba       : ((27, 500, 15),)          → model(X)
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
        macs_g    = round(_flops.total() / 2e9, 3)   # FLOPs / 2 = MACs
        macs_note = 'fvcore (Mamba selective_scan_cuda excluded if present)'
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
