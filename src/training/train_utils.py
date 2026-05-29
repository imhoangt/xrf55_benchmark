import math
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from src.training.amp_utils import torch_load_checkpoint

ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def configure_speed_mode():
    """Kaggle speed mode: faster, not bit-level deterministic.

    NOTE: Some `set_seed()` implementations (e.g. baselines/tf_mamba_base) set
    `cudnn.deterministic = True` for reproducibility. This function INTENTIONALLY
    overrides that to enable cuDNN auto-tuning and TF32 matmul — required for
    fair speed comparison across baselines. Always call AFTER `set_seed()`.
    """
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


def disable_windows_sleep():
    if platform.system() == 'Windows':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)


def restore_windows_sleep():
    if platform.system() == 'Windows':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def make_optimizer(model, lr=5e-4, weight_decay=1e-4, betas=(0.9, 0.95)):
    """No weight decay for: biases, norms, LayerScale (.gamma), gate scalars,
    Mamba A_log/D, embeddings/cls/pos tokens."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (param.ndim <= 1
                or name.endswith('.bias')
                or name.endswith('.gamma')
                or name.endswith('.alpha')
                or name.endswith('.pos_alpha')
                or name.endswith('.pos_beta')
                or name.endswith('.gate_bias')
                or name.endswith('.pos_bias')
                or 'norm' in name.lower()
                or 'A_log' in name
                or name.endswith('.D')
                or 'embedding' in name
                or 'cls_token' in name
                or 'pos_embed' in name):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW([
        {'params': decay,    'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=lr, betas=betas, eps=1e-8)


def save_checkpoint(path, epoch, model, optimizer, scheduler=None, **extra):
    """Save model + optimizer + scheduler state for resume."""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        **extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch_load_checkpoint(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt


def save_environment_info(output_path, seed=None):
    lines = [
        f"Timestamp : {datetime.now().isoformat()}",
        f"OS        : {platform.platform()}",
        f"Python    : {sys.version.split()[0]}",
        f"PyTorch   : {torch.__version__}",
    ]
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines += [
            f"GPU       : {p.name}  {p.total_memory / 1e9:.1f} GB",
            f"CUDA      : {torch.version.cuda} / cuDNN {torch.backends.cudnn.version()}",
        ]
    lines += [
        f"Seed      : {seed}",
        f"cudnn.benchmark     : {torch.backends.cudnn.benchmark}",
        f"cudnn.deterministic : {torch.backends.cudnn.deterministic}",
    ]
    text = '\n'.join(lines)
    print(text)
    Path(output_path).write_text(text, encoding='utf-8')


def sanity_check_apwmamba(model, train_loader, criterion):
    """First-batch verification specific to APWMamba (hardcoded shapes 27/18, num_classes=11).

    Run once on fresh start only. Other baselines should not call this.
    """
    print("=" * 60)
    print("FIRST-BATCH SANITY CHECK")
    print("=" * 60)

    Xa, Xp, y = next(iter(train_loader))
    print(f"Xa: shape={tuple(Xa.shape)}, range=[{Xa.min():.2f}, {Xa.max():.2f}]")
    print(f"Xp: shape={tuple(Xp.shape)}, range=[{Xp.min():.2f}, {Xp.max():.2f}]")
    print(f"y:  shape={tuple(y.shape)}, unique={y.unique().tolist()}")

    B = Xa.shape[0]
    assert Xa.shape == (B, 27, 500, 15)
    assert Xp.shape == (B, 18, 500, 15)
    assert y.dtype == torch.long and y.min() >= 0 and y.max() <= 10
    assert not torch.isnan(Xa).any() and not torch.isnan(Xp).any()

    model.train()
    Xa, Xp, y = Xa.cuda(), Xp.cuda(), y.cuda()
    logits = model(Xa, Xp)
    assert logits.shape == (B, 11)

    loss = criterion(logits, y)
    assert not torch.isnan(loss)

    print(f"Initial loss: {loss.item():.4f} (expected ~2.40 = ln(11))")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print("SANITY CHECK PASSED")
    print("=" * 60)


def build_lr_scheduler(optimizer, warmup_epochs, total_epochs, floor_ratio=0.1):
    """LambdaLR: linear warmup → cosine decay to floor_ratio of base_lr."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor_ratio + (1.0 - floor_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)

