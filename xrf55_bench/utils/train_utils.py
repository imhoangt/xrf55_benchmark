"""Seeding, runtime speed configuration, and checkpoint I/O for the XRF55 benchmark."""
import os
import random

import numpy as np
import torch


def torch_load_checkpoint(path, map_location=None):
    """Load checkpoints with weights_only=False when supported."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def configure_speed_mode():
    """Kaggle speed mode: faster, not bit-level deterministic.

    Enables cuDNN auto-tuning and TF32 matmul for fair speed comparison across
    baselines. Call AFTER set_seed() (set_seed here does not touch cudnn flags,
    so the order is not load-bearing, but keep this convention).
    """
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
