"""Training configuration and protocol presets for the XRF55 benchmark."""
import copy
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainCfg:
    protocol:         str            = '01'    # '01' | '02' | '03'

    # Hyperparameters
    lr:               float          = 1e-3
    batch_size:       int            = 32
    num_epochs:       int            = 40
    grad_clip:        Optional[float]= None    # None = no clipping
    weight_decay:     float          = 0.0

    # Optimizer: 'adamw' | 'adam' | 'sgd'
    optimizer:        str            = 'adamw'
    betas:            tuple          = (0.9, 0.999)
    eps:              float          = 1e-8

    # Scheduler: None | 'cosine' | 'multistep' | 'warmup_cosine'
    scheduler:        Optional[str]  = None
    warmup_epochs:    int            = 0
    floor_lr:         float          = 1e-5
    scheduler_kwargs: Optional[dict] = None

    # Loss: 'ce' (label smoothing controlled by label_smoothing below)
    criterion:        str            = 'ce'
    label_smoothing:  float          = 0.0

    # Data mode: 'raw' | 'proc' | None (None = auto-infer from stats.json meta)
    data_mode:        Optional[str]  = None

    # Seeds — (42,) single default, (4, 8, 17, 42) multi
    seeds:            tuple          = (42,)


_PROTOCOL_DEFAULTS = {
    '01': dict(                         # plain — tf_mamba paper
        optimizer='adamw', lr=1e-4,  batch_size=32, num_epochs=40,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
        scheduler=None, warmup_epochs=0,
        grad_clip=None, criterion='ce', label_smoothing=0.0,
    ),
    '02': dict(                         # xrf55 paper
        optimizer='adam',  lr=1e-3,  batch_size=64, num_epochs=200,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        scheduler='multistep',
        scheduler_kwargs={'milestones': [40, 80, 120, 160], 'gamma': 0.5},
        warmup_epochs=0,
        grad_clip=None, criterion='ce', label_smoothing=0.0,
    ),
    '03': dict(                         # apwmamba paper
        optimizer='adamw', lr=5e-4,  batch_size=32, num_epochs=200,
        betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-3,
        scheduler='warmup_cosine', warmup_epochs=10, floor_lr=4e-5,
        grad_clip=None, criterion='ce', label_smoothing=0.1,
    ),
}


def TrainCfg_for_protocol(protocol: str, **overrides) -> TrainCfg:
    """Return TrainCfg with protocol-specific defaults, optionally overridden."""
    if protocol not in _PROTOCOL_DEFAULTS:
        raise ValueError(
            f"Unknown protocol {protocol!r}. Choose from: {list(_PROTOCOL_DEFAULTS)}")
    defaults = copy.deepcopy(_PROTOCOL_DEFAULTS[protocol])
    defaults.update(overrides)
    return TrainCfg(protocol=protocol, **defaults)
