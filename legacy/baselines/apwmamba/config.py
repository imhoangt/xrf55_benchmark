"""APWMamba training configuration — loaded from configs/apwmamba.yaml.

All hyperparameters live in that YAML file. This module exposes the same
flat constants so the rest of the codebase can do simple imports:

    from baselines.apwmamba.config import MAX_EPOCHS, LR, ...
"""
import math
from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).parent.parent.parent / 'configs' / 'apwmamba.yaml'
try:
    _cfg = yaml.safe_load(_YAML_PATH.read_text(encoding='utf-8'))
except FileNotFoundError as e:
    raise FileNotFoundError(
        f'APWMamba config YAML not found at {_YAML_PATH}. '
        f'On Kaggle, ensure the code dataset is attached and includes configs/apwmamba.yaml.'
    ) from e
except yaml.YAMLError as e:
    raise RuntimeError(f'Malformed APWMamba YAML at {_YAML_PATH}: {e}') from e

_tr  = _cfg['training']
_inf = _cfg['infrastructure']

# ── Training ──────────────────────────────────────────────────────────────────
SEEDS           = list(_tr['seeds'])           # [4, 8, 17]
MAX_EPOCHS      = int(_tr['max_epochs'])        # 100
BATCH_SIZE      = int(_tr['batch_size'])        # 32 (fixed)
BASE_BATCH_SIZE = int(_tr['base_batch_size'])   # 16 (LR scaling reference)
BASE_LR         = float(_tr['base_lr'])         # 5e-4
LR_CAP          = float(_tr['lr_cap'])          # 1e-3 (matches TF-Mamba/ResNet1D)
LR              = min(BASE_LR * math.sqrt(BATCH_SIZE / BASE_BATCH_SIZE), LR_CAP)
WEIGHT_DECAY    = float(_tr['weight_decay'])    # 1e-4
BETAS           = tuple(_tr['betas'])           # (0.9, 0.95)
WARMUP_EPOCHS   = int(_tr['warmup_epochs'])     # 5
GRAD_CLIP_NORM  = float(_tr['grad_clip_norm'])  # 1.0
LABEL_SMOOTHING = float(_tr['label_smoothing']) # 0.10
PATIENCE        = int(_tr['early_stopping']['patience'])  # 10

# ── Infrastructure ────────────────────────────────────────────────────────────
NUM_WORKERS        = int(_inf['num_workers'])
PIN_MEMORY         = bool(_inf['pin_memory'])
PERSISTENT_WORKERS = bool(_inf['persistent_workers'])
PREFETCH_FACTOR    = int(_inf['prefetch_factor'])

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = _PROJECT_ROOT / _cfg['paths']['data_processed']

if Path('/kaggle/working').exists():
    OUTPUT_DIR = Path('/kaggle/working') / _cfg['paths']['runs_dir'] / _cfg['paths']['current_run']
else:
    OUTPUT_DIR = _PROJECT_ROOT / _cfg['paths']['runs_dir'] / _cfg['paths']['current_run']
