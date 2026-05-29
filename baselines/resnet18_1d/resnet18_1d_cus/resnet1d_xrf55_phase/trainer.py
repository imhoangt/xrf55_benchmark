"""Train ResNet18-1D-XRF55-Phase — multi-protocol, multi-seed, resume-safe.

Thin wrapper around `src.training.xrf55_trainer.run_all_experiments`; this file
holds only the baseline-specific config (model architecture, input shape,
loader builders) and delegates the training loop to the shared module.

Protocols
---------
  "split" : rep-split with val — train=reps 1-12, val=reps 13-14, test=reps 15-20  (1 fold)
  "loso"  : LOSO-5fold rotation — test=G[i], val=G[(i+1)%5], train=3 remaining groups

Seeds: n_seeds=1 → [4]; n_seeds=3 → [4, 8, 17]

Usage:
    cd har_csi
    python baselines/resnet1d_xrf55_phase/trainer.py --protocol split --n_seeds 1
    python baselines/resnet1d_xrf55_phase/trainer.py --protocol loso  --n_seeds 3
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.base_models.resnet1d_base.train_utils import measure_efficiency
from baselines.base_models.resnet1d_base.model import resnet18
from baselines.resnet1d_xrf55_phase.dataset import (
    build_noval_loaders_with_val, build_loso_loaders_with_val,
)
from src.training.xrf55_trainer import ExperimentConfig, run_all_experiments

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT  = PROJECT_ROOT / 'dataset' / 'xrf55' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'runs' / 'resnet1d_xrf55_phase'

INCHANNEL   = 180
NUM_CLASSES = 11
SEQ_LEN     = 1000


def _model_factory():
    return resnet18(inchannel=INCHANNEL, num_classes=NUM_CLASSES)


def _measure_efficiency(model, device):
    return measure_efficiency(model, device, x_shape=(INCHANNEL, SEQ_LEN))


CONFIG = ExperimentConfig(
    model_name='resnet1d_xrf55_phase',
    output_dir=OUTPUT_DIR,
    data_root=DATA_ROOT,
    build_loaders_split=build_noval_loaders_with_val,
    build_loaders_loso=build_loso_loaders_with_val,
    model_factory=_model_factory,
    measure_efficiency_fn=_measure_efficiency,
    num_classes=NUM_CLASSES,
)


def run_all(protocol: str = 'split', n_seeds: int = 1,
            fold_range=None, output_dir=None, data_root=None) -> dict:
    return run_all_experiments(CONFIG, protocol=protocol, n_seeds=n_seeds,
                               fold_range=fold_range,
                               output_dir=output_dir, data_root=data_root)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol',   default='split', choices=['split', 'loso'])
    parser.add_argument('--n_seeds',    type=int, default=1, choices=[1, 3])
    parser.add_argument('--fold_range', type=int, nargs='*', default=None)
    args = parser.parse_args()
    run_all(args.protocol, args.n_seeds, args.fold_range)


if __name__ == '__main__':
    main()
