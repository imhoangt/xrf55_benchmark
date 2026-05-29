"""Train TF-Mamba-XRF55-Amp — multi-protocol, multi-seed, resume-safe.

Thin wrapper around `src.training.xrf55_trainer.run_all_experiments`; this file
holds only the baseline-specific config (model architecture, input shapes,
loader builders) and delegates the training loop to the shared module.

Protocols
---------
  "split" : rep-split with val — train=reps 1-12, val=reps 13-14, test=reps 15-20  (1 fold)
  "loso"  : LOSO-5fold rotation — test=G[i], val=G[(i+1)%5], train=3 remaining groups

Seeds: n_seeds=1 → [4]; n_seeds=3 → [4, 8, 17]

Usage:
    cd har_csi
    python baselines/tf_mamba_xrf55_amp/trainer.py --protocol split --n_seeds 1
    python baselines/tf_mamba_xrf55_amp/trainer.py --protocol loso  --n_seeds 3
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.base_models.tf_mamba_base.train_utils import measure_efficiency
from baselines.tf_mamba_xrf55_amp.dataset import (
    build_noval_loaders_with_val, build_loso_loaders_with_val,
)
from baselines.tf_mamba_xrf55_amp.model import TFMamba
from src.training.xrf55_trainer import ExperimentConfig, run_all_experiments

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT  = PROJECT_ROOT / 'dataset' / 'xrf55' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'runs' / 'tf_mamba_xrf55_amp'

D_MODEL     = 64
NUM_LAYERS  = 3
NUM_CLASSES = 11
XH_FEATURES = 135   # features per token in XH stream (last dim)
XV_FEATURES = 500   # features per token in XV stream (last dim)
XH_SEQ_LEN  = 500   # seq_len of XH stream (axis 1) — used by positional embedding
XV_SEQ_LEN  = 135   # seq_len of XV stream (axis 1) — used by positional embedding


def _model_factory():
    return TFMamba(
        xh_features=XH_FEATURES, xv_features=XV_FEATURES,
        d_model=D_MODEL, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
        max_len_t=XH_SEQ_LEN, max_len_f=XV_SEQ_LEN,
    )


def _measure_efficiency(model, device):
    return measure_efficiency(model, device,
                              xh_shape=(XH_SEQ_LEN, XH_FEATURES),
                              xv_shape=(XV_SEQ_LEN, XV_FEATURES))


CONFIG = ExperimentConfig(
    model_name='tf_mamba_xrf55_amp',
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
