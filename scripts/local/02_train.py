"""Training entry point for APWMamba (local).

Usage examples:
    python scripts/local/02_train.py                              # split, 1 seed (default)
    python scripts/local/02_train.py --protocol loso --n_seeds 3  # full LOSO, 3 seeds
    python scripts/local/02_train.py --protocol loso --fold_range 0 1  # LOSO folds 0 & 1 only
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.apwmamba.trainer import run_all


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--protocol',   default='split', choices=['split', 'loso'],
                   help="Eval protocol: 'split' (1 fold) or 'loso' (5 folds). Default: split")
    p.add_argument('--n_seeds',    type=int, default=1, choices=[1, 3],
                   help='Number of seeds: 1 -> [4]; 3 -> [4, 8, 17]. Default: 1')
    p.add_argument('--fold_range', type=int, nargs='*', default=None,
                   help='Restrict to specific fold indices (LOSO only), e.g. --fold_range 0 1')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_all(protocol=args.protocol,
            n_seeds=args.n_seeds,
            fold_range=args.fold_range)
