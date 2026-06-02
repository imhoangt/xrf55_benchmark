"""Verify XRF55 dataset action IDs match expected [31..41]."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.verify_ids import scan_action_ids

if __name__ == '__main__':
    raw_dir = PROJECT_ROOT / 'dataset' / 'xrf55' / 'raw' / 'scene_01'
    if not raw_dir.exists():
        raise FileNotFoundError(f'Raw dataset not found: {raw_dir}')
    scan_action_ids(raw_dir)
