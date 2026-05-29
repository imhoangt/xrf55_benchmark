"""LOSO-5fold split generator for XRF55.

30 subjects, 11 actions, 20 reps = 6 600 samples total.
5 folds of 6 subjects each (sequential grouping).

vol_id and rep_id reconstruction are purely analytical — derived from the
deterministic loop order in 01_preprocess.py (vol_id outer, action_id middle,
rep_id inner). No extra files need to be saved or preprocessing re-run.

Preprocessing loop order:
  for vol_id in range(1, 31):
    for action_id in ACTION_IDS_USED:      # 11 actions
      for rep_id in range(1, 21):          # 20 reps

Cached split files contain samples in that order, filtered by rep_id:
  train  → reps  1-12  →  12 × 11 = 132 samples per subject  (3 960 total)
  val    → reps 13-14  →   2 × 11 =  22 samples per subject  (  660 total)
  test   → reps 15-20  →   6 × 11 =  66 samples per subject  (1 980 total)

Validation scheme (rotation, 1 inner fold per outer fold):
  fold i: test=G[i], val=G[(i+1)%5], train=remaining 3 groups
  Val subjects use ALL 20 reps (all three split files), same as train/test.
"""
import numpy as np

# 5 groups of 6 subjects each — sequential grouping
LOSO_FOLD_SUBJECTS: list[list[int]] = [
    list(range(1,  7)),   # fold 0 / G1: subjects  1– 6
    list(range(7,  13)),  # fold 1 / G2: subjects  7–12
    list(range(13, 19)),  # fold 2 / G3: subjects 13–18
    list(range(19, 25)),  # fold 3 / G4: subjects 19–24
    list(range(25, 31)),  # fold 4 / G5: subjects 25–30
]

_N_GROUPS   = len(LOSO_FOLD_SUBJECTS)   # 5
_N_SUBJECTS = 30
_N_ACTIONS  = 11
_REPS_PER_SPLIT = {'train': 12, 'val': 2, 'test': 6}
_FIRST_REP      = {'train':  1, 'val': 13, 'test': 15}


def get_loso_val_subjects(fold_idx: int) -> list[int]:
    """Return the 6 val subjects for fold *fold_idx* (rotation scheme).

    Rotation: test=G[i], val=G[(i+1)%5], train=remaining 3 groups.
    """
    if not 0 <= fold_idx < _N_GROUPS:
        raise ValueError(f"fold_idx={fold_idx} out of range [0, {_N_GROUPS - 1}]")
    return LOSO_FOLD_SUBJECTS[(fold_idx + 1) % _N_GROUPS]


def get_vol_ids(split: str) -> np.ndarray:
    """Return vol_id (1-indexed) for every sample in the cached split file.

    The array is parallel to the cached .npy arrays — same index = same sample.
    """
    n_per_subj = _REPS_PER_SPLIT[split] * _N_ACTIONS
    return np.repeat(np.arange(1, _N_SUBJECTS + 1), n_per_subj)


def get_rep_ids(split: str) -> np.ndarray:
    """Return rep_id (1-indexed) for every sample in the cached split file.

    Analytically derived from preprocessing loop order: vol→action→rep.
    Within each (vol, action) block of n_reps samples, rep cycles
    1..12 (train), 13..14 (val), or 15..20 (test).
    """
    n_reps    = _REPS_PER_SPLIT[split]
    first_rep = _FIRST_REP[split]
    return np.tile(np.arange(first_rep, first_rep + n_reps),
                   _N_SUBJECTS * _N_ACTIONS)


def get_all_vol_ids() -> np.ndarray:
    """vol_ids for the full dataset (train + val + test concatenated)."""
    return np.concatenate([
        get_vol_ids('train'),
        get_vol_ids('val'),
        get_vol_ids('test'),
    ])


def loso_fold_masks(vol_ids: np.ndarray, fold_idx: int):
    """Return boolean (train_mask, test_mask) for fold *fold_idx*.

    train_mask: True for all subjects NOT in this fold's test group
    test_mask:  True for the 6 test subjects of this fold
    """
    if not 0 <= fold_idx < _N_GROUPS:
        raise ValueError(
            f"fold_idx={fold_idx} out of range [0, {_N_GROUPS - 1}]")
    test_subjects = set(LOSO_FOLD_SUBJECTS[fold_idx])
    test_mask  = np.isin(vol_ids, list(test_subjects))
    train_mask = ~test_mask
    return train_mask, test_mask
