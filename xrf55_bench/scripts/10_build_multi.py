"""Build packed Haar-3 (S4.1) bench arrays for HUST-HAR / UT-HAR / NTU-Fi.

Output layout matches PreprocWavMambaDataset (dataset.py), so S4.1 trains via the
existing, tested model_name='wavdualmamba' path — no new dataset/model code.

Output per (dataset, mode): UN-NORMALIZED packed arrays + stats.json (all-reps).
Normalization (z-score per channel,bin) is applied at load time, matching XRF55.

    dataset/<DS>/bench/<mode>/
        wavmamba/X_<split>.npy   (N, 3*n_per_sub, T//2, sub//2)  float32 (un-norm)
        y_<split>.npy            (N,)  int64
        stats.json               { 'wavmamba': {mean,std (C, F2)}, 'meta': {...} }

Usage:
    python xrf55_bench/scripts/10_build_multi.py --dataset hust --mode proc
    python xrf55_bench/scripts/10_build_multi.py --dataset hust --mode raw
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xrf55_bench.preprocessing.multi_dataset import pack_haar3

DATA_ROOT = PROJECT_ROOT / 'dataset'
SPLIT_SEED = 42   # fixed 80/20 split, reproducible across the 5 model-init seeds


# ── Dataset loaders → (X_all (N, n_ant*sub, time) float32, y_all (N,) int64) ────
# Each loader takes a `root` and AUTO-DETECTS its files via rglob, so it works for
# both the local layout (dataset/<DIR>/...) and the Kaggle dataset mounts
# (which nest the files under a different prefix, e.g. /kaggle/input/hust_dataset/
# HUST-HAR/..., /kaggle/input/ut_har_dataset/data/..., .../ntu_fi_dataset/train_amp/).

def _find(root: Path, name: str) -> Path:
    """First path under root whose final component == name (file or dir)."""
    hit = next(Path(root).rglob(name), None)
    if hit is None:
        raise FileNotFoundError(f"'{name}' not found under {root}")
    return hit


def load_hust(root):
    """HUST-HAR: (3600, 270, 1000) float32, 6 classes. Random 80/20 (seed 42)."""
    import torch
    ptf = next(Path(root).rglob('HUST_HAR_dataset*.pt'), None) \
        or _find(root, 'HUST_HAR_dataset-001.pt')
    lbf = _find(root, 'HUST_HAR_labels.pt')
    d = torch.load(ptf, map_location='cpu', weights_only=False)
    y = torch.load(lbf, map_location='cpu', weights_only=False)
    X = d.numpy().astype(np.float32)             # (3600, 270, 1000)
    y = y.numpy().astype(np.int64)
    n = len(X)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n)                    # pure-random split (matches orig)
    n_tr = int(0.8 * n)
    splits = {'train': perm[:n_tr], 'test': perm[n_tr:]}
    return X, y, splits


def load_uthar(root):
    """UT-HAR: .npy stored as .csv. X (N,250,90)=time x (3ant*30sub); needs
    TRANSPOSE -> (N,90,250) feature-major. 7 classes. Official train/val/test;
    per user decision val is MERGED INTO TEST (train=3977, test=500+496=996)."""
    def ld(name):
        return np.load(_find(root, f'{name}.csv'), allow_pickle=True)
    Xtr = ld('X_train').astype(np.float32).transpose(0, 2, 1)   # (3977,90,250)
    Xte = ld('X_test').astype(np.float32).transpose(0, 2, 1)    # (500,90,250)
    Xva = ld('X_val').astype(np.float32).transpose(0, 2, 1)     # (496,90,250)
    ytr = ld('y_train').astype(np.int64)
    yte = ld('y_test').astype(np.int64)
    yva = ld('y_val').astype(np.int64)
    X = np.concatenate([Xtr, Xte, Xva], axis=0)
    y = np.concatenate([ytr, yte, yva], axis=0)
    n_tr, n_te = len(Xtr), len(Xte) + len(Xva)        # val folded into test
    splits = {'train': np.arange(n_tr),
              'test':  np.arange(n_tr, n_tr + n_te)}
    return X, y, splits


def load_ntufi(root):
    """NTU-Fi: per-class folders of .mat, key 'CSIamp' (342,2000)=(3ant*114sub) x time.
    Already feature-major (no transpose). Official train_amp/test_amp folders.

    DOWNSAMPLE time x[:, ::4]: 2000 -> 500, matching the SenseFi benchmark
    dataloader (x = x[:, ::4]; reshape(3,114,500)). The benchmark sample is
    500 packets @ 500Hz over 1s, so DATASETS['ntufi']['fs']=500 for the proc LPF."""
    import os
    import scipy.io
    train_dir = _find(root, 'train_amp')
    test_dir  = _find(root, 'test_amp')
    classes = sorted(os.listdir(train_dir))   # box,circle,clean,fall,run,walk
    cls2idx = {c: i for i, c in enumerate(classes)}
    Xs, ys, split_idx = [], [], {'train': [], 'test': []}
    k = 0
    for split, folder in [('train', train_dir), ('test', test_dir)]:
        for c in classes:
            cdir = Path(folder) / c
            for fn in sorted(os.listdir(cdir)):
                if not fn.endswith('.mat'):
                    continue
                arr = scipy.io.loadmat(str(cdir / fn))['CSIamp'].astype(np.float32)
                assert arr.shape == (342, 2000), f'bad NTU shape {arr.shape} in {fn}'
                arr = arr[:, ::4]                       # 2000 -> 500 (benchmark ::4)
                Xs.append(arr); ys.append(cls2idx[c]); split_idx[split].append(k); k += 1
    X = np.stack(Xs, axis=0)                            # (1200, 342, 500)
    y = np.array(ys, dtype=np.int64)
    splits = {sp: np.array(ix, dtype=np.int64) for sp, ix in split_idx.items()}
    return X, y, splits


DATASETS = {
    # n_per_sub = n_ant (channels per subband).  fs drives the proc LPF only.
    'hust':  dict(loader=load_hust,  n_ant=9, sub=30,  n_per_sub=9, fs=200.0,  classes=6),
    'uthar': dict(loader=load_uthar, n_ant=3, sub=30,  n_per_sub=3, fs=100.0,  classes=7),
    'ntufi': dict(loader=load_ntufi, n_ant=3, sub=114, n_per_sub=3, fs=500.0,  classes=6),
}

# Class names in LABEL ORDER (0..N-1) for confusion-matrix display.
#   ntufi  : CERTAIN — sorted train_amp/ folder names (= load_ntufi cls2idx order).
#   uthar  : documented benchmark/Yousefi order (alphabetical activity names).
#   hust   : HUST_HAR README + paper order (lie/pick/sit/stand/standup/walk).
CLASS_NAMES = {
    'hust':  ['lie down', 'pick up', 'sit down', 'stand', 'stand up', 'walk'],
    # UT-HAR order VERIFIED against SenseFi listing + per-class window counts
    # (walk=label2=most windows, run=label4; transient actions fewer). NOT alphabetical.
    'uthar': ['lie down', 'fall', 'walk', 'pick up', 'run', 'sit down', 'stand up'],
    'ntufi': ['box', 'circle', 'clean', 'fall', 'run', 'walk'],
}

SPLIT_DESC = {
    'hust':  'random 80/20 (seed 42, subject-mixed)',
    'uthar': 'official train; test = official test + val',
    'ntufi': 'official train_amp / test_amp folders',
}


DIRMAP = {'hust': 'HUST-HAR', 'uthar': 'UT_HAR', 'ntufi': 'NTU-Fi_HAR'}


def build(dataset: str, mode: str, raw_root=None, out_root=None):
    """raw_root: where the raw dataset lives (default local dataset/<DIR>; on Kaggle
    pass the mounted dataset path). out_root: where bench/ is written (default local
    dataset/<DIR>; on Kaggle pass e.g. /kaggle/working)."""
    cfg = DATASETS[dataset]
    do_filter = (mode == 'proc')
    n_ant, sub, n_per_sub, fs = cfg['n_ant'], cfg['sub'], cfg['n_per_sub'], cfg['fs']

    raw_root = Path(raw_root) if raw_root else DATA_ROOT / DIRMAP[dataset]
    base_out = Path(out_root) / DIRMAP[dataset] if out_root else DATA_ROOT / DIRMAP[dataset]

    print(f'Loading {dataset} from {raw_root} ...')
    X, y, splits = cfg['loader'](raw_root)
    N, AxS, time = X.shape
    assert AxS == n_ant * sub, f'axis1 {AxS} != n_ant*sub {n_ant*sub}'
    C = 3 * n_per_sub
    T2, F2 = time // 2, sub // 2
    print(f'  N={N}  raw=({AxS},{time})  -> packed=({C},{T2},{F2})  classes={cfg["classes"]}')
    print(f'  mode={mode}  do_filter={do_filter}  fs={fs}')
    print(f'  split: ' + '  '.join(f'{k}={len(v)}' for k, v in splits.items()))

    out_dir = base_out / 'bench' / mode
    wav_dir = out_dir / 'wavmamba'           # PreprocWavMambaDataset reads wavmamba/X_*.npy
    wav_dir.mkdir(parents=True, exist_ok=True)

    # memmaps per split + all-reps stat accumulators (per channel,bin over time+samples)
    mm = {}
    for sp, idx in splits.items():
        mm[sp] = np.lib.format.open_memmap(
            str(wav_dir / f'X_{sp}.npy'), mode='w+', dtype=np.float32,
            shape=(len(idx), C, T2, F2))
        np.save(out_dir / f'y_{sp}.npy', y[idx].astype(np.int64))

    s  = np.zeros((C, F2), dtype=np.float64)
    s2 = np.zeros((C, F2), dtype=np.float64)
    n_acc = np.int64(0)

    for sp, idx in splits.items():
        for j, i in enumerate(tqdm(idx, desc=f'  [{sp}] pack', unit='smp')):
            x = pack_haar3(X[i], n_ant, sub, n_per_sub, fs=fs, do_filter=do_filter)  # (C,T2,F2)
            mm[sp][j] = x
            xd = x.astype(np.float64)
            s  += xd.sum(axis=1)              # sum over time -> (C,F2)
            s2 += (xd * xd).sum(axis=1)
            n_acc += T2

    mean = s / n_acc
    std  = np.maximum(np.sqrt(np.maximum(s2 / n_acc - mean * mean, 0.0)), 1e-6)
    meta = dict(dataset=dataset, mode=mode, fs=fs, do_filter=do_filter,
                n_ant=n_ant, sub=sub, n_per_sub=n_per_sub, C=C, T2=T2, F2=F2,
                classes=cfg['classes'], class_names=CLASS_NAMES[dataset],
                split_seed=SPLIT_SEED, split=SPLIT_DESC[dataset],
                subband_order='LL|HL|LH', norm='all-reps per (channel,bin)',
                # source string drives infer_data_mode() in dataset.py:
                source=('multi_hampel_lpf' if do_filter else 'multi_raw'))
    if do_filter:
        meta['filter'] = dict(hampel_window=8, hampel_nsigma=3.0,
                              lpf_order=4, lpf_cutoff_hz=20.0)
    stats = {
        # nested under 'wavmamba' so PreprocWavMambaDataset picks it up directly.
        'wavmamba': {
            'mean': mean.astype(np.float32).tolist(),     # (C, F2)
            'std':  std.astype(np.float32).tolist(),
        },
        'meta': meta,
    }
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f)
    for sp in splits:
        del mm[sp]
    print(f'  saved -> {out_dir}  (wavmamba/X_*, y_*, stats.json)  un-normalized')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--mode', required=True, choices=['raw', 'proc'])
    ap.add_argument('--raw-root', default=None,
                    help='Raw dataset root (default local dataset/<DIR>; on Kaggle: mount path)')
    ap.add_argument('--out-root', default=None,
                    help='Where bench/ is written (default local dataset/<DIR>; on Kaggle: /kaggle/working)')
    args = ap.parse_args()
    build(args.dataset, args.mode, raw_root=args.raw_root, out_root=args.out_root)
