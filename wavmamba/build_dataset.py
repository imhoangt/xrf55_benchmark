"""Build packed Haar bench arrays for UT-HAR and NTU-Fi.

Output layout matches PreprocWavMambaDataset (dataset.py). Packs the two Haar
subbands {HL, LH} used by WavMamba.

Output per (dataset, mode, prenorm, z_gran): UN-NORMALIZED packed arrays +
stats.json. z-norm is applied at load time.

Protocol note: stats.json stores all-reps z-normalization statistics computed
over all official split samples in the bench build (train + test, and UT-HAR
val when --merge-val is used). This is kept for exact paper-protocol
reproduction; it is not a generic train-only normalization recipe.

    <out_root>/<DIR>/bench/<mode>_<prenorm>_<z_gran>/
        wavmamba/X_<split>.npy   (N, 2*n_per_sub, T//2, sub//2) float32
        y_<split>.npy            (N,)  int64
        stats.json               { 'wavmamba': {mean,std}, 'meta': {...} }

Normalization is split into TWO ORTHOGONAL flags (see --prenorm / --z-gran):
  prenorm : pre-norm on RAW (BEFORE DWT).
      'sensefi' = UT-HAR min-max/split, NTU-Fi (x-42.32)/4.98.
      'none'    = no raw pre-normalization.
  z_gran  : granularity of z-norm AFTER DWT (applied at load, always on).
      'perpos' = per-position (C,T2,F2).
      'pcb'    = per-channel-bin (C,F2), collapsing time.

The 4 combinations write to 4 separate bench dirs so builds never overwrite.

Usage:
    python build_dataset.py --dataset uthar --mode raw
    python build_dataset.py --dataset ntufi --mode raw --prenorm none --z-gran pcb
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from preprocess import haar3_subbands, to_maps

# Default raw dataset root (sibling of this package). Override with --raw-root.
DATA_ROOT = Path(__file__).parent.parent / 'dataset'


# ── Dataset loaders -> (X_all (N, n_ant*sub, time) float32, y_all (N,) int64) ──
# Each loader takes a `root` and AUTO-DETECTS its files via rglob, so it works
# for both the local layout and Kaggle dataset mounts (which nest files under a
# different prefix, e.g. /kaggle/input/ut_har_dataset/data/...).

def _listing(root, n=50):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file())[:n]


def _find(root: Path, name: str) -> Path:
    """First path under root whose final component == name (file or dir)."""
    hit = next(Path(root).rglob(name), None)
    if hit is None:
        raise FileNotFoundError(f"'{name}' not found under {root}. Files seen: {_listing(root)}")
    return hit


def load_uthar(root, merge_val=False):
    """UT-HAR: .npy stored as .csv. X (N,250,90)=time x (3ant*30sub); needs
    TRANSPOSE -> (N,90,250) feature-major (for DWT). 7 classes.

    merge_val=False: train=X_train(3977), test=X_test(500); val(496) unused.
    merge_val=True : test = X_test + X_val (500+496=996)."""
    def ld(name):
        return np.load(_find(root, f'{name}.csv'), allow_pickle=True)
    Xtr = ld('X_train').astype(np.float32).transpose(0, 2, 1)   # (3977,90,250)
    Xte = ld('X_test').astype(np.float32).transpose(0, 2, 1)    # (500,90,250)
    parts_X = [Xtr, Xte]
    parts_y = [ld('y_train').astype(np.int64), ld('y_test').astype(np.int64)]
    if merge_val:
        parts_X.append(ld('X_val').astype(np.float32).transpose(0, 2, 1))   # (496,90,250)
        parts_y.append(ld('y_val').astype(np.int64))
    X = np.concatenate(parts_X, axis=0)
    y = np.concatenate(parts_y, axis=0)
    n_tr = len(Xtr); n_te = len(X) - n_tr
    splits = {'train': np.arange(n_tr),
              'test':  np.arange(n_tr, n_tr + n_te)}
    return X, y, splits


def load_ntufi(root):
    """NTU-Fi: per-class folders of .mat, key 'CSIamp' (342,2000)=(3ant*114sub) x time.
    Already feature-major (no transpose). Official train_amp/test_amp folders.

    DOWNSAMPLE time x[:, ::4]: 2000 -> 500, matching the public benchmark
    dataloader convention (x = x[:, ::4]; reshape(3,114,500)). The sample is
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
                if arr.shape != (342, 2000):
                    raise ValueError(
                        f'bad NTU-Fi shape {arr.shape} in {fn}; expected (342, 2000)'
                    )
                arr = arr[:, ::4]                       # 2000 -> 500 (benchmark ::4)
                Xs.append(arr); ys.append(cls2idx[c]); split_idx[split].append(k); k += 1
    X = np.stack(Xs, axis=0)                            # (1200, 342, 500)
    y = np.array(ys, dtype=np.int64)
    splits = {sp: np.array(ix, dtype=np.int64) for sp, ix in split_idx.items()}
    return X, y, splits


DATASETS = {
    # n_per_sub = n_ant (channels per subband).  fs drives the proc LPF only.
    'uthar': dict(loader=load_uthar, n_ant=3, sub=30,  n_per_sub=3, fs=100.0, classes=7),
    'ntufi': dict(loader=load_ntufi, n_ant=3, sub=114, n_per_sub=3, fs=500.0, classes=6),
}

# Class names in LABEL ORDER (0..N-1) for confusion-matrix display.
#   ntufi : sorted train_amp/ folder names (= load_ntufi cls2idx order).
#   uthar : documented UT-HAR benchmark label order; not alphabetical.
CLASS_NAMES = {
    # UT-HAR order checked against public label listings and per-class window counts
    # (walk=label2=most windows, run=label4; transient actions fewer).
    'uthar': ['lie down', 'fall', 'walk', 'pick up', 'run', 'sit down', 'stand up'],
    'ntufi': ['box', 'circle', 'clean', 'fall', 'run', 'walk'],
}

SPLIT_DESC = {
    'uthar': 'official train=X_train; test=X_test (val unused by default)',
    'ntufi': 'official train_amp / test_amp folders',
}

# Optional RAW pre-normalization applied BEFORE the Haar DWT when --prenorm=sensefi.
# This flag preserves the public benchmark convention used by these datasets:
#   uthar : min-max global PER split  (x-min)/(max-min)
#   ntufi : fixed constant (x-42.3199)/4.9802
NTUFI_MEAN, NTUFI_STD = 42.3199, 4.9802


def _sensefi_prenorm(dataset, X, splits):
    """Apply RAW pre-normalization before DWT. Returns (X, prenorm_applied: bool).
      uthar : min-max global PER split  (x-min)/(max-min)
      ntufi : fixed constant (x-42.3199)/4.9802"""
    if dataset == 'uthar':
        for idx in splits.values():
            seg = X[idx]
            mn, mx = float(seg.min()), float(seg.max())
            X[idx] = (seg - mn) / (mx - mn)
        return X, True
    if dataset == 'ntufi':
        return ((X - np.float32(NTUFI_MEAN)) / np.float32(NTUFI_STD)).astype(np.float32), True
    return X, False


DIRMAP = {'uthar': 'UT_HAR', 'ntufi': 'NTU-Fi_HAR'}


def _finalize(s, s2, n):
    """all-reps mean/std from running sums; std floored at 1e-6."""
    mean = s / n
    std = np.maximum(np.sqrt(np.maximum(s2 / n - mean * mean, 0.0)), 1e-6)
    return mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()


def build(dataset: str, mode: str, raw_root=None, out_root=None,
          merge_val=False, prenorm='sensefi', z_gran='perpos'):
    """Build packed Haar bench arrays for one dataset.

    Packs the two Haar subbands {HL, LH} used by WavMamba.
    See module docstring for the prenorm x z_gran normalization scheme.
    """
    if prenorm not in ('none', 'sensefi'):
        raise ValueError(f"prenorm must be 'none' | 'sensefi', got {prenorm!r}")
    if z_gran not in ('perpos', 'pcb'):
        raise ValueError(f"z_gran must be 'perpos' | 'pcb', got {z_gran!r}")
    wav_subs = ('HL', 'LH')   # fixed — matches WavMamba
    per_channel_bin = (z_gran == 'pcb')   # pcb: z-norm collapse time axis
    cfg = DATASETS[dataset]
    do_filter = (mode == 'proc')
    n_ant, sub, n_per_sub, fs = cfg['n_ant'], cfg['sub'], cfg['n_per_sub'], cfg['fs']

    raw_root = Path(raw_root) if raw_root else DATA_ROOT / DIRMAP[dataset]
    base_out = Path(out_root) / DIRMAP[dataset] if out_root else DATA_ROOT / DIRMAP[dataset]

    print(f'Loading {dataset} from {raw_root} ...')
    if dataset == 'uthar':
        X, y, splits = cfg['loader'](raw_root, merge_val=merge_val)
    else:
        X, y, splits = cfg['loader'](raw_root)
    N, AxS, time = X.shape
    expected_axis = n_ant * sub
    if AxS != expected_axis:
        raise ValueError(
            f'loaded feature axis {AxS} != n_ant*sub {expected_axis} '
            f'({n_ant} antennas x {sub} subcarriers) for {dataset}'
        )
    if prenorm == 'none':
        prenorm_applied = False      # none: no raw pre-normalization for any dataset
        print(f'  prenorm=none: skip raw pre-normalization for {dataset}')
    else:
        X, prenorm_applied = _sensefi_prenorm(dataset, X, splits)
        if prenorm_applied:
            print(f'  prenorm=sensefi: raw pre-normalization applied for {dataset}')
    C, T2, F2 = len(wav_subs) * n_per_sub, time // 2, sub // 2   # C = #subbands * n_per_sub
    print(f'  N={N}  raw=({AxS},{time})  wav_subs={wav_subs}  -> wav=({C},{T2},{F2})')
    print(f'  mode={mode}  do_filter={do_filter}  fs={fs}  classes={cfg["classes"]}  '
          f'prenorm={prenorm}  z_gran={z_gran}')
    print(f'  split: ' + '  '.join(f'{k}={len(v)}' for k, v in splits.items()))

    # 4 combos prenorm x z_gran -> 4 separate bench dirs, never overwrite.
    mode_sub = f'{mode}_{prenorm}_{z_gran}'
    out_dir = base_out / 'bench' / mode_sub
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp, idx in splits.items():
        np.save(out_dir / f'y_{sp}.npy', y[idx].astype(np.int64))

    (out_dir / 'wavmamba').mkdir(exist_ok=True)
    mm = {sp: np.lib.format.open_memmap(str(out_dir / 'wavmamba' / f'X_{sp}.npy'),
          mode='w+', dtype=np.float32, shape=(len(idx), C, T2, F2))
          for sp, idx in splits.items()}
    # z_gran='perpos': per-position stats (C,T2,F2) — one mean per time position.
    # z_gran='pcb':    per-channel-bin stats (C,F2), collapsing time.
    _wsh = (C, F2) if per_channel_bin else (C, T2, F2)
    s = np.zeros(_wsh); s2 = np.zeros(_wsh); n_wav = np.int64(0)

    for sp, idx in splits.items():
        for j, i in enumerate(tqdm(idx, desc=f'  [{sp}]', unit='smp')):
            LL, HL, LH = haar3_subbands(X[i], n_ant, sub, fs=fs, do_filter=do_filter)
            _sb = {'LL': LL, 'HL': HL, 'LH': LH}                 # pack only wav_subs, in order
            x = np.concatenate([to_maps(_sb[s], n_per_sub) for s in wav_subs],
                               axis=0).astype(np.float32, copy=False)
            mm[sp][j] = x
            xd = x.astype(np.float64)
            if per_channel_bin:                       # collapse time -> (C,F2)
                s += xd.sum(axis=1); s2 += (xd * xd).sum(axis=1); n_wav += T2
            else:                                     # per-position (C,T2,F2)
                s += xd; s2 += xd * xd; n_wav += 1

    meta = dict(dataset=dataset, mode=mode, fs=fs, do_filter=do_filter,
                n_ant=n_ant, sub=sub, n_per_sub=n_per_sub, C=C, T2=T2, F2=F2,
                classes=cfg['classes'], class_names=CLASS_NAMES[dataset],
                split=('official train=X_train; test=X_test+X_val (merged)'
                       if (dataset == 'uthar' and merge_val) else SPLIT_DESC[dataset]),
                merge_val=(merge_val if dataset == 'uthar' else None),
                subband_order='|'.join(wav_subs), norm='all-reps',
                prenorm=prenorm,           # 'none' | 'sensefi' — pre-norm raw (before DWT)
                z_gran=z_gran,             # 'perpos' | 'pcb' — granularity z-norm after DWT
                sensefi_prenorm=prenorm_applied,  # bool: True if pre-norm was applied
                source=('multi_hampel_lpf' if do_filter else 'multi_raw'))
    if do_filter:
        meta['filter'] = dict(hampel_window=8, hampel_nsigma=3.0,
                              lpf_order=4, lpf_cutoff_hz=20.0)
    stats = {'meta': meta}
    mean, std = _finalize(s, s2, n_wav)            # pcb:(C,F2) | perpos:(C,T2,F2)
    stats['wavmamba'] = {'mean': mean, 'std': std}
    meta['wavmamba_norm'] = 'per-channel-bin' if per_channel_bin else 'per-position'
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f)
    for sp in list(mm):
        del mm[sp]
    print(f'  saved -> {out_dir}  (y_*, stats.json)  un-normalized')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('Usage:')[0])
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--mode', required=True, choices=['raw', 'proc'])
    ap.add_argument('--raw-root', default=None,
                    help='Raw dataset root (default ../dataset/<DIR>; on Kaggle: mount path)')
    ap.add_argument('--out-root', default=None,
                    help='Where bench/ is written (default ../dataset/<DIR>; on Kaggle: /kaggle/working)')
    ap.add_argument('--merge-val', action='store_true',
                    help='ONLY UT-HAR: merge X_val into test (default: use X_test only)')
    ap.add_argument('--prenorm', default='sensefi', choices=['none', 'sensefi'],
                    help="pre-norm RAW before DWT: 'sensefi'=UT-HAR min-max / NTU-Fi (x-42.32)/4.98 "
                         "| 'none'=no raw pre-normalization")
    ap.add_argument('--z-gran', default='perpos', choices=['perpos', 'pcb'],
                    help="granularity of z-norm AFTER DWT (at load): 'perpos'=per-position (C,T2,F2) "
                         "| 'pcb'=per-channel-bin (C,F2), time collapsed")
    args = ap.parse_args()
    build(args.dataset, args.mode, raw_root=args.raw_root, out_root=args.out_root,
          merge_val=args.merge_val, prenorm=args.prenorm, z_gran=args.z_gran)
