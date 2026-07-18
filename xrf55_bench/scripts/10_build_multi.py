"""Build packed Haar bench arrays for HUST-HAR / UT-HAR / NTU-Fi.

Output layout matches PreprocWavMambaDataset (dataset.py), so WavDualMamba trains
via the existing, tested model_name='wavdualmamba' path — no new dataset/model code.
Subbands packed are selectable via --wav-subbands: 'LL,HL,LH' (S4.1, 3 bang) or
'HL,LH' (S4, 2 bang). Stats + meta auto-track the actual channel count C.

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

from xrf55_bench.preprocessing.multi_dataset import (
    pack_haar3, haar3_subbands, to_maps,
)

DATA_ROOT = PROJECT_ROOT / 'dataset'
SPLIT_SEED = 42   # fixed 80/20 split, reproducible across the 5 model-init seeds


# ── Dataset loaders → (X_all (N, n_ant*sub, time) float32, y_all (N,) int64) ────
# Each loader takes a `root` and AUTO-DETECTS its files via rglob, so it works for
# both the local layout (dataset/<DIR>/...) and the Kaggle dataset mounts
# (which nest the files under a different prefix, e.g. /kaggle/input/hust_dataset/
# HUST-HAR/..., /kaggle/input/ut_har_dataset/data/..., .../ntu_fi_dataset/train_amp/).

def _listing(root, n=50):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file())[:n]


def _find(root: Path, name: str) -> Path:
    """First path under root whose final component == name (file or dir)."""
    hit = next(Path(root).rglob(name), None)
    if hit is None:
        raise FileNotFoundError(f"'{name}' not found under {root}. Files seen: {_listing(root)}")
    return hit


def load_hust(root):
    """HUST-HAR: (3600, 270, 1000) float32, 6 classes. Random 80/20 (seed 42).

    Robust file discovery (mount co the gop NHIEU dataset chung): labels = file co
    'label' trong ten. data = file (KHONG label, khong text) duoc cham diem uu tien:
      (1) ten chua 'hust'  (2) cung thu muc voi labels  (3) duoi .pt/.pth  (4) size.
    Tranh vo phai file dataset khac (vd X_train.npy 5GB) o noi khac trong mount.
    Ho tro ca .pt (torch) lan .npy (numpy)."""
    import torch
    root  = Path(root)
    files = [p for p in root.rglob('*') if p.is_file()]
    lbf   = next((p for p in files if 'label' in p.name.lower()), None)
    if lbf is None:
        raise FileNotFoundError(
            f"HUST labels (*label*) not found under {root}. Files seen: {_listing(root)}")
    cand = [p for p in files if p is not lbf and 'label' not in p.name.lower()
            and p.suffix.lower() not in ('.txt', '.md', '.json', '.csv')]
    if not cand:
        raise FileNotFoundError(
            f"HUST data not found next to {lbf.name}. Dir: {[p.name for p in lbf.parent.iterdir()]}")
    ptf = max(cand, key=lambda p: ('hust' in p.name.lower(), p.parent == lbf.parent,
                                   p.suffix.lower() in ('.pt', '.pth'), p.stat().st_size))
    print(f'  HUST data={ptf.name} ({ptf.stat().st_size/1e9:.2f}GB)  labels={lbf.name}')
    if ptf.suffix.lower() == '.npy':
        X = np.load(ptf, allow_pickle=False).astype(np.float32)   # (3600, 270, 1000)
    else:
        d = torch.load(ptf, map_location='cpu', weights_only=False)
        X = (d.numpy() if hasattr(d, 'numpy') else np.asarray(d)).astype(np.float32)
    yv = torch.load(lbf, map_location='cpu', weights_only=False)
    y = (yv.numpy() if hasattr(yv, 'numpy') else np.asarray(yv)).astype(np.int64)
    n = len(X)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n)                    # pure-random split (matches orig)
    n_tr = int(0.8 * n)
    splits = {'train': perm[:n_tr], 'test': perm[n_tr:]}
    return X, y, splits


def load_uthar(root, merge_val=False):
    """UT-HAR: .npy stored as .csv. X (N,250,90)=time x (3ant*30sub); needs
    TRANSPOSE -> (N,90,250) feature-major (for DWT). 7 classes.

    merge_val=False (git SenseFi): train=X_train(3977), test=X_test(500); val(496) unused.
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
    'uthar': 'official train=X_train; test=X_test (val unused, SenseFi git)',
    'ntufi': 'official train_amp / test_amp folders',
}

# SenseFi RAW normalization (UT_HAR_dataset / CSI_Dataset). Applied BEFORE the Haar
# DWT for uthar/ntufi so the build matches the TF-Mamba authors' acknowledged toolchain
# (https://github.com/.../WiFi-CSI-Sensing-Benchmark). HUST has no SenseFi pre-norm.
NTUFI_MEAN, NTUFI_STD = 42.3199, 4.9802   # CSI_Dataset: (x - 42.3199)/4.9802


def _sensefi_prenorm(dataset, X, splits):
    """Norm RAW (truoc DWT) giong het SenseFi. Tra (X, prenorm_applied: bool).
      uthar : min-max toan cuc TUNG split  (x-min)/(max-min)   [UT_HAR_dataset]
      ntufi : hang so co dinh (x-42.3199)/4.9802               [CSI_Dataset]
      hust  : khong pre-norm (dung data_norm z sau DWT)."""
    if dataset == 'uthar':
        for idx in splits.values():
            seg = X[idx]
            mn, mx = float(seg.min()), float(seg.max())
            X[idx] = (seg - mn) / (mx - mn)
        return X, True
    if dataset == 'ntufi':
        return ((X - np.float32(NTUFI_MEAN)) / np.float32(NTUFI_STD)).astype(np.float32), True
    return X, False


DIRMAP = {'hust': 'HUST-HAR', 'uthar': 'UT_HAR', 'ntufi': 'NTU-Fi_HAR'}


def _finalize(s, s2, n):
    """all-reps mean/std from running sums; std floored at 1e-6."""
    mean = s / n
    std = np.maximum(np.sqrt(np.maximum(s2 / n - mean * mean, 0.0)), 1e-6)
    return mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()


def build(dataset: str, mode: str, raw_root=None, out_root=None, fmt='wavmamba',
          merge_val=False, wav_subs=('LL', 'HL', 'LH'), norm_style='sensefi'):
    """raw_root: where the raw dataset lives (default local dataset/<DIR>; on Kaggle
    pass the mounted dataset path). out_root: where bench/ is written.
    fmt: 'wavmamba' (packed [subbands] for WavDualMamba) | 'tfmamba' (2-stream
    xh=HL, xv=LH flat, for the original TF-Mamba) | 'both'. Haar computed once.
    wav_subs: which Haar subbands to pack for the wavmamba format, IN ORDER:
      ('LL','HL','LH') = S4.1 (3 bang, mac dinh) | ('HL','LH') = S4 (2 bang, no LL).
      Chi anh huong fmt wavmamba; thu tu nay = subband order model phai khop.
    merge_val: CHI UT-HAR — True thi gop X_val vao test (mac dinh False = git SenseFi).
    norm_style: 'sensefi' (mac dinh) = SenseFi pre-norm (uthar/ntufi) + z-norm
      per-position (C,T2,F2)/(T2,M). 'xrf55' = KHONG pre-norm (moi dataset) + z-norm
      per-channel-bin (C,F2)/(M,) gop truc thoi gian — giong het build XRF55 (02).
      z-norm ap luc load; pre-norm bi nuong vao X_*.npy nen 2 style KHONG dung chung bench."""
    _VALID = ('LL', 'HL', 'LH')
    wav_subs = tuple(wav_subs)
    if not wav_subs or any(s not in _VALID for s in wav_subs):
        raise ValueError(f'wav_subs phai la tap con co thu tu cua {_VALID}, got {wav_subs}')
    if norm_style not in ('sensefi', 'xrf55'):
        raise ValueError(f"norm_style phai la 'sensefi' | 'xrf55', got {norm_style!r}")
    per_channel_bin = (norm_style == 'xrf55')   # xrf55: z-norm gop truc thoi gian
    cfg = DATASETS[dataset]
    do_filter = (mode == 'proc')
    n_ant, sub, n_per_sub, fs = cfg['n_ant'], cfg['sub'], cfg['n_per_sub'], cfg['fs']
    do_wav, do_tf = fmt in ('wavmamba', 'both'), fmt in ('tfmamba', 'both')

    raw_root = Path(raw_root) if raw_root else DATA_ROOT / DIRMAP[dataset]
    base_out = Path(out_root) / DIRMAP[dataset] if out_root else DATA_ROOT / DIRMAP[dataset]

    print(f'Loading {dataset} from {raw_root} ...')
    if dataset == 'uthar':
        X, y, splits = cfg['loader'](raw_root, merge_val=merge_val)
    else:
        X, y, splits = cfg['loader'](raw_root)
    N, AxS, time = X.shape
    assert AxS == n_ant * sub, f'axis1 {AxS} != n_ant*sub {n_ant*sub}'
    if norm_style == 'xrf55':
        prenorm = False        # xrf55: KHONG pre-norm cho bat ky dataset nao
        print(f'  norm_style=xrf55: SKIP SenseFi pre-norm for {dataset}')
    else:
        X, prenorm = _sensefi_prenorm(dataset, X, splits)   # SenseFi raw norm (uthar/ntufi)
        if prenorm:
            print(f'  SenseFi pre-norm (raw, before DWT) applied for {dataset}')
    C, T2, F2 = len(wav_subs) * n_per_sub, time // 2, sub // 2   # C = #subbands * n_per_sub
    M = n_per_sub * F2                      # tfmamba flat feature dim (= n_ant*sub//2)
    print(f'  N={N}  raw=({AxS},{time})  fmt={fmt}  wav_subs={wav_subs}  '
          f'-> wav=({C},{T2},{F2}) / tf xh,xv=({T2},{M})')
    print(f'  mode={mode}  do_filter={do_filter}  fs={fs}  classes={cfg["classes"]}')
    print(f'  split: ' + '  '.join(f'{k}={len(v)}' for k, v in splits.items()))

    # sensefi giu nguyen bench/{mode} (tuong thich run cu); xrf55 -> bench/{mode}_xrf55
    # de build gate (xrf55) KHONG de len build sensefi ma tfmamba/s4.nogn doc.
    mode_sub = mode if norm_style == 'sensefi' else f'{mode}_{norm_style}'
    out_dir = base_out / 'bench' / mode_sub
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp, idx in splits.items():
        np.save(out_dir / f'y_{sp}.npy', y[idx].astype(np.int64))

    mm = xh_mm = xv_mm = None
    if do_wav:
        (out_dir / 'wavmamba').mkdir(exist_ok=True)
        mm = {sp: np.lib.format.open_memmap(str(out_dir / 'wavmamba' / f'X_{sp}.npy'),
              mode='w+', dtype=np.float32, shape=(len(idx), C, T2, F2))
              for sp, idx in splits.items()}
        # sensefi: PER-POSITION (C,T2,F2) de cong bang voi tfmamba goc (cung per-position).
        # xrf55: PER-CHANNEL-BIN (C,F2) gop truc thoi gian, giong het build XRF55 (02).
        _wsh = (C, F2) if per_channel_bin else (C, T2, F2)
        s = np.zeros(_wsh); s2 = np.zeros(_wsh); n_wav = np.int64(0)
    if do_tf:
        (out_dir / 'tfmamba').mkdir(exist_ok=True)
        xh_mm = {sp: np.lib.format.open_memmap(str(out_dir / 'tfmamba' / f'X_{sp}_xh.npy'),
                 mode='w+', dtype=np.float32, shape=(len(idx), T2, M)) for sp, idx in splits.items()}
        xv_mm = {sp: np.lib.format.open_memmap(str(out_dir / 'tfmamba' / f'X_{sp}_xv.npy'),
                 mode='w+', dtype=np.float32, shape=(len(idx), T2, M)) for sp, idx in splits.items()}
        # sensefi: PER-POSITION (T2,M) = data_norm cua git TF-Mamba.
        # xrf55: PER-FEATURE (M,) gop truc thoi gian, giong build XRF55 (01/02).
        _tsh = (M,) if per_channel_bin else (T2, M)
        hs = np.zeros(_tsh); hs2 = np.zeros(_tsh)
        vs = np.zeros(_tsh); vs2 = np.zeros(_tsh); n_tf = np.int64(0)

    for sp, idx in splits.items():
        for j, i in enumerate(tqdm(idx, desc=f'  [{sp}] {fmt}', unit='smp')):
            LL, HL, LH = haar3_subbands(X[i], n_ant, sub, fs=fs, do_filter=do_filter)  # each (T2,M)
            if do_wav:
                _sb = {'LL': LL, 'HL': HL, 'LH': LH}                 # pack chi cac bang trong wav_subs, dung thu tu
                x = np.concatenate([to_maps(_sb[s], n_per_sub) for s in wav_subs],
                                   axis=0).astype(np.float32, copy=False)
                mm[sp][j] = x
                xd = x.astype(np.float64)
                if per_channel_bin:                       # gop truc thoi gian -> (C,F2)
                    s += xd.sum(axis=1); s2 += (xd * xd).sum(axis=1); n_wav += T2
                else:                                     # per-position (C,T2,F2)
                    s += xd; s2 += xd * xd; n_wav += 1
            if do_tf:
                xh_mm[sp][j] = HL; xv_mm[sp][j] = LH
                hd = HL.astype(np.float64); vd = LH.astype(np.float64)
                if per_channel_bin:                       # gop truc thoi gian -> (M,)
                    hs += hd.sum(axis=0); hs2 += (hd * hd).sum(axis=0)
                    vs += vd.sum(axis=0); vs2 += (vd * vd).sum(axis=0); n_tf += T2
                else:                                     # per-position (T2,M)
                    hs += hd; hs2 += hd * hd
                    vs += vd; vs2 += vd * vd; n_tf += 1

    meta = dict(dataset=dataset, mode=mode, fs=fs, do_filter=do_filter,
                n_ant=n_ant, sub=sub, n_per_sub=n_per_sub, C=C, T2=T2, F2=F2, M=M,
                classes=cfg['classes'], class_names=CLASS_NAMES[dataset],
                split_seed=SPLIT_SEED,
                split=('official train=X_train; test=X_test+X_val (merged)'
                       if (dataset == 'uthar' and merge_val) else SPLIT_DESC[dataset]),
                merge_val=(merge_val if dataset == 'uthar' else None),
                subband_order='|'.join(wav_subs), norm='all-reps',
                norm_style=norm_style,     # 'sensefi' | 'xrf55' (no pre-norm + per-channel-bin z)
                sensefi_prenorm=prenorm,   # True cho uthar/ntufi: z-norm load-time bo qua khi norm_mode='author'
                source=('multi_hampel_lpf' if do_filter else 'multi_raw'))
    if do_filter:
        meta['filter'] = dict(hampel_window=8, hampel_nsigma=3.0,
                              lpf_order=4, lpf_cutoff_hz=20.0)
    stats = {'meta': meta}
    if do_wav:
        mean, std = _finalize(s, s2, n_wav)            # xrf55:(C,F2) | sensefi:(C,T2,F2)
        stats['wavmamba'] = {'mean': mean, 'std': std}
        meta['wavmamba_norm'] = 'per-channel-bin' if per_channel_bin else 'per-position'
    if do_tf:
        xh_m, xh_s = _finalize(hs, hs2, n_tf)          # (T2,M) per-position (data_norm)
        xv_m, xv_s = _finalize(vs, vs2, n_tf)
        stats['tfmamba'] = {'xh_mean': xh_m, 'xh_std': xh_s, 'xv_mean': xv_m, 'xv_std': xv_s}
        meta['tfmamba_subband_naming'] = 'paper-eq5'   # xh file holds HL content
        meta['tfmamba_norm'] = 'per-feature' if per_channel_bin else 'per-position'
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f)
    for d in (mm, xh_mm, xv_mm):
        if d:
            for sp in list(d):
                del d[sp]
    print(f'  saved -> {out_dir}  (fmt={fmt}, y_*, stats.json)  un-normalized')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--mode', required=True, choices=['raw', 'proc'])
    ap.add_argument('--raw-root', default=None,
                    help='Raw dataset root (default local dataset/<DIR>; on Kaggle: mount path)')
    ap.add_argument('--out-root', default=None,
                    help='Where bench/ is written (default local dataset/<DIR>; on Kaggle: /kaggle/working)')
    ap.add_argument('--format', default='wavmamba', choices=['wavmamba', 'tfmamba', 'both'],
                    help='wavmamba=packed [subbands] | tfmamba=2-stream xh/xv | both')
    ap.add_argument('--wav-subbands', default='LL,HL,LH',
                    help='wavmamba pack: "LL,HL,LH"=S4.1 (3 bang) | "HL,LH"=S4 (2 bang, no LL)')
    ap.add_argument('--merge-val', action='store_true',
                    help='CHI UT-HAR: gop X_val vao test (mac dinh khong, giong git SenseFi)')
    ap.add_argument('--norm-style', default='sensefi', choices=['sensefi', 'xrf55'],
                    help='sensefi=SenseFi pre-norm+z per-position (mac dinh) | '
                         'xrf55=KHONG pre-norm + z per-channel-bin (giong build XRF55)')
    args = ap.parse_args()
    wav_subs = tuple(s.strip() for s in args.wav_subbands.split(',') if s.strip())
    build(args.dataset, args.mode, raw_root=args.raw_root, out_root=args.out_root,
          fmt=args.format, merge_val=args.merge_val, wav_subs=wav_subs,
          norm_style=args.norm_style)
