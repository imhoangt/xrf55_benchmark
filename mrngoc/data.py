"""Pipeline dữ liệu cho TF-Mamba gốc — raw (chỉ Haar DWT, KHÔNG Hampel/LPF).

raw (N, n_ant*sub, time)
  -> [UT-HAR/NTU-Fi] SenseFi pre-norm trên RAW (giống UT_HAR_dataset / CSI_Dataset):
       uthar = min-max từng split (x-min)/(max-min);  ntufi = (x-42.3199)/4.9802
       (HUST không pre-norm)
  -> Haar 2-D DWT (pywt 'periodization'):  HL = cV.T (paper XH),  LH = cH.T (paper XV)
  -> xh, xv  (N, T/2, M)  với M = n_ant*sub/2
  -> z-score per-POSITION (= `data_norm` git TF-Mamba: (x-mean_dim0)/(std_dim0+1e-6)),
     theo NORM_MODE:
       'author' : HUST có z; UT-HAR/NTU-Fi KHÔNG z (chỉ SenseFi norm) — đúng tác giả
       'double' : cả 3 đều z sau DWT (UT-HAR/NTU-Fi = SenseFi norm + z = 2 lần)
  -> TensorDataset(xh, xv, y)  trả (S_T=HL, S_F=LH, label)

3 dataset: HUST-HAR / UT-HAR / NTU-Fi. Mỗi loader tự rglob tìm file nên chạy
được cả local lẫn Kaggle (file bị lồng dưới prefix mount khác nhau).

Subband: vì Haar 2-tap và mỗi antenna có số subcarrier CHẴN, dwt2 trên trục
gộp (antenna-major) không trộn lẫn antenna — tương đương dwt2 từng antenna.
"""
import os
from pathlib import Path

import numpy as np
import pywt
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

SPLIT_SEED = 42   # split 80/20 cố định cho HUST (UT-HAR/NTU-Fi dùng split chính thức)

# n_per_sub = số kênh không-gian/subband (= n_ant). classes = số lớp.
DATASETS = {
    'hust':  dict(n_ant=9, sub=30,  n_per_sub=9, classes=6),
    'uthar': dict(n_ant=3, sub=30,  n_per_sub=3, classes=7),
    'ntufi': dict(n_ant=3, sub=114, n_per_sub=3, classes=6),
}

# Tên lớp theo thứ tự nhãn (0..N-1) — để vẽ confusion matrix.
CLASS_NAMES = {
    'hust':  ['lie down', 'pick up', 'sit down', 'stand', 'stand up', 'walk'],
    'uthar': ['lie down', 'fall', 'walk', 'pick up', 'run', 'sit down', 'stand up'],
    'ntufi': ['box', 'circle', 'clean', 'fall', 'run', 'walk'],
}


# ── tìm file (local + Kaggle) ────────────────────────────────────────────────
def _listing(root, n=50):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob('*')
                  if p.is_file())[:n]


def _find(root: Path, name: str) -> Path:
    hit = next(Path(root).rglob(name), None)
    if hit is None:
        raise FileNotFoundError(
            f"'{name}' khong thay duoi {root}. Files: {_listing(root)}")
    return hit


# ── loaders → (X (N, n_ant*sub, time) float32, y (N,) int64, splits) ─────────
def load_hust(root):
    """HUST-HAR: (3600, 270, 1000), 6 lop. Random 80/20 (seed 42).

    Mount co the gop nhieu dataset -> chon file data uu tien: ten chua 'hust' >
    cung thu muc voi labels > .pt > size. Ho tro .pt (torch) lan .npy (numpy).
    Tranh vo phai file dataset khac (vd X_train.npy 5GB)."""
    root = Path(root)
    files = [p for p in root.rglob('*') if p.is_file()]
    lbf = next((p for p in files if 'label' in p.name.lower()), None)
    if lbf is None:
        raise FileNotFoundError(f"HUST labels (*label*) khong thay duoi {root}. Files: {_listing(root)}")
    cand = [p for p in files if p is not lbf and 'label' not in p.name.lower()
            and p.suffix.lower() not in ('.txt', '.md', '.json', '.csv')]
    if not cand:
        raise FileNotFoundError(
            f"HUST data khong thay canh {lbf.name}. Dir: {[p.name for p in lbf.parent.iterdir()]}")
    ptf = max(cand, key=lambda p: ('hust' in p.name.lower(), p.parent == lbf.parent,
                                   p.suffix.lower() in ('.pt', '.pth'), p.stat().st_size))
    print(f"  HUST data={ptf.name} ({ptf.stat().st_size/1e9:.2f}GB)  labels={lbf.name}")
    if ptf.suffix.lower() == '.npy':
        X = np.load(ptf, allow_pickle=False).astype(np.float32)
    else:
        d = torch.load(ptf, map_location='cpu', weights_only=False)
        X = (d.numpy() if hasattr(d, 'numpy') else np.asarray(d)).astype(np.float32)
    yv = torch.load(lbf, map_location='cpu', weights_only=False)
    y = (yv.numpy() if hasattr(yv, 'numpy') else np.asarray(yv)).astype(np.int64)
    n = len(X)
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    n_tr = int(0.8 * n)
    return X, y, {'train': perm[:n_tr], 'test': perm[n_tr:]}


def load_uthar(root, merge_val=False):
    """UT-HAR: .npy luu duoi .csv. X (N,250,90) time x (3ant*30sub) -> TRANSPOSE
    (N,90,250) feature-major (de DWT). 7 lop.

    merge_val=False (giong git SenseFi): train=X_train(3977), test=X_test(500); val bo.
    merge_val=True : test = X_test + X_val (500+496=996)."""
    def ld(name):
        return np.load(_find(root, f'{name}.csv'), allow_pickle=True)
    Xtr = ld('X_train').astype(np.float32).transpose(0, 2, 1)
    Xte = ld('X_test').astype(np.float32).transpose(0, 2, 1)
    parts_X, parts_y = [Xtr, Xte], [ld('y_train').astype(np.int64),
                                    ld('y_test').astype(np.int64)]
    if merge_val:
        parts_X.append(ld('X_val').astype(np.float32).transpose(0, 2, 1))
        parts_y.append(ld('y_val').astype(np.int64))
    X = np.concatenate(parts_X, axis=0)
    y = np.concatenate(parts_y, axis=0)
    n_tr = len(Xtr); n_te = len(X) - n_tr
    return X, y, {'train': np.arange(n_tr), 'test': np.arange(n_tr, n_tr + n_te)}


def load_ntufi(root):
    """NTU-Fi: folder theo lop, .mat key 'CSIamp' (342,2000)=(3ant*114sub) x time.
    Da feature-major. Downsample time x[:, ::4]: 2000->500 (theo SenseFi)."""
    import scipy.io
    train_dir = _find(root, 'train_amp')
    test_dir = _find(root, 'test_amp')
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
                assert arr.shape == (342, 2000), f"bad NTU shape {arr.shape} in {fn}"
                arr = arr[:, ::4]                    # 2000 -> 500
                Xs.append(arr); ys.append(cls2idx[c]); split_idx[split].append(k); k += 1
    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    splits = {sp: np.array(ix, dtype=np.int64) for sp, ix in split_idx.items()}
    return X, y, splits


LOADERS = {'hust': load_hust, 'uthar': load_uthar, 'ntufi': load_ntufi}


# ── Haar DWT + đóng gói ──────────────────────────────────────────────────────
def haar_hl_lh(flat, n_ant, sub):
    """(n_ant*sub, time) -> (HL, LH), moi cai (time//2, n_ant*sub//2) float32.

    HL = cV.T = paper XH (chi tiet doc truc subcarrier).
    LH = cH.T = paper XV (chi tiet doc truc thoi gian).
    """
    flat = np.asarray(flat, dtype=np.float32)
    assert flat.shape[0] == n_ant * sub, \
        f"flat axis0 {flat.shape[0]} != n_ant*sub {n_ant*sub}"
    _, (cH, cV, _) = pywt.dwt2(flat, 'haar', mode='periodization')
    HL = cV.T.astype(np.float32)
    LH = cH.T.astype(np.float32)
    return HL, LH


def build_xhxv(X, n_ant, sub):
    """(N, n_ant*sub, time) -> xh, xv (N, T//2, M) float32 (chua chuan hoa)."""
    N = len(X)
    HL0, LH0 = haar_hl_lh(X[0], n_ant, sub)
    T2, M = HL0.shape
    xh = np.empty((N, T2, M), dtype=np.float32)
    xv = np.empty((N, T2, M), dtype=np.float32)
    xh[0], xv[0] = HL0, LH0
    for i in tqdm(range(1, N), desc='  Haar DWT', unit='smp'):
        xh[i], xv[i] = haar_hl_lh(X[i], n_ant, sub)
    return xh, xv


def _pos_stats(a):
    """mean/std per-POSITION (T2,M) theo truc samples (axis 0) — giong data_norm
    cua git TF-Mamba. std unbiased (ddof=1) de khop torch.std mac dinh."""
    ad = a.astype(np.float64)
    mean = ad.mean(axis=0)
    std = ad.std(axis=0, ddof=1)
    return mean.astype(np.float32), std.astype(np.float32)


# Hang so SenseFi cho NTU-Fi (CSI_Dataset): x = (x - 42.3199)/4.9802
NTUFI_MEAN, NTUFI_STD = 42.3199, 4.9802


def _sensefi_prenorm(name, X, splits):
    """Norm RAW truoc DWT, GIONG HET SenseFi (chi UT-HAR/NTU-Fi):
      uthar : min-max toan cuc TUNG split  (x-min)/(max-min)   [UT_HAR_dataset]
      ntufi : hang so co dinh (x-42.3199)/4.9802               [CSI_Dataset]
    HUST khong pre-norm. Tra X (da norm tai cho cho uthar; mang moi cho ntufi)."""
    if name == 'uthar':
        for idx in (splits['train'], splits['test']):
            seg = X[idx]
            mn, mx = float(seg.min()), float(seg.max())
            X[idx] = (seg - mn) / (mx - mn)
    elif name == 'ntufi':
        X = (X - np.float32(NTUFI_MEAN)) / np.float32(NTUFI_STD)
    return X


def make_datasets(name, raw_root, norm_mode='author', merge_val=False):
    """Tra (train_ds, test_ds, meta). Dataset tra (S_T=HL, S_F=LH, label).

    norm_mode (chi anh huong UT-HAR/NTU-Fi; HUST LUON = data_norm per-position z):
      'author' : SenseFi norm (raw) -> DWT, KHONG z-norm lai   (1 lan, dung tac gia)
      'double' : SenseFi norm (raw) -> DWT -> z-norm per-position (2 lan)
    z-norm per-position = `data_norm` git TF-Mamba: (x-mean_dim0)/(std_dim0+1e-6),
    all-reps (tinh tren CA train+test).
    merge_val: CHI UT-HAR — True thi gop X_val vao test; False (mac dinh) = git SenseFi.
    """
    assert norm_mode in ('author', 'double'), f"norm_mode la {norm_mode!r}"
    cfg = DATASETS[name]
    print(f"[{name}] load raw tu {raw_root}  (norm_mode={norm_mode}, merge_val={merge_val})")
    if name == 'uthar':
        X, y, splits = LOADERS[name](raw_root, merge_val=merge_val)
    else:
        X, y, splits = LOADERS[name](raw_root)
    n_ant, sub = cfg['n_ant'], cfg['sub']
    assert X.shape[1] == n_ant * sub, \
        f"axis1 {X.shape[1]} != n_ant*sub {n_ant*sub}"
    X = _sensefi_prenorm(name, X, splits)        # SenseFi norm raw (uthar/ntufi)
    print(f"  N={len(X)}  raw={X.shape[1:]}  -> DWT...")
    xh, xv = build_xhxv(X, n_ant, sub)
    del X
    T2, M = xh.shape[1], xh.shape[2]

    # z-norm per-position (data_norm): HUST luon; UT-HAR/NTU-Fi chi khi 'double'
    apply_z = (name == 'hust') or (norm_mode == 'double')
    if apply_z:
        xh_m, xh_s = _pos_stats(xh); xv_m, xv_s = _pos_stats(xv)
        xh -= xh_m; xh /= (xh_s + 1e-6)
        xv -= xv_m; xv /= (xv_s + 1e-6)
    print(f"  z-norm(post-DWT)={'YES' if apply_z else 'NO'}  xh,xv=({T2},{M})  "
          f"classes={cfg['classes']}  train={len(splits['train'])} test={len(splits['test'])}")

    tr, te = splits['train'], splits['test']
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    train_ds = TensorDataset(to_t(xh[tr]), to_t(xv[tr]), torch.from_numpy(y[tr]))
    test_ds = TensorDataset(to_t(xh[te]), to_t(xv[te]), torch.from_numpy(y[te]))
    meta = dict(name=name, M=M, T2=T2, num_classes=cfg['classes'],
                class_names=CLASS_NAMES[name], norm_mode=norm_mode,
                merge_val=merge_val, n_train=len(tr), n_test=len(te))
    return train_ds, test_ds, meta
