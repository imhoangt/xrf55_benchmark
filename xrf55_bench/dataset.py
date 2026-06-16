"""Dataset loaders for the XRF55 benchmark.

Loads pre-built bench arrays from bench/raw_nosc/ or bench/processed_nosc/.
Build these with 01_build_dataset_raw.py (raw) or 02_build_dataset_processed.py (processed).

Split: train=reps 1-14 (4620 samples), test=reps 15-20 (1980 samples). No val.

Model input shapes after normalization:
  resnet    → (270, 1000)          per-channel z-score (270,)
  tfmamba   → XH (500, 135)        per-channel z-score on XH = L·S·Hᵀ (pywt cV.T)
               XV (500, 135)        per-channel z-score on XV = H·S·Lᵀ (pywt cH.T)
  wavdualmamba_haar → (18, 500, 15)  tfmamba Haar arrays re-packed as
               packed WavDualMamba input [HL ‖ LH] (ablation ladder S4)
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pywt
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_stats(bench_dir) -> dict:
    with open(Path(bench_dir) / 'stats.json') as f:
        return json.load(f)


def infer_data_mode(stats: dict) -> str:
    """Infer 'proc' | 'raw' from stats.json meta.

    Processed stats (02_build_dataset_processed.py) carry a 'filter' key and
    meta.source='raw_npy_nosc_270_hampel_lpf'.  Raw stats (01_build_dataset_raw.py)
    use meta.source='raw_npy_nosc_270' with no filter key.
    """
    meta = stats.get('meta', {})
    if 'filter' in meta or 'hampel' in str(meta.get('source', '')).lower():
        return 'proc'
    return 'raw'


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _kw(num_workers):
    kw = dict(pin_memory=True, num_workers=num_workers,
              persistent_workers=(num_workers > 0))
    if num_workers > 0:
        kw['worker_init_fn'] = _worker_init_fn
    return kw


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset classes — load from pre-built bench arrays
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocResNetDataset(Dataset):
    """Loads resnet/X_{split}.npy, applies per-channel z-score. Returns (X, label)."""

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.X   = np.load(bench_dir / 'resnet' / f'X_{split}.npy', mmap_mode='r')
        self.y   = np.load(bench_dir / f'y_{split}.npy')
        self.mu  = np.array(stats['resnet']['mean'], dtype=np.float32)  # (270,)
        self.sig = np.array(stats['resnet']['std'],  dtype=np.float32)

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mu[:, None]) / self.sig[:, None]   # (270, 1000)
        return torch.from_numpy(x), int(self.y[idx])


def _tfmamba_hl_is_xh(stats: dict) -> bool:
    """True if the tfmamba xh file holds HL content (builds after the cH/cV
    naming fix, marked meta.tfmamba_subband_naming=='paper-eq5'); False for
    legacy builds where the xh file holds LH content. Single source for the
    HL/LH vintage decision, shared by PreprocTFMambaDataset + the S4 adapter."""
    return stats.get('meta', {}).get('tfmamba_subband_naming') == 'paper-eq5'


class PreprocTFMambaDataset(Dataset):
    """Loads tfmamba XH/XV arrays (N,500,135), normalizes, returns (S_T, S_F, label).

    The two returned streams are CANONICALISED to (HL content, LH content) via
    the tfmamba_subband_naming marker, so stream_T always carries the HL subband
    regardless of build vintage. This lets TFMamba(subband_kernels=True) apply
    WavDualMamba's physical stem kernels (stream_T=HL (3,7), stream_F=LH (7,3)) to
    the matching content — same routing as the S4 adapter. Each file is z-scored
    with its own tfmamba stats before the (symmetric) reorder.
    """

    def __init__(self, bench_dir: Path, split: str, stats: dict, norm_mode: str = 'double'):
        bench_dir = Path(bench_dir)
        self.XH = np.load(bench_dir / 'tfmamba' / f'X_{split}_xh.npy', mmap_mode='r')  # (N, 500, 135)
        self.XV = np.load(bench_dir / 'tfmamba' / f'X_{split}_xv.npy', mmap_mode='r')  # (N, 500, 135)
        self.y  = np.load(bench_dir / f'y_{split}.npy')
        s = stats['tfmamba']
        # Stats co the la (M,) per-feature [XRF55 builds] hoac (T2,M) per-position
        # [multi-dataset = data_norm cua git TF-Mamba]. Dua ve dang broadcast duoc
        # voi (T2,M): (M,) -> (1,M); (T2,M) -> giu nguyen.
        def _bcast(a):
            a = np.array(a, dtype=np.float32)
            return a[None, :] if a.ndim == 1 else a
        self.xh_mean = _bcast(s['xh_mean']); self.xh_std = _bcast(s['xh_std'])
        self.xv_mean = _bcast(s['xv_mean']); self.xv_std = _bcast(s['xv_std'])
        self._hl_is_xh = _tfmamba_hl_is_xh(stats)
        # norm_mode='author': UT-HAR/NTU-Fi chi SenseFi pre-norm (build), KHONG z-norm
        # lai. Chi bo z khi build co SenseFi pre-norm (uthar/ntufi); HUST/XRF55 luon z.
        self._skip_z = (norm_mode == 'author'
                        and stats.get('meta', {}).get('sensefi_prenorm', False))

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        xh = np.array(self.XH[idx], dtype=np.float32)                     # copy -> writable
        xv = np.array(self.XV[idx], dtype=np.float32)
        if not self._skip_z:
            xh = (xh - self.xh_mean) / self.xh_std        # mean/std da broadcast (1,M)|(T2,M)
            xv = (xv - self.xv_mean) / self.xv_std
        # Canonical order: stream_T = HL content, stream_F = LH content.
        st, sf = (xh, xv) if self._hl_is_xh else (xv, xh)
        return torch.from_numpy(st), torch.from_numpy(sf), int(self.y[idx])


class PreprocTFMambaHaarAsWavDataset(Dataset):
    """[Ablation ladder S4] TF-Mamba Haar arrays re-packed as WavDualMamba input.

    Loads tfmamba X_{split}_xh/xv.npy (N, 500, 135), z-scores each file with its
    own tfmamba stats, unflattens 135 = 9 links × 15 bins (link-major — Haar
    pairs never straddle link boundaries since 30 is even), and stacks the two
    detail subbands → (18, 500, 15), packed channel layout [HL(9) ‖ LH(9)]
    matching WavDualMamba's canonical subband order for subbands=('HL','LH').

    HL/LH content mapping depends on the dataset build vintage:
      - builds AFTER the cH/cV naming fix carry
        meta.tfmamba_subband_naming='paper-eq5'  → xh file holds HL content;
      - LEGACY builds (no marker) stored the swapped naming → xh file holds LH.
    Branch assignment only matters for WavDualMamba's per-subband stem kernels
    {HL:(3,7), LH:(7,3)}; everything else is branch-symmetric.

    Returns (X, label) with X float32 (18, 500, 15).
    """

    N_LINKS = 9

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.XH = np.load(bench_dir / 'tfmamba' / f'X_{split}_xh.npy', mmap_mode='r')
        self.XV = np.load(bench_dir / 'tfmamba' / f'X_{split}_xv.npy', mmap_mode='r')
        self.y  = np.load(bench_dir / f'y_{split}.npy')
        s = stats['tfmamba']
        self.xh_mean = np.array(s['xh_mean'], dtype=np.float32)  # (135,)
        self.xh_std  = np.array(s['xh_std'],  dtype=np.float32)
        self.xv_mean = np.array(s['xv_mean'], dtype=np.float32)
        self.xv_std  = np.array(s['xv_std'],  dtype=np.float32)

        M = self.XH.shape[-1]
        if M % self.N_LINKS != 0:
            raise ValueError(f'Feature dim {M} not divisible by {self.N_LINKS} links')
        self.f2 = M // self.N_LINKS

        self._hl_is_xh = _tfmamba_hl_is_xh(stats)

    def __len__(self):  return len(self.y)

    def _to_maps(self, a: np.ndarray) -> np.ndarray:
        """(T, 135) → (9, T, 15) — unflatten the link-major feature axis."""
        T = a.shape[0]
        return a.reshape(T, self.N_LINKS, self.f2).transpose(1, 0, 2)

    def __getitem__(self, idx):
        xh = (self.XH[idx] - self.xh_mean[None, :]) / self.xh_std[None, :]
        xv = (self.XV[idx] - self.xv_mean[None, :]) / self.xv_std[None, :]
        hl, lh = (xh, xv) if self._hl_is_xh else (xv, xh)
        x = np.concatenate(
            [self._to_maps(hl), self._to_maps(lh)], axis=0
        ).astype(np.float32, copy=False)                  # (18, 500, 15)
        return torch.from_numpy(x), int(self.y[idx])


class PreprocWavMambaDataset(Dataset):
    """Loads wavmamba/X_{split}.npy, applies per-channel z-score. Returns (X, label)."""

    def __init__(self, bench_dir: Path, split: str, stats: dict, norm_mode: str = 'double'):
        bench_dir = Path(bench_dir)
        self.X   = np.load(bench_dir / 'wavmamba' / f'X_{split}.npy', mmap_mode='r')
        self.y   = np.load(bench_dir / f'y_{split}.npy')
        s = stats['wavmamba']
        # Stats (C,F2) per-channel-bin [XRF55] hoac (C,T2,F2) per-position [multi-dataset].
        # Dua ve dang broadcast voi (C,T2,F2): (C,F2)->(C,1,F2); (C,T2,F2) giu nguyen.
        def _bcast(a):
            a = np.array(a, dtype=np.float32)
            return a[:, None, :] if a.ndim == 2 else a
        self.mu = _bcast(s['mean']); self.sig = _bcast(s['std'])
        # author: bo z-norm khi build co SenseFi pre-norm (uthar/ntufi); xem TFMamba ds.
        self._skip_z = (norm_mode == 'author'
                        and stats.get('meta', {}).get('sensefi_prenorm', False))

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        x = np.array(self.X[idx], dtype=np.float32)                 # copy -> writable
        if not self._skip_z:
            x = (x - self.mu) / self.sig          # mu/sig da broadcast (C,1,F2)|(C,T2,F2)
        return torch.from_numpy(x), int(self.y[idx])


# ── Haar LL subband helper (3-branch Haar ablation S4.1/S4.2) ───────────────────
_LL_STATS_CACHE: dict = {}   # str(bench_dir) -> (mean(135,), std(135,)), all-reps


def _haar_ll(flat: np.ndarray) -> np.ndarray:
    """(270,1000) amplitude -> Haar LL subband (500,135) = pywt cA.T.

    Uses the SAME dwt2 the build applied for xh/xv; since resnet/X holds the exact
    `flat` that produced xh/xv, this LL is consistent with the stored HL/LH."""
    cA = pywt.dwt2(np.asarray(flat, dtype=np.float32), 'haar', mode='periodization')[0]
    return cA.T                                                # (500, 135)


def _haar_ll_stats(bench_dir: Path):
    """Per-feature (135,) mean/std of the Haar LL subband, fitted on ALL reps
    (train+test) — same protocol the build uses for xh/xv. Cached per bench_dir."""
    key = str(bench_dir)
    if key in _LL_STATS_CACHE:
        return _LL_STATS_CACHE[key]
    s  = np.zeros(135, dtype=np.float64)
    s2 = np.zeros(135, dtype=np.float64)
    n  = np.int64(0)
    for split in ('train', 'test'):
        R = np.load(bench_dir / 'resnet' / f'X_{split}.npy', mmap_mode='r')
        for i in range(R.shape[0]):
            ll = _haar_ll(R[i]).astype(np.float64)             # (500, 135)
            s  += ll.sum(axis=0)
            s2 += (ll * ll).sum(axis=0)
            n  += ll.shape[0]
    mean = s / n
    std  = np.maximum(np.sqrt(np.maximum(s2 / n - mean * mean, 0.0)), 1e-6)
    out = (mean.astype(np.float32), std.astype(np.float32))
    _LL_STATS_CACHE[key] = out
    return out


class PreprocTFMambaHaar3AsWavDataset(Dataset):
    """[Ablation S4.1/S4.2] Haar 3-branch (LL,HL,LH) packed for WavDualMamba.

    HL,LH reuse tfmamba X_{split}_xh/xv.npy z-scored with their own stats
    (identical to S4). LL is computed on-the-fly from resnet/X_{split}.npy
    (the exact `flat` the build fed to dwt2, so LL matches the stored HL/LH)
    and z-scored with all-reps LL stats. Returns (X, label) with X float32
    (27, 500, 15) in canonical subband order [LL | HL | LH].
    """

    N_LINKS = 9

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.XH = np.load(bench_dir / 'tfmamba' / f'X_{split}_xh.npy', mmap_mode='r')
        self.XV = np.load(bench_dir / 'tfmamba' / f'X_{split}_xv.npy', mmap_mode='r')
        self.R  = np.load(bench_dir / 'resnet'  / f'X_{split}.npy',    mmap_mode='r')
        self.y  = np.load(bench_dir / f'y_{split}.npy')
        s = stats['tfmamba']
        self.xh_mean = np.array(s['xh_mean'], dtype=np.float32)
        self.xh_std  = np.array(s['xh_std'],  dtype=np.float32)
        self.xv_mean = np.array(s['xv_mean'], dtype=np.float32)
        self.xv_std  = np.array(s['xv_std'],  dtype=np.float32)

        M = self.XH.shape[-1]
        if M % self.N_LINKS != 0:
            raise ValueError(f'Feature dim {M} not divisible by {self.N_LINKS} links')
        self.f2 = M // self.N_LINKS
        if len(self.R) != len(self.XH):
            raise ValueError(
                f'resnet ({len(self.R)}) and tfmamba ({len(self.XH)}) sample counts differ')

        self._hl_is_xh = _tfmamba_hl_is_xh(stats)
        self.ll_mean, self.ll_std = _haar_ll_stats(bench_dir)   # all-reps, cached

    def __len__(self):  return len(self.y)

    def _to_maps(self, a: np.ndarray) -> np.ndarray:
        """(T, 135) -> (9, T, 15) — unflatten the link-major feature axis."""
        T = a.shape[0]
        return a.reshape(T, self.N_LINKS, self.f2).transpose(1, 0, 2)

    def __getitem__(self, idx):
        xh = (self.XH[idx] - self.xh_mean[None, :]) / self.xh_std[None, :]
        xv = (self.XV[idx] - self.xv_mean[None, :]) / self.xv_std[None, :]
        ll = (_haar_ll(self.R[idx]) - self.ll_mean[None, :]) / self.ll_std[None, :]
        hl, lh = (xh, xv) if self._hl_is_xh else (xv, xh)
        x = np.concatenate(
            [self._to_maps(ll), self._to_maps(hl), self._to_maps(lh)], axis=0
        ).astype(np.float32, copy=False)                  # (27, 500, 15) = LL|HL|LH
        return torch.from_numpy(x), int(self.y[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# Loader factory
# ═══════════════════════════════════════════════════════════════════════════════

_PREPROC_DS = {
    'resnet':            PreprocResNetDataset,
    'tfmamba':           PreprocTFMambaDataset,
    'wavdualmamba':      PreprocWavMambaDataset,
    'wavdualmamba_haar': PreprocTFMambaHaarAsWavDataset,   # [S4] tfmamba Haar files
    'wavdualmamba_haar3': PreprocTFMambaHaar3AsWavDataset,  # [S4.1/S4.2] Haar LL+HL+LH
}
_PREPROC_SENTINEL = {
    'resnet':            'resnet/X_train.npy',
    'tfmamba':           'tfmamba/X_train_xh.npy',
    'wavdualmamba':      'wavmamba/X_train.npy',
    'wavdualmamba_haar': 'tfmamba/X_train_xh.npy',   # [S4] same files as tfmamba
    'wavdualmamba_haar3': 'resnet/X_train.npy',       # [S4.1/S4.2] needs resnet for Haar LL
}


def build_loaders(model_name: str, stats: dict, bench_dir,
                  batch_size: int = 32, num_workers: int = 4,
                  norm_mode: str = 'double'):
    """Build (train_loader, test_loader) from pre-built bench arrays.

    Args:
        model_name : 'resnet', 'tfmamba', 'wavdualmamba', or 'wavdualmamba_haar'
        stats      : dict from load_stats(bench_dir)
        bench_dir  : path to bench/raw_nosc/ or bench/processed_nosc/
        norm_mode  : 'double' (mac dinh, z-norm luon — giu hanh vi cu) | 'author'
                     (bo z-norm cho build co SenseFi pre-norm = UT-HAR/NTU-Fi).
                     Chi tfmamba & wavdualmamba ho tro; cac ds khac bo qua.
    """
    if model_name not in _PREPROC_DS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_PREPROC_DS)}")

    bench_dir = Path(bench_dir)
    sentinel  = bench_dir / _PREPROC_SENTINEL[model_name]
    if not sentinel.exists():
        raise FileNotFoundError(
            f'Bench arrays not found: {sentinel}\n'
            'Run 01_build_dataset_raw.py or 02_build_dataset_processed.py first.')

    data_label = ('Processed CSI Amplitude (Hampel + Butterworth LPF)'
                  if infer_data_mode(stats) == 'proc' else 'Raw CSI Amplitude')
    print(f'  Data  : {data_label}')
    print(f'  Loaded: {bench_dir}')

    DS = _PREPROC_DS[model_name]
    kw = _kw(num_workers)
    # norm_mode chi ap dung cho 2 ds co duong z-norm theo SenseFi pre-norm.
    extra = {'norm_mode': norm_mode} if model_name in ('tfmamba', 'wavdualmamba') else {}
    train_ds = DS(bench_dir, 'train', stats, **extra)
    test_ds  = DS(bench_dir, 'test',  stats, **extra)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)
    return train_loader, test_loader
