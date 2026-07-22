"""Preprocessing: raw CSI amplitude -> packed Haar DWT subbands.

Vectorized Hampel filter (outlier removal) + Butterworth LPF, and the 2-D
Haar DWT that splits a sample into (LL, HL, LH) subbands and packs the
selected ones for WavMamba.

Subband convention:
    cA, (cH, cV, cD) = pywt.dwt2(flat, 'haar', 'periodization')
        LL = cA.T   (approximation)
        HL = cV.T   (paper XH — detail along subcarrier axis)
        LH = cH.T   (paper XV — detail along time axis)
Packed channel order is canonical [LL | HL | LH], each n_per_sub maps,
so WavMamba's per-subband stem kernels {LL:(7,5), HL:(3,7), LH:(7,3)} line up.

Per-sample input to the transform is the merged amplitude `flat`:
    (n_ant * sub, time)   antenna-major, subcarrier-minor
Haar is 2-tap and each antenna has an EVEN subcarrier count, so dwt2 on the
merged axis never mixes antennas — identical to a per-antenna dwt2.
"""
import numpy as np
import pywt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

HAMPEL_WINDOW  = 8
HAMPEL_NSIGMA  = 3.0
LPF_ORDER      = 4
LPF_CUTOFF_HZ  = 20.0


def hampel_vectorized(X, window=7, n_sigma=3.0, eps=1e-6):
    """Vectorized Hampel filter along axis=1 (time). Replaces outliers
    (|x - median| > n_sigma*1.4826*MAD) with the local median.
    """
    pad_width = [(0, 0)] * X.ndim
    pad_width[1] = (window, window)
    X_pad = np.pad(X, pad_width, mode='reflect')
    Xw = sliding_window_view(X_pad, 2 * window + 1, axis=1)
    median = np.median(Xw, axis=-1)
    mad = np.maximum(np.median(np.abs(Xw - median[..., None]), axis=-1), eps)
    threshold = n_sigma * 1.4826 * mad
    return np.where(np.abs(X - median) > threshold, median, X)


def _filter_per_antenna(x_ats, fs):
    """Hampel + Butterworth LPF along time. x_ats: (n_ant, time, sub)."""
    x = hampel_vectorized(x_ats, window=HAMPEL_WINDOW, n_sigma=HAMPEL_NSIGMA)
    sos = butter(LPF_ORDER, LPF_CUTOFF_HZ, btype='low', fs=fs, output='sos')
    x = sosfiltfilt(sos, x, axis=1).astype(np.float32)
    return x


def haar3_subbands(flat, n_ant, sub, fs=None, do_filter=False):
    """(n_ant*sub, time) amplitude -> (LL, HL, LH), each (time//2, n_ant*sub//2).

    do_filter=True applies Hampel + LPF (needs fs) before the DWT, matching the
    'proc' build; do_filter=False is the 'raw' build.
    """
    flat = np.asarray(flat, dtype=np.float32)
    time = flat.shape[1]
    expected = n_ant * sub
    if flat.shape[0] != expected:
        raise ValueError(
            f"flat axis0 {flat.shape[0]} != n_ant*sub {expected} "
            f"({n_ant} antennas x {sub} subcarriers)"
        )

    if do_filter:
        if fs is None:
            raise ValueError("do_filter=True requires fs")
        x = flat.reshape(n_ant, sub, time).transpose(0, 2, 1)   # (n_ant, time, sub)
        x = _filter_per_antenna(x, fs)
        flat = x.transpose(0, 2, 1).reshape(n_ant * sub, time)

    cA, (cH, cV, _) = pywt.dwt2(flat, 'haar', mode='periodization')
    LL = cA.T.astype(np.float32)   # (time//2, n_ant*sub//2)
    HL = cV.T.astype(np.float32)
    LH = cH.T.astype(np.float32)
    return LL, HL, LH


def to_maps(a, n_per_sub):
    """(T, n_per_sub*f2) -> (n_per_sub, T, f2). Unflatten link-major feature axis."""
    T, M = a.shape
    f2 = M // n_per_sub
    return a.reshape(T, n_per_sub, f2).transpose(1, 0, 2)
