"""Generalized Haar 3-branch (S4.1) preprocessing for multiple CSI-HAR datasets.

Replicates the XRF55 S4.1 pipeline (WavDualMamba on Haar {LL,HL,LH}) but
parametrized by (n_antennas, subcarriers, time, fs) so it works on HUST-HAR,
UT-HAR and NTU-Fi as well.

Subband convention — IDENTICAL to scripts/02_build_dataset_processed.py:
    cA, (cH, cV, cD) = pywt.dwt2(flat, 'haar', 'periodization')
        LL = cA.T   (approximation)
        HL = cV.T   (paper XH — detail along subcarrier axis)
        LH = cH.T   (paper XV — detail along time axis)
Packed channel order is canonical [LL | HL | LH], each n_per_sub maps,
so WavDualMamba's per-subband kernels {LL:(7,5), HL:(3,7), LH:(7,3)} line up.

Per-sample input to the transform is the merged amplitude `flat`:
    (n_ant * sub, time)   antenna-major, subcarrier-minor
Haar is 2-tap and each antenna has an EVEN subcarrier count, so dwt2 on the
merged axis never mixes antennas — identical to a per-antenna dwt2.
"""
import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

from xrf55_bench.preprocessing.amplitude import hampel_vectorized

HAMPEL_WINDOW  = 8
HAMPEL_NSIGMA  = 3.0
LPF_ORDER      = 4
LPF_CUTOFF_HZ  = 20.0


def _filter_per_antenna(x_ats, fs):
    """Hampel + Butterworth LPF along time. x_ats: (n_ant, time, sub)."""
    x = hampel_vectorized(x_ats, window=HAMPEL_WINDOW, n_sigma=HAMPEL_NSIGMA)
    sos = butter(LPF_ORDER, LPF_CUTOFF_HZ, btype='low', fs=fs, output='sos')
    x = sosfiltfilt(sos, x, axis=1).astype(np.float32)
    return x


def haar3_subbands(flat, n_ant, sub, fs=None, do_filter=False):
    """(n_ant*sub, time) amplitude -> (LL, HL, LH), each (time//2, n_ant*sub//2).

    do_filter=True applies Hampel + LPF (needs fs) before the DWT, matching the
    XRF55 'proc' build; do_filter=False is the 'raw' build.
    """
    flat = np.asarray(flat, dtype=np.float32)
    time = flat.shape[1]
    assert flat.shape[0] == n_ant * sub, \
        f"flat axis0 {flat.shape[0]} != n_ant*sub {n_ant*sub}"

    if do_filter:
        assert fs is not None, "do_filter=True requires fs"
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


def pack_haar3(flat, n_ant, sub, n_per_sub, fs=None, do_filter=False):
    """One sample (n_ant*sub, time) -> packed (3*n_per_sub, T//2, sub//2) float32.

    n_per_sub = spatial channels per subband (= n_ant here). Packed order [LL|HL|LH].
    """
    LL, HL, LH = haar3_subbands(flat, n_ant, sub, fs=fs, do_filter=do_filter)
    x = np.concatenate(
        [to_maps(LL, n_per_sub), to_maps(HL, n_per_sub), to_maps(LH, n_per_sub)],
        axis=0,
    ).astype(np.float32, copy=False)
    return x   # (3*n_per_sub, time//2, sub//2)
