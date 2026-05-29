import numpy as np
from scipy.signal import butter, sosfiltfilt
from numpy.lib.stride_tricks import sliding_window_view


def hampel_vectorized(X, window=7, n_sigma=3.0, eps=1e-6):
    pad_width = [(0, 0)] * X.ndim
    pad_width[1] = (window, window)
    X_pad = np.pad(X, pad_width, mode='reflect')
    Xw = sliding_window_view(X_pad, 2 * window + 1, axis=1)
    median = np.median(Xw, axis=-1)
    mad = np.maximum(np.median(np.abs(Xw - median[..., None]), axis=-1), eps)
    threshold = n_sigma * 1.4826 * mad
    return np.where(np.abs(X - median) > threshold, median, X)


def extract_amplitude_raw(H):
    """Step 1: Hampel + LPF (NO normalization).
    Args:   H: (1000, 30, 3, 3) complex
    Returns: X: (9, 1000, 30) float32
    """
    assert H.shape == (1000, 30, 3, 3), \
        f"Expected H shape (1000, 30, 3, 3), got {H.shape}"
    X = np.abs(H).reshape(1000, 30, 9).transpose(2, 0, 1).astype(np.float32)
    X = hampel_vectorized(X, window=11, n_sigma=3.0, eps=1e-6)
    sos = butter(4, 20.0, btype='low', fs=200.0, output='sos')
    X = sosfiltfilt(sos, X, axis=1).astype(np.float32)
    return X


def fit_amplitude_stats(X_train_all):
    """Fit per-channel-per-subcarrier stats on TRAIN ONLY.
    Args:   X_train_all: (N_train, C, T2, F2) float32
    Returns: {'mean': list[C][F2], 'std': list[C][F2]}
    """
    mean = X_train_all.mean(axis=(0, 2))                       # (C, F2)
    std  = np.maximum(X_train_all.std(axis=(0, 2)), 1e-6)     # (C, F2)
    return {'mean': mean.tolist(), 'std': std.tolist()}


def apply_amplitude_stats(X, stats):
    """Apply per-channel-per-subcarrier z-score. X: (N, C, T2, F2)"""
    mean = np.array(stats['mean'], dtype=np.float32)[:, None, :]  # (C,1,F2)
    std  = np.array(stats['std'],  dtype=np.float32)[:, None, :]  # (C,1,F2)
    return ((X - mean) / std).astype(np.float32)
