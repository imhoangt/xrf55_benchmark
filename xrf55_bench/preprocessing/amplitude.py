import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def hampel_vectorized(X, window=7, n_sigma=3.0, eps=1e-6):
    """Vectorized Hampel filter along axis=1 (time). Replaces outliers
    (|x - median| > n_sigma·1.4826·MAD) with the local median.

    Used by 02_build_dataset_processed.py and 03_plot_amplitude.py (window=8).
    """
    pad_width = [(0, 0)] * X.ndim
    pad_width[1] = (window, window)
    X_pad = np.pad(X, pad_width, mode='reflect')
    Xw = sliding_window_view(X_pad, 2 * window + 1, axis=1)
    median = np.median(Xw, axis=-1)
    mad = np.maximum(np.median(np.abs(Xw - median[..., None]), axis=-1), eps)
    threshold = n_sigma * 1.4826 * mad
    return np.where(np.abs(X - median) > threshold, median, X)
