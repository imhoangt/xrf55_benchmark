import numpy as np
import pywt


def apply_dwt2_stack(X, wavelet='db4', mode='periodization'):
    """X: (B, C, T=1000, F=30) → (B, C*3, T2=500, F2=15) — drops HH (diagonal detail).

    Stack order along channel axis:
        [LL_0..LL_{C-1} | HL_0..HL_{C-1} | LH_0..LH_{C-1}]

    PyWavelets mapping for dwt2(x) where x[rows=time, cols=subcarrier]:
        cA → LL  (approx: low-pass time  × low-pass freq)
        cH → HL  (high-pass time × low-pass freq  — PyWavelets "horizontal")
        cV → LH  (low-pass time  × high-pass freq — PyWavelets "vertical")
        cD → HH  (diagonal detail) — DROPPED

    Vectorized via axes=(-2,-1): one PyWavelets call over all (B,C) samples,
    avoiding the Python-level B*C loop (≥10× faster on large batches).
    """
    B, C, T, F = X.shape
    assert T == 1000 and F == 30, \
        f"Expected T=1000, F=30 (XRF55 spec); got T={T}, F={F}"
    cA, (cH, cV, _) = pywt.dwt2(X, wavelet=wavelet, mode=mode, axes=(-2, -1))
    # cA, cH, cV: (B, C, T2, F2)
    out = np.concatenate([cA, cH, cV], axis=1).astype(np.float32, copy=False)
    return out
