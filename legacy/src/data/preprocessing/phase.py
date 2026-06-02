import numpy as np


def extract_phase_raw(H, ref_ant=0):
    """Conjugate diff + subcarrier-domain LSQ detrend (NO time-domain detrend, NO LPF).

    SFO residual after conjugate diff manifests as a per-packet linear slope
    across subcarrier indices (ε·k), not across time. LSQ per (stream, timestep)
    removes this slope without disturbing low-frequency body motion (0.5–1 Hz).

    Args:    H: (1000, 30, 3, 3) complex
    Returns: X: (6, 1000, 30) float32
    """
    assert H.shape == (1000, 30, 3, 3), \
        f"Expected H shape (1000, 30, 3, 3), got {H.shape}"

    T, F, n_rx, _ = H.shape
    streams = []
    for m in range(n_rx):
        H_ref = H[:, :, m, ref_ant]                        # (T, F)
        for a in range(3):
            if a == ref_ant:
                continue
            Z = H[:, :, m, a] * np.conj(H_ref)            # (T, F)
            streams.append(np.angle(Z))                    # (T, F)

    # phi: (6, T, F) — stack 6 streams
    phi = np.stack(streams, axis=0).astype(np.float64)

    # Unwrap along SUBCARRIER axis (axis=2) per stream
    phi = np.unwrap(phi, axis=2)

    # LSQ linear fit per (stream, timestep) to remove residual SFO slope ε·k
    k     = np.arange(F, dtype=np.float64)    # (F,) hardware-agnostic indices
    k_bar = k.mean()                           # scalar
    k_c   = k - k_bar                         # centered (F,)
    Skk   = (k_c ** 2).sum()                  # scalar

    phi_mean = phi.mean(axis=2, keepdims=True)                                  # (6, T, 1)
    phi_c    = phi - phi_mean                                                   # (6, T, F)
    eps      = (k_c[None, None, :] * phi_c).sum(axis=2, keepdims=True) / Skk   # (6, T, 1)
    tau      = phi_mean - eps * k_bar                                           # (6, T, 1)
    phi_detr = phi - (eps * k[None, None, :] + tau)                            # (6, T, F)

    return phi_detr.astype(np.float32)


def fit_phase_stats(X_train_all):
    """Fit per-channel-per-subcarrier stats on TRAIN ONLY.
    Args:   X_train_all: (N_train, C, T2, F2) float32
    Returns: {'mean': list[C][F2], 'std': list[C][F2]}
    """
    mean = X_train_all.mean(axis=(0, 2))                       # (C, F2)
    std  = np.maximum(X_train_all.std(axis=(0, 2)), 1e-6)     # (C, F2)
    return {'mean': mean.tolist(), 'std': std.tolist()}


def apply_phase_stats(X, stats):
    """Apply per-channel-per-subcarrier z-score. X: (N, C, T2, F2)"""
    mean = np.array(stats['mean'], dtype=np.float32)[:, None, :]  # (C,1,F2)
    std  = np.array(stats['std'],  dtype=np.float32)[:, None, :]  # (C,1,F2)
    return ((X - mean) / std).astype(np.float32)
