"""TF-Mamba — paper-faithful implementation.

Reference
─────────
Liu et al., "TF-Mamba: A Lightweight State-Space Model for Wi-Fi-Based
Human Activity Recognition," IEEE Sensors Journal, Vol. 25, No. 13, 2025.
https://doi.org/10.1109/JSEN.2024.3520857

Input convention (caller's responsibility)
──────────────────────────────────────────
Both XH and XV are delivered in the same orientation:

    XH, XV  ∈  R^{B × L × M}    where  L = T/2,  M = N/2

HUST-HAR example (S ∈ R^{270×1000}):
    After 2-D Haar DWT: XH_raw, XV_raw ∈ R^{135×500}
    Caller transposes BOTH:
        XH = XH_raw.permute(0,2,1)   →  (B, 500, 135)
        XV = XV_raw.permute(0,2,1)   →  (B, 500, 135)

Because L_T = L_F = L, the full paper-faithful sequence-level flow is used:
    AdaptiveFusion operates on (B, L, D) — not on pre-pooled vectors.

Paper-faithful data flow (Fig. 5, Sec. IV)
──────────────────────────────────────────
XH ──► stream_T ──► S_T (B, L, D)
                                   ──► AdaptiveFusion [Eq.15] ──► S2 (B, L, D)
XV ──► stream_F ──► S_F (B, L, D)
                                         │ proj_s3 + tanh
                                        S3 (B, L, D)
                                         │ GAP  mean(dim=1)
                                        (B, D)
                                         │ classifier
                                        Ŝ  (B, C)

Differences from original model.py
════════════════════════════════════
[Fix 1] Global Average Pooling position
    OLD: pool S_T and S_F BEFORE AdaptiveFusion
         → fusion(mean(S_T), mean(S_F)) ≠ mean(fusion(S_T, S_F))
    NEW: AdaptiveFusion on full sequences (B, L, D); GAP applied AFTER proj_s3.

[Fix 2] AdaptiveFusion weight indexing
    OLD: w[:, 0:1], w[:, 1:2]  → slices axis-1 (the L axis) for 3-D input
    NEW: w[..., 0:1], w[..., 1:2]  → slices last axis (feature dim) correctly.

[Fix 3] proj_s3 output dimension D′
    Paper: "higher dimensional space" refers to the raw feature space M,
    not d_model. Optimal D′ = D = 64 confirmed by Table I.
    Linear(d_model, d_model) is correct; d_proj parameter kept for flexibility.
"""

import math
import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError as _e:
    raise ImportError(
        'mamba_ssm not installed. Run: pip install mamba-ssm\n'
        'Requires CUDA + compatible GPU.'
    ) from _e


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding [Vaswani et al. 2017; Sec. IV-A].

    Stores a (1, max_len, d_model) buffer — moves with the model device
    automatically via register_buffer.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, L, *) — only L is used.
        Returns: (1, L, d_model)."""
        return self.pe[:, :x.size(1), :]


class EmbeddingLayer(nn.Module):
    """Linear projection + positional embedding — Eq. 10, Sec. IV-A.

        S_1 = ReLU(W·x + b) + P_pos

    Maps each time-step from num_features to embed_dim (d_model)
    and injects positional encodings.
    """

    def __init__(self, num_features: int, embed_dim: int, max_len: int = 500):
        super().__init__()
        self.pos = PositionalEmbedding(embed_dim, max_len)
        self.fc  = nn.Linear(num_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, num_features)  →  (B, L, embed_dim)."""
        return torch.relu(self.fc(x)) + self.pos(x)


class TFMambaStream(nn.Module):
    """Single Mamba stream: EmbeddingLayer + P stacked pre-norm Mamba blocks.

    Scans along the sequence axis L (temporal dimension).
    Mamba d_model = D (embed dim); sequence length = L.

    Each block implements Eq. 11–14 (Fig. 6):
        Forward path:   M_f = SSM(σ(Conv(Linear(LN(M)))))     [Eq. 11]
        Independent:    M_i = σ(Linear(LN(M)))                 [Eq. 12]
        Gating:         M_g = M_f ⊙ M_i                       [Eq. 13]
        Skip conn.:     M_o = M + Linear(M_g)                  [Eq. 14]

    Implementation note
    ───────────────────
    mamba_ssm.Mamba encapsulates Eq. 11–13 internally:
      in_proj  → splits to x_branch (→ Conv → SSM) and z_gate (→ SiLU)
      hadamard product of both + out_proj  =  Linear(M_g)
    The outer `x = x + mamba(norm(x))` is the skip of Eq. 14:
      M_o = M + Linear(M_g)   ← exact match.

    Output shape: (B, L, d_model) — sequence is NEVER pooled here.
    """

    def __init__(
        self,
        num_features: int,
        d_model:      int,
        d_state:      int = 16,
        d_conv:       int = 4,
        expand:       int = 2,
        num_layers:   int = 3,
        max_len:      int = 500,
    ):
        super().__init__()
        self.emb    = EmbeddingLayer(num_features, d_model, max_len)
        self.norms  = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )
        self.mambas = nn.ModuleList(
            [
                Mamba(d_model=d_model, d_state=d_state,
                      d_conv=d_conv, expand=expand)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, num_features)
        Returns:
            (B, L, d_model)  — full sequence, never pooled.
        """
        x = self.emb(x)
        for norm, mamba in zip(self.norms, self.mambas):
            x = x + mamba(norm(x))   # pre-norm residual, Eq. 14
        return x


class AdaptiveFusion(nn.Module):
    """Soft-weighted adaptive fusion — Sec. IV-E, Eq. 15.

        S_cat    = Concat(S_T, S_F)                  (..., 2D)
        α_T, α_F = Softmax( Linear_{2D→2}(S_cat) )
        S_fused  = α_T ⊙ S_T  +  α_F ⊙ S_F

    Input shape: (B, L, D) for both S_T and S_F.
    Output shape: (B, L, D) — same as inputs.

    [Fix 2] w[..., 0:1] and w[..., 1:2] slice the LAST axis (feature dim),
    broadcasting correctly over all preceding dimensions including L.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(2 * d_model, 2)

    def forward(self, ST: torch.Tensor, SF: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ST: (B, L, D)
            SF: (B, L, D)
        Returns:
            S_fused: (B, L, D)
        """
        w = torch.softmax(
            self.linear(torch.cat([ST, SF], dim=-1)), dim=-1
        )                                              # (B, L, 2)
        return w[..., 0:1] * ST + w[..., 1:2] * SF    # (B, L, D)


# ─────────────────────────────────────────────────────────────────────────────
# Full TF-Mamba model
# ─────────────────────────────────────────────────────────────────────────────

class TFMamba(nn.Module):
    """TF-Mamba dual-stream SSM for Wi-Fi CSI-based HAR — Fig. 5, Sec. IV.

    Both streams receive inputs of identical shape (B, L, M) and scan along L.
    The two streams differ only in their input subband: XH (time-domain Haar
    horizontal detail) vs XV (freq-domain Haar vertical detail).

    Args
    ────
    num_features : feature dim M shared by both XH and XV inputs
                   (= N/2 = 135 for HUST-HAR after DWT + transpose)
    d_model      : hidden dimension D = D′ = 64                  (default 64)
    num_layers   : Mamba layers per stream P                      (default 3)
    num_classes  : number of activity classes c                   (default 6)
    d_state      : Mamba SSM state dimension                      (default 16)
    d_conv       : Mamba depthwise-conv kernel size               (default 4)
    expand       : Mamba inner-dimension expansion factor         (default 2)
    max_len      : positional-encoding max sequence length L      (default 500)
    """

    def __init__(
        self,
        num_features: int,
        d_model:      int = 64,
        num_layers:   int = 3,
        num_classes:  int = 6,
        d_state:      int = 16,
        d_conv:       int = 4,
        expand:       int = 2,
        max_len:      int = 500,
    ):
        super().__init__()

        shared_kwargs = dict(
            num_features=num_features,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            max_len=max_len,
        )

        # Time-Mamba stream — XH (horizontal-detail DWT subband)
        self.stream_T = TFMambaStream(**shared_kwargs)

        # Freq-Mamba stream — XV (vertical-detail DWT subband)
        self.stream_F = TFMambaStream(**shared_kwargs)

        # Adaptive fusion  [Eq. 15]
        self.fusion = AdaptiveFusion(d_model)

        # proj_s3: S2 → S3 with tanh  (D′ = D = 64, confirmed by Table I)
        self.proj_s3 = nn.Linear(d_model, d_model)

        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, XH: torch.Tensor, XV: torch.Tensor) -> torch.Tensor:
        """
        Args
        ────
        XH : (B, L, M)  — DWT horizontal-detail subband, transposed by caller
        XV : (B, L, M)  — DWT vertical-detail subband,   transposed by caller
             Both must have identical shape.

        Returns
        ───────
        logits : (B, num_classes)
        """
        # ── Step 1: stream processing ─────────────────────────────────────────
        S_T = self.stream_T(XH)              # (B, L, d_model)
        S_F = self.stream_F(XV)              # (B, L, d_model)

        # ── Step 2: sequence-level adaptive fusion [Eq. 15] ──────────────────
        S2 = self.fusion(S_T, S_F)           # (B, L, d_model)

        # ── Step 3: linear projection + tanh ─────────────────────────────────
        S3 = torch.tanh(self.proj_s3(S2))    # (B, L, d_model)

        # ── Step 4: Global Average Pooling ────────────────────────────────────
        S3 = S3.mean(dim=1)                  # (B, d_model)

        # ── Step 5: classifier ────────────────────────────────────────────────
        return self.classifier(S3)           # (B, num_classes)
