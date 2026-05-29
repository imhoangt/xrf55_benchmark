"""TF-Mamba — paper-faithful implementation (final version4).

Reference
─────────
Liu et al., "TF-Mamba: A Lightweight State-Space Model for Wi-Fi-Based
Human Activity Recognition," IEEE Sensors Journal, Vol. 25, No. 13, 2025.
https://doi.org/10.1109/JSEN.2024.3520857

Input convention (caller's responsibility)
──────────────────────────────────────────
Both XH and XV are delivered in the same orientation:

    XH, XV  ∈  R^{B × L × M}    where  L = T/2 = 500,  M = N/2 = 135

HUST-HAR example (S ∈ R^{270×1000} after 200-Hz downsampling):
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

[Fix 4] Frequency-Mamba scan direction (Fig. 7)
    OLD: stream_F used TFMambaStream — scanned along L=500, same as stream_T.
         Both streams were identical; "freq" was only in the input subband.
    NEW: stream_F uses TFMambaStreamFreq — after EmbeddingLayer produces
         (B, L, D), transposes to (B, D, L), Mamba scans D=64 steps (each
         carrying L=500 features), then transposes back to (B, L, D).
         This matches Fig. 7(c): vertical arrows on X_V mean scan along the
         feature/embedding axis, not the temporal axis.
         Consequence: stream_F Mamba uses d_model=max_len=L, not d_model=D.
"""

import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba


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
    """Time-Mamba stream: EmbeddingLayer + P stacked pre-norm Mamba blocks.

    Scans along the temporal axis L (Fig. 7b — horizontal arrows on X_H).
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


class TFMambaStreamFreq(nn.Module):
    """Frequency-Mamba stream: EmbeddingLayer + P stacked pre-norm Mamba blocks.

    Scans along the embedding/feature axis D (Fig. 7c — vertical arrows on X_V).
    After EmbeddingLayer maps (B, L, M) → (B, L, D), transposes to (B, D, L)
    so Mamba scans D=d_model steps, each carrying L features.
    Transposes back to (B, L, D) before returning.

    Mamba d_model = max_len = L (NOT D); LayerNorm = LayerNorm(max_len).

    Shape contract:
        Input  (B, L, M)  →  EmbeddingLayer  →  (B, L, D)
        transpose(1, 2)   →  (B, D, L)
        P × Mamba block   →  (B, D, L)         [Mamba d_model=L, seq_len=D]
        transpose(1, 2)   →  (B, L, D)         ← same shape as TFMambaStream
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
        # After transpose: last dim = max_len (= L); normalise along L.
        self.norms  = nn.ModuleList(
            [nn.LayerNorm(max_len) for _ in range(num_layers)]
        )
        # Mamba scans D=d_model steps; each step has max_len features.
        self.mambas = nn.ModuleList(
            [
                Mamba(d_model=max_len, d_state=d_state,
                      d_conv=d_conv, expand=expand)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, num_features)
        Returns:
            (B, L, d_model)  — same shape contract as TFMambaStream.
        """
        x = self.emb(x)              # (B, L, D)
        x = x.transpose(1, 2)        # (B, D, L) — scan along D
        for norm, mamba in zip(self.norms, self.mambas):
            x = x + mamba(norm(x))   # pre-norm residual, Eq. 14
        x = x.transpose(1, 2)        # (B, L, D)
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

    Both streams receive inputs of identical shape (B, L, M) and the full
    paper-faithful sequence-level flow is used throughout.

    Args
    ────
    num_features : feature dim M shared by both XH and XV inputs
                   (= N/2 = 135 for HUST-HAR after DWT + transpose)
    d_model      : hidden dimension D = D′ = 64                  (default 64)
                   stream_T Mamba d_model = D; stream_F Mamba d_model = max_len.
    num_layers   : Mamba layers per stream P                      (default 3)
    num_classes  : number of activity classes c                   (default 6)
    d_state      : Mamba SSM state dimension                      (default 16)
    d_conv       : Mamba depthwise-conv kernel size               (default 4)
    expand       : Mamba inner-dimension expansion factor         (default 2)
    max_len      : positional-encoding max sequence length L      (default 500)
                   Also serves as Mamba d_model for stream_F (freq-domain scan).
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

        # Time-Mamba stream — scans DWT horizontal-detail subband XH
        self.stream_T = TFMambaStream(
            num_features=num_features,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            max_len=max_len,
        )

        # Freq-Mamba stream — scans along D (embedding dim) after transpose [Fix 4]
        self.stream_F = TFMambaStreamFreq(
            num_features=num_features,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            max_len=max_len,
        )

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
        # L_T == L_F == L  →  weights α_T, α_F computed from full (B,L,D) context
        S2 = self.fusion(S_T, S_F)           # (B, L, d_model)

        # ── Step 3: linear projection + tanh ─────────────────────────────────
        S3 = torch.tanh(self.proj_s3(S2))    # (B, L, d_model)

        # ── Step 4: Global Average Pooling ────────────────────────────────────
        # Reduces (B, L, D) → (B, D) before classifier.
        # Placed AFTER proj_s3 as in Fig. 5 (S3 → Classifier → Ŝ).
        S3 = S3.mean(dim=1)                  # (B, d_model)

        # ── Step 5: classifier ────────────────────────────────────────────────
        return self.classifier(S3)           # (B, num_classes)