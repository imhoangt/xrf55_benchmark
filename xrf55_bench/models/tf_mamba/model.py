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

Ablation flags (TF-Mamba → WavDualMamba; defaults = TF-Mamba baseline)
─────────────────────────────────────────────────────────────────────────────
    use_cnn=True      EmbeddingLayer → WavDualMamba CNN front-end: per-subband
                      SubbandStem (HL (3,7) / LH (7,3)) + dilated TFBlocks. PE kept.
    mamba='bi'        uni-Mamba×num_layers (d_state=16) → WavDualMamba gated
                      BiMamba stack as one unit (×2 layers, d_state=32, final
                      RMSNorm) — swaps direction, depth AND d_state together.
    pool='attnstat'   GAP → AttnStatPool (attentive mean+std; classifier input
                      doubles to 2·d_model).
    use_proj_s3=False Skip the paper-faithful proj_s3 + tanh step (S2 → S3).
                      Default True keeps original TFMamba behaviour. Set False
                      to test AttnStatPool without tanh clamping to [-1,1] —
                      tanh collapses temporal variance and disables AttnStatPool's
                      σ component (see analysis_s3_vs_s4.md).
    use_pos_emb=False Drop the sinusoidal positional embedding (both streams,
                      CNN and non-CNN paths). Default True = original behaviour.
                      Isolates the effect of PE (e.g. rung S2.npe); Mamba already
                      encodes order via recurrence so PE is usually redundant.

Each flag swaps ONE TF-Mamba block for its WavDualMamba counterpart. Blocks are
IMPORTED from wavdualmamba.model (not copied), so they are byte-identical, and
the flags compose freely. The canonical ladder — rung order, which flags are
cumulative, and the higher rungs that move to the full WavDualMamba class on
Haar data (via PreprocTFMambaHaarAsWavDataset, rung S4) — is defined in
notebooks/ablation_ladder.ipynb, the single source of truth for rung numbering.
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

from xrf55_bench.models.wavdualmamba.model import (
    AttnStatPool, BiMamba, SubbandStem, TFBlock, _SUBBAND_KERNEL,
)


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

    def __init__(self, num_features: int, embed_dim: int, max_len: int = 500,
                 use_pos_emb: bool = True):
        super().__init__()
        self.pos = PositionalEmbedding(embed_dim, max_len) if use_pos_emb else None
        self.fc  = nn.Linear(num_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, num_features)  →  (B, L, embed_dim)."""
        h = torch.relu(self.fc(x))
        return h + self.pos(x) if self.pos is not None else h    # PE optional (use_pos_emb)


class CNNFrontEnd(nn.Module):
    """WavDualMamba CNN front-end grafted onto a TF-Mamba stream (use_cnn=True).

    Replaces EmbeddingLayer's Linear(M → d_model). The flat per-step feature
    vector M = n_links · f2 (link-major: XRF55 has 135 = 9 links × 15 Haar
    bins, and the Haar pairs never straddle link boundaries) is unflattened to
    per-link 2-D maps, processed by the exact WavDualMamba blocks, then
    flattened back and projected — mirroring BranchBackbone:

        (B, L, M) → (B, n_links, L, f2)
                  → SubbandStem(n_links → d_stem)
                  → TFBlock × len(dilations)  (dilated, DropPath)
                  → flatten (B, L, d_stem·f2)
                  → Linear → SiLU → Dropout   (= BranchBackbone.embed)
                  → (B, L, d_model)

    Stem kernel is configurable via `kernel` (default (5,5)). TFMamba passes
    WavDualMamba's exact per-subband kernels — HL (3,7) to stream_T, LH (7,3) to
    stream_F — so this front-end is byte-identical to the WavDualMamba branch it
    mirrors (subband_kernels=True; set False for a neutral (5,5) on both streams).
    """

    def __init__(self, num_features: int, d_model: int, n_links: int = 9,
                 d_stem: int = 16, dilations: tuple = (1, 2, 4),
                 dp_cnn: tuple = (0.0, 0.05, 0.1), embed_drop: float = 0.1,
                 kernel: tuple = (5, 5)):
        super().__init__()
        if num_features % n_links != 0:
            raise ValueError(
                f"num_features={num_features} not divisible by n_links={n_links}")
        if len(dp_cnn) != len(dilations):
            raise ValueError(
                f"len(dp_cnn)={len(dp_cnn)} must equal len(dilations)={len(dilations)}")
        self.n_links = n_links
        self.f2      = num_features // n_links
        self.stem    = SubbandStem(n_links, d_stem, kernel=kernel)
        self.blocks  = nn.ModuleList([
            TFBlock(d_stem, dilation=dilations[i], drop_path=dp_cnn[i])
            for i in range(len(dilations))
        ])
        self.embed   = nn.Sequential(
            nn.Linear(d_stem * self.f2, d_model),
            nn.SiLU(),
            nn.Dropout(embed_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, num_features)  →  (B, L, d_model)."""
        B, L, M = x.shape
        x = x.reshape(B, L, self.n_links, self.f2).permute(0, 2, 1, 3)  # (B,n,L,f2)
        x = self.stem(x)                                  # (B, d_stem, L, f2)
        for blk in self.blocks:
            x = blk(x)                                    # (B, d_stem, L, f2)
        B, C, T, Fd = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fd)   # (B, L, d_stem·f2)
        return self.embed(x)                              # (B, L, d_model)


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

    Ablation flags (defaults = TF-Mamba baseline):
        use_cnn=True  EmbeddingLayer → CNNFrontEnd + PE (PE kept so this flag
                      changes only the embedding; PE removal is part of the
                      full-WavDualMamba rung).
        mamba='bi'    uni-Mamba×num_layers → WavDualMamba BiMamba stack
                      (bi_layers × gated BiMambaLayer + final RMSNorm; swaps
                      the whole Mamba block as one unit: direction 1→2,
                      depth num_layers→bi_layers, d_state→bi_d_state).
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
        use_cnn:      bool = False,
        mamba:        str = 'uni',
        n_links:      int = 9,
        d_stem:       int = 16,
        dilations:    tuple = (1, 2, 4),
        dp_cnn:       tuple = (0.0, 0.05, 0.1),
        embed_drop:   float = 0.1,
        stem_kernel:  tuple = (5, 5),
        bi_layers:    int = 2,
        bi_d_state:   int = 32,
        dp_bimamba:   tuple = (0.0, 0.10),
        use_pos_emb:  bool = True,
    ):
        super().__init__()
        if mamba not in ('uni', 'bi'):
            raise ValueError(f"mamba must be 'uni' or 'bi', got {mamba!r}")

        if use_cnn:
            self.frontend = CNNFrontEnd(
                num_features, d_model, n_links=n_links, d_stem=d_stem,
                dilations=dilations, dp_cnn=dp_cnn, embed_drop=embed_drop,
                kernel=stem_kernel)
            self.pos = PositionalEmbedding(d_model, max_len) if use_pos_emb else None
            self.emb = None
        else:
            self.frontend = None
            self.pos      = None
            self.emb      = EmbeddingLayer(num_features, d_model, max_len,
                                           use_pos_emb=use_pos_emb)

        if mamba == 'bi':
            self.bimamba = BiMamba(
                d_model, n_layers=bi_layers, d_state=bi_d_state,
                d_conv=d_conv, expand=expand,
                drop_path_rates=dp_bimamba, bidirectional=True)
            self.norms  = None
            self.mambas = None
        else:
            self.bimamba = None
            self.norms   = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(num_layers)]
            )
            self.mambas  = nn.ModuleList(
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
        if self.frontend is not None:
            h = self.frontend(x)
            x = h + self.pos(h) if self.pos is not None else h    # PE optional (use_pos_emb)
        else:
            x = self.emb(x)
        if self.bimamba is not None:
            return self.bimamba(x)
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

    Ablation flags (all defaults = TF-Mamba baseline)
    ─────────────────────────────────────────────────────────
    pool         : 'gap' (baseline) | 'attnstat'
                   'attnstat' replaces GAP with WavDualMamba's AttnStatPool
                   (context=True); classifier input becomes 2·d_model.
    use_proj_s3  : True (default, paper-faithful) | False
                   False bypasses the Linear+tanh projection before pooling.
                   Required for a fair AttnStatPool test (rung S1.2b).
    use_cnn      : False (baseline) | True
                   Per-stream WavDualMamba CNN front-end replaces the Linear
                   embedding (PE kept). See CNNFrontEnd.
    subband_kernels : True (default) | False
                   With use_cnn, give each stream WavDualMamba's exact physical
                   stem kernel — stream_T=HL (3,7), stream_F=LH (7,3) — so the CNN
                   front-end is byte-identical to WavDualMamba. False → neutral
                   (5,5). Assumes the loader feeds HL content to stream_T
                   (PreprocTFMambaDataset canonicalises this via the marker).
    mamba        : 'uni' (baseline) | 'bi'
                   Swaps the whole Mamba stack for WavDualMamba's gated
                   BiMamba (bi_layers=2, bi_d_state=32, final RMSNorm).
    n_links, d_stem, dilations, dp_cnn, embed_drop   : CNNFrontEnd knobs
    bi_layers, bi_d_state, dp_bimamba                : BiMamba knobs
    Flags compose freely — see notebooks/ablation_ladder.ipynb for the
    canonical rung order and numbering (the single source of truth).
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
        pool:         str = 'gap',
        use_proj_s3:  bool = True,
        use_pos_emb:  bool = True,
        use_cnn:      bool = False,
        subband_kernels: bool = True,
        mamba:        str = 'uni',
        n_links:      int = 9,
        d_stem:       int = 16,
        dilations:    tuple = (1, 2, 4),
        dp_cnn:       tuple = (0.0, 0.05, 0.1),
        embed_drop:   float = 0.1,
        bi_layers:    int = 2,
        bi_d_state:   int = 32,
        dp_bimamba:   tuple = (0.0, 0.10),
    ):
        super().__init__()
        if pool not in ('gap', 'attnstat'):
            raise ValueError(f"pool must be 'gap' or 'attnstat', got {pool!r}")

        shared_kwargs = dict(
            num_features=num_features,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            max_len=max_len,
            use_cnn=use_cnn,
            mamba=mamba,
            n_links=n_links,
            d_stem=d_stem,
            dilations=dilations,
            dp_cnn=dp_cnn,
            embed_drop=embed_drop,
            bi_layers=bi_layers,
            bi_d_state=bi_d_state,
            dp_bimamba=dp_bimamba,
            use_pos_emb=use_pos_emb,
        )

        # Per-stream stem kernel: when subband_kernels, hand each stream the
        # EXACT WavDualMamba physical kernel (stream_T=XH=HL, stream_F=XV=LH).
        # The loader feeds HL content to stream_T canonically (see
        # PreprocTFMambaDataset), so kernel matches subband content and the
        # S1–S3 CNN front-end is byte-identical to WavDualMamba's. Only takes
        # effect with use_cnn=True; False → neutral (5,5) on both streams.
        kT, kF = ((_SUBBAND_KERNEL['HL'], _SUBBAND_KERNEL['LH'])
                  if subband_kernels else ((5, 5), (5, 5)))

        # Time-Mamba stream — XH = HL (horizontal-detail DWT subband)
        self.stream_T = TFMambaStream(**shared_kwargs, stem_kernel=kT)

        # Freq-Mamba stream — XV = LH (vertical-detail DWT subband)
        self.stream_F = TFMambaStream(**shared_kwargs, stem_kernel=kF)

        # Adaptive fusion  [Eq. 15]
        self.fusion = AdaptiveFusion(d_model)

        # proj_s3: S2 → S3 with tanh (D′ = D = 64, confirmed by Table I).
        # use_proj_s3=False bypasses this for fair AttnStatPool ablations.
        self.proj_s3 = nn.Linear(d_model, d_model) if use_proj_s3 else None

        # Temporal pooling: GAP (baseline) or AttnStatPool (pool='attnstat')
        if pool == 'attnstat':
            self.tpool = AttnStatPool(d_model, context=True)
            head_in    = 2 * d_model
        else:
            self.tpool = None
            head_in    = d_model

        # Classifier
        self.classifier = nn.Linear(head_in, num_classes)

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

        # ── Step 3: proj_s3 + tanh (paper-faithful; skipped when use_proj_s3=False) ──
        h = torch.tanh(self.proj_s3(S2)) if self.proj_s3 is not None else S2

        # ── Step 4: temporal pooling — GAP (baseline) or AttnStatPool ─────────
        h = self.tpool(h) if self.tpool is not None else h.mean(dim=1)

        # ── Step 5: classifier ────────────────────────────────────────────────
        return self.classifier(h)            # (B, num_classes)
