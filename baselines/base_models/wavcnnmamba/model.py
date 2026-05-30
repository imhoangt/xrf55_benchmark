"""
wavmamba_har_v4.py — WavMambaHAR-v4
====================================

WiFi-CSI human activity recognition on XRF55 (11 classes). A compact
CNN + bidirectional-Mamba model (~0.75M params) operating on the amplitude
of a 1-level 2-D DWT (subbands LL | HL | LH; HH dropped).

v4 incorporates four targeted upgrades over v3 (all backward-compatible
at initialization step):

  1. TFBlock — LayerScale-style channel-wise gating on the parallel axial
     depthwise branches (γ_t, γ_f, init=1.0). Lets each channel learn
     whether to weight temporal or frequency mixing more, before the
     pointwise channel projection.

  2. FreqAttnPool — added output projection Linear(d, d) with identity
     init. Turns the multi-head pool from "4 disjoint single-head pools"
     into a real multi-head module where heads can communicate.

  3. WavMambaHAR — learned 1-D absolute positional embedding injected
     after FreqAttnPool, before BiMamba. Addresses the symmetry issue
     of bidirectional Mamba + sum-fusion where absolute position info
     can be washed out at center positions.

  4. Classifier — default dropout_in changed from 0.1 → 0.0. LayerNorm
     immediately before the first Linear already provides standardization;
     a hidden Dropout(0.2) between the two Linears still regularizes.

Architecture
------------
    (B, 27, T, F2)                    27 = 9 ant-pairs × 3 subbands (LL|HL|LH)
        ▼ Encoder:
          per-subband stems  LL(7,5)/HL(3,7)/LH(7,3)  → (B, 96, T, F2)
          1×1 proj + GN      96 → d_model             → (B, 128, T, F2)
          TFBlock ×3         γ_t/γ_f + dil[1,2,4] + SE → (B, 128, T, F2)  ★v4
        ▼ FreqAttnPool       4-head + out_proj         → (B, T, 128)      ★v4
        ▼ + PositionalEmb    learned, (1, 500, 128)    → (B, T, 128)      ★v4
        ▼ BiMamba            2 layers, sum, no FFN      → (B, T, 128)
        ▼ AttnStatPool       ECAPA mean+std             → (B, 256)
        ▼ Classifier         LN → MLP (no input drop)   → (B, 11)          ★v4

Input : X (B, 27, T, F2). Channel order [LL ant1-9 | HL ant1-9 | LH ant1-9].
        T = 500 (after DWT on T_raw=1000); F2 = 15 or 32 (F_raw=30 or 64).
Output: logits (B, num_classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    _MAMBA_IMPORT_ERROR = None
except ImportError as e:                          # pragma: no cover
    HAS_MAMBA = False
    _MAMBA_IMPORT_ERROR = e


# ─── Stochastic depth ─────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Drop entire samples from the residual branch during training."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


# ─── Channel attention (Squeeze-and-Excite) ───────────────────────────────────

class ChannelGate(nn.Module):
    """Squeeze-and-Excite channel recalibration for 2-D feature maps.

    Squeeze : global average pool over (T, F) → one statistic per channel.
    Excite  : Linear(d, d/r) → SiLU → Linear(d/r, d) → Sigmoid → gates ∈ (0,1).
    Scale   : x ← x ⊙ gates  (broadcast over T, F).

    Used INSIDE each TFBlock (SE-ResNet pattern). After the encoder's 1×1
    projection the channels are no longer subband-aligned, so this performs
    generic per-channel recalibration — hence "ChannelGate", not "SubBandGate".

    Init note: with trunc_normal(std=0.02) weights and zero bias the excite
    MLP outputs ≈ 0 → Sigmoid(0) = 0.5, so gates start near 0.5 where the
    sigmoid gradient is maximal (0.25), letting them move quickly to the
    correct per-channel regime. Because the gate sits on the residual *delta*
    (not the residual stream), the early 0.5 scaling only attenuates the
    block's contribution, never the preserved stem/identity features.

    Parameters (d=128, r=4): Linear(128,32)=4,128 + Linear(32,128)=4,224 = 8,352
    """

    def __init__(self, d: int = 128, reduction: int = 4):
        super().__init__()
        hidden = max(8, d // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Flatten(1),                       # (B, d, 1, 1) → (B, d)
            nn.Linear(d, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d),
            nn.Sigmoid(),                        # independent gates ∈ (0, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, d, T, F2)
        w = self.excite(self.squeeze(x))                   # (B, d)
        return x * w.unsqueeze(-1).unsqueeze(-1)           # (B, d, T, F2)


# ─── M1: Encoder ──────────────────────────────────────────────────────────────

class TFBlock(nn.Module):
    """Pre-norm axial-depthwise block for 2-D (time × freq) feature maps,
    with a LayerScale-style channel-wise gate on the two axial branches
    AND an SE ChannelGate on the residual branch (SE-ResNet pattern).

    Flow (Pre-GroupNorm, single residual sub-block):
        y = GroupNorm(x)
        y = γ_t ⊙ dw_t(y) + γ_f ⊙ dw_f(y)     # ★v4: gated axial mixing
        y = pw(SiLU(y))                        # pointwise channel projection
        y = ChannelGate(y)                     # SE channel recalibration
        out = x + DropPath(y)                  # residual

    v4 ★ — Channel-wise LayerScale gates (γ_t, γ_f, shape (1, d, 1, 1)):
        With init = 1.0 the block is bit-exact equivalent to v3 at step 0,
        but the optimizer can subsequently shift channels toward time-axial
        or frequency-axial preference. SE later in the block reweights the
        entire residual delta on a per-channel basis; γ_t, γ_f instead
        decide PER CHANNEL whether the T-axial or F-axial information is
        more useful BEFORE the two are combined and passed to pw. This
        decouples a knob (pre-mix balance) that pw alone cannot easily
        learn because pw sees an already-merged signal.

    Parallel depthwise convolutions along time (dw_t, dilated) and frequency
    (dw_f) are gated and summed, then mixed by a 1×1 pointwise conv. The SE
    gate then reweights channels of the block's contribution before residual.

    Dilation schedule [1, 2, 4] across the three blocks gives compound
    temporal receptive fields of 7 → 19 → 43 timesteps — local patterns
    suited for T=500 before BiMamba handles long-range dependencies.
    """

    def __init__(self, d: int = 128, k_t: int = 7, k_f: int = 3,
                 dilation: int = 1, drop_path: float = 0.0,
                 se_reduction: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, d)
        self.dw_t = nn.Conv2d(d, d, (k_t, 1),
                              padding=(k_t // 2 * dilation, 0),
                              dilation=(dilation, 1), groups=d)
        self.dw_f = nn.Conv2d(d, d, (1, k_f), padding=(0, k_f // 2), groups=d)

        # ── v4: LayerScale-style channel-wise gating on axial branches ──
        # Init = 1.0 → equivalent to v3 at step 0 (backward-compat).
        # 2 × d params per block = 256 per block × 3 blocks = 768 total.
        self.gamma_t = nn.Parameter(torch.ones(1, d, 1, 1))
        self.gamma_f = nn.Parameter(torch.ones(1, d, 1, 1))
        # ────────────────────────────────────────────────────────────────

        self.act  = nn.SiLU()
        self.pw   = nn.Conv2d(d, d, 1)
        self.gate = ChannelGate(d, reduction=se_reduction)   # SE on the delta
        self.dp   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        # ★v4 — gated parallel axial mixing (γ broadcast over T, F2)
        y = self.gamma_t * self.dw_t(y) + self.gamma_f * self.dw_f(y)
        y = self.pw(self.act(y))            # channel projection
        y = self.gate(y)                    # SE channel recalibration
        return x + self.dp(y)


class Encoder(nn.Module):
    """Per-subband stems → 1×1 cross-subband proj → TFBlocks (with SE + LS).

    Channel layout (MUST match data pipeline):
        x[:, 0:9,   :, :]  = LL subband (9 antenna pairs)
        x[:, 9:18,  :, :]  = HL subband
        x[:, 18:27, :, :]  = LH subband

    Stem kernel rationale (physically motivated):
        LL (7,5) — large temporal + medium freq: slow signal envelope
        HL (3,7) — small temporal + large freq:  temporal burst onsets
        LH (7,3) — large temporal + small freq:  Doppler micro-structure

    Flow:
        cat(LL_out, HL_out, LH_out) → (B, 96, T, F2)
            ↓ 1×1 proj + GN     cross-subband channel mixing (96 → d_model)
            ↓ TFBlocks ×3       multi-scale spatio-temporal + LS + SE

    Input : (B, 27, T, F2)
    Output: (B, d_model, T, F2)
    """

    # Specialized kernels per DWT subband: LL | HL | LH   (format (k_t, k_f))
    _STEM_KERNELS = [(7, 5), (3, 7), (7, 3)]

    def __init__(self, d_subband: int = 32, d_model: int = 128,
                 dilations: tuple = (1, 2, 4),
                 drop_path_rates: tuple = (0.0, 0.05, 0.10),
                 se_reduction: int = 4):
        super().__init__()
        if len(dilations) != len(drop_path_rates):
            raise ValueError(
                f"dilations ({len(dilations)}) and drop_path_rates "
                f"({len(drop_path_rates)}) must have the same length"
            )

        n_per_sub = 9                                # 9 antenna-pairs per subband
        d_cat     = 3 * d_subband                    # 96 channels after cat

        self.stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_per_sub, d_subband, (kt, kf),
                          padding=(kt // 2, kf // 2)),
                nn.GroupNorm(8, d_subband),
                nn.SiLU(),
            )
            for kt, kf in self._STEM_KERNELS
        ])

        # Cross-subband channel mixing (96 → d_model)
        self.proj = nn.Sequential(
            nn.Conv2d(d_cat, d_model, 1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            TFBlock(d_model, dilation=dilations[i],
                    drop_path=drop_path_rates[i], se_reduction=se_reduction)
            for i in range(len(dilations))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, hl, lh = x.chunk(3, dim=1)               # split by subband
        f = torch.cat([
            self.stems[0](ll),
            self.stems[1](hl),
            self.stems[2](lh),
        ], dim=1)                                      # (B, 96, T, F2)

        f = self.proj(f)                               # (B, d_model, T, F2)
        for blk in self.blocks:
            f = blk(f)
        return f                                       # (B, d_model, T, F2)


# ─── M2: Multi-head frequency attention pooling ───────────────────────────────

class FreqAttnPool(nn.Module):
    """Multi-head content attention pool over the subcarrier axis F2.

    A single Linear(d_model → n_heads) produces n_heads scores per (T, F2)
    position; each head softmaxes over F2 independently and weights its own
    disjoint channel group of size head_dim = d_model // n_heads.

    v4 ★ — Output projection (Linear(d_model, d_model), init=identity):
        v3 used `out_heads.view(B, T, C)` as the only "concat" step. That
        means the 4 heads operated on disjoint channel groups and their
        outputs NEVER interacted — a degenerate form of MHA. v4 adds a
        d×d output projection (init = eye matrix + zero bias) so:
          • At step 0 the module is bit-exact equivalent to v3 (identity).
          • The optimizer can subsequently learn cross-head mixing if it
            helps; otherwise out_proj stays near identity at no cost.
        For n_heads=1, out_proj is nn.Identity (no extra params).

    Param cost (d_model=128, n_heads=4):
        score    = 128 × 4 + 4   =      516
        out_proj = 128 × 128 + 128 = 16,512
        total                       = 17,028

    Setting n_heads=1 yields a single-head pool whose output goes through
    Identity (no out_proj), useful as a lean baseline for ablation.

    Input  : (B, d_model, T, F2)
    Output : (B, T, d_model)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.score    = nn.Linear(d_model, n_heads)

        # ── v4: output projection so heads can communicate ──────────────
        # Init as identity → at step 0 this layer is a no-op, equivalent
        # to v3's `out_heads.view(B, T, C)`. Optimizer may subsequently
        # rotate / mix channels across heads if it helps the loss.
        if n_heads > 1:
            self.out_proj = nn.Linear(d_model, d_model)
            nn.init.eye_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:
            self.out_proj = nn.Identity()
        # ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F2 = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()              # (B, T, F2, C)

        # Per-head attention weights over F2 (subcarriers)
        weights = self.score(x).softmax(dim=2)               # (B, T, F2, H)
        x_heads = x.view(B, T, F2, self.n_heads, self.head_dim)
        # Weighted sum: (B, T, F2, H, 1) ⊙ (B, T, F2, H, Dh) → (B, T, H, Dh)
        out = (weights.unsqueeze(-1) * x_heads).sum(dim=2).reshape(B, T, C)

        return self.out_proj(out)                            # ★v4


# ─── M3: Bidirectional Mamba (no FFN) ─────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )


class BiMambaLayer(nn.Module):
    """One bidirectional Mamba layer — Vim-style sum fusion, NO FFN.

    Architecture (Pre-RMSNorm, single residual sub-block):
        h    = RMSNorm(x)
        y    = Mamba_fwd(h) + flip(Mamba_bwd(flip(h)))
        out  = x + DropPath(y)

    No FFN: the majority of HAR-Mamba comparables omit it; Mamba's internal
    in_proj → conv → SSM → ×gate → out_proj with expand=2 already provides
    gated channel mixing.

    d_state=32: T=500 is long; extra SSM state improves temporal memory
    without overfitting on 19,800 samples.
    """

    def __init__(self, d_model: int = 128, d_state: int = 32, d_conv: int = 4,
                 expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is required.\n"
                "Install: pip install mamba-ssm[causal-conv1d] --no-build-isolation\n"
                f"Original error: {_MAMBA_IMPORT_ERROR}"
            )
        self.norm = RMSNorm(d_model)
        self.fwd  = Mamba(d_model=d_model, d_state=d_state,
                          d_conv=d_conv, expand=expand)
        self.bwd  = Mamba(d_model=d_model, d_state=d_state,
                          d_conv=d_conv, expand=expand)
        self.dp   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y = self.fwd(h) + self.bwd(h.flip(1)).flip(1)
        return x + self.dp(y)


class BiMamba(nn.Module):
    """Stack of n_layers bidirectional Mamba layers + final RMSNorm."""

    def __init__(self, d_model: int = 128, n_layers: int = 2, d_state: int = 32,
                 drop_path_rates: tuple = (0.0, 0.05)):
        super().__init__()
        if len(drop_path_rates) != n_layers:
            raise ValueError(
                f"len(drop_path_rates)={len(drop_path_rates)} must equal "
                f"n_layers={n_layers}"
            )
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model, d_state=d_state, drop_path=drop_path_rates[i])
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─── M4: Temporal attentive statistics pooling ────────────────────────────────

class AttnStatPool(nn.Module):
    """ECAPA-TDNN style: per-channel temporal attention → [mean ‖ std].

    Each channel independently weights time steps; concatenating mean and std
    doubles the representation to (B, 2·d_model), capturing both the location
    and the spread of the temporal activity distribution.

    Numerically-stable variance: (var + 1e-6).sqrt() rather than
    .clamp(min=1e-12).sqrt() — the latter has a sqrt gradient ≈ 5e5 at the
    clamp floor, the former keeps it ≤ 5e2 everywhere.

    Input : (B, T, d_model)
    Output: (B, 2·d_model)
    """

    def __init__(self, dim: int = 128, bn: int = 32):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, bn), nn.Tanh(), nn.Linear(bn, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w    = self.score(x).softmax(dim=1)                   # (B, T, C)
        mean = (w * x).sum(dim=1)                             # (B, C)
        var  = (w * (x - mean.unsqueeze(1)).pow(2)).sum(dim=1)
        return torch.cat([mean, (var + 1e-6).sqrt()], dim=-1) # (B, 2C)


# ─── M5: Classifier ───────────────────────────────────────────────────────────

class Classifier(nn.Module):
    """MLP head: LayerNorm → [Dropout]? → Linear → SiLU → Dropout → Linear.

    v4 ★ — Default `dropout_in` changed from 0.1 → 0.0:
        LayerNorm immediately before the first Linear already standardizes
        features (zero mean, unit variance per channel), so dropping
        randomly *standardized* values via Dropout(0.1) was redundant with
        the hidden Dropout(0.2) that follows. Pattern aligns with ConvNeXt,
        Vim, MambaVision heads — they only have one dropout, after the
        hidden activation.

        Wrapped behind `if dropout_in > 0:` so the layer is omitted
        entirely when 0, keeping `state_dict` clean. Easy to revert by
        passing `dropout_in=0.1` if overfit is observed empirically.
    """

    def __init__(self, d: int = 256, hidden: int = 128, num_classes: int = 11,
                 dropout: float = 0.2, dropout_in: float = 0.0):   # ★v4
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(d)]
        if dropout_in > 0.0:
            layers.append(nn.Dropout(dropout_in))
        layers.extend([
            nn.Linear(d, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─── Full model ───────────────────────────────────────────────────────────────

class WavMambaHAR(nn.Module):
    """WavMambaHAR-v4 — WiFi-CSI HAR on XRF55.

    Input  : X  (B, 27, T, F2)
                 27 = 9 antenna-pairs × 3 DWT subbands (LL | HL | LH grouped)
                 T  = 500 (after 2-D DWT on T_raw=1000)
                 F2 = 15 or 32 (after 2-D DWT on F_raw=30 or 64)
    Output : logits  (B, num_classes)

    v4 ★ — Learned absolute positional embedding (shape (1, T_MAX, d_model)):
        Bidirectional Mamba with sum-fusion is structurally weak at encoding
        absolute position: every position receives forward-state ("t steps
        processed") + backward-state ("T-t steps processed"), so the
        information sums to a value that depends only on T (a constant
        within a sample). HAR actions with identical motion-shape but
        different absolute timing (e.g. "walk then idle" vs "idle then
        walk") can be confused without an explicit position anchor.

        Init = trunc_normal(std=0.02) → magnitude ≪ feature scale at step
        0, but well-conditioned for gradient flow. Empirical evidence:
        Vim (+spatial PE), VideoMamba (+spatio-temporal PE), and Mamba
        GitHub issue #51 all report +3-4% from adding PE.

        Position is injected ONCE — after FreqAttnPool, before BiMamba.
        Adding it inside the encoder is unnecessary (CNN has implicit
        position info via padding boundaries); inside each Mamba layer
        would be redundant.

        Param cost: T_MAX × d_model = 500 × 128 = 64,000 params (~9.6%
        of total). For T_actual ≤ T_MAX we slice; for T_actual > T_MAX
        we linearly interpolate (a rare edge case for this dataset).

    Args:
        num_classes       : output classes (default 11 for XRF55).
        d_model           : feature width (default 128).
        d_state           : Mamba SSM state size (default 32 for T=500).
        n_mamba_layers    : number of BiMamba layers (default 2).
        n_freq_attn_heads : heads in FreqAttnPool (default 4; 1 ≡ original).
        se_reduction      : SE bottleneck reduction inside TFBlocks (default 4).
        dp_cnn            : DropPath rates for the 3 TFBlocks.
        dp_mamba          : DropPath rates for the BiMamba layers.
        use_pos_emb       : whether to add learned positional embedding (★v4).
    """

    C_IN  = 27    # hard-coded: amplitude only, 9 ant-pairs × 3 subbands
    T_MAX = 500   # ★v4 — buffer length for positional embedding
                  #     (= T_raw 1000 / 2 after 1-level DWT)

    def __init__(
        self,
        num_classes:        int   = 11,
        d_model:            int   = 128,
        d_state:            int   = 32,
        n_mamba_layers:     int   = 2,
        n_freq_attn_heads:  int   = 4,
        se_reduction:       int   = 4,
        dp_cnn:             tuple = (0.0, 0.05, 0.10),
        dp_mamba:           tuple = (0.0, 0.05),
        use_pos_emb:        bool  = True,                       # ★v4
    ):
        super().__init__()
        if len(dp_cnn) != 3:
            raise ValueError(
                f"dp_cnn must have 3 entries (one per TFBlock), got {len(dp_cnn)}"
            )
        if len(dp_mamba) != n_mamba_layers:
            raise ValueError(
                f"len(dp_mamba)={len(dp_mamba)} must equal "
                f"n_mamba_layers={n_mamba_layers}"
            )

        self.encoder = Encoder(
            d_subband=32,
            d_model=d_model,
            drop_path_rates=dp_cnn,
            se_reduction=se_reduction,
        )
        self.fpool = FreqAttnPool(d_model, n_heads=n_freq_attn_heads)

        # ── v4: learned 1-D absolute positional embedding ───────────────
        # Buffer sized T_MAX=500 (= T_raw / 2 after 1-level DWT).
        # For shorter T we slice; for longer T we interpolate (rare).
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.T_MAX, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        else:
            self.register_parameter("pos_emb", None)
        # ────────────────────────────────────────────────────────────────

        self.mamba = BiMamba(d_model, n_layers=n_mamba_layers,
                             d_state=d_state, drop_path_rates=dp_mamba)
        self.tpool = AttnStatPool(d_model, bn=32)
        self.head  = Classifier(2 * d_model, hidden=128,
                                num_classes=num_classes)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned absolute PE to (B, T, d_model). No-op if disabled.
        For T ≤ T_MAX we slice the buffer; for T > T_MAX (rare) we
        linearly interpolate the PE to match.
        """
        if self.pos_emb is None:
            return x
        T_actual = x.size(1)
        if T_actual <= self.T_MAX:
            return x + self.pos_emb[:, :T_actual]
        pe = F.interpolate(
            self.pos_emb.transpose(1, 2),                # (1, d, T_MAX)
            size=T_actual, mode="linear", align_corners=False,
        ).transpose(1, 2)                                 # (1, T_actual, d)
        return x + pe

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, 27, T, F2)
               Channel order: [ant1-9 of LL] | [ant1-9 of HL] | [ant1-9 of LH]
        Returns:
            logits: (B, num_classes)
        """
        if X.ndim != 4:
            raise ValueError(
                f"Expected 4-D input (B, 27, T, F2), got shape {tuple(X.shape)}"
            )
        if X.shape[1] != self.C_IN:
            raise ValueError(
                f"Expected {self.C_IN} channels "
                f"(amp, 9 ant-pairs × 3 subbands LL|HL|LH), got {X.shape[1]}"
            )

        x = self.encoder(X)         # (B, d_model, T, F2)
        x = self.fpool(x)           # (B, T, d_model)
        x = self._apply_pos_emb(x)  # ★v4 — inject absolute position info
        x = self.mamba(x)           # (B, T, d_model)
        z = self.tpool(x)           # (B, 2·d_model)
        return self.head(z)         # (B, num_classes)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    import sys

    def count_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    if not HAS_MAMBA:
        print("mamba_ssm not installed — sanity check skipped.")
        print("Install: pip install mamba-ssm[causal-conv1d] --no-build-isolation")
        sys.exit(0)

    print("WavMambaHAR-v4 parameter breakdown")
    print("=" * 50)
    model = WavMambaHAR()
    total = count_params(model)

    # Child modules
    accounted = 0
    for name, child in model.named_children():
        p = count_params(child)
        accounted += p
        print(f"  {name:10s}: {p:>9,}  ({100 * p / total:5.1f}%)")

    # pos_emb is a Parameter (not a child module) — print separately
    if model.pos_emb is not None:
        pe = model.pos_emb.numel()
        accounted += pe
        print(f"  {'pos_emb':10s}: {pe:>9,}  ({100 * pe / total:5.1f}%)")

    print("-" * 50)
    print(f"  {'sum':10s}: {accounted:>9,}  (check: matches TOTAL? "
          f"{'YES' if accounted == total else 'NO'})")
    print(f"  {'TOTAL':10s}: {total:>9,}")

    # Shape check (requires CUDA)
    print("\nShape check:")
    B, T, F2 = 2, 500, 15
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = torch.randn(B, 27, T, F2, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"  device      : {device}")
    print(f"  input shape : {tuple(x.shape)}")
    print(f"  output shape: {tuple(out.shape)}  (expected: ({B}, 11))")