"""
wavmamba_har.py — WavMambaHAR-v5
================================

WiFi-CSI human activity recognition on XRF55 (11 classes). A compact
CNN + bidirectional-Mamba model (~0.75M params) operating on the amplitude
of a 1-level 2-D DWT (subbands LL | HL | LH; HH dropped).

Changes over v4 (three, all deliberate)
---------------------------------------
  1. TFBlock — REMOVED the per-branch LayerScale gates (gamma_t, gamma_f).
     For a depthwise conv, a per-output-channel static scalar (init=1.0)
     folds into that channel's kernel, so the gates added zero expressivity;
     and the LayerScale benefit requires small init on deep stacks, not
     init=1.0 on a 3-block encoder. The pointwise conv + SE already supply
     all the channel reweighting the block needs.

  2. Configurable input channels — C_IN is derived from
     (n_links M, n_antennas A, n_subbands) as C_IN = M * A * n_subbands
     instead of being hard-coded to 27. Default (3,3,3) -> 27 (XRF55
     scene_01), but A=4 -> 36, or keeping HH (n_subbands=4), now work
     without editing the model. n_per_sub = M * A.

  3. SubbandGate — a per-SAMPLE squeeze-excite gate over the DWT subbands,
     inserted between the per-subband stems and the 1x1 cross-subband
     projection. Because it is conditioned on the input it differs per
     sample, so it can boost/suppress a whole subband (e.g. HL for
     impulsive actions, LL for slow ones) — something the static proj
     (identical weights for all samples) and the per-channel SE inside
     TFBlocks (which runs after mixing) cannot express. ~1.2k params;
     toggle with `use_subband_gate`.

Unchanged from v4 (all sound): FreqAttnPool multi-head + identity-init
out_proj; learned absolute positional embedding (kept behind `use_pos_emb`,
default on — worth ablating); Vim-style bidirectional Mamba (sum fusion,
no FFN, d_state=32); ECAPA attentive mean+std pooling; LN->MLP head with
no input dropout; dilation schedule [1,2,4]; GroupNorm throughout.

Architecture (default config: M=3, A=3, 3 subbands -> C_IN=27)
-------------------------------------------------------------
    (B, 27, T, F2)                    27 = 9 ant-pairs × 3 subbands (LL|HL|LH)
        ▼ Encoder:
          per-subband stems  LL(7,5)/HL(3,7)/LH(7,3)  → (B, 96, T, F2)
          SubbandGate        per-sample LL/HL/LH gate → (B, 96, T, F2)  ★v5
          1×1 proj + GN      96 → d_model             → (B, 128, T, F2)
          TFBlock ×3         dil[1,2,4] + SE          → (B, 128, T, F2)  ★v5
        ▼ FreqAttnPool       4-head + out_proj         → (B, T, 128)
        ▼ + PositionalEmb    learned, (1, 500, 128)    → (B, T, 128)
        ▼ BiMamba            2 layers, sum, no FFN      → (B, T, 128)
        ▼ AttnStatPool       ECAPA mean+std             → (B, 256)
        ▼ Classifier         LN → MLP (no input drop)   → (B, 11)

Input : X (B, C_IN, T, F2). Subband-major channel order:
        [ (M·A) chans LL | (M·A) chans HL | (M·A) chans LH ].
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


# ─── Cross-subband reweighting gate ───────────────────────────────────────────

class SubbandGate(nn.Module):
    """Per-sample Squeeze-and-Excite gate over the DWT subbands.

    Applied to the concatenated stem outputs (B, n_subbands*d_sub, T, F2)
    BEFORE the 1×1 cross-subband projection, so each scalar gate maps to
    exactly one subband (LL / HL / LH):

        squeeze : global avg pool over (T, F2)  → (B, n_subbands*d_sub)
        excite  : Linear → SiLU → Linear(→ n_subbands) → Sigmoid → gates
        scale   : multiply each subband's d_sub-channel block by its gate.

    Why it is NOT redundant with the downstream projection or the SE inside
    TFBlock: the gate is conditioned on the input, so it differs per sample
    and can boost/suppress a whole subband per sample (e.g. HL for impulsive
    actions, LL for slow ones). The 1×1 proj applies identical weights to
    every sample, and the per-channel SE in TFBlock runs AFTER subbands are
    mixed (no subband boundary left), so neither can express this.

    Init: the final Linear's bias is zeroed → gates start at Sigmoid(0)=0.5
    (a uniform 0.5 scaling, not a literal no-op; the following projection
    absorbs the constant factor during training).

    Params (n_subbands=3, d_sub=32): ≈ 1.2 K.

    Input  : (B, n_subbands*d_sub, T, F2)
    Output : same shape, subband blocks rescaled.
    """

    def __init__(self, n_subbands: int = 3, d_sub: int = 32):
        super().__init__()
        self.n_subbands = n_subbands
        self.d_sub      = d_sub
        hidden = max(n_subbands, (n_subbands * d_sub) // 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1     = nn.Linear(n_subbands * d_sub, hidden)
        self.act     = nn.SiLU()
        self.fc2     = nn.Linear(hidden, n_subbands)
        nn.init.zeros_(self.fc2.bias)                # gates ≈ 0.5 at init
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, n*d_sub, T, F2)
        B, C, T, F2 = x.shape
        s = self.squeeze(x).flatten(1)                   # (B, n*d_sub)
        g = self.sigmoid(self.fc2(self.act(self.fc1(s)))) # (B, n_subbands)
        x = x.view(B, self.n_subbands, self.d_sub, T, F2)
        x = x * g.view(B, self.n_subbands, 1, 1, 1)      # per-subband scaling
        return x.reshape(B, C, T, F2)


# ─── M1: Encoder ──────────────────────────────────────────────────────────────

class TFBlock(nn.Module):
    """Pre-norm axial-depthwise block for 2-D (time × freq) feature maps,
    with an SE ChannelGate on the residual branch (SE-ResNet pattern).

    Flow (Pre-GroupNorm, single residual sub-block):
        y = GroupNorm(x)
        y = dw_t(y) + dw_f(y)                  # parallel axial depthwise mixing
        y = pw(SiLU(y))                        # pointwise channel projection
        y = ChannelGate(y)                     # SE channel recalibration
        out = x + DropPath(y)                  # residual

    Parallel depthwise convolutions along time (dw_t, dilated) and frequency
    (dw_f) are summed, then mixed by a 1×1 pointwise conv. The SE gate then
    reweights channels of the block's contribution before the residual add.

    Note: v4 placed learnable LayerScale gates (γ_t, γ_f, init=1.0) on the two
    axial branches. They were removed — for a depthwise conv a per-output-
    channel scalar folds into that channel's kernel, so at init=1.0 they add
    no expressivity (only a no-op reparameterisation), and LayerScale's actual
    benefit needs small init on deep stacks, not a 3-block encoder. The pw conv
    and the SE gate already provide the channel reweighting the block needs.

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
        self.act  = nn.SiLU()
        self.pw   = nn.Conv2d(d, d, 1)
        self.gate = ChannelGate(d, reduction=se_reduction)   # SE on the delta
        self.dp   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.dw_t(y) + self.dw_f(y)     # parallel axial mixing
        y = self.pw(self.act(y))            # channel projection
        y = self.gate(y)                    # SE channel recalibration
        return x + self.dp(y)


class Encoder(nn.Module):
    """Per-subband stems → SubbandGate → 1×1 cross-subband proj → TFBlocks.

    Channel layout is subband-major (MUST match data pipeline); with
    n_per_sub = M·A antenna pairs per subband, for the default M=3, A=3,
    3 subbands:
        x[:, 0:9,   :, :]  = LL subband (9 antenna pairs)
        x[:, 9:18,  :, :]  = HL subband
        x[:, 18:27, :, :]  = LH subband

    Stem kernels (canonical 3-subband case, physically motivated):
        LL (7,5) — large temporal + medium freq: slow signal envelope
        HL (3,7) — small temporal + large freq:  temporal burst onsets
        LH (7,3) — large temporal + small freq:  Doppler micro-structure
    For any other subband count a uniform (5,5) kernel is used unless
    `stem_kernels` is given explicitly.

    Flow:
        cat(stem_i(subband_i))  → (B, n_subbands*d_subband, T, F2)
            ↓ SubbandGate       per-sample subband reweighting
            ↓ 1×1 proj + GN     cross-subband channel mixing (→ d_model)
            ↓ TFBlocks ×3       multi-scale spatio-temporal + SE

    Input : (B, n_per_sub*n_subbands, T, F2)
    Output: (B, d_model, T, F2)
    """

    # Canonical kernels for the 3-subband case: LL | HL | LH   ((k_t, k_f))
    _CANONICAL_3 = [(7, 5), (3, 7), (7, 3)]

    def __init__(self, n_per_sub: int = 9, n_subbands: int = 3,
                 d_subband: int = 32, d_model: int = 128,
                 dilations: tuple = (1, 2, 4),
                 drop_path_rates: tuple = (0.0, 0.05, 0.10),
                 se_reduction: int = 4,
                 use_subband_gate: bool = True,
                 stem_kernels: tuple = None):
        super().__init__()
        if len(dilations) != len(drop_path_rates):
            raise ValueError(
                f"dilations ({len(dilations)}) and drop_path_rates "
                f"({len(drop_path_rates)}) must have the same length"
            )

        if stem_kernels is None:
            stem_kernels = (tuple(self._CANONICAL_3) if n_subbands == 3
                            else ((5, 5),) * n_subbands)
        if len(stem_kernels) != n_subbands:
            raise ValueError(
                f"stem_kernels has {len(stem_kernels)} entries but "
                f"n_subbands={n_subbands}"
            )

        self.n_subbands = n_subbands
        self.n_per_sub  = n_per_sub
        d_cat = n_subbands * d_subband               # channels after cat

        self.stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_per_sub, d_subband, (kt, kf),
                          padding=(kt // 2, kf // 2)),
                nn.GroupNorm(8, d_subband),
                nn.SiLU(),
            )
            for (kt, kf) in stem_kernels
        ])

        # Per-sample cross-subband reweighting before channel mixing
        self.subband_gate = (SubbandGate(n_subbands, d_subband)
                             if use_subband_gate else nn.Identity())

        # Cross-subband channel mixing (n_subbands*d_subband → d_model)
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
        subbands = x.chunk(self.n_subbands, dim=1)   # subband-major split
        f = torch.cat([stem(sb) for stem, sb in zip(self.stems, subbands)],
                      dim=1)                          # (B, n_sub*d_sub, T, F2)
        f = self.subband_gate(f)                      # per-sample subband gates
        f = self.proj(f)                              # (B, d_model, T, F2)
        for blk in self.blocks:
            f = blk(f)
        return f                                      # (B, d_model, T, F2)


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
    """WavMambaHAR-v5 — WiFi-CSI HAR on XRF55.

    Input  : X  (B, C_IN, T, F2),  C_IN = n_links · n_antennas · n_subbands
                 (default 3·3·3 = 27). Subband-major channel order:
                 [ (M·A) chans LL | (M·A) chans HL | (M·A) chans LH ].
                 T  = 500 (after 2-D DWT on T_raw=1000)
                 F2 = 15 or 32 (after 2-D DWT on F_raw=30 or 64)
    Output : logits  (B, num_classes)

    Learned absolute positional embedding (shape (1, T_MAX, d_model)):
        A learned 1-D PE is added once, after FreqAttnPool and before BiMamba.
        Whether an SSM needs a PE is contested (Vim adds one; Vim-F argues a
        recurrent SSM fed in fixed order does not need it), and this model
        already has a convolutional encoder that leaks some absolute position.
        It is kept behind `use_pos_emb` (default True) and is worth ABLATING —
        it costs T_MAX·d_model = 500·128 = 64,000 params (~8.5% of the model).
        For T ≤ T_MAX we slice the buffer; for T > T_MAX we interpolate.

    Args:
        num_classes       : output classes (default 11 for XRF55).
        n_links           : number of Wi-Fi receivers/links M (default 3).
        n_antennas        : antennas per receiver A (default 3).
        n_subbands        : DWT subbands kept (default 3 = LL|HL|LH).
        d_model           : feature width (default 128).
        d_subband         : per-subband stem width (default 32).
        d_state           : Mamba SSM state size (default 32 for T=500).
        n_mamba_layers    : number of BiMamba layers (default 2).
        n_freq_attn_heads : heads in FreqAttnPool (default 4; 1 = single-head).
        se_reduction      : SE bottleneck reduction inside TFBlocks (default 4).
        dp_cnn            : DropPath rates for the 3 TFBlocks.
        dp_mamba          : DropPath rates for the BiMamba layers.
        use_pos_emb       : add learned positional embedding (default True).
        use_subband_gate  : per-sample DWT-subband SE gate (default True).
        stem_kernels      : optional per-subband (k_t, k_f) tuples.
    """

    T_MAX = 500   # buffer length for positional embedding
                  #     (= T_raw 1000 / 2 after 1-level DWT)

    def __init__(
        self,
        num_classes:        int   = 11,
        n_links:            int   = 3,
        n_antennas:         int   = 3,
        n_subbands:         int   = 3,
        d_model:            int   = 128,
        d_subband:          int   = 32,
        d_state:            int   = 32,
        n_mamba_layers:     int   = 2,
        n_freq_attn_heads:  int   = 4,
        se_reduction:       int   = 4,
        dp_cnn:             tuple = (0.0, 0.05, 0.10),
        dp_mamba:           tuple = (0.0, 0.05),
        use_pos_emb:        bool  = True,
        use_subband_gate:   bool  = True,
        stem_kernels:       tuple = None,
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

        # Input channels derived from the sensing layout (subband-major):
        # C_IN = (links · antennas) per subband × number of subbands.
        self.n_per_sub  = n_links * n_antennas
        self.n_subbands = n_subbands
        self._c_in      = self.n_per_sub * n_subbands

        self.encoder = Encoder(
            n_per_sub=self.n_per_sub,
            n_subbands=n_subbands,
            d_subband=d_subband,
            d_model=d_model,
            drop_path_rates=dp_cnn,
            se_reduction=se_reduction,
            use_subband_gate=use_subband_gate,
            stem_kernels=stem_kernels,
        )
        self.fpool = FreqAttnPool(d_model, n_heads=n_freq_attn_heads)

        # ── Learned 1-D absolute positional embedding (optional) ────────
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

    # ── Public property ───────────────────────────────────────────────────────

    @property
    def C_IN(self) -> int:
        """Expected number of input channels (= n_links·n_antennas·n_subbands)."""
        return self._c_in

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
            X: (B, C_IN, T, F2), subband-major channel order
               [ (M·A) LL | (M·A) HL | (M·A) LH ].
        Returns:
            logits: (B, num_classes)
        """
        if X.ndim != 4:
            raise ValueError(
                f"Expected 4-D input (B, {self._c_in}, T, F2), "
                f"got shape {tuple(X.shape)}"
            )
        if X.shape[1] != self._c_in:
            raise ValueError(
                f"Expected {self._c_in} channels "
                f"(= n_links·n_antennas·n_subbands), got {X.shape[1]}"
            )

        x = self.encoder(X)         # (B, d_model, T, F2)
        x = self.fpool(x)           # (B, T, d_model)
        x = self._apply_pos_emb(x)  # inject absolute position info (optional)
        x = self.mamba(x)           # (B, T, d_model)
        z = self.tpool(x)           # (B, 2·d_model)
        return self.head(z)         # (B, num_classes)