"""
wavdualmamba.py — WavDualMamba
==============================

WiFi-CSI human activity recognition on XRF55 (default 11 classes). A multi-branch
(one branch per DWT subband) CNN + bidirectional-Mamba model with adaptive
late fusion. ~0.5M params in the default 3-branch config.

Motivation vs WavMambaHAR
─────────────────────────
WavMambaHAR collapses the subcarrier axis F with `FreqAttnPool` BEFORE
Mamba, so the SSM never sees frequency structure. WavDualMamba instead
**flattens (channel × frequency) into the per-timestep feature vector** and lets
a Linear embed it — frequency is preserved as features (not pooled away).

Relation to TF-Mamba (accurate): same *spirit* — keep frequency as features and
let Mamba scan the TIME axis. In the repo's TF-Mamba BOTH streams scan time;
neither scans the frequency axis as a sequence. WavDualMamba differs by using N
branches, one per DWT subband, instead of TF-Mamba's two fixed cH/cV streams.
Note the embed Linear already mixes ALL subcarriers *linearly* per timestep; the
optional `freq_mix='mlp'` flag adds an explicit *nonlinear* spectral mix to test
whether deeper frequency modelling helps on XRF55 (prior: small gain, since the
linear mix exists and F=15 is short).

Design (finalized, literature-grounded)
───────────────────────────────────────
References: TF-Mamba (IEEE Sensors 2025, dual-stream DWT Mamba), Vim / HARMamba
(bidirectional SSM), PTM-Mamba (bidirectional *gated* Mamba blocks),
PerceptionNet (late fusion > early fusion in HAR), ECAPA-TDNN (attentive stats).

Input  : X (B, 27, T2, F2), subband-major channels
         [ (M*A) LL | (M*A) HL | (M*A) LH ], default (B, 27, 500, 15).
         (T_raw 1000 → T2 500, F_raw 30 → F2 15.)

Per-branch (one per SELECTED subband s):
    subband_s (B, M*A, T2, F2)
      → Stem_s         per-subband conv (LL/HL/LH kernel) + GN + SiLU
      → TFBlock        1 light pre-norm axial-depthwise block (local context)
      → [FreqMix]      optional nonlinear subcarrier mix (freq_mix='mlp')
      → flatten F×C    (B, d_stem, T2, F2) → (B, T2, d_stem*F2)   ← KEEP frequency
      → Linear embed   d_stem*F2 → d_model               [+ optional PosEmb]
      → BiMamba ×L      fwd/bwd merged by a per-channel ZERO-INIT gate
                        (starts at ½(fwd+bwd), specialises if data supports it)
    ⇒ S_s (B, T2, d_model)

Fusion + head:
    AdaptiveFusion(S_s …)   N-way softmax gating (zero-init → uniform at step 0)
      → AttnStatPool        ECAPA mean‖std over time → (B, 2*d_model)
      → Classifier          LN → Dropout → Linear → logits (B, num_classes)

Ablation flags:
    subbands       — any subset of ('LL','HL','LH'); enables the 4 studies
                     {HL,LH} {LL,HL} {LL,LH} {LL,HL,LH}.
    share_branches — tie the TFBlock+embed+BiMamba across subbands (stems stay
                     per-subband). Default False (separate, more expressive).
    use_pos_emb    — sinusoidal absolute PE (default False; Mamba encodes order
                     via its recurrence, so PE is usually redundant here).
    bidirectional  — backward Mamba branch + gate (default True).
    freq_mix       — None | 'mlp'. 'mlp' inserts a nonlinear MLP-Mixer over the
                     F subcarriers before flatten (zero-init ⇒ identity at step 0).
    expand, d_conv — Mamba SSM inner-expansion / local-conv width (capacity knobs).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    _MAMBA_IMPORT_ERROR = None
except Exception as e:                                   # pragma: no cover
    Mamba = None
    HAS_MAMBA = False
    _MAMBA_IMPORT_ERROR = e


# Canonical channel layout of the bench array (subband-major) and the
# physically-motivated stem kernel per subband.
_SUBBAND_ORDER  = ('LL', 'HL', 'LH')
_SUBBAND_KERNEL = {'LL': (7, 5), 'HL': (3, 7), 'LH': (7, 3)}
#   LL (7,5) slow envelope · HL (3,7) temporal burst onsets · LH (7,3) Doppler


def _gn_groups(d: int) -> int:
    """GroupNorm groups targeting ~8 channels per group, guaranteed to divide d.

    Returns the largest divisor of d that is ≤ d//8 (so ≥ ~8 channels/group);
    falls back to 1 (≡ LayerNorm over channels) when no such divisor exists.
    This keeps `d % groups == 0` for any d, not just multiples of 8.
    """
    for g in range(max(1, d // 8), 0, -1):
        if d % g == 0:
            return g
    return 1


def _sincos_pe(length: int, d_model: int) -> torch.Tensor:
    """Sinusoidal absolute positional encoding [Vaswani et al. 2017].

    Returns (1, length, d_model) — parameter-free and well-defined for any
    length (generalises beyond the cached table). Handles odd d_model.
    """
    pe  = torch.zeros(length, d_model)
    pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


# ─── Stochastic depth ─────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Drop whole samples from a residual branch during training."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


# ─── CNN: per-subband stem + light axial-depthwise block ──────────────────────

class SubbandStem(nn.Module):
    """Conv2d(in_ch → d_stem) + GroupNorm + SiLU, one per subband."""

    def __init__(self, in_ch: int, d_stem: int = 32, kernel=(5, 5)):
        super().__init__()
        kt, kf = kernel
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d_stem, (kt, kf), padding=(kt // 2, kf // 2)),
            nn.GroupNorm(_gn_groups(d_stem), d_stem),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TFBlock(nn.Module):
    """Pre-norm sequential axial-depthwise block for 2-D (time × freq) maps.

        y   = GroupNorm(x)
        y   = dw_f(dw_t(y))        # sequential axial: temporal first, then freq
        y   = pw(SiLU(y))          # pointwise channel projection
        out = x + DropPath(y)
    """

    def __init__(self, d: int, k_t: int = 7, k_f: int = 3,
                 dilation: int = 1, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(_gn_groups(d), d)
        self.dw_t = nn.Conv2d(d, d, (k_t, 1),
                              padding=(k_t // 2 * dilation, 0),
                              dilation=(dilation, 1), groups=d)
        self.dw_f = nn.Conv2d(d, d, (1, k_f), padding=(0, k_f // 2), groups=d)
        self.act  = nn.SiLU()
        self.pw   = nn.Conv2d(d, d, 1)
        self.dp   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.dw_f(self.dw_t(y))
        y = self.pw(self.act(y))
        return x + self.dp(y)


# ─── Bidirectional Mamba with per-channel zero-init gated fwd/bwd merge ────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )


class BiMambaLayer(nn.Module):
    """One (bi)directional Mamba layer — NO FFN — with a learned gate fusing the
    two directions instead of a plain sum.

        h = RMSNorm(x)
        f = Mamba_fwd(h)
        b = flip(Mamba_bwd(flip(h)))
        g = σ(W·[f ‖ b] + c)        # per-channel gate, W=0,c=0 at init ⇒ g≡0.5
        y = g ⊙ f + (1 − g) ⊙ b      # starts as ½(f+b), specialises later
        out = x + DropPath(y)

    With bidirectional=False the layer is a plain unidirectional residual
    Mamba (no gate), for ablation.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, drop_path: float = 0.0,
                 bidirectional: bool = True):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is required to build BiMambaLayer.\n"
                "Install: pip install mamba-ssm[causal-conv1d] --no-build-isolation\n"
                f"Original import error: {_MAMBA_IMPORT_ERROR}"
            )
        self.bidirectional = bidirectional
        self.norm = RMSNorm(d_model)
        self.fwd  = Mamba(d_model=d_model, d_state=d_state,
                          d_conv=d_conv, expand=expand)
        if bidirectional:
            self.bwd  = Mamba(d_model=d_model, d_state=d_state,
                              d_conv=d_conv, expand=expand)
            self.gate = nn.Linear(2 * d_model, d_model)
            nn.init.zeros_(self.gate.weight)     # g ≡ 0.5 at init → ½(fwd+bwd)
            nn.init.zeros_(self.gate.bias)
        else:
            self.bwd  = None
            self.gate = None
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        f = self.fwd(h)
        if self.bwd is None:
            y = f
        else:
            b = self.bwd(h.flip(1)).flip(1)
            g = torch.sigmoid(self.gate(torch.cat([f, b], dim=-1)))   # (B,T,d)
            y = g * f + (1.0 - g) * b
        return x + self.dp(y)


class BiMamba(nn.Module):
    """Stack of gated (bi)directional Mamba layers + final RMSNorm."""

    def __init__(self, d_model: int, n_layers: int = 2, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 drop_path_rates=(0.0, 0.05), bidirectional: bool = True):
        super().__init__()
        if len(drop_path_rates) != n_layers:
            raise ValueError(
                f"len(drop_path_rates)={len(drop_path_rates)} != n_layers={n_layers}"
            )
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         drop_path=drop_path_rates[i], bidirectional=bidirectional)
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─── Optional nonlinear frequency-axis mixing (MLP-Mixer over subcarriers) ────

class FreqMix(nn.Module):
    """Nonlinear mixing across the subcarrier axis F (MLP-Mixer token-mixing).

    The embed Linear after flatten already mixes (channel × frequency) *linearly*
    per timestep; this block adds an explicit *nonlinear* mix over the F
    subcarriers, shared across (channel, time), with a residual. The second
    Linear is zero-initialised ⇒ the block is identity at step 0 (safe baseline),
    consistent with the model's zero-init philosophy.

    Input / Output: (B, C, T, F) — same shape (operates on the last, F, axis).
    """

    def __init__(self, f2: int, hidden: int = None, drop_path: float = 0.0):
        super().__init__()
        hidden = hidden or max(8, f2 * 2)
        self.norm = nn.LayerNorm(f2)
        self.fc1  = nn.Linear(f2, hidden)
        self.act  = nn.SiLU()
        self.fc2  = nn.Linear(hidden, f2)
        nn.init.zeros_(self.fc2.weight)            # identity at init
        nn.init.zeros_(self.fc2.bias)
        self.dp   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(self.act(self.fc1(self.norm(x))))   # mix over F (last axis)
        return x + self.dp(y)


# ─── Per-subband branch backbone (shareable) ──────────────────────────────────

class BranchBackbone(nn.Module):
    """TFBlock → flatten F×C → Linear embed [+ PosEmb] → BiMamba.

    Input  : (B, d_stem, T2, F2)    Output: (B, T2, d_model)
    """

    def __init__(self, d_stem: int, f2: int, d_model: int = 64,
                 dp_cnn: float = 0.0, n_mamba_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dp_mamba=(0.0, 0.05), bidirectional: bool = True,
                 use_pos_emb: bool = False, freq_mix: str = None,
                 t_max: int = 500):
        super().__init__()
        if freq_mix not in (None, 'mlp'):
            raise ValueError(f"freq_mix must be None or 'mlp', got {freq_mix!r}")
        self.block    = TFBlock(d_stem, drop_path=dp_cnn)
        self.freq_mix = FreqMix(f2) if freq_mix == 'mlp' else None
        self.embed = nn.Sequential(
            nn.Linear(d_stem * f2, d_model),
            nn.SiLU(),
        )
        self.use_pos_emb = use_pos_emb
        self.t_max = t_max
        if use_pos_emb:
            # Sinusoidal (Vaswani) absolute PE — parameter-free, length-robust.
            # Non-persistent buffer: deterministic, recomputed on load (not saved).
            self.register_buffer("pos_emb", _sincos_pe(t_max, d_model),
                                 persistent=False)
        else:
            self.pos_emb = None
        self.mamba = BiMamba(d_model, n_layers=n_mamba_layers, d_state=d_state,
                             d_conv=d_conv, expand=expand,
                             drop_path_rates=dp_mamba, bidirectional=bidirectional)

    def _add_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_emb is None:
            return x
        T = x.size(1)
        if T <= self.t_max:
            return x + self.pos_emb[:, :T]
        # Longer than the cached table (rare): build sinusoidal PE on the fly.
        pe = _sincos_pe(T, x.size(-1)).to(device=x.device, dtype=x.dtype)
        return x + pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)                                # (B, d_stem, T2, F2)
        if self.freq_mix is not None:
            x = self.freq_mix(x)                         # nonlinear subcarrier mix
        B, C, T, Fd = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fd)  # flatten F×C → feature
        x = self.embed(x)                                # (B, T2, d_model)
        x = self._add_pos_emb(x)
        return self.mamba(x)                             # (B, T2, d_model)


# ─── Adaptive N-way fusion (zero-init → uniform at step 0) ─────────────────────

class AdaptiveFusion(nn.Module):
    """Soft-weighted fusion of N streams (generalises TF-Mamba's 2-way fusion).

        S_cat = concat(S_1 … S_N)                 (B, T, N·d)
        α     = softmax(Linear_{N·d→N}(S_cat))    (B, T, N)   per-token weights
        out   = Σ_i α_i ⊙ S_i                      (B, T, d)

    The Linear is zero-initialised, so α is uniform (1/N) at step 0.
    """

    def __init__(self, d_model: int, n_branches: int):
        super().__init__()
        self.n_branches = n_branches
        if n_branches > 1:
            self.linear = nn.Linear(n_branches * d_model, n_branches)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        else:
            self.linear = None

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        if self.linear is None:
            return streams[0]
        cat = torch.cat(streams, dim=-1)                 # (B, T, N·d)
        w = self.linear(cat).softmax(dim=-1)             # (B, T, N)
        out = sum(w[..., i:i + 1] * streams[i]
                  for i in range(self.n_branches))
        return out                                       # (B, T, d)


# ─── Temporal attentive statistics pooling (ECAPA-style, zero-init) ───────────

class AttnStatPool(nn.Module):
    """Per-channel temporal attention → [weighted mean ‖ weighted std].

    Input : (B, T, d)    Output: (B, 2*d)
    """

    def __init__(self, dim: int, bn: int = None):
        super().__init__()
        bn = bn or max(8, dim // 2)
        self.score = nn.Sequential(
            nn.Linear(dim, bn), nn.Tanh(), nn.Linear(bn, dim)
        )
        nn.init.zeros_(self.score[-1].weight)            # uniform attention init
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.score(x).softmax(dim=1)                 # (B, T, d)
        mean = (w * x).sum(dim=1)                        # (B, d)
        var = (w * (x - mean.unsqueeze(1)).pow(2)).sum(dim=1)
        return torch.cat([mean, (var + 1e-6).sqrt()], dim=-1)


# ─── Classifier head ──────────────────────────────────────────────────────────

class Classifier(nn.Module):
    """LayerNorm → Dropout → Linear."""

    def __init__(self, in_dim: int, num_classes: int = 11, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─── Full model ───────────────────────────────────────────────────────────────

class WavDualMamba(nn.Module):
    """Multi-branch (per-subband) CNN + gated-BiMamba with adaptive late fusion.

    Args:
        num_classes    : output classes (default 11).
        n_links        : Wi-Fi receivers M (default 3).
        n_antennas     : antennas per receiver A (default 3).
        subbands       : subset of ('LL','HL','LH') to use as branches
                         (default all three). Order is normalised to LL,HL,LH.
        d_model        : branch feature width (default 64).
        d_stem         : per-subband stem width (default 32, %8==0 preferred).
        d_state        : Mamba SSM state size (default 16).
        n_mamba_layers : BiMamba layers per branch (default 2).
        f2             : subcarrier axis length after DWT (default 15).
        dp_cnn         : DropPath for the single TFBlock (default 0.0).
        dp_mamba       : DropPath per BiMamba layer (len == n_mamba_layers).
        bidirectional  : gated backward Mamba branch (default True).
        use_pos_emb    : sinusoidal positional embedding (default False;
                         parameter-free, length-robust).
        share_branches : tie TFBlock+embed+BiMamba across subbands; stems stay
                         per-subband (default False). When True, the branches are
                         run in ONE batched backbone call (≈ N× throughput).
        freq_mix       : None | 'mlp' — nonlinear subcarrier mix before flatten.
        expand         : Mamba inner-expansion factor (default 2, capacity knob).
        d_conv         : Mamba local conv1d width (default 4).
        dropout        : classifier dropout (default 0.2).
        t_max          : positional-embedding buffer length (default 500 = T2).

    Input  : X (B, 27, T2, F2), subband-major [LL | HL | LH] (27 = 3·M·A).
             Only the channels of the SELECTED subbands are processed.
    Output : logits (B, num_classes).
    """

    def __init__(
        self,
        num_classes: int = 11,
        n_links: int = 3,
        n_antennas: int = 3,
        subbands=('LL', 'HL', 'LH'),
        d_model: int = 64,
        d_stem: int = 32,
        d_state: int = 16,
        n_mamba_layers: int = 2,
        f2: int = 15,
        dp_cnn: float = 0.0,
        dp_mamba=(0.0, 0.05),
        bidirectional: bool = True,
        use_pos_emb: bool = False,
        share_branches: bool = False,
        freq_mix: str = None,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.2,
        t_max: int = 500,
    ):
        super().__init__()
        if len(dp_mamba) != n_mamba_layers:
            raise ValueError("len(dp_mamba) must equal n_mamba_layers")

        # Normalise & validate the selected subbands.
        sel = [s for s in _SUBBAND_ORDER if s in subbands]   # keep canonical order
        unknown = [s for s in subbands if s not in _SUBBAND_ORDER]
        if unknown:
            raise ValueError(f"Unknown subband(s) {unknown}; choose from {_SUBBAND_ORDER}")
        if len(sel) < 1:
            raise ValueError("subbands must select at least one of ('LL','HL','LH')")
        self.subbands   = tuple(sel)
        self.n_branches = len(sel)

        self.f2         = f2                                 # subcarrier axis length
        self.n_per_sub  = n_links * n_antennas               # channels per subband
        self.n_data_sub = len(_SUBBAND_ORDER)                # bench array has all 3
        self._c_in      = self.n_per_sub * self.n_data_sub   # = 27 (full input)
        self._sb_index  = {s: _SUBBAND_ORDER.index(s) for s in self.subbands}

        # Per-subband stems (always separate — physically-motivated kernels).
        self.stems = nn.ModuleDict({
            s: SubbandStem(self.n_per_sub, d_stem, kernel=_SUBBAND_KERNEL[s])
            for s in self.subbands
        })

        # Branch backbones: shared (one) or separate (one per subband).
        bb_kwargs = dict(
            d_stem=d_stem, f2=f2, d_model=d_model, dp_cnn=dp_cnn,
            n_mamba_layers=n_mamba_layers, d_state=d_state,
            d_conv=d_conv, expand=expand, dp_mamba=dp_mamba,
            bidirectional=bidirectional, use_pos_emb=use_pos_emb,
            freq_mix=freq_mix, t_max=t_max,
        )
        self.share_branches = share_branches
        if share_branches:
            self.shared_backbone = BranchBackbone(**bb_kwargs)
            self.backbones = None
        else:
            self.shared_backbone = None
            self.backbones = nn.ModuleDict({
                s: BranchBackbone(**bb_kwargs) for s in self.subbands
            })

        self.fusion = AdaptiveFusion(d_model, self.n_branches)
        self.tpool  = AttnStatPool(d_model)
        self.head   = Classifier(2 * d_model, num_classes=num_classes,
                                 dropout=dropout)

    @property
    def C_IN(self) -> int:
        """Expected input channels = n_links * n_antennas * 3 subbands (= 27)."""
        return self._c_in

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 4:
            raise ValueError(
                f"Expected 4-D input (B, {self._c_in}, T2, F2), got {tuple(X.shape)}"
            )
        if X.shape[1] != self._c_in:
            raise ValueError(
                f"Expected {self._c_in} channels (= n_links*n_antennas*3), "
                f"got {X.shape[1]}"
            )
        if X.shape[-1] != self.f2:
            raise ValueError(
                f"Expected F2={self.f2} subcarriers (model built with f2={self.f2}); "
                f"got {X.shape[-1]}. Rebuild the model with f2={X.shape[-1]} to match "
                f"your DWT output."
            )

        # Per-subband stems (always separate, physical kernels).
        stem_outs = []
        for s in self.subbands:
            i  = self._sb_index[s]
            sb = X[:, i * self.n_per_sub:(i + 1) * self.n_per_sub]   # (B, M*A, T2, F2)
            stem_outs.append(self.stems[s](sb))                     # (B, d_stem, T2, F2)

        if self.share_branches:
            # One backbone call on the N stem outputs stacked along the batch axis
            # (≈ N× throughput vs a Python loop). Mathematically identical: the
            # backbone has tied weights and the samples are independent.
            h       = torch.cat(stem_outs, dim=0)                   # (N*B, d_stem, T2, F2)
            out     = self.shared_backbone(h)                       # (N*B, T2, d_model)
            streams = list(out.chunk(self.n_branches, dim=0))       # N × (B, T2, d_model)
        else:
            streams = [self.backbones[s](stem_outs[k])              # (B, T2, d_model)
                       for k, s in enumerate(self.subbands)]

        z = self.fusion(streams)                                    # (B, T2, d_model)
        z = self.tpool(z)                                           # (B, 2*d_model)
        return self.head(z)                                         # (B, num_classes)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
