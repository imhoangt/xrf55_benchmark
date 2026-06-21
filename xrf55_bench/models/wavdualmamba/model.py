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
      → Stem_s         per-subband conv (LL/HL/LH kernel) + [GN] + SiLU
                       (GroupNorm dropped by default since 2026-06; stem_norm=True restores it)
                       [temporal_stride=2 → T2 500→250]
      → TFBlock×3      dilation [1,2,4] → RF 7→19→43 timesteps (drop_path 0.0,0.05,0.1)
      → [FreqMix]      optional nonlinear subcarrier mix (freq_mix='mlp')
      → flatten F×C    (B, d_stem, T2, F2) → (B, T2, d_stem*F2)   ← KEEP frequency
      → Linear embed   d_stem*F2 → d_model + SiLU + Dropout(0.1) [+ optional PosEmb]
      → BiMamba ×L      fwd/bwd merged by a per-channel ZERO-INIT gate
                        (starts at ½(fwd+bwd), specialises if data supports it)
    ⇒ S_s (B, T2, d_model)

Fusion + head:
    AdaptiveFusion(S_s …)   convex | gate | concat merge (zero-init → mean at step 0)
      → AttnStatPool        ECAPA mean‖std over time → (B, 2*d_model)
      → Classifier          LN → Dropout → Linear → logits (B, num_classes)

Ablation flags:
    subbands       — any subset of ('LL','HL','LH'); enables the 4 studies
                     {HL,LH} {LL,HL} {LL,LH} {LL,HL,LH}.
    share_branches — tie the TFBlock+embed+BiMamba across subbands (stems stay
                     per-subband). Default False (separate, more expressive).
    fusion         — 'convex' | 'gate' | 'concat' branch-merge (default 'gate'
                     since 2026-06; 'convex' = TF-Mamba baseline). 'gate' =
                     per-channel convex routing; 'concat' = static full-matrix
                     mix. See AdaptiveFusion.
    use_pos_emb    — sinusoidal absolute PE (default False; Mamba encodes order
                     via its recurrence, so PE is usually redundant here).
    bidirectional  — backward Mamba branch + gate (default True).
    freq_mix       — None | 'mlp'. 'mlp' inserts a nonlinear MLP-Mixer over the
                     F subcarriers before flatten (zero-init ⇒ identity at step 0).
    expand, d_conv — Mamba SSM inner-expansion / local-conv width (capacity knobs).
    use_eca        — [C7] ECA channel gate on raw 27-ch input before stems (default False).
    pool_context   — [C8] full ECAPA [x‖μ‖σ] context pooling in AttnStatPool (default True).
    use_final_attn — [C6] one MHSA layer after fusion, before pooling (default False).
    use_post_fusion_proj — [S4.a/S4.b/S4.c] Linear(d→d) after fusion, before pooling
                     (default False).
    post_fusion_proj_tanh — [S4.c] add tanh after that Linear ⇒ full TF-Mamba proj_s3
                     head (Linear+tanh) grafted onto WavDualMamba (default False;
                     only meaningful when use_post_fusion_proj=True).
    stem_norm        [S4.nogn] Keep/drop the GroupNorm inside SubbandStem. False
                     (default since 2026-06) ⇒ stem = Conv→SiLU (no GroupNorm);
                     the TFBlock pre-norm GN still normalises right after. True =
                     original behaviour (GroupNorm in the stem).
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

    def __init__(self, in_ch: int, d_stem: int = 16, kernel=(5, 5),
                 temporal_stride: int = 1, norm: bool = True):
        super().__init__()
        kt, kf = kernel
        layers = [nn.Conv2d(in_ch, d_stem, (kt, kf),
                            stride=(temporal_stride, 1),
                            padding=(kt // 2, kf // 2))]
        if norm:                                        # norm=False -> bo GN cua stem (rung S4.nogn)
            layers.append(nn.GroupNorm(_gn_groups(d_stem), d_stem))
        layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

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


# ─── [C7] Efficient Channel Attention (ECA) — applied on raw 27-ch input ────────

class ECA(nn.Module):
    """[C7] Efficient Channel Attention (~5 params). Applied on raw input before stems.

    GAP over (T, F) → Conv1d(k=5) → sigmoid → per-channel scale.
    Input / Output: (B, C, T, F) — 4D, same shape.
    """

    def __init__(self, k: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=(2, 3))                                   # (B, C) global avg
        y = torch.sigmoid(self.conv(y.unsqueeze(1))).squeeze(1)  # (B, C) weights
        return x * y.unsqueeze(-1).unsqueeze(-1)                 # (B, C, T, F) scale


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
    """One (bi)directional Mamba layer with optional FFN sub-block.

    SSM sub-block (always present):
        h = RMSNorm(x)
        f = Mamba_fwd(h)
        b = flip(Mamba_bwd(flip(h)))
        g = σ(W·[f ‖ b] + c)        # per-channel gate, W=0,c=0 at init ⇒ g≡0.5
        y = g ⊙ f + (1 − g) ⊙ b      # starts as ½(f+b), specialises later
        x = x + DropPath(y)

    FFN sub-block (ffn_ratio > 0):
        h = RMSNorm(x)
        x = x + DropPath(fc2(SiLU(fc1(h))))   # fc2 zero-init ⇒ identity at init

    With bidirectional=False the SSM is unidirectional (no gate), for ablation.
    """

    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4,
                 expand: int = 2, drop_path: float = 0.0,
                 bidirectional: bool = True, ffn_ratio: int = 0):
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
        if ffn_ratio > 0:
            self.ffn_norm = RMSNorm(d_model)
            self.ffn_fc1  = nn.Linear(d_model, ffn_ratio * d_model)
            self.ffn_act  = nn.SiLU()
            self.ffn_fc2  = nn.Linear(ffn_ratio * d_model, d_model)
            nn.init.zeros_(self.ffn_fc2.weight)  # identity at init
            nn.init.zeros_(self.ffn_fc2.bias)
            self.ffn_dp   = DropPath(drop_path)
        else:
            self.ffn_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        f = self.fwd(h)
        if self.bwd is None:
            y = f
        else:
            b = self.bwd(h.flip(1)).flip(1)
            g = torch.sigmoid(self.gate(torch.cat([f, b], dim=-1)))   # (B,T,d)
            y = g * f + (1.0 - g) * b
        x = x + self.dp(y)
        if self.ffn_norm is not None:
            h = self.ffn_norm(x)
            x = x + self.ffn_dp(self.ffn_fc2(self.ffn_act(self.ffn_fc1(h))))
        return x


class BiMamba(nn.Module):
    """Stack of gated (bi)directional Mamba layers + final RMSNorm."""

    def __init__(self, d_model: int, n_layers: int = 2, d_state: int = 32,
                 d_conv: int = 4, expand: int = 2,
                 drop_path_rates=(0.0, 0.10), bidirectional: bool = True,
                 ffn_ratio: int = 0):
        super().__init__()
        if len(drop_path_rates) != n_layers:
            raise ValueError(
                f"len(drop_path_rates)={len(drop_path_rates)} != n_layers={n_layers}"
            )
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         drop_path=drop_path_rates[i], bidirectional=bidirectional,
                         ffn_ratio=ffn_ratio)
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
    """TFBlock×3 (dilation 1,2,4) → flatten F×C → Linear embed [+ PosEmb] → BiMamba.

    Input  : (B, d_stem, T2, F2)    Output: (B, T2, d_model)
    """

    def __init__(self, d_stem: int, f2: int, d_model: int = 64,
                 dp_cnn: tuple = (0.0, 0.05, 0.1), dilations: tuple = (1, 2, 4),
                 n_mamba_layers: int = 2,
                 d_state: int = 32, d_conv: int = 4, expand: int = 2,
                 dp_mamba=(0.0, 0.10), bidirectional: bool = True,
                 use_pos_emb: bool = False, freq_mix: str = None,
                 embed_drop: float = 0.1, t_max: int = 500,
                 embed_hidden: int = None, ffn_ratio: int = 0):
        super().__init__()
        if freq_mix not in (None, 'mlp'):
            raise ValueError(f"freq_mix must be None or 'mlp', got {freq_mix!r}")
        if len(dp_cnn) != len(dilations):
            raise ValueError(
                f"len(dp_cnn)={len(dp_cnn)} must equal len(dilations)={len(dilations)}"
            )
        self.blocks   = nn.ModuleList([
            TFBlock(d_stem, dilation=dilations[i], drop_path=dp_cnn[i])
            for i in range(len(dilations))
        ])
        self.freq_mix = FreqMix(f2) if freq_mix == 'mlp' else None
        in_dim = d_stem * f2
        if embed_hidden:
            self.embed = nn.Sequential(
                nn.Linear(in_dim, embed_hidden),
                nn.SiLU(),
                nn.Linear(embed_hidden, d_model),
                nn.Dropout(embed_drop),
            )
        else:
            self.embed = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.SiLU(),
                nn.Dropout(embed_drop),
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
                             drop_path_rates=dp_mamba, bidirectional=bidirectional,
                             ffn_ratio=ffn_ratio)

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
        for blk in self.blocks:
            x = blk(x)                                   # (B, d_stem, T2, F2)
        if self.freq_mix is not None:
            x = self.freq_mix(x)                         # nonlinear subcarrier mix
        B, C, T, Fd = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fd)  # flatten F×C → feature
        x = self.embed(x)                                # (B, T2, d_model)
        x = self._add_pos_emb(x)
        return self.mamba(x)                             # (B, T2, d_model)


# ─── Adaptive N-way fusion (zero-init → mean of streams at step 0) ─────────────

class AdaptiveFusion(nn.Module):
    """Merge N branch streams → (B, T, d). Three modes, all zero-initialised so
    the block starts as the plain mean of the streams (uniform 1/N weighting) at
    step 0 — every variant departs from the SAME function, so an ablation
    measures what is *learned*, not init luck.

        cat = concat(S_1 … S_N)                       (B, T, N·d)

      'convex' (default = baseline, TF-Mamba style):
        α   = softmax(Linear_{N·d→N}(cat))            (B, T, N)   one scalar weight
        out = Σ_i α_i ⊙ S_i                            per BRANCH per token → all d
              channels of a branch share one weight (the bottleneck 'gate' lifts).

      'gate' (per-channel convex routing):
        a   = Linear_{N·d→N·d}(cat) → (B, T, N, d)
        α   = softmax(a, dim=branch)                  per token, per CHANNEL
        out = Σ_i α_i ⊙ S_i                            each channel mixes the N
              branches independently (take channel j from one, k from another).
              Still input-adaptive and convex (α ≥ 0, Σ_i α_i = 1).

      'concat' (static full-matrix mix):
        out = Linear_{N·d→d}(cat)                      one learned matrix; can add,
              subtract and cross-mix channels, but is the SAME at every timestep
              (not input-adaptive). Init = averaging (each d×d block = I/N).

    'convex' keeps the original `self.linear` parameter name so baseline
    checkpoints load unchanged; 'gate'/'concat' use `self.proj`.
    """

    def __init__(self, d_model: int, n_branches: int, mode: str = 'convex'):
        super().__init__()
        if mode not in ('convex', 'gate', 'concat'):
            raise ValueError(
                f"fusion mode must be 'convex'|'gate'|'concat', got {mode!r}"
            )
        self.n_branches = n_branches
        self.d_model    = d_model
        self.mode       = mode
        self.linear     = None          # 'convex'  (original name → ckpt-compatible)
        self.proj       = None          # 'gate' / 'concat'
        if n_branches == 1:
            return                       # single branch — nothing to fuse
        if mode == 'convex':
            self.linear = nn.Linear(n_branches * d_model, n_branches)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        elif mode == 'gate':
            self.proj = nn.Linear(n_branches * d_model, n_branches * d_model)
            nn.init.zeros_(self.proj.weight)     # softmax → 1/N per channel ⇒ mean
            nn.init.zeros_(self.proj.bias)
        else:  # 'concat'
            self.proj = nn.Linear(n_branches * d_model, d_model)
            with torch.no_grad():                # init = averaging ⇒ mean at step 0
                self.proj.weight.zero_()
                self.proj.bias.zero_()
                for i in range(n_branches):
                    self.proj.weight[:, i * d_model:(i + 1) * d_model] = \
                        torch.eye(d_model) / n_branches

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        if self.n_branches == 1:
            return streams[0]
        cat = torch.cat(streams, dim=-1)                     # (B, T, N·d)
        if self.mode == 'convex':
            w = self.linear(cat).softmax(dim=-1)             # (B, T, N)
            return sum(w[..., i:i + 1] * streams[i]
                       for i in range(self.n_branches))      # (B, T, d)
        if self.mode == 'gate':
            B, T, _ = cat.shape
            w = self.proj(cat).view(B, T, self.n_branches, self.d_model)
            w = w.softmax(dim=2)                             # (B, T, N, d)
            s = torch.stack(streams, dim=2)                  # (B, T, N, d)
            return (w * s).sum(dim=2)                        # (B, T, d)
        return self.proj(cat)                                # 'concat' → (B, T, d)


# ─── [C6] FinalAttention — optional MHSA after fusion, before pooling ────────────

class FinalAttention(nn.Module):
    """[C6] Pre-norm MHSA after AdaptiveFusion, before temporal pooling.

    out_proj zero-init → residual starts as identity at step 0 (MambaVision-style).
    Default OFF (use_final_attn=False). Input / Output: (B, T, d_model).
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 attn_drop: float = 0.1, drop_path: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=attn_drop, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + self.dp(h)


# ─── Temporal attentive statistics pooling (ECAPA-style, zero-init) ───────────

class AttnStatPool(nn.Module):
    """Per-channel temporal attention → [weighted mean ‖ weighted std].

    Input : (B, T, d)    Output: (B, 2*d)

    context=True  [C8]: score sees [x‖μ‖σ] (3×dim input, full ECAPA style).
    context=False      : score sees x only (old v1 behaviour).
    """

    def __init__(self, dim: int, bn: int = None, context: bool = True):
        super().__init__()
        self.context = context
        bn = bn or max(8, dim // 2)
        in_dim = 3 * dim if context else dim
        self.score = nn.Sequential(
            nn.Linear(in_dim, bn), nn.Tanh(), nn.Linear(bn, dim)
        )
        nn.init.zeros_(self.score[-1].weight)            # uniform attention init
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.context:
            mu = x.mean(dim=1, keepdim=True).expand_as(x)
            sg = x.var(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6).sqrt().expand_as(x)
            h = torch.cat([x, mu, sg], dim=-1)          # (B, T, 3*d)
        else:
            h = x
        w = self.score(h).softmax(dim=1)                 # (B, T, d)
        mean = (w * x).sum(dim=1)                        # (B, d)
        var = (w * (x - mean.unsqueeze(1)).pow(2)).sum(dim=1)
        return torch.cat([mean, var.clamp(min=1e-6).sqrt()], dim=-1)


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
        d_stem         : per-subband stem width (default 16).
        d_state        : Mamba SSM state size (default 32).
        n_mamba_layers : BiMamba layers per branch (default 2).
        f2             : subcarrier axis length after DWT (default 15).
        dp_cnn         : DropPath per TFBlock (default (0.0, 0.05, 0.1) for 3 blocks).
        dilations      : dilation per TFBlock (default (1, 2, 4) → RF 7, 19, 43 timesteps).
        embed_drop     : Dropout after Linear embed, before Mamba (default 0.1).
        temporal_stride: stride along time axis in the stem conv (default 1).
                         Set to 2 to halve T2 (500→250) for faster training.
        dp_mamba       : DropPath per BiMamba layer (len == n_mamba_layers).
        bidirectional  : gated backward Mamba branch (default True).
        use_pos_emb    : sinusoidal positional embedding (default False;
                         parameter-free, length-robust).
        share_branches : tie TFBlock+embed+BiMamba across subbands; stems stay
                         per-subband (default False). When True, the branches are
                         run in ONE batched backbone call (≈ N× throughput).
        fusion         : 'convex' | 'gate' | 'concat' — how the N branch streams
                         are merged (all (B,T,d)→(B,T,d), zero-init → mean at
                         step 0). 'gate' = per-channel convex routing (default
                         since 2026-06); 'convex' = per-branch scalar (TF-Mamba
                         baseline); 'concat' = static full-matrix mix.
                         See AdaptiveFusion.
        freq_mix       : None | 'mlp' — nonlinear subcarrier mix before flatten.
        expand         : Mamba inner-expansion factor (default 2, capacity knob).
        d_conv         : Mamba local conv1d width (default 4).
        dropout        : classifier dropout (default 0.2).
        t_max          : positional-embedding buffer length (default 500 = T2).
        embed_hidden   : if set, adds an intermediate Linear(d_stem*f2 → embed_hidden)
                         before the final embed projection, splitting the 3.75× compression
                         into two stages. Default None (single-stage, current behaviour).
        ffn_ratio      : if > 0, appends an FFN sub-block (RMSNorm → Linear(d→ratio·d)
                         → SiLU → Linear(ratio·d→d), fc2 zero-init) after each Mamba
                         layer. Default 0 (no FFN, current behaviour).
        use_eca        : [C7] ECA channel gate on raw 27-ch input before stems.
                         Default False (headline benchmark config); True = ablation.
        pool           : 'attnstat' (default) | 'gap'. 'gap' replaces AttnStatPool
                         with global average pooling (head input d_model, not 2*d_model).
        pool_context   : [C8] pass [x‖μ‖σ] into AttnStatPool score (full ECAPA).
                         Default True. Set False to reproduce old v1 behaviour.
        use_final_attn : [C6] insert one MHSA layer after fusion, before pooling.
                         Default False (off); True enables the ablation.
        attn_heads     : number of MHSA heads for FinalAttention (default 4).
        attn_drop      : attention dropout in FinalAttention (default 0.1).
        attn_drop_path : DropPath rate for FinalAttention residual (default 0.1).

    Input  : X (B, 27, T2, F2), subband-major [LL | HL | LH] (27 = 3·M·A).
             Only the channels of the SELECTED subbands are processed.
             Also accepts a PACKED input with 9·len(subbands) channels holding
             only the selected subbands in canonical order (ablation adapters).
    Output : logits (B, num_classes).
    """

    def __init__(
        self,
        num_classes: int = 11,
        n_links: int = 3,
        n_antennas: int = 3,
        subbands=('LL', 'HL', 'LH'),
        d_model: int = 64,
        d_stem: int = 16,
        d_state: int = 32,
        n_mamba_layers: int = 2,
        f2: int = 15,
        dp_cnn: tuple = (0.0, 0.05, 0.1),
        dilations: tuple = (1, 2, 4),
        dp_mamba=(0.0, 0.10),
        embed_drop: float = 0.1,
        temporal_stride: int = 1,
        bidirectional: bool = True,
        use_pos_emb: bool = False,
        share_branches: bool = False,
        fusion: str = 'gate',
        freq_mix: str = None,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.2,
        t_max: int = 500,
        embed_hidden: int = None,
        ffn_ratio: int = 0,
        use_eca: bool = False,
        pool: str = 'attnstat',
        pool_context: bool = True,
        use_final_attn: bool = False,
        attn_heads: int = 4,
        attn_drop: float = 0.1,
        attn_drop_path: float = 0.1,
        use_post_fusion_proj: bool = False,
        post_fusion_proj_tanh: bool = False,
        stem_norm: bool = False,
    ):
        super().__init__()
        if len(dp_mamba) != n_mamba_layers:
            raise ValueError("len(dp_mamba) must equal n_mamba_layers")
        if pool not in ('attnstat', 'gap'):
            raise ValueError(f"pool must be 'attnstat' or 'gap', got {pool!r}")

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
        self.temporal_stride = temporal_stride
        self.stems = nn.ModuleDict({
            s: SubbandStem(self.n_per_sub, d_stem, kernel=_SUBBAND_KERNEL[s],
                           temporal_stride=temporal_stride, norm=stem_norm)
            for s in self.subbands
        })

        # Branch backbones: shared (one) or separate (one per subband).
        bb_kwargs = dict(
            d_stem=d_stem, f2=f2, d_model=d_model,
            dp_cnn=dp_cnn, dilations=dilations,
            n_mamba_layers=n_mamba_layers, d_state=d_state,
            d_conv=d_conv, expand=expand, dp_mamba=dp_mamba,
            bidirectional=bidirectional, use_pos_emb=use_pos_emb,
            freq_mix=freq_mix, embed_drop=embed_drop,
            t_max=t_max // max(temporal_stride, 1),
            embed_hidden=embed_hidden, ffn_ratio=ffn_ratio,
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

        self.eca = ECA() if use_eca else None
        self.fusion = AdaptiveFusion(d_model, self.n_branches, mode=fusion)
        # [S4.a/S4.b] phep chieu tuyen tinh sau fusion, truoc pooling.
        #   post_fusion_proj_tanh=False -> Linear THUAN (= proj_s3 BO tanh): S4.a/S4.b
        #   post_fusion_proj_tanh=True  -> Linear + tanh (= FULL proj_s3 cua TF-Mamba): S4.c
        # Mac dinh tat hoan toan.
        self.post_fusion_proj = nn.Linear(d_model, d_model) if use_post_fusion_proj else None
        self.post_fusion_proj_tanh = post_fusion_proj_tanh
        self.final_attn = (FinalAttention(d_model, n_heads=attn_heads,
                                          attn_drop=attn_drop, drop_path=attn_drop_path)
                           if use_final_attn else None)
        # Temporal pooling: AttnStatPool (default, -> 2*d_model) or GAP (-> d_model).
        if pool == 'gap':
            self.tpool = None
            head_in    = d_model
        else:
            self.tpool = AttnStatPool(d_model, context=pool_context)
            head_in    = 2 * d_model
        self.head   = Classifier(head_in, num_classes=num_classes,
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
        packed_c = self.n_per_sub * self.n_branches
        if X.shape[1] not in (self._c_in, packed_c):
            raise ValueError(
                f"Expected {self._c_in} channels (full bench layout LL|HL|LH) or "
                f"{packed_c} (packed: only the selected subbands {self.subbands}, "
                f"in that order), got {X.shape[1]}"
            )
        is_packed = X.shape[1] == packed_c
        if X.shape[-1] != self.f2:
            raise ValueError(
                f"Expected F2={self.f2} subcarriers (model built with f2={self.f2}); "
                f"got {X.shape[-1]}. Rebuild the model with f2={X.shape[-1]} to match "
                f"your DWT output."
            )

        if self.eca is not None:
            X = self.eca(X)                                         # [C7] ECA on raw 27-ch

        # Per-subband stems (always separate, physical kernels).
        stem_outs = []
        for k, s in enumerate(self.subbands):
            i  = k if is_packed else self._sb_index[s]
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
        if self.post_fusion_proj is not None:
            z = self.post_fusion_proj(z)                            # [S4.a/S4.b] Linear(d->d)
            if self.post_fusion_proj_tanh:
                z = torch.tanh(z)                                   # [S4.c] + tanh = full TF-Mamba proj_s3 head
        if self.final_attn is not None:
            z = self.final_attn(z)                                  # [C6] optional MHSA
        z = self.tpool(z) if self.tpool is not None else z.mean(dim=1)  # AttnStatPool or GAP
        return self.head(z)                                         # (B, num_classes)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
