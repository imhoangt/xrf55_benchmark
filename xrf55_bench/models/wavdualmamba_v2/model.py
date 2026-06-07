"""
wavdualmamba_v2.py — WavDualMambaV2
=====================================

WiFi-CSI HAR on XRF55 (default 11 classes). Upgraded from WavDualMamba based
on TF-Mamba (IEEE Sensors 2025), HARMamba (2024), Vim (ICML 2024),
MambaVision (CVPR 2025), ECAPA-TDNN (Interspeech 2020), ECA-Net (CVPR 2020).

Input : X (B, 27, T2, F2) — subband-major [LL | HL | LH], default (B, 27, 500, 15)
Output: logits (B, num_classes)

Changes vs WavDualMamba v1 (each has an enable/disable flag for ablation):
──────────────────────────────────────────────────────────────────────────────
C1  arch='trunk' (default): concat 3 subbands AFTER stems into 1 trunk instead
    of 3 parallel branch backbones. Rationale: (a) concentrates the parameter
    budget instead of splitting 3-ways (allows d_model 64→96 same budget);
    (b) cross-subband mixing at every TFBlock (pointwise conv) not just at
    AdaptiveFusion; (c) TF-Mamba (reference paper) also feeds subbands into a
    shared stream. PerceptionNet late-fusion advantage applies to distinct
    MODALITIES, not subbands of the same signal.
    arch='branch' preserves exact v1 behaviour (incl. AdaptiveFusion).

C2  TemporalDownsample ×2 (learnable Conv2d stride=2) placed AFTER 2 TFBlocks:
    CNN encodes high-frequency content into channels first, then decimates
    ("encode-then-decimate"). Reduces T 500→250, cutting Mamba compute ~2×,
    freeing budget for wider d_model.

C3  d_state 32→16 (Mamba-1 default; TF-Mamba / Vim both use 16). Saves params
    without accuracy loss at this scale.

C4  ffn_ratio=2 ON by default (v1 code already exists, fc2 zero-init). Channel-
    mixing after each SSM layer — lesson from hybrid architectures (MambaVision).

C5  d_model 64→96, using budget freed by C1+C3.

C6  FinalAttention: 1 MHSA layer pre-norm (out_proj zero-init → identity at
    init) after Mamba, before pooling — MambaVision: attention in FINAL layers
    improves global token-to-token retrieval that SSM state cannot do.
    Default ON at T=250; the smaller sequence makes it less data-hungry.

C7  ECA (~5 params) after subband concat: lightweight channel gate on the
    (subband × stem channel) axis — ECA-Net style, safe and cheap.

C8  AttnStatPool upgraded to full ECAPA context-aware form: scores see
    [x ‖ μ ‖ σ] global stats; std uses clamp instead of +eps. Zero-init →
    uniform attention at init (identical to plain statistics pooling at step 0).

Preserved from v1 (already correct, literature-grounded):
    • Per-subband stems with physically-motivated kernels LL(7,5)/HL(3,7)/LH(7,3)
    • TFBlock axial-depthwise pre-norm + DropPath
    • BiMamba bidirectional, per-channel zero-init gate (PTM-Mamba style)
    • Zero-init philosophy on all new branches (safe at initialisation)
    • GN in CNN / RMSNorm in SSM; no PE (Mamba encodes order via recurrence)
    • Flatten (C×F) keeps frequency as features (TF-Mamba time-stream spirit)
    • Classifier LN → Dropout → Linear

To reproduce v1 exactly:
    arch='branch', d_model=64, d_state=32, ffn_ratio=0,
    dp_mamba=(0.0, 0.10), use_final_attn=False, use_eca=False

Default param estimate: ~0.57M — comparable to v1 (~0.58M) but Mamba compute
    is ~3-4× lower (1 trunk @ T=250 vs 3 branches @ T=500).

NOTE — verify DWT band naming before training:
    pywt returns (cA, (cH, cV, cD)). Run a high-frequency sine along the TIME
    axis to confirm which band is "high-pass in time", then map correctly to the
    [LL|HL|LH] layout used by the bench array. If swapped, just exchange the HL
    and LH stem kernels.
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


_SUBBAND_ORDER  = ('LL', 'HL', 'LH')
_SUBBAND_KERNEL = {'LL': (7, 5), 'HL': (3, 7), 'LH': (7, 3)}
#   LL (7,5) slow envelope · HL (3,7) temporal burst onsets · LH (7,3) Doppler


def _gn_groups(d: int) -> int:
    """GroupNorm groups targeting ~8 channels per group, guaranteed to divide d."""
    for g in range(max(1, d // 8), 0, -1):
        if d % g == 0:
            return g
    return 1


def _sincos_pe(length: int, d_model: int) -> torch.Tensor:
    """Sinusoidal absolute PE [Vaswani 2017] — (1, length, d_model)."""
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
        keep  = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )


# ─── CNN: per-subband stem + axial-depthwise block ────────────────────────────

class SubbandStem(nn.Module):
    """Conv2d(in_ch → d_stem) + GroupNorm + SiLU, one per subband."""

    def __init__(self, in_ch: int, d_stem: int = 16, kernel=(5, 5),
                 temporal_stride: int = 1):
        super().__init__()
        kt, kf = kernel
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d_stem, (kt, kf),
                      stride=(temporal_stride, 1),
                      padding=(kt // 2, kf // 2)),
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


class ECA(nn.Module):
    """[C7] Efficient Channel Attention (ECA-Net, CVPR 2020) — ~5-param gate.

    GAP over (T, F) → Conv1d(k) along the channel axis → sigmoid → scale.
    At init the gate is ≈0.5 everywhere, a pure scale absorbed by GroupNorm,
    so the initial behaviour matches the un-gated model.

    Input / Output: (B, C, T, F).
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=(2, 3))                                    # (B, C)
        y = torch.sigmoid(self.conv(y.unsqueeze(1))).squeeze(1)   # (B, C)
        return x * y.unsqueeze(-1).unsqueeze(-1)


class TemporalDownsample(nn.Module):
    """[C2] Learnable temporal downsampling + channel projection.

    stride=2: Conv2d kernel=(4,1), stride=(2,1), pad=(1,0) → T → T/2 (even T)
    stride=1: Conv2d kernel=(3,1), stride=(1,1), pad=(1,0) → channel proj only
    Followed by GroupNorm + SiLU.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        if stride == 2:
            conv = nn.Conv2d(in_ch, out_ch, (4, 1), stride=(2, 1), padding=(1, 0))
        elif stride == 1:
            conv = nn.Conv2d(in_ch, out_ch, (3, 1), stride=(1, 1), padding=(1, 0))
        else:
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        self.net = nn.Sequential(
            conv,
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── BiMamba: bidirectional, per-channel zero-init gate, optional FFN ─────────

class BiMambaLayer(nn.Module):
    """One (bi)directional Mamba layer with optional FFN sub-block.

    SSM sub-block (always present):
        h = RMSNorm(x)
        f = Mamba_fwd(h);  b = flip(Mamba_bwd(flip(h)))
        g = σ(W·[f ‖ b] + c)   W=0,c=0 at init → g≡0.5 → y=(f+b)/2
        x = x + DropPath(g·f + (1−g)·b)

    FFN sub-block (ffn_ratio > 0):
        h = RMSNorm(x)
        x = x + DropPath(fc2(SiLU(fc1(h))))   fc2 zero-init → identity at init

    bidirectional=False: unidirectional SSM (no gate), for ablation.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, drop_path: float = 0.0,
                 bidirectional: bool = True, ffn_ratio: int = 2):
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
            nn.init.zeros_(self.gate.weight)
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
            nn.init.zeros_(self.ffn_fc2.weight)
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
            g = torch.sigmoid(self.gate(torch.cat([f, b], dim=-1)))
            y = g * f + (1.0 - g) * b
        x = x + self.dp(y)
        if self.ffn_norm is not None:
            h = self.ffn_norm(x)
            x = x + self.ffn_dp(self.ffn_fc2(self.ffn_act(self.ffn_fc1(h))))
        return x


class BiMamba(nn.Module):
    """Stack of gated (bi)directional Mamba layers + final RMSNorm."""

    def __init__(self, d_model: int, n_layers: int = 2, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 drop_path_rates=(0.05, 0.10), bidirectional: bool = True,
                 ffn_ratio: int = 2):
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


# ─── [C6] Final MHSA layer (MambaVision-style) ────────────────────────────────

class FinalAttention(nn.Module):
    """One pre-norm MHSA layer, out_proj zero-init → identity at step 0.

    Placed AFTER Mamba stack, before pooling — MambaVision finding: attention
    in the FINAL layers adds exact token-to-token retrieval that a fixed-size
    SSM state cannot do. No positional encoding needed: input already carries
    position implicitly from the recurrence.
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 attn_drop: float = 0.1, drop_path: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.norm = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=attn_drop, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        return x + self.dp(y)


# ─── Optional nonlinear frequency mixing (MLP-Mixer over subcarriers) ─────────

class FreqMix(nn.Module):
    """Nonlinear mixing across the subcarrier axis F, residual, fc2 zero-init.

    Input / Output: (B, C, T, F).
    """

    def __init__(self, f2: int, hidden: int = None, drop_path: float = 0.0):
        super().__init__()
        hidden = hidden or max(8, f2 * 2)
        self.norm = nn.LayerNorm(f2)
        self.fc1  = nn.Linear(f2, hidden)
        self.act  = nn.SiLU()
        self.fc2  = nn.Linear(hidden, f2)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.dp   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(self.act(self.fc1(self.norm(x))))
        return x + self.dp(y)


# ─── [C1] Trunk backbone (new, default) ───────────────────────────────────────

class TrunkBackbone(nn.Module):
    """[ECA] → TFBlock×2 → TemporalDownsample(T/2, in_ch→d_mid) → TFBlock_post
       → [FreqMix] → flatten(C×F) → Linear embed [+PE] → BiMamba.

    Input : (B, in_ch, T2, F2)    Output: (B, T2', d_model)
    where T2' = T2//2 if downsample else T2.
    """

    def __init__(self, in_ch: int, f2: int, d_mid: int = 64, d_model: int = 96,
                 dp_cnn: tuple = (0.0, 0.05), dilations: tuple = (1, 2),
                 post_dilation: int = 2, post_drop_path: float = 0.05,
                 downsample: bool = True, use_post: bool = True,
                 n_mamba_layers: int = 2, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 dp_mamba=(0.05, 0.10), bidirectional: bool = True,
                 use_pos_emb: bool = False, freq_mix: str = None,
                 embed_drop: float = 0.1, t_max: int = 250,
                 embed_hidden: int = None, ffn_ratio: int = 2,
                 use_eca: bool = True):
        super().__init__()
        if freq_mix not in (None, 'mlp'):
            raise ValueError(f"freq_mix must be None or 'mlp', got {freq_mix!r}")
        if len(dp_cnn) != len(dilations):
            raise ValueError(
                f"len(dp_cnn)={len(dp_cnn)} must equal len(dilations)={len(dilations)}"
            )
        self.eca    = ECA() if use_eca else None
        self.blocks = nn.ModuleList([
            TFBlock(in_ch, dilation=dilations[i], drop_path=dp_cnn[i])
            for i in range(len(dilations))
        ])
        self.down  = TemporalDownsample(in_ch, d_mid,
                                        stride=2 if downsample else 1)
        self.post  = (TFBlock(d_mid, dilation=post_dilation,
                              drop_path=post_drop_path)
                      if use_post else None)
        self.freq_mix = FreqMix(f2) if freq_mix == 'mlp' else None

        in_dim = d_mid * f2
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
        pe = _sincos_pe(T, x.size(-1)).to(device=x.device, dtype=x.dtype)
        return x + pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.eca is not None:
            x = self.eca(x)                              # [C7] channel gate
        for blk in self.blocks:
            x = blk(x)                                   # (B, in_ch, T2, F2)
        x = self.down(x)                                 # (B, d_mid, T2', F2)
        if self.post is not None:
            x = self.post(x)                             # refine at lower resolution
        if self.freq_mix is not None:
            x = self.freq_mix(x)
        B, C, T, Fd = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fd)  # keep freq as features
        x = self.embed(x)                                # (B, T2', d_model)
        x = self._add_pos_emb(x)
        return self.mamba(x)                             # (B, T2', d_model)


# ─── Branch backbone (legacy v1, kept for ablation comparison) ────────────────

class BranchBackbone(nn.Module):
    """TFBlock×2 → flatten(C×F) → Linear embed [+PE] → BiMamba (v1 layout).

    Input : (B, d_stem, T2, F2)    Output: (B, T2, d_model)
    """

    def __init__(self, d_stem: int, f2: int, d_model: int = 64,
                 dp_cnn: tuple = (0.0, 0.05), dilations: tuple = (1, 2),
                 n_mamba_layers: int = 2, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 dp_mamba=(0.05, 0.10), bidirectional: bool = True,
                 use_pos_emb: bool = False, freq_mix: str = None,
                 embed_drop: float = 0.1, t_max: int = 500,
                 embed_hidden: int = None, ffn_ratio: int = 2):
        super().__init__()
        if freq_mix not in (None, 'mlp'):
            raise ValueError(f"freq_mix must be None or 'mlp', got {freq_mix!r}")
        if len(dp_cnn) != len(dilations):
            raise ValueError(
                f"len(dp_cnn)={len(dp_cnn)} must equal len(dilations)={len(dilations)}"
            )
        self.blocks = nn.ModuleList([
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
        pe = _sincos_pe(T, x.size(-1)).to(device=x.device, dtype=x.dtype)
        return x + pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.freq_mix is not None:
            x = self.freq_mix(x)
        B, C, T, Fd = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fd)
        x = self.embed(x)
        x = self._add_pos_emb(x)
        return self.mamba(x)


# ─── Adaptive N-way fusion (branch mode only) ─────────────────────────────────

class AdaptiveFusion(nn.Module):
    """Soft-weighted fusion of N streams, zero-init → uniform at step 0.

        S_cat = concat(S_1 … S_N)                 (B, T, N·d)
        α     = softmax(Linear_{N·d→N}(S_cat))    (B, T, N)
        out   = Σ_i α_i ⊙ S_i                     (B, T, d)
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
        cat = torch.cat(streams, dim=-1)
        w   = self.linear(cat).softmax(dim=-1)
        return sum(w[..., i:i + 1] * streams[i] for i in range(self.n_branches))


# ─── [C8] Context-aware attentive statistics pooling (full ECAPA) ─────────────

class AttnStatPool(nn.Module):
    """Per-channel temporal attention → [weighted mean ‖ weighted std].

    context=True (always on): scores see [x ‖ μ_global ‖ σ_global] — each
    frame is judged RELATIVE to the recording's own baseline, which transfers
    better across subjects/link gains. Zero-init on last score layer → uniform
    attention at step 0 (= plain statistics pooling, safe baseline).
    std uses clamp(min=1e-6) instead of +eps: no gradient spike on near-zero
    variance channels (common early in training with zero-init residuals).

    Input : (B, T, d)    Output: (B, 2*d)
    """

    def __init__(self, dim: int, bn: int = None):
        super().__init__()
        bn = bn or max(8, dim // 2)
        self.score = nn.Sequential(
            nn.Linear(3 * dim, bn), nn.Tanh(), nn.Linear(bn, dim)
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)                          # (B, 1, d)
        sg = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-6).sqrt()
        h  = torch.cat([x, mu.expand_as(x), sg.expand_as(x)], dim=-1)
        w    = self.score(h).softmax(dim=1)                       # (B, T, d)
        mean = (w * x).sum(dim=1)                                 # (B, d)
        var  = (w * (x - mean.unsqueeze(1)).pow(2)).sum(dim=1)    # (B, d)
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

class WavDualMambaV2(nn.Module):
    """CNN + gated-BiMamba for WiFi-CSI HAR, two architecture modes.

    arch='trunk' (default, proposed):
        stems/subband → concat channels → [ECA] → TFBlock×2
        → TemporalDownsample(T/2) → TFBlock_post → flatten → embed
        → BiMamba → [FinalAttention] → AttnStatPool → Classifier

    arch='branch' (legacy v1, for ablation):
        stems/subband → backbone/subband [or shared] → AdaptiveFusion
        → [FinalAttention] → AttnStatPool → Classifier

    Args:
        num_classes     : output classes (default 11).
        n_links         : Wi-Fi receivers M (default 3).
        n_antennas      : antennas per receiver A (default 3).
        subbands        : subset of ('LL','HL','LH') to use (default all three).
        arch            : 'trunk' (default) or 'branch' (legacy v1).
        d_stem          : per-subband stem width (default 16).
        d_mid           : trunk intermediate width after downsample (default 64).
        d_model         : Mamba feature width (default 96).              [C5]
        d_state         : Mamba SSM state size (default 16).             [C3]
        n_mamba_layers  : BiMamba layers (default 2).
        f2              : subcarrier axis length after DWT (default 15).
        dp_cnn          : DropPath per TFBlock (default (0.0, 0.05)).
        dilations       : dilation per TFBlock (default (1, 2)).
        post_dilation   : dilation of the post-downsample TFBlock (default 2).
        post_drop_path  : DropPath of the post-downsample TFBlock (default 0.05).
        dp_mamba        : DropPath per BiMamba layer (default (0.05, 0.10)).
        embed_drop      : Dropout after embed Linear (default 0.1).
        temporal_stride : stride in stem conv time axis (default 1).
        downsample      : apply TemporalDownsample T/2 in trunk (default True). [C2]
        use_post        : apply post-downsample TFBlock in trunk (default True).
        bidirectional   : gated backward Mamba branch (default True).
        use_pos_emb     : sinusoidal PE (default False).
        share_branches  : tie branch weights in branch mode (default False).
        freq_mix        : None | 'mlp' — nonlinear subcarrier mix (default None).
        expand          : Mamba inner-expansion (default 2).
        d_conv          : Mamba local conv width (default 4).
        ffn_ratio       : FFN sub-block ratio after each Mamba layer (default 2). [C4]
        use_final_attn  : MHSA layer after Mamba (default True).         [C6]
        attn_heads      : number of attention heads (default 4).
        attn_drop       : attention dropout (default 0.1).
        attn_drop_path  : DropPath for FinalAttention (default 0.1).
        use_eca         : ECA channel gate after subband concat (default True). [C7]
        dropout         : classifier dropout (default 0.2).
        t_max           : PE buffer length (default 500 = T2 before downsample).
        embed_hidden    : two-stage embed hidden dim (default None = single stage).

    To reproduce v1 exactly:
        arch='branch', d_model=64, d_state=32, ffn_ratio=0,
        dp_mamba=(0.0, 0.10), use_final_attn=False, use_eca=False

    Input  : X (B, 27, T2, F2), subband-major [LL | HL | LH] (27 = 3·M·A).
    Output : logits (B, num_classes).
    """

    def __init__(
        self,
        num_classes: int = 11,
        n_links: int = 3,
        n_antennas: int = 3,
        subbands=('LL', 'HL', 'LH'),
        arch: str = 'trunk',
        d_stem: int = 16,
        d_mid: int = 64,
        d_model: int = 96,
        d_state: int = 16,
        n_mamba_layers: int = 2,
        f2: int = 15,
        dp_cnn: tuple = (0.0, 0.05),
        dilations: tuple = (1, 2),
        post_dilation: int = 2,
        post_drop_path: float = 0.05,
        dp_mamba=(0.05, 0.10),
        embed_drop: float = 0.1,
        temporal_stride: int = 1,
        downsample: bool = True,
        use_post: bool = True,
        bidirectional: bool = True,
        use_pos_emb: bool = False,
        share_branches: bool = False,
        freq_mix: str = None,
        expand: int = 2,
        d_conv: int = 4,
        ffn_ratio: int = 2,
        use_final_attn: bool = True,
        attn_heads: int = 4,
        attn_drop: float = 0.1,
        attn_drop_path: float = 0.1,
        use_eca: bool = True,
        dropout: float = 0.2,
        t_max: int = 500,
        embed_hidden: int = None,
    ):
        super().__init__()
        if arch not in ('trunk', 'branch'):
            raise ValueError(f"arch must be 'trunk' or 'branch', got {arch!r}")
        if len(dp_mamba) != n_mamba_layers:
            raise ValueError("len(dp_mamba) must equal n_mamba_layers")
        if arch == 'branch' and (use_eca or downsample or not use_post):
            import warnings
            flags = ([f for f, v in (('use_eca', use_eca), ('downsample', downsample)) if v]
                     + (['use_post=False'] if not use_post else []))
            warnings.warn(
                f"arch='branch': {flags} are ignored (trunk-only flags). "
                "Use arch='trunk' to enable them.", UserWarning, stacklevel=2)

        sel     = [s for s in _SUBBAND_ORDER if s in subbands]
        unknown = [s for s in subbands if s not in _SUBBAND_ORDER]
        if unknown:
            raise ValueError(f"Unknown subband(s) {unknown}; choose from {_SUBBAND_ORDER}")
        if len(sel) < 1:
            raise ValueError("subbands must select at least one of ('LL','HL','LH')")
        self.subbands   = tuple(sel)
        self.n_branches = len(sel)
        self.arch       = arch

        self.f2        = f2
        self.n_per_sub = n_links * n_antennas               # = 9
        self._c_in     = self.n_per_sub * len(_SUBBAND_ORDER)  # = 27
        self._sb_index = {s: _SUBBAND_ORDER.index(s) for s in self.subbands}

        self.temporal_stride = temporal_stride
        self.share_branches  = share_branches
        self.stems = nn.ModuleDict({
            s: SubbandStem(self.n_per_sub, d_stem, kernel=_SUBBAND_KERNEL[s],
                           temporal_stride=temporal_stride)
            for s in self.subbands
        })

        t_after_stem = t_max // max(temporal_stride, 1)

        if arch == 'trunk':
            in_ch = self.n_branches * d_stem
            t_eff = t_after_stem // (2 if downsample else 1)
            self.trunk = TrunkBackbone(
                in_ch=in_ch, f2=f2, d_mid=d_mid, d_model=d_model,
                dp_cnn=dp_cnn, dilations=dilations,
                post_dilation=post_dilation, post_drop_path=post_drop_path,
                downsample=downsample, use_post=use_post,
                n_mamba_layers=n_mamba_layers, d_state=d_state,
                d_conv=d_conv, expand=expand, dp_mamba=dp_mamba,
                bidirectional=bidirectional, use_pos_emb=use_pos_emb,
                freq_mix=freq_mix, embed_drop=embed_drop, t_max=t_eff,
                embed_hidden=embed_hidden, ffn_ratio=ffn_ratio,
                use_eca=use_eca,
            )
            self.shared_backbone = None
            self.backbones       = None
            self.fusion          = None
        else:
            bb_kwargs = dict(
                d_stem=d_stem, f2=f2, d_model=d_model,
                dp_cnn=dp_cnn, dilations=dilations,
                n_mamba_layers=n_mamba_layers, d_state=d_state,
                d_conv=d_conv, expand=expand, dp_mamba=dp_mamba,
                bidirectional=bidirectional, use_pos_emb=use_pos_emb,
                freq_mix=freq_mix, embed_drop=embed_drop,
                t_max=t_after_stem,
                embed_hidden=embed_hidden, ffn_ratio=ffn_ratio,
            )
            self.trunk = None
            if share_branches:
                self.shared_backbone = BranchBackbone(**bb_kwargs)
                self.backbones       = None
            else:
                self.shared_backbone = None
                self.backbones       = nn.ModuleDict({
                    s: BranchBackbone(**bb_kwargs) for s in self.subbands
                })
            self.fusion = AdaptiveFusion(d_model, self.n_branches)

        self.final_attn = (FinalAttention(d_model, n_heads=attn_heads,
                                          attn_drop=attn_drop,
                                          drop_path=attn_drop_path)
                           if use_final_attn else None)
        self.tpool = AttnStatPool(d_model)
        self.head  = Classifier(2 * d_model, num_classes=num_classes,
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
                f"got {X.shape[-1]}. Rebuild the model with f2={X.shape[-1]}."
            )

        stem_outs = []
        for s in self.subbands:
            i  = self._sb_index[s]
            sb = X[:, i * self.n_per_sub:(i + 1) * self.n_per_sub]
            stem_outs.append(self.stems[s](sb))              # (B, d_stem, T2, F2)

        if self.arch == 'trunk':
            h = torch.cat(stem_outs, dim=1)                  # (B, n*d_stem, T2, F2)
            z = self.trunk(h)                                # (B, T2', d_model)
        else:
            if self.share_branches:
                h       = torch.cat(stem_outs, dim=0)
                out     = self.shared_backbone(h)
                streams = list(out.chunk(self.n_branches, dim=0))
            else:
                streams = [self.backbones[s](stem_outs[k])
                           for k, s in enumerate(self.subbands)]
            z = self.fusion(streams)                         # (B, T2, d_model)

        if self.final_attn is not None:
            z = self.final_attn(z)                           # (B, T', d_model)
        z = self.tpool(z)                                    # (B, 2*d_model)
        return self.head(z)                                  # (B, num_classes)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
