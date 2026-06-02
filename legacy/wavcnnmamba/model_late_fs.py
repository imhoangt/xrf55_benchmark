"""
wavmamba_har.py — WavMambaHAR
=============================

WiFi-CSI human activity recognition on XRF55 (default 11 classes), built as a
LATE-FUSION model over DWT subbands with a SHARED (Siamese) CNN + bidirectional
Mamba backbone. ~0.68M params in the default config.

Design (finalized)
-------------------
Preprocessing (done in the data pipeline, NOT here):
    Per antenna-pair (M*A), 1-level 2-D DWT on the (time x subcarrier) plane,
    keep LL | HL | LH (drop HH). Antennas stay as CHANNELS, frequency stays a
    spatial axis. Channel layout fed to the model is SUBBAND-MAJOR:
        [ (M*A) chans LL | (M*A) chans HL | (M*A) chans LH ]
    Input tensor: X (B, C_in, T2, F2),  C_in = n_subbands * (M*A).
    Default: (B, 27, 500, 15)  (T_raw=1000 -> T2=500, F_raw=30 -> F2=15).

Per-subband path (LATE fusion):
    subband_s (B, M*A, T2, F2)
      -> Stem_s        per-subband conv (LL/HL/LH kernels)   (B, 32, T2, F2)   [NOT shared]
      |---- shared backbone (same weights for every subband) ----------------|
      -> 1x1 proj+GN   32 -> d_model                          (B, 128, T2, F2)
      -> TFBlock x3    depthwise axial (dil 1/2/4) + SE        (B, 128, T2, F2)
      -> FreqAttnPool  1-head, collapse F                      (B, T2, 128)
      -> [PosEmb]      learned, optional (default OFF)         (B, T2, 128)
      -> BiMamba x2    fwd+bwd, no FFN, d_state=16             (B, T2, 128)
      -> AttnStatPool  ECAPA mean||std over time               (B, 256)
    => per-subband vector v_s (B, 256)

Fusion + head:
    concat[v_s] (B, 256 * n_subbands) -> LN -> MLP -> logits (B, num_classes)

`fusion="early"` gives the strong baseline (Paper-1 style): all subband channels
go through a SINGLE stem + the same backbone (no per-subband split), head input
is (B, 256). Use it to verify whether late fusion actually helps.

Ablation flags: fusion {late|early}, bidirectional {True|False},
use_pos_emb {True|False}, n_subbands (+ matching channels), n_mamba_layers,
d_state, stem_kernels, dropout.

Training notes (not in this file): cross-subject split is the headline metric;
report accuracy + macro-F1; use label_smoothing~0.1 and weight_decay~0.02-0.05.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    _MAMBA_IMPORT_ERROR = None
except Exception as e:                                   # pragma: no cover
    Mamba = None
    HAS_MAMBA = False
    _MAMBA_IMPORT_ERROR = e


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


# ─── Squeeze-and-Excite channel gate (best attention per ActivityMamba) ───────

class SqueezeExcite(nn.Module):
    """SE channel recalibration for 2-D feature maps (B, d, T, F)."""

    def __init__(self, d: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, d // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, d), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))                        # (B, d)
        return x * w.unsqueeze(-1).unsqueeze(-1)


# ─── CNN block: pre-norm axial depthwise + SE (residual) ──────────────────────

class TFBlock(nn.Module):
    """Parallel axial depthwise mixing along time (dilated) and frequency,
    a pointwise channel projection, an SE gate, and a residual connection.

    Dilation schedule [1, 2, 4] across the 3 blocks gives compound temporal
    receptive fields of 7 -> 19 -> 43 steps (k_t=7) — local context before
    BiMamba models long-range temporal dependencies.
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
        self.act = nn.SiLU()
        self.pw = nn.Conv2d(d, d, 1)
        self.se = SqueezeExcite(d, se_reduction)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.dw_t(y) + self.dw_f(y)                  # parallel axial mixing
        y = self.pw(self.act(y))                         # channel projection
        y = self.se(y)                                   # SE recalibration
        return x + self.dp(y)


# ─── Frequency attention pooling (collapse the subcarrier axis) ───────────────

class FreqAttnPool(nn.Module):
    """Content attention pool over the frequency axis F.

    Default n_heads=1: a single Linear(d, 1) scores each (T, F) position,
    softmax over F, weighted sum -> (B, T, d). For F2=15 one head is plenty.
    n_heads>1 adds disjoint head groups + an identity-initialised out_proj.

    Input : (B, d, T, F)    Output: (B, T, d)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) % n_heads ({n_heads}) != 0")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.score = nn.Linear(d_model, n_heads)
        if n_heads > 1:
            self.out_proj = nn.Linear(d_model, d_model)
            nn.init.eye_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:
            self.out_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, Fd = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()           # (B, T, F, C)
        w = self.score(x).softmax(dim=2)                 # (B, T, F, H)
        xh = x.view(B, T, Fd, self.n_heads, self.head_dim)
        out = (w.unsqueeze(-1) * xh).sum(dim=2).reshape(B, T, C)
        return self.out_proj(out)


# ─── Bidirectional Mamba (no FFN) ─────────────────────────────────────────────

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
    """One (bi)directional Mamba layer — Vim-style sum fusion, NO FFN.

        h   = RMSNorm(x)
        y   = Mamba_fwd(h) [ + flip(Mamba_bwd(flip(h))) if bidirectional ]
        out = x + DropPath(y)
    """

    def __init__(self, d_model: int = 128, d_state: int = 16, d_conv: int = 4,
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
        self.fwd = Mamba(d_model=d_model, d_state=d_state,
                         d_conv=d_conv, expand=expand)
        self.bwd = (Mamba(d_model=d_model, d_state=d_state,
                          d_conv=d_conv, expand=expand)
                    if bidirectional else None)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y = self.fwd(h)
        if self.bwd is not None:
            y = y + self.bwd(h.flip(1)).flip(1)
        return x + self.dp(y)


class BiMamba(nn.Module):
    """Stack of (bi)directional Mamba layers + final RMSNorm."""

    def __init__(self, d_model: int = 128, n_layers: int = 2, d_state: int = 16,
                 drop_path_rates=(0.0, 0.05), bidirectional: bool = True):
        super().__init__()
        if len(drop_path_rates) != n_layers:
            raise ValueError(
                f"len(drop_path_rates)={len(drop_path_rates)} != "
                f"n_layers={n_layers}"
            )
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model, d_state=d_state, drop_path=drop_path_rates[i],
                         bidirectional=bidirectional)
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─── Temporal attentive statistics pooling (ECAPA-style) ──────────────────────

class AttnStatPool(nn.Module):
    """Per-channel temporal attention -> [weighted mean || weighted std].

    Input : (B, T, d)    Output: (B, 2*d)
    """

    def __init__(self, dim: int = 128, bn: int = 32):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, bn), nn.Tanh(), nn.Linear(bn, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.score(x).softmax(dim=1)                 # (B, T, d)
        mean = (w * x).sum(dim=1)                        # (B, d)
        var = (w * (x - mean.unsqueeze(1)).pow(2)).sum(dim=1)
        return torch.cat([mean, (var + 1e-6).sqrt()], dim=-1)


# ─── Per-subband stem (only NON-shared part) ──────────────────────────────────

class SubbandStem(nn.Module):
    """Conv2d(in_ch -> d_stem) + GroupNorm + SiLU. One stem per subband, with a
    physically-motivated kernel per subband (LL/HL/LH)."""

    def __init__(self, in_ch: int, d_stem: int = 32, kernel=(5, 5)):
        super().__init__()
        kt, kf = kernel
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d_stem, (kt, kf), padding=(kt // 2, kf // 2)),
            nn.GroupNorm(8, d_stem),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Shared backbone (applied to each subband; weights tied) ──────────────────

class SharedBackbone(nn.Module):
    """proj -> TFBlock x N -> FreqAttnPool -> [PosEmb] -> BiMamba -> AttnStatPool.

    Input : (B, d_stem, T2, F2)    Output: (B, 2*d_model)
    """

    def __init__(self, d_stem: int = 32, d_model: int = 128,
                 n_tfblocks: int = 3, dilations=(1, 2, 4),
                 dp_cnn=(0.0, 0.05, 0.10), se_reduction: int = 4,
                 n_freq_heads: int = 1, n_mamba_layers: int = 2,
                 d_state: int = 16, dp_mamba=(0.0, 0.05),
                 bidirectional: bool = True, use_pos_emb: bool = False,
                 t_max: int = 500):
        super().__init__()
        if not (len(dilations) == len(dp_cnn) == n_tfblocks):
            raise ValueError("dilations, dp_cnn must both have length n_tfblocks")

        self.proj = nn.Sequential(
            nn.Conv2d(d_stem, d_model, 1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            TFBlock(d_model, dilation=dilations[i], drop_path=dp_cnn[i],
                    se_reduction=se_reduction)
            for i in range(n_tfblocks)
        ])
        self.fpool = FreqAttnPool(d_model, n_heads=n_freq_heads)

        self.use_pos_emb = use_pos_emb
        self.t_max = t_max
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, t_max, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        else:
            self.register_parameter("pos_emb", None)

        self.mamba = BiMamba(d_model, n_layers=n_mamba_layers, d_state=d_state,
                             drop_path_rates=dp_mamba, bidirectional=bidirectional)
        self.tpool = AttnStatPool(d_model)
        self.out_dim = 2 * d_model

    def _add_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_emb is None:
            return x
        T = x.size(1)
        if T <= self.t_max:
            return x + self.pos_emb[:, :T]
        pe = F.interpolate(self.pos_emb.transpose(1, 2), size=T,
                           mode="linear", align_corners=False).transpose(1, 2)
        return x + pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                                 # (B, d_model, T2, F2)
        for blk in self.blocks:
            x = blk(x)
        x = self.fpool(x)                                # (B, T2, d_model)
        x = self._add_pos_emb(x)
        x = self.mamba(x)                                # (B, T2, d_model)
        return self.tpool(x)                             # (B, 2*d_model)


# ─── Classifier head ──────────────────────────────────────────────────────────

class Classifier(nn.Module):
    """LayerNorm -> Linear -> SiLU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 11,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─── Full model ───────────────────────────────────────────────────────────────

class WavMambaHAR(nn.Module):
    """Late-fusion (default) WiFi-CSI HAR model with a shared CNN+BiMamba backbone.

    Args:
        num_classes    : output classes (default 11).
        n_links        : Wi-Fi receivers M (default 3).
        n_antennas     : antennas per receiver A (default 3).
        n_subbands     : DWT subbands kept (default 3 = LL|HL|LH).
        fusion         : "late" (per-subband shared backbone + concat) or
                         "early" (all subband channels -> one backbone; baseline).
        d_model        : backbone feature width (default 128, must be %8==0).
        d_stem         : stem width (default 32, must be %8==0).
        d_state        : Mamba SSM state size (default 16; try 32 if underfitting).
        n_mamba_layers : BiMamba layers (default 2; 1 = leaner ~0.45M).
        n_freq_heads   : FreqAttnPool heads (default 1).
        se_reduction   : SE bottleneck reduction in TFBlocks (default 4).
        dp_cnn         : DropPath rates for the 3 TFBlocks.
        dp_mamba       : DropPath rates for the BiMamba layers (len == n_mamba_layers).
        bidirectional  : use backward Mamba branch (default True; ablate to False).
        use_pos_emb    : learned positional embedding (default False).
        stem_kernels   : per-subband (k_t, k_f) tuples; default LL/HL/LH kernels
                         for n_subbands==3, else (5,5) each. For 2-subband ablations
                         pass the kernels of the subbands you actually feed.
        head_hidden    : hidden width of the classifier MLP (default 128).
        dropout        : classifier dropout (default 0.3).
        t_max          : positional-embedding buffer length (default 500 = T2).

    Input  : X (B, C_in, T2, F2), subband-major channels
             [ (M*A) LL | (M*A) HL | (M*A) LH ], C_in = n_subbands * M*A.
    Output : logits (B, num_classes).
    """

    _CANONICAL_3 = [(7, 5), (3, 7), (7, 3)]              # LL | HL | LH

    def __init__(
        self,
        num_classes: int = 11,
        n_links: int = 3,
        n_antennas: int = 3,
        n_subbands: int = 3,
        fusion: str = "late",
        d_model: int = 128,
        d_stem: int = 32,
        d_state: int = 16,
        n_mamba_layers: int = 2,
        n_freq_heads: int = 1,
        se_reduction: int = 4,
        dp_cnn=(0.0, 0.05, 0.10),
        dp_mamba=(0.0, 0.05),
        bidirectional: bool = True,
        use_pos_emb: bool = False,
        stem_kernels=None,
        head_hidden: int = 128,
        dropout: float = 0.3,
        t_max: int = 500,
    ):
        super().__init__()
        if fusion not in ("late", "early"):
            raise ValueError(f"fusion must be 'late' or 'early', got {fusion!r}")
        if d_model % 8 != 0 or d_stem % 8 != 0:
            raise ValueError("d_model and d_stem must be divisible by 8 (GroupNorm)")
        if len(dp_mamba) != n_mamba_layers:
            raise ValueError("len(dp_mamba) must equal n_mamba_layers")

        self.fusion = fusion
        self.n_per_sub = n_links * n_antennas
        self.n_subbands = n_subbands
        self._c_in = self.n_per_sub * n_subbands

        # Per-subband stem kernels
        if stem_kernels is None:
            stem_kernels = (list(self._CANONICAL_3) if n_subbands == 3
                            else [(5, 5)] * n_subbands)
        if len(stem_kernels) != n_subbands:
            raise ValueError(
                f"stem_kernels has {len(stem_kernels)} entries but "
                f"n_subbands={n_subbands}"
            )

        if fusion == "late":
            self.stems = nn.ModuleList([
                SubbandStem(self.n_per_sub, d_stem, kernel=k) for k in stem_kernels
            ])
            self.stem = None
        else:  # early: one stem over all subband channels (baseline)
            self.stem = SubbandStem(self._c_in, d_stem, kernel=(5, 5))
            self.stems = None

        self.backbone = SharedBackbone(
            d_stem=d_stem, d_model=d_model, dp_cnn=dp_cnn, se_reduction=se_reduction,
            n_freq_heads=n_freq_heads, n_mamba_layers=n_mamba_layers,
            d_state=d_state, dp_mamba=dp_mamba, bidirectional=bidirectional,
            use_pos_emb=use_pos_emb, t_max=t_max,
        )

        feat_dim = self.backbone.out_dim * (n_subbands if fusion == "late" else 1)
        self.head = Classifier(feat_dim, hidden=head_hidden,
                               num_classes=num_classes, dropout=dropout)

    @property
    def C_IN(self) -> int:
        """Expected input channels = n_links * n_antennas * n_subbands."""
        return self._c_in

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 4:
            raise ValueError(
                f"Expected 4-D input (B, {self._c_in}, T2, F2), got {tuple(X.shape)}"
            )
        if X.shape[1] != self._c_in:
            raise ValueError(
                f"Expected {self._c_in} channels (= n_links*n_antennas*n_subbands), "
                f"got {X.shape[1]}"
            )

        if self.fusion == "late":
            subs = X.chunk(self.n_subbands, dim=1)        # subband-major split
            vecs = [self.backbone(stem(sb))               # shared weights, per subband
                    for stem, sb in zip(self.stems, subs)]
            z = torch.cat(vecs, dim=-1)                   # (B, 256*n_subbands)
        else:
            z = self.backbone(self.stem(X))               # (B, 256)
        return self.head(z)


# ─── Helpers / self-test ──────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Self-test: requires mamba_ssm (CUDA). Run on a GPU machine.
    if not HAS_MAMBA:
        print("mamba_ssm not available — skipping forward self-test.")
        print(f"(import error: {_MAMBA_IMPORT_ERROR})")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        B, MA, T2, F2 = 2, 9, 500, 15

        print("=== late fusion (LL/HL/LH), default ===")
        m = WavMambaHAR(num_classes=11, n_subbands=3, fusion="late").to(device)
        x = torch.randn(B, m.C_IN, T2, F2, device=device)
        y = m(x)
        print(f"  in {tuple(x.shape)} -> out {tuple(y.shape)} | params {count_params(m):,}")

        print("=== early fusion baseline (Paper-1 style) ===")
        mb = WavMambaHAR(num_classes=11, n_subbands=3, fusion="early").to(device)
        xb = torch.randn(B, mb.C_IN, T2, F2, device=device)
        print(f"  in {tuple(xb.shape)} -> out {tuple(mb(xb).shape)} | params {count_params(mb):,}")

        print("=== ablation: 2 subbands {LL, HL}, unidirectional ===")
        m2 = WavMambaHAR(num_classes=11, n_subbands=2,
                         stem_kernels=[(7, 5), (3, 7)], bidirectional=False).to(device)
        x2 = torch.randn(B, m2.C_IN, T2, F2, device=device)
        print(f"  in {tuple(x2.shape)} -> out {tuple(m2(x2).shape)} | params {count_params(m2):,}")
