"""APWMamba — Amplitude-Phase Wavelet Mamba for WiFi CSI HAR.

Architecture (6 modules):
  M1  SubbandAwareCNNEncoder × 2  (amplitude + phase, DWT subbands LL/HL/LH)
  M2  CrossModalGatedFusion        (bilateral: G_a·amp + G_p·phase, independent gates)
  M3  FreqStatGatedPool             (SE freq-profile recalibration + local attention, F2=15)
  M4  BidirectionalMambaStack      (2 layers BiMamba + SimplifiedBiMambaGate,
                                    d_state=32, RMSNorm at end)
  M5  AttentiveStatPool1D          (ECAPA-style channel-dependent attention → [B, 256])
  M6  Classifier                   (LN → 256→128→11, dropout_in before fc1)

Dimension flow:
  CNN        → [B, 128, T, F2]
  Fusion     → [B, 128, T, F2]
  FreqMean   → [B, T, 128]
  BiMamba    → [B, T, 128]
  AttnPool   → [B, 256]
  Classifier → [B, 11]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    _MAMBA_IMPORT_ERROR = None
except ImportError as e:
    HAS_MAMBA = False
    _MAMBA_IMPORT_ERROR = e


# ─── Stochastic Depth ────────────────────────────────────────────────────────

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep  = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


# ─── M1: Subband-Aware CNN Encoder ───────────────────────────────────────────

class TFBlock(nn.Module):
    """Pre-norm Summed Depthwise Conv block with dilation + LayerScale + DropPath.

    Parallel dw_t (time axis) and dw_f (freq axis) are summed then projected
    with a shared pw(C→C). LayerScale [Fix 3] initialises the residual
    contribution to near zero, preventing early-training instability.
    """

    def __init__(self, d_model: int = 128, k_t: int = 7, k_f: int = 3,
                 time_dilation: int = 1, drop_path: float = 0.0,
                 layer_scale_init: float = 1e-4):
        super().__init__()
        self.norm  = nn.GroupNorm(8, d_model)
        self.dw_t  = nn.Conv2d(d_model, d_model,
                               kernel_size=(k_t, 1),
                               padding=(k_t // 2 * time_dilation, 0),
                               dilation=(time_dilation, 1),
                               groups=d_model)
        self.dw_f  = nn.Conv2d(d_model, d_model,
                               kernel_size=(1, k_f),
                               padding=(0, k_f // 2),
                               groups=d_model)
        self.act   = nn.GELU()
        self.pw    = nn.Conv2d(d_model, d_model, kernel_size=1)
        # [Fix 3] Layer scale: (d_model, 1, 1) broadcasts over (B, C, T, F2)
        self.ls    = nn.Parameter(
            torch.full((d_model, 1, 1), layer_scale_init)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.dw_t(y) + self.dw_f(y)
        y = self.act(y)
        y = self.pw(y)
        # [Fix 3] scale residual before DropPath
        return x + self.drop_path(self.ls * y)


class SubbandAwareCNNEncoder(nn.Module):
    """Per-subband CNN encoder with subband-tailored stem kernels.

    DWT stack order: LL | HL | LH
      LL (slow/smooth):              large temporal + medium freq  (7,5)
      HL (bursty temporal):          small temporal + large freq   (3,7)
      LH (slow temporal, freq-rich): large temporal + small freq   (7,3)
    """

    DEFAULT_STEM_KERNELS = [(7, 5), (3, 7), (7, 3)]

    def __init__(self, n_per_subband: int, d_subband: int = 32,
                 d_model: int = 128, n_tf_blocks: int = 2,
                 dilations=None, drop_path_rates=None,
                 stem_kernels=None, layer_scale_init: float = 1e-4):
        super().__init__()
        if dilations is None:
            dilations = [1] * n_tf_blocks
        if drop_path_rates is None:
            drop_path_rates = [0.0] * n_tf_blocks
        if stem_kernels is None:
            stem_kernels = self.DEFAULT_STEM_KERNELS
        assert len(stem_kernels) == 3, \
            f"stem_kernels must have exactly 3 entries (LL/HL/LH), got {len(stem_kernels)}"
        assert len(dilations) == n_tf_blocks
        assert len(drop_path_rates) == n_tf_blocks

        self.subband_stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_per_subband, d_subband,
                          kernel_size=(kt, kf), padding=(kt // 2, kf // 2)),
                nn.GroupNorm(8, d_subband),
                nn.GELU(),
            )
            for kt, kf in stem_kernels
        ])
        self.joint_stem = nn.Sequential(
            nn.Conv2d(3 * d_subband, d_model, kernel_size=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
        self.tf_blocks = nn.ModuleList([
            TFBlock(d_model, time_dilation=dilations[i],
                    drop_path=drop_path_rates[i],
                    layer_scale_init=layer_scale_init)
            for i in range(n_tf_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll, x_hl, x_lh = x.chunk(3, dim=1)   # DWT order: LL | HL | LH
        f = torch.cat([
            self.subband_stems[0](x_ll),
            self.subband_stems[1](x_hl),
            self.subband_stems[2](x_lh),
        ], dim=1)
        f = self.joint_stem(f)
        for block in self.tf_blocks:
            f = block(f)
        return f


# ─── M2: Cross-modal Bilateral Gated Fusion ──────────────────────────────────

class CrossModalGatedFusion(nn.Module):
    """Bilateral gated fusion with full-resolution gate inputs.

    gate_input = cat[F_amp, F_phase]  (B, 2*d_model, T, F2)
    G_a = sigmoid(Conv1x1(gate_input))
    G_p = sigmoid(Conv1x1(gate_input))
    output = GroupNorm(G_a * F_amp + G_p * F_phase)

    G_a and G_p are independent — can both exceed 0.5 (amplify both)
    or suppress one selectively. Zero-init → G_a = G_p = 0.5 at init.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        gate_in       = 2 * d_model
        self.gate_a   = nn.Conv2d(gate_in, d_model, kernel_size=1)
        self.gate_p   = nn.Conv2d(gate_in, d_model, kernel_size=1)
        self.norm_out = nn.GroupNorm(8, d_model)

    def forward(self, F_amp: torch.Tensor,
                F_phase: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([F_amp, F_phase], dim=1)
        G_a = torch.sigmoid(self.gate_a(gate_input))
        G_p = torch.sigmoid(self.gate_p(gate_input))
        return self.norm_out(G_a * F_amp + G_p * F_phase)


# ─── M3: Frequency Statistics-Gated Pooling ──────────────────────────────────

class FreqStatGatedPool(nn.Module):
    """Frequency Statistics-Gated Pooling: collapses F2 → [B, T, C].

    Step 1  mean + std over F2       → global frequency profile   (B, T, 2C)
    Step 2  SE-style MLP + sigmoid   → per-channel scale          (B, T, C)
    Step 3  x * scale                → recalibrated features      (B, T, F2, C)
    Step 4  content-local attention  → softmax weights over F2    (B, T, F2, 1)
    Step 5  weighted sum             → output                     (B, T, C)

    [Fix 1] OLD: var().clamp(min=1e-6).sqrt()
              → clamp on variance (x²) gives std_floor = sqrt(1e-6) = 1e-3
    NEW: var().clamp(min=1e-12).sqrt()
              → std_floor = sqrt(1e-12) = 1e-6 as intended

    Input:  (B, C, T, F2)
    Output: (B, T, C)
    """

    def __init__(self, d_model: int = 128, reduction: int = 8):
        super().__init__()
        r = d_model // reduction
        self.stats_fc1   = nn.Linear(2 * d_model, r)
        self.stats_fc2   = nn.Linear(r, d_model)
        self.local_score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F2)
        x_t = x.permute(0, 2, 3, 1).contiguous()            # (B, T, F2, C)

        # Step 1: global frequency profile
        f_mean = x_t.mean(dim=2)                             # (B, T, C)
        # [Fix 1] clamp on variance, not std
        f_std  = x_t.var(dim=2, unbiased=False).clamp(min=1e-12).sqrt()
        stats  = torch.cat([f_mean, f_std], dim=-1)          # (B, T, 2C)

        # Step 2: SE-style channel recalibration
        scale = self.stats_fc2(
            F.gelu(self.stats_fc1(stats))
        ).sigmoid()                                          # (B, T, C)

        # Step 3: apply scale
        x_cal = x_t * scale.unsqueeze(2)                    # (B, T, F2, C)

        # Step 4-5: content-local attention + weighted sum
        w = self.local_score(x_cal).softmax(dim=2)          # (B, T, F2, 1)
        return (w * x_cal).sum(dim=2)                        # (B, T, C)


# ─── M4: Bidirectional Mamba ──────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )


class SimplifiedBiMambaGate(nn.Module):
    """Token-wise fwd/bwd gate: content projection + learnable position prior.

    Convex combination: a·fwd + (1−a)·bwd  (magnitude preserved).
    Content: Linear(2*dim → 1) per token.
    Position prior: α·t_norm + β, t_norm ∈ [-1,1], T-independent (2 scalars).
    Zero-init → gate = 0.5 at init (equal fwd/bwd).

    [Fix 8] Slice pos_t buffer when T ≤ max_seq_len instead of
    creating a new linspace tensor every forward call.
    """

    def __init__(self, d_model: int = 128, max_seq_len: int = 500):
        super().__init__()
        self.content_proj = nn.Linear(2 * d_model, 1, bias=True)
        self.pos_alpha    = nn.Parameter(torch.zeros(1))
        self.pos_beta     = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            'pos_t',
            torch.linspace(-1.0, 1.0, max_seq_len),
            persistent=False,
        )

    def forward(self, y_fwd: torch.Tensor,
                y_bwd: torch.Tensor) -> torch.Tensor:
        T = y_fwd.shape[1]
        # [Fix 8] slice buffer when possible, avoiding tensor allocation
        if T <= self.pos_t.shape[0]:
            pos_t = self.pos_t[:T]
        else:
            pos_t = torch.linspace(-1.0, 1.0, T, device=y_fwd.device)
        pos_t    = pos_t.to(dtype=y_fwd.dtype)               # AMP-safe
        gate_pos = (self.pos_alpha * pos_t[None, :, None]
                    + self.pos_beta)                          # (1, T, 1)
        gate_c   = self.content_proj(
            torch.cat([y_fwd, y_bwd], dim=-1)
        )                                                     # (B, T, 1)
        a = torch.sigmoid(gate_c + gate_pos)                  # (B, T, 1)
        return a * y_fwd + (1.0 - a) * y_bwd                 # (B, T, D)


class BidirectionalMambaLayer(nn.Module):
    """Bidirectional Mamba with separate fwd/bwd norms and LayerScale.

    [Fix 5] Separate norm_fwd / norm_bwd: each direction learns independent
    statistics (+2×d_model = 256 params, negligible vs total ~1.86M).

    [Fix 4] Layer scale: ls (init=1e-4, shape (d_model,)) multiplies the
    residual contribution before DropPath, starting near-identity.
    """

    def __init__(self, d_model: int = 128, d_state: int = 32,
                 d_conv: int = 4, expand: int = 2,
                 drop_path: float = 0.0,
                 layer_scale_init: float = 1e-4):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is required.\n"
                "Install: pip install mamba-ssm[causal-conv1d] --no-build-isolation\n"
                f"Original error: {_MAMBA_IMPORT_ERROR}"
            )
        # [Fix 5] separate norms for fwd and bwd
        self.norm_fwd  = RMSNorm(d_model)
        self.norm_bwd  = RMSNorm(d_model)
        self.fwd       = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.bwd       = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.gate      = SimplifiedBiMambaGate(d_model=d_model)
        # [Fix 4] layer scale: (d_model,) broadcasts over (B, T, d_model)
        self.ls        = nn.Parameter(torch.full((d_model,), layer_scale_init))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [Fix 5] each direction uses its own norm
        y_fwd = self.fwd(self.norm_fwd(x))
        y_bwd = self.bwd(self.norm_bwd(x).flip(1)).flip(1)
        y     = self.gate(y_fwd, y_bwd)
        # [Fix 4] scale residual before DropPath
        return x + self.drop_path(self.ls * y)


class BidirectionalMambaStack(nn.Module):
    """Stack of n_layers BiMamba layers followed by final RMSNorm."""

    def __init__(self, d_model: int = 128, n_layers: int = 2,
                 d_state: int = 32, d_conv: int = 4, expand: int = 2,
                 drop_path_rates=None, layer_scale_init: float = 1e-4):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * n_layers
        self.layers = nn.ModuleList([
            BidirectionalMambaLayer(
                d_model=d_model, d_state=d_state, d_conv=d_conv,
                expand=expand, drop_path=drop_path_rates[i],
                layer_scale_init=layer_scale_init,
            )
            for i in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ─── M5: Attentive Statistics Pooling ────────────────────────────────────────

class AttentiveStatPool1D(nn.Module):
    """Channel-dependent attentive statistics pooling → [B, 2*dim].

    ECAPA-TDNN style: each channel gets its own temporal attention weight.

    [Fix 2] OLD: std = sqrt(E[X²] - E[X]²) — catastrophic cancellation
            when mean >> std (e.g. mean=50, std=0.1 → 125% fp32 error,
            >90% fp16 error).
    NEW: x_c = x - mean  →  std = sqrt(E[x_c²])
         No cancellation. Error < 0.01% at same test case.

    Output: cat[mean, std] (B, 2*dim)
    """

    def __init__(self, dim: int = 128, bn: int = 32):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, bn),
            nn.Tanh(),
            nn.Linear(bn, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w       = self.score(x).softmax(dim=1)         # (B, T, dim)
        mean    = (w * x).sum(dim=1)                   # (B, dim)
        # [Fix 2] centered variance — no catastrophic cancellation
        x_c     = x - mean.unsqueeze(1)                # (B, T, dim)
        std     = (w * x_c * x_c).sum(dim=1).clamp(min=1e-12).sqrt()
        return torch.cat([mean, std], dim=-1)           # (B, 2*dim)


# ─── M6: Classifier ───────────────────────────────────────────────────────────

class Classifier(nn.Module):
    """MLP classifier head with LayerNorm input normalisation.

    [Fix 6] LN(d) before fc1: AttentiveStatPool outputs cat[mean, std]
    which have different scales. LayerNorm brings both to the same distribution,
    improving gradient conditioning. Standard in ECAPA-TDNN classifiers.
    """

    def __init__(self, d: int = 256, hidden: int = 128,
                 num_classes: int = 11, dropout: float = 0.2,
                 dropout_in: float = 0.1):
        super().__init__()
        self.norm        = nn.LayerNorm(d)              # [Fix 6]
        self.dropout_in  = nn.Dropout(dropout_in)
        self.fc1         = nn.Linear(d, hidden)
        self.act         = nn.GELU()
        self.drop        = nn.Dropout(dropout)
        self.fc2         = nn.Linear(hidden, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(
            self.drop(self.act(self.fc1(self.dropout_in(self.norm(z)))))
        )


# ─── Full Model ───────────────────────────────────────────────────────────────

class APWMamba(nn.Module):
    """Amplitude-Phase Wavelet Mamba for WiFi CSI Human Activity Recognition.

    Input:
        X_amp   (B, 27, T, F2)  — 3 DWT subbands × 9 amplitude channels
        X_phase (B, 18, T, F2)  — 3 DWT subbands × 6 phase channels
    Output:
        logits  (B, num_classes)

    [Fix 9] drop_path_rates and layer_scale_init exposed as constructor
    parameters for hyperparameter search. Defaults match v1 behaviour.
    """

    def __init__(
        self,
        num_classes:      int   = 11,
        dp_cnn:           list  = None,   # DropPath for CNN blocks
        dp_mamba:         list  = None,   # DropPath for BiMamba layers
        layer_scale_init: float = 1e-4,
    ):
        super().__init__()
        # [Fix 9] default drop-path schedules (same as v1 hardcoded values)
        if dp_cnn   is None: dp_cnn   = [0.00, 0.05, 0.10]
        if dp_mamba is None: dp_mamba = [0.00, 0.05]

        self.cnn_amp    = SubbandAwareCNNEncoder(
            n_per_subband=9,  n_tf_blocks=3,
            dilations=[1, 3, 9], drop_path_rates=dp_cnn,
            layer_scale_init=layer_scale_init)
        self.cnn_phase  = SubbandAwareCNNEncoder(
            n_per_subband=6,  n_tf_blocks=3,
            dilations=[1, 3, 9], drop_path_rates=dp_cnn,
            layer_scale_init=layer_scale_init)
        self.fusion     = CrossModalGatedFusion(d_model=128)
        self.f_agg      = FreqStatGatedPool(d_model=128, reduction=8)
        self.bi_mamba   = BidirectionalMambaStack(
            d_model=128, n_layers=2, d_state=32,
            drop_path_rates=dp_mamba,
            layer_scale_init=layer_scale_init)
        self.attn_pool  = AttentiveStatPool1D(dim=128, bn=32)
        self.classifier = Classifier(
            d=256, hidden=128, num_classes=num_classes)
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _mark_no_reinit(self):
        """Tag all submodules inside Mamba to preserve SSM initialisation."""
        if HAS_MAMBA:
            for m in self.modules():
                if isinstance(m, Mamba):
                    for child in m.modules():
                        child._no_reinit = True

    def _init_weights(self):
        self._mark_no_reinit()
        for m in self.modules():
            if getattr(m, '_no_reinit', False):
                continue
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                # [Fix 7] nonlinearity='relu' for GELU activation (gain=sqrt(2))
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # RMSNorm: weight initialised to ones by nn.Parameter(torch.ones(d))

        # Zero-init bilateral gate → G_a = G_p = sigmoid(0) = 0.5 (neutral blend)
        # Must run AFTER generic loop (loop runs trunc_normal first, then we override)
        for gate_conv in (self.fusion.gate_a, self.fusion.gate_p):
            nn.init.zeros_(gate_conv.weight)
            nn.init.zeros_(gate_conv.bias)

        # Zero-init BiMamba content gate → a = sigmoid(0) = 0.5 (equal fwd/bwd)
        for layer in self.bi_mamba.layers:
            nn.init.zeros_(layer.gate.content_proj.weight)
            nn.init.zeros_(layer.gate.content_proj.bias)
            # pos_alpha, pos_beta already zeros (nn.Parameter(torch.zeros(1)))

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, X_amp: torch.Tensor,
                X_phase: torch.Tensor) -> torch.Tensor:
        # [Fix 10] input shape assertions — catch silent mis-use immediately
        assert X_amp.ndim   == 4, f"X_amp must be 4-D, got {X_amp.shape}"
        assert X_phase.ndim == 4, f"X_phase must be 4-D, got {X_phase.shape}"
        assert X_amp.shape[1]   == 27, \
            f"X_amp must have 27 channels (3 subbands × 9), got {X_amp.shape[1]}"
        assert X_phase.shape[1] == 18, \
            f"X_phase must have 18 channels (3 subbands × 6), got {X_phase.shape[1]}"
        assert X_amp.shape[2:] == X_phase.shape[2:], \
            f"X_amp and X_phase spatial dims must match: {X_amp.shape} vs {X_phase.shape}"

        F_amp   = self.cnn_amp(X_amp)           # (B, 128, T, F2)
        F_phase = self.cnn_phase(X_phase)        # (B, 128, T, F2)
        F_fused = self.fusion(F_amp, F_phase)    # (B, 128, T, F2)
        x       = self.f_agg(F_fused)           # (B, T, 128)
        x       = self.bi_mamba(x)              # (B, T, 128)
        z       = self.attn_pool(x)             # (B, 256)
        return  self.classifier(z)              # (B, num_classes)
