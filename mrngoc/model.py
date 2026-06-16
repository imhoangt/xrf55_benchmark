"""TF-Mamba — bản gốc (paper), dual-stream, KHÔNG ablation.

Liu et al., "TF-Mamba: A Lightweight State-Space Model for Wi-Fi-Based Human
Activity Recognition," IEEE Sensors Journal, Vol. 25, No. 13, 2025.
https://doi.org/10.1109/JSEN.2024.3520857

Đây là bản TÁCH RA độc lập (chỉ TF-Mamba gốc) — đã bỏ toàn bộ phần ablation/
WavDualMamba (CNN front-end, BiMamba, AttnStatPool, Haar-3 ...). Chỉ phụ thuộc
torch + mamba_ssm. Luồng đúng paper (Fig. 5, Sec. IV):

    XH ─► stream_T ─► S_T (B,L,D)
                                  ─► AdaptiveFusion [Eq.15] ─► S2 (B,L,D)
    XV ─► stream_F ─► S_F (B,L,D)
                                        │ proj_s3 + tanh ─► S3 (B,L,D)
                                        │ GAP mean(dim=1) ─► (B,D)
                                        │ classifier      ─► logits (B,C)

Quy ước input (do caller chuẩn bị): XH, XV ∈ R^{B×L×M}, L=T/2, M=N/2.
XH = subband HL (paper XH), XV = subband LH (paper XV).
"""
import math

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as _e:
    raise ImportError(
        "mamba_ssm chưa được cài. Chạy: pip install mamba-ssm causal-conv1d\n"
        "Yêu cầu CUDA + GPU tương thích."
    ) from _e


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding [Vaswani 2017; Sec. IV-A]."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1), :]              # (1, L, d_model)


class EmbeddingLayer(nn.Module):
    """Linear projection + positional embedding — Eq. 10:  S_1 = ReLU(W·x+b) + P."""

    def __init__(self, num_features: int, embed_dim: int, max_len: int = 500):
        super().__init__()
        self.pos = PositionalEmbedding(embed_dim, max_len)
        self.fc = nn.Linear(num_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x)) + self.pos(x)   # (B, L, embed_dim)


class TFMambaStream(nn.Module):
    """Một stream: EmbeddingLayer + P khối Mamba pre-norm (uni-directional).

    Mỗi khối (Eq. 11-14, Fig. 6):  M_o = M + Linear(SSM(σ(Conv(Linear(LN(M))))) ⊙ σ(Linear(LN(M))))
    mamba_ssm.Mamba gói gọn Eq. 11-13; phần `x = x + mamba(norm(x))` là skip Eq.14.
    KHÔNG bao giờ pooling trong stream — trả về cả chuỗi (B, L, d_model).
    """

    def __init__(self, num_features: int, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, num_layers: int = 3,
                 max_len: int = 500):
        super().__init__()
        self.emb = EmbeddingLayer(num_features, d_model, max_len)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.mambas = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        for norm, mamba in zip(self.norms, self.mambas):
            x = x + mamba(norm(x))                    # pre-norm residual, Eq. 14
        return x                                      # (B, L, d_model)


class AdaptiveFusion(nn.Module):
    """Soft-weighted adaptive fusion — Sec. IV-E, Eq. 15.

        α_T, α_F = Softmax(Linear_{2D→2}(Concat(S_T, S_F)))
        S_fused  = α_T ⊙ S_T + α_F ⊙ S_F
    Input/Output: (B, L, D). Slice last axis (feature dim) để broadcast theo L.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(2 * d_model, 2)

    def forward(self, ST: torch.Tensor, SF: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.linear(torch.cat([ST, SF], dim=-1)), dim=-1)  # (B,L,2)
        return w[..., 0:1] * ST + w[..., 1:2] * SF                          # (B,L,D)


class TFMamba(nn.Module):
    """TF-Mamba dual-stream cho Wi-Fi CSI HAR — Fig. 5, Sec. IV (bản gốc baseline).

    num_features : M = N/2 (135 cho HUST sau DWT + transpose)
    d_model      : D = D' = 64           num_layers : P = 3 khối Mamba/stream
    num_classes  : số lớp c              d_state/d_conv/expand : Mamba (16/4/2)
    max_len      : độ dài chuỗi tối đa cho PE (>= L)
    """

    def __init__(self, num_features: int, d_model: int = 64, num_layers: int = 3,
                 num_classes: int = 6, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, max_len: int = 500):
        super().__init__()
        shared = dict(num_features=num_features, d_model=d_model, d_state=d_state,
                      d_conv=d_conv, expand=expand, num_layers=num_layers,
                      max_len=max_len)
        self.stream_T = TFMambaStream(**shared)       # XH = HL (time-detail)
        self.stream_F = TFMambaStream(**shared)       # XV = LH (freq-detail)
        self.fusion = AdaptiveFusion(d_model)         # Eq. 15
        self.proj_s3 = nn.Linear(d_model, d_model)    # S2 -> S3 (+ tanh)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, XH: torch.Tensor, XV: torch.Tensor) -> torch.Tensor:
        S_T = self.stream_T(XH)                       # (B, L, D)
        S_F = self.stream_F(XV)                       # (B, L, D)
        S2 = self.fusion(S_T, S_F)                    # (B, L, D)  Eq. 15
        S3 = torch.tanh(self.proj_s3(S2))             # (B, L, D)
        S3 = S3.mean(dim=1)                           # (B, D)     GAP
        return self.classifier(S3)                    # (B, C)
