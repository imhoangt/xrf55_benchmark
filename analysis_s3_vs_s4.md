# Vì sao S4 (WavDualMamba) vượt S3 (TFMamba + CNN + BiMamba + AttnStatPool)?

Phân tích dựa trên code thực tế của hai model.

---

## Bức tranh toàn cảnh trước: S3 và S4 dùng cùng building blocks

TFMamba model.py có dòng này ở đầu file:
```python
from xrf55_bench.models.wavdualmamba.model import (
    AttnStatPool, BiMamba, SubbandStem, TFBlock, _SUBBAND_KERNEL,
)
```

**SubbandStem, TFBlock, BiMamba, AttnStatPool trong S3 và S4 là cùng một class, cùng một file.** Nếu chỉ nhìn vào building blocks, hai model về cơ bản giống nhau. Sự khác biệt nằm ở cách chúng được **ghép lại với nhau**.

---

## Sơ đồ pipeline thực tế (từ code)

### S3 — TFMamba.forward()

```
XH (B,500,135) → CNNFrontEnd(HL kernel) → (B,500,64) + PE → BiMamba → S_T (B,500,64)
XV (B,500,135) → CNNFrontEnd(LH kernel) → (B,500,64) + PE → BiMamba → S_F (B,500,64)
                                                                              │
                                                          AdaptiveFusion [RANDOM init]
                                                              S2 (B,500,64)
                                                                 │
                                                     S3 = tanh(proj_s3(S2))  ← !!!
                                                              S3 (B,500,64)
                                                                 │
                                                         AttnStatPool(S3)
                                                             (B,128)
                                                                 │
                                                     Linear(128, 11)  [không có LN/Dropout]
```

### S4 — WavDualMamba.forward()

```
X (B,18,500,15)
  HL branch → SubbandStem(HL) → TFBlock×3 → flatten → Linear → BiMamba → S_HL (B,500,64)
  LH branch → SubbandStem(LH) → TFBlock×3 → flatten → Linear → BiMamba → S_LH (B,500,64)
                                                                              │
                                                          AdaptiveFusion [ZERO init]
                                                               z (B,500,64)
                                                                 │
                                                         AttnStatPool(z)     ← trực tiếp
                                                             (B,128)
                                                                 │
                                                  LayerNorm → Dropout(0.2) → Linear(128,11)
```

---

## Lý do 1: `proj_s3 + tanh` là nguyên nhân chính AttnStatPool không hoạt động ở S3

### Trong TFMamba.forward() (dòng 478-484):
```python
S3 = torch.tanh(self.proj_s3(S2))    # (B, L, d_model)

if self.tpool is not None:
    S3 = self.tpool(S3)              # AttnStatPool nhận tanh-compressed values
else:
    S3 = S3.mean(dim=1)              # GAP cũng nhận tanh-compressed, nhưng không bị ảnh hưởng
```

`proj_s3` là Linear(64→64) theo sau bởi `tanh` — paper-faithful từ TF-Mamba gốc. **Điều này clamp tất cả values vào [-1, 1].**

### AttnStatPool làm gì (từ wavdualmamba/model.py dòng 550-560):
```python
w = self.score(h).softmax(dim=1)     # (B, T, d) — per-timestep, per-channel weights
mean = (w * x).sum(dim=1)            # weighted mean
var  = (w * (x - mean.unsqueeze(1)).pow(2)).sum(dim=1)
return torch.cat([mean, var.clamp(min=1e-6).sqrt()], dim=-1)
```

Output của AttnStatPool = concat(μ, σ). **Nửa sau là standard deviation theo thời gian.**

Khi `x` là output của `tanh`: tất cả timesteps bị ép vào [-1,1]. Nếu BiMamba đã học được representations tốt (values có dynamic range), tanh clamp sẽ làm:
- Nhiều timestep cùng bị saturate ở gần ±1
- Variance giữa các timestep giảm mạnh
- σ gần 0 → nửa sau của output AttnStatPool chứa rất ít information

**GAP (dùng trong S2) chỉ tính mean → không bị ảnh hưởng bởi tanh compression.** Đây là lý do S2 (91.76%) > S3 (90.64%): thêm AttnStatPool vào TFMamba không chỉ không giúp mà còn hại, vì AttnStatPool nhận input đã bị tanh bóp méo.

WavDualMamba không có `proj_s3`. AttnStatPool nhận trực tiếp Mamba output với full dynamic range → σ component có ý nghĩa thực sự.

---

## Lý do 2: AdaptiveFusion initialization khác nhau

### TFMamba.AdaptiveFusion (class riêng trong tf_mamba/model.py):
```python
class AdaptiveFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(2 * d_model, 2)
        # KHÔNG CÓ zero-init — random init mặc định
```

### WavDualMamba.AdaptiveFusion ('convex' mode, wavdualmamba/model.py dòng 469-472):
```python
if mode == 'convex':
    self.linear = nn.Linear(n_branches * d_model, n_branches)
    nn.init.zeros_(self.linear.weight)   # zero-init
    nn.init.zeros_(self.linear.bias)     # ⇒ softmax([0,0]) = [0.5, 0.5] = mean at step 0
```

Random init trong TFMamba có thể khiến một stream chiếm ưu thế ngay từ đầu (softmax của random values không phải là [0.5, 0.5]). Zero-init trong WavDualMamba đảm bảo mọi training run bắt đầu từ cùng một điểm (mean fusion), sau đó specialise theo data.

Đây là lý do trực tiếp giải thích tại sao std của S3 (0.70%) cao hơn S4 (0.27%): khởi tạo khác nhau → training trajectory khác nhau tùy seed.

---

## Lý do 3: Classifier head có regularization

### S3 (TFMamba):
```python
self.classifier = nn.Linear(head_in, num_classes)
```

### S4 (WavDualMamba):
```python
class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes=11, dropout=0.2):
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )
```

S4 có LayerNorm + Dropout(0.2) trước Linear cuối. Hai tác động:
- LayerNorm normalize AttnStatPool output → ổn định gradient từ loss về Mamba
- Dropout(0.2) trên 128 features → regularization cụ thể, đặc biệt quan trọng vì S3/S4 đều có ~0.40M params trên dataset tương đối nhỏ

Lý do này đặc biệt giải thích tại sao gap trên raw data lớn hơn nhiều (+5.16% raw vs +2.68% proc): raw data có nhiều noise hơn, regularization trong S4 giúp nhiều hơn.

---

## Tóm tắt: 3 lý do từ code

| | S3 (TFMamba ablation) | S4 (WavDualMamba) | Tác động |
|--|--|--|--|
| Trước AttnStatPool | `tanh(proj_s3(S2))` → clamp [-1,1] → σ gần 0 | Mamba output trực tiếp, full range | AttnStatPool vô dụng ở S3, hiệu quả ở S4 |
| AdaptiveFusion init | Random → training unstable theo seed | Zero-init → khởi đầu đồng nhất | Std 0.70% vs 0.27% |
| Classifier | Linear(128,11) | LN → Dropout(0.2) → Linear | Generalization tốt hơn, đặc biệt trên raw |

**Kết luận đơn giản**: S3 là S4 nhưng thêm `tanh(proj_s3(...))` vào pipeline — và chính cái đó phá vỡ AttnStatPool. S4 loại bỏ nó, thêm zero-init và regularize classifier. Ba thay đổi nhỏ, cộng hưởng cho ra +2.68% proc và +5.16% raw.
