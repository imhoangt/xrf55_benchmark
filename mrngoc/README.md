# TF-Mamba gốc (dual-stream) — bản tách độc lập

Bản **tách hoàn toàn** của TF-Mamba **gốc theo paper** (Liu et al., IEEE Sensors
Journal 2025), chạy trên 3 dataset Wi-Fi CSI HAR: **HUST-HAR / UT-HAR / NTU-Fi**.
Thư mục này **không phụ thuộc** `xrf55_bench` — copy đi nơi khác vẫn chạy.

> CHỈ CÓ TF-Mamba gốc: dual-stream Mamba + AdaptiveFusion + proj_s3 + GAP.
> Không có bất kỳ phần ablation/WavDualMamba nào (CNN front-end, BiMamba,
> AttnStatPool, Haar-3...).

## Mô hình (`model.py`)
```
XH=HL ─► stream_T (EmbeddingLayer + uni-Mamba×3) ─► S_T
                                                        ─► AdaptiveFusion(Eq.15) ─► S2
XV=LH ─► stream_F (EmbeddingLayer + uni-Mamba×3) ─► S_F
                                                              ─► proj_s3 + tanh ─► GAP ─► Linear ─► logits
```
d_model=64, num_layers=3, d_state=16, d_conv=4, expand=2. Params ~0.09–0.10M.

## Pipeline dữ liệu (`data.py`) — raw (chỉ Haar DWT)
```
raw (N, n_ant·sub, time)
  → [UT-HAR/NTU-Fi] SenseFi pre-norm trên RAW (giống UT_HAR_dataset / CSI_Dataset):
       uthar = min-max từng split;  ntufi = (x-42.3199)/4.9802   (HUST: không pre-norm)
  → Haar 2-D DWT (pywt 'periodization'):  HL=cV.T (paper XH),  LH=cH.T (paper XV)
  → xh, xv (N, T/2, M),  M = n_ant·sub/2
  → z-norm per-POSITION (= data_norm git TF-Mamba) theo NORM_MODE
```
**`NORM_MODE`** (chỉ ảnh hưởng UT-HAR/NTU-Fi; HUST luôn data_norm):
- `author` (mặc định) — UT-HAR/NTU-Fi chỉ SenseFi norm, **không** z-norm sau DWT → **đúng tác giả** (acknowledgment dùng UT_HAR_dataset/CSI_Dataset). HUST vẫn z-norm.
- `double` — UT-HAR/NTU-Fi = SenseFi norm → DWT → z-norm (2 lần); đồng scale với HUST.
| Dataset | n_ant×sub | M | T/2 | lớp | split (giống hệt git họ) |
|---|---|---|---|---|---|
| HUST-HAR | 9×30 | 135 | 500 | 6 | random 80/20 (seed 42; git họ no-seed, method giống) |
| UT-HAR | 3×30 | 45 | 125 | 7 | official: train=X_train(3977), test=X_test(500); val bỏ (`MERGE_VAL=True` để gộp→996) |
| NTU-Fi | 3×114 | 171 | 250 | 6 | official train_amp / test_amp |

## Protocol (`train.py`) = "theirs"
AdamW **lr=1e-4** (paper; code release dùng 1e-3 — ta theo paper), wd=0.01,
betas (0.9,0.999), CrossEntropyLoss, 40 epoch, bs=32, **grad_clip=1.0**,
early-stop khi train-loss<0.01, **không scheduler**. **Seed cố định = 42**.

## Chạy

### Kaggle (khuyến nghị — cần GPU)
Mở `tfmamba_kaggle.ipynb`, Add Input 3 dataset RAW, bật GPU, Run All.
(Notebook clone repo này rồi import `mrngoc/`; cần đã push `mrngoc/` lên GitHub.)

### Local (cần GPU + CUDA cho mamba-ssm)
```bash
pip install -r requirements.txt
python run.py --dataset hust --raw-root /path/to/HUST-HAR
# hoặc cả 3:
python run.py --dataset all --hust /p/HUST --uthar /p/UT --ntufi /p/NTU
```
Kết quả lưu `results.json` (accuracy / precision / recall / F1 / AUC / confusion).

## Files
| File | Nội dung |
|---|---|
| `model.py` | TF-Mamba dual-stream thuần (chỉ torch + mamba_ssm) |
| `data.py` | 3 loaders + Haar DWT + chuẩn hoá → TensorDataset |
| `train.py` | protocol theirs + train/eval + metrics |
| `run.py` | CLI chạy local |
| `tfmamba_kaggle.ipynb` | runner Kaggle (orchestrator mỏng) |
| `requirements.txt` | phụ thuộc |
