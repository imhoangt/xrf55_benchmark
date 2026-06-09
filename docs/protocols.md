# Training protocols

Ba protocol huấn luyện, mỗi cái tái hiện cấu hình của một paper. Preset nằm trong
[`xrf55_bench/config.py`](../xrf55_bench/config.py) (`TrainCfg_for_protocol`).

## Cách chạy

Chạy 3 baseline với **raw** và **processed** CSI amplitude (dataset:
[xrf55-amp-dataset](https://www.kaggle.com/datasets/imhoangt/xrf55-amp-dataset)),
theo 3 protocol bên dưới → tổng **6 run** mỗi model.

```python
from xrf55_bench.config  import TrainCfg_for_protocol
from xrf55_bench.trainer import run
cfg = TrainCfg_for_protocol('03', seeds=(42,))
run(model_name='wavdualmamba', bench_dir=BENCH_DIR, output_dir=OUTPUT_DIR, train_cfg=cfg)
```

## Bảng protocol

| # | Nguồn | Optimizer | betas | LR | Batch | Epochs | WD | Scheduler | Label smooth |
|---|---|---|---|---|---|---|---|---|---|
| 01 | tf_mamba | AdamW | (0.9, 0.999) | 1e-4 | 32 | 40  | 0.01 | None | 0.0 |
| 02 | XRF55    | Adam  | (0.9, 0.999) | 1e-3 | 64 | 200 | 0.0  | MultiStepLR `[40,80,120,160]`, γ=0.5 | 0.0 |
| 03 | APWMamba | AdamW | (0.9, 0.99)  | 5e-4 | 32 | 200 | 1e-3 | warmup_cosine, warmup=10ep, floor_lr=4e-5 | 0.0 |

Mọi protocol: `eps=1e-8`, `clip=None`, loss `CE`, label smoothing `0.0`, không early-stop, FP32.
Protocol 03: weight decay không áp cho `bias / A_log / D / pos_emb`.

## Output mỗi run
```
output_dir/
├── metrics.json                 # config + per_seed + summary
├── run_config.json
├── plots/                       # training_curve, confusion_matrix, [seed_comparison]
└── seeds/{seed:03d}/            # training_log.csv, last_model.pt, best_model.pt, test_predictions.npz
```
`last_model.pt` = epoch cuối (model chính cho final eval); `best_model.pt` = test acc cao nhất lúc train.
