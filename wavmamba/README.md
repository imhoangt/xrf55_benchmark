# wavmamba — WavMamba for WiFi-CSI HAR (UT-HAR & NTU-Fi)

Public code accompanying the paper. Trains **WavMamba** — a multi-branch
(one branch per Haar DWT subband) CNN + bidirectional-Mamba model with adaptive
late fusion — for WiFi-CSI human activity recognition on **UT-HAR** and
**NTU-Fi**.

The architecture is **fixed** to the configuration reported in the paper:

| Flag | Value |
|------|-------|
| `subbands`   | `('HL', 'LH')`  — Haar 2-branch (no LL) |
| `pool`       | `attnstat`      — attentive statistics pooling |
| `stem_norm`  | `False`         — no GroupNorm in the stem |
| `fusion`     | `gate`          — per-channel softmax branch gate |

Only the dataset-dependent dimensions (`num_classes`, `n_links`, `n_antennas`,
`f2`) and four width knobs (`d_model`, `d_stem`, `d_state`, `n_mamba_layers`)
are configurable; any attempt to change a fixed flag raises `ValueError`.

## Requirements

- Python 3.10+
- CUDA GPU (the Mamba SSM kernels are CUDA-only)
- See `requirements.txt` for the regular Python dependency list:
  - `torch` (2.7+), `numpy`, `scipy`, `pywavelets`, `scikit-learn`, `tqdm`,
    `matplotlib`
  - `fvcore` (optional — for FLOPs/MACs measurement; falls back gracefully)
- Install `mamba-ssm` and `causal-conv1d` manually as described below because
  their CUDA wheels must match the torch/CUDA/C++ ABI.

## Install

`mamba-ssm` and `causal-conv1d` ship prebuilt CUDA wheels that must match the
torch C++ ABI. With torch 2.7+ from PyPI (`cxx11abi=TRUE`), install the matching
`...cxx11abiTRUE` wheels with `--no-deps` so pip does not re-resolve torch or
replace an ABI-compatible wheel:

```bash
pip install torch==2.7.0
pip install mamba-ssm --no-build-isolation --no-deps
pip install causal-conv1d --no-deps
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` are intentionally **not** listed in
`requirements.txt`; installing them through `pip install -r requirements.txt`
can trigger dependency re-resolution and hard-to-debug CUDA/ABI failures.

> Do **not** build `mamba-ssm` from source unless you have matched the exact
> CUDA/torch ABI — the prebuilt wheels are the reliable path.

## Layout

```
wavmamba/
├── model.py            WavMamba (fixed paper configuration)
├── preprocess.py       Hampel/LPF + 2-D Haar DWT (hampel_vectorized, haar3_subbands, to_maps)
├── build_dataset.py    Build packed Haar bench arrays (UT-HAR / NTU-Fi) + CLI
├── dataset.py          PreprocWavMambaDataset + build_loaders (z-norm at load)
├── config.py           TrainCfg — the paper training protocol
├── train.py            Train -> eval -> metrics -> plots + CLI
├── notebooks/
│   └── wavmamba_kaggle.ipynb   Kaggle companion notebook
├── README.md
└── requirements.txt
```

This is a **flat script-first** package. The CLI examples below assume you run
from inside this directory:

```bash
cd wavmamba
```

## Datasets

Expected local layout by default:

```
dataset/
├── UT_HAR/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── X_val.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   └── y_val.csv
└── NTU-Fi_HAR/
    ├── train_amp/
    └── test_amp/
```

You can also keep the data anywhere and pass `--raw-root <path>`; the loaders
recursively search for the expected marker files/folders (`X_train.csv` for
UT-HAR and `train_amp/` for NTU-Fi), which also works with Kaggle dataset mounts.

| Dataset | Classes | n_ant x sub | fs | Split |
|---------|---------|-------------|----|-------|
| UT-HAR  | 7 | 3 x 30  | 100 Hz | official train=X_train / test=X_test (`--merge-val` merges val into test) |
| NTU-Fi  | 6 | 3 x 114 | 500 Hz | official train_amp / test_amp (time downsampled 2000->500) |

Raw CSI amplitude -> 2-D Haar DWT -> packed `[HL | LH]` subband-major input
`(B, 2*n_antennas, T2, F2)`.

Before public release, replace the dataset placeholders in the Citation /
Provenance section with the official dataset URLs and license/citation notes
required by the dataset providers.

## Normalization — two orthogonal flags

Normalization is split into two independent stages, controlled by `PRENORM`
and `Z_GRAN`. z-norm after the DWT is **always applied**; the flags only choose
the scheme:

| Flag | Values | Meaning |
|------|--------|---------|
| `PRENORM` | `sensefi` \| `none` | Pre-norm on raw amplitude **before** the DWT. `sensefi` = UT-HAR min-max (per split) / NTU-Fi `(x-42.32)/4.98`. `none` = no raw pre-normalization. |
| `Z_GRAN`  | `perpos` \| `pcb`    | Granularity of z-norm **after** the DWT. `perpos` = per-position `(C,T2,F2)`. `pcb` = per-channel-bin `(C,F2)`, collapsing time. |

Four combinations, each writing to its own bench dir
`bench/<mode>_<prenorm>_<z_gran>/`:

| Combination | Meaning |
|-------------|---------|
| `sensefi + perpos` | raw pre-norm + per-position z statistics |
| `none + pcb`       | no raw pre-norm + per-channel-bin z statistics |
| `sensefi + pcb`    | raw pre-norm + per-channel-bin z statistics |
| `none + perpos`    | no raw pre-norm + per-position z statistics |

**Protocol note.** For exact reproduction of the reported protocol, the DWT
z-normalization statistics are computed over all official split samples in the
bench build (`train + test`, and UT-HAR `val` when `--merge-val` is used). The
loader applies those stored statistics to every split. This all-reps protocol is
kept intentionally so the public code matches the reported preprocessing; it is
not presented as a generic train-only normalization recipe.

## Usage

### CLI (primary entry point)

All commands below assume:

```bash
cd wavmamba
```

Build the bench arrays, then train:

```bash
# UT-HAR, no raw pre-norm + per-channel-bin z statistics
python build_dataset.py --dataset uthar --mode raw --prenorm none --z-gran pcb --merge-val
python train.py --dataset uthar --mode raw --prenorm none --z-gran pcb --merge-val

# NTU-Fi, raw pre-norm + per-position z statistics
python build_dataset.py --dataset ntufi --mode raw --prenorm sensefi --z-gran perpos
python train.py --dataset ntufi --mode raw --prenorm sensefi --z-gran perpos
```

`train.py --help` lists all options (`--seeds`, `--num-epochs`, `--batch-size`,
`--lr`, `--num-workers`, `--raw-root`, `--out-root`, `--no-build`,
`--bench-dir`). By default `train.py` builds the bench first, then trains; pass
`--no-build` or `--bench-dir <path>` to reuse an existing build.

When `--bench-dir` is supplied, `train.py` reads `stats.json` from that bench and
checks that the CLI dataset/mode/normalization labels match the metadata before
naming the output directory.

Output goes to `<out_root>/outputs/wavmamba_<ds>_<mode>_<prenorm>_<z_gran>[_mv]/`:

```
metrics.json                     config + per_seed + summary (acc, f1, CM, efficiency)
plots/training_curve.png
plots/confusion_matrix.png
seeds/<seed:03d>/
    training_log.csv
    last_model.pt                headline result (final epoch)
    best_model.pt                diagnostic only (peeked at test acc)
    test_predictions.npz
```

### Kaggle notebook (companion)

`notebooks/wavmamba_kaggle.ipynb` runs both datasets end-to-end on Kaggle: clone
the repo, install the dependencies, set `PRENORM`/`Z_GRAN`/`MERGE_VAL`/`SEEDS`,
and run all cells. Before public release, replace `REPO_URL` in the notebook
with the final public repository URL; optionally set `REPO_REF` to the paper
release tag.

**Reported results** come from `last_model.pt` (final epoch). The
`best_epoch` / `best_test_acc` fields are train-time diagnostics selected by
peeking at test accuracy and **must not** be used as headline results.

## Training protocol (fixed)

`config.py` ships a single protocol:

| | |
|---|---|
| Optimizer  | AdamW, lr=5e-4, betas=(0.9, 0.95), wd=1e-3 |
| Scheduler  | warmup_cosine, warmup=5 epochs, floor_lr=1e-6 |
| Epochs     | 30, batch_size=32, grad_clip=1.0 |
| Loss       | CrossEntropy (no label smoothing) |
| WD exclude | norm/bias/A_log/D/pos_emb excluded from weight decay |
| Seeds      | (0, 4, 8, 17, 42) |

Seeds are fixed for statistical reproducibility, but training is not bitwise
deterministic by default: cuDNN benchmarking and TF32 matmul are enabled for
speed, and CUDA kernel versions can change exact trajectories.

## Citation / provenance

If you use this code, cite the accompanying WavMamba paper. Replace this block
with the final BibTeX before public release:

```bibtex
@article{wavmamba2026,
  title  = {WavMamba: Wavelet-Guided Bidirectional Mamba for WiFi-CSI Human Activity Recognition},
  author = {<authors>},
  year   = {2026},
  note   = {Code: https://github.com/<owner>/wavmamba}
}
```

Dataset and preprocessing provenance to fill with final official links before
release:

- UT-HAR: add official dataset URL, citation, and license/usage terms.
- NTU-Fi: add official dataset URL, citation, and license/usage terms.
- The raw pre-normalization option `sensefi` follows the public benchmark-style
  preprocessing used for these WiFi-CSI datasets: UT-HAR split-wise min-max and
  NTU-Fi fixed `(x - 42.3199) / 4.9802` scaling.
- `perpos` and `pcb` name the two z-stat granularities used in the experiments;
  the implementation documents their shapes directly so the code remains
  understandable without relying on prior-work labels.

## License

This package is released under the MIT License; see `LICENSE`.
