# Kaggle Lite Notebooks

Each notebook in this directory is a "lite" variant — **no model/dataset code inline**.
Instead, the notebook attaches a Kaggle dataset that contains the project source code, then
calls `run_all()` (or `run()`) from the package. All training logic lives in `baselines/`.

---

## Datasets required per notebook

| Notebook | Required datasets |
|---|---|
| `apwmamba.ipynb` | `imhoangt/4-baselines-code` + `imhoangt/xrf55_processed_dataset` |
| `tf_mamba_xrf55_amp.ipynb` | `imhoangt/4-baselines-code` + `imhoangt/xrf55_processed_dataset` |
| `tf_mamba_xrf55_phase.ipynb` | `imhoangt/4-baselines-code` + `imhoangt/xrf55_processed_dataset` |
| `resnet1d_xrf55_amp.ipynb` | `imhoangt/4-baselines-code` + `imhoangt/xrf55_processed_dataset` |
| `resnet1d_xrf55_phase.ipynb` | `imhoangt/4-baselines-code` + `imhoangt/xrf55_processed_dataset` |
| `tf_mamba_hust_har.ipynb` | `imhoangt/4-baselines-code` + `imhoangt/hust_dataset` |

---

## How to create the `4-baselines-code` dataset on Kaggle

This dataset contains the entire project source code.

### Step 1 — Upload to Kaggle

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) -> **New Dataset**
2. Name: `4-baselines-code` (use this exact slug — notebooks point to the matching path)
3. Upload a **zip** of the `har_csi/` directory (or upload individual files/folders)
4. Set visibility: **Private**

### Step 2 — Required structure inside the dataset

The dataset must contain the following layout at **root** (or inside a `har_csi/`
subfolder — the notebooks auto-detect both):

```
baselines/
  __init__.py
  apwmamba/                 __init__.py  config.py  dataset.py  model.py  trainer.py
  tf_mamba_hust_har/        __init__.py  dataset.py  model.py  trainer.py
  tf_mamba_xrf55_amp/       __init__.py  dataset.py  model.py  trainer.py
  tf_mamba_xrf55_phase/     __init__.py  dataset.py  model.py  trainer.py
  resnet1d_xrf55_amp/       __init__.py  dataset.py  trainer.py
  resnet1d_xrf55_phase/     __init__.py  dataset.py  trainer.py
  base_models/
    __init__.py
    tf_mamba_base/          __init__.py  model.py  train_utils.py
    resnet1d_base/          __init__.py  model.py  train_utils.py
configs/
  apwmamba.yaml
src/
  training/
    train_utils.py
    amp_utils.py
  data/
    loso_splits.py
    splits.py
```

> `configs/apwmamba.yaml` is required — `baselines/apwmamba/config.py` loads it.

### Step 3 — Updating code

When the source code changes:
1. **Create a new version** of the Kaggle dataset (Upload -> New version)
2. Notebooks will automatically use the latest version on next run

---

## Attach datasets to a notebook

1. Open the notebook on Kaggle -> **+ Add data** (top right)
2. Find `4-baselines-code` (Your datasets) -> Add
3. Find `xrf55_processed_dataset` or `hust_dataset` -> Add
4. Verify the mount path in the "Mount code" cell

---

## Run configuration

Each notebook has a **Config** cell to tweak:

| Variable | Meaning | Valid values |
|---|---|---|
| `PROTOCOL` | Evaluation protocol | `'split'` or `'loso'` |
| `N_SEEDS` | Number of seeds | `1` (seed=4) or `3` (seeds [4,8,17]) |

After editing -> **Run All**.
