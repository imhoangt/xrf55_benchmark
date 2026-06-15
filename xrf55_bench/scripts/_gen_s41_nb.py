"""Generate xrf55_bench/notebooks/s41_multidataset.ipynb (nbformat v4).

S4.1 (WavDualMamba on Haar-3 LL|HL|LH + AttnStatPool) applied to HUST / UT-HAR /
NTU-Fi. Parameterised by DATASET + MODE; everything else is read from the
self-describing stats.json built by 10_build_multi.py.
"""
import json
from pathlib import Path


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def _src(lines):
    flat = []
    for ln in lines:
        flat.extend(ln.split("\n"))
    return [l + "\n" for l in flat[:-1]] + [flat[-1]]


cells = []

cells.append(md(
    "# S4.1 Multi-dataset — WavDualMamba (Haar-3 LL|HL|LH + AttnStatPool)",
    "",
    "Áp dụng mô hình tốt nhất **S4.1** (đạt 93.99% trên XRF55) cho **HUST / UT-HAR / NTU-Fi**.",
    "Chọn `DATASET` + `MODE` ở Cell 3; mọi cấu hình còn lại đọc tự động từ `stats.json`.",
    "",
    "| Dataset | classes | packed (C,T2,F2) | fs | split |",
    "|---|---|---|---|---|",
    "| HUST | 6 | (27,500,15) | 200 | random 80/20 seed 42 |",
    "| UT-HAR | 7 | (9,125,15) | 100 | official train; test=test+val |",
    "| NTU-Fi | 6 | (9,250,57) | 500 (::4) | official train/test |",
    "",
    "**Protocol** = giống S4.1 ablation: AdamW lr 5e-4→1e-6, warmup+cosine, "
    "betas (0.9,0.95), grad_clip 1.0, 80 epochs, eval = last_model.",
    "",
    "Tiền xử lý (Hampel+LPF theo fs + Haar-3 + z-score per channel,bin all-reps) "
    "đã build sẵn & verify ở local bằng `10_build_multi.py`.",
))

cells.append(code(
    "# Cell 1 — Install mamba-ssm (required by WavDualMamba BiMamba layers)",
    "!pip install -q ninja packaging wheel",
    "!pip install -q triton",
    "!pip install -q causal-conv1d>=1.2.0 --no-build-isolation",
    "!pip install -q mamba-ssm --no-build-isolation",
    "print('Install done')",
))

cells.append(code(
    "# Cell 2 — Clone / update latest code from GitHub",
    "import sys, subprocess",
    "from pathlib import Path",
    "",
    "CODE_PATH = Path('/kaggle/working/xrf55_benchmark')",
    "if not CODE_PATH.exists():",
    "    subprocess.run(['git', 'clone', '--depth', '1',",
    "                    'https://github.com/imhoangt/xrf55_benchmark.git',",
    "                    str(CODE_PATH)], check=True)",
    "else:",
    "    subprocess.run(['git', 'pull'], cwd=str(CODE_PATH), check=True)",
    "",
    "sys.path.insert(0, str(CODE_PATH))",
    "from xrf55_bench.trainer import run",
    "print('Import OK : xrf55_bench.trainer.run')",
))

cells.append(code(
    "# Cell 3 — Configuration (chọn DATASET + MODE; phần còn lại đọc từ stats.json)",
    "import json",
    "from pathlib import Path",
    "",
    "DATASET = 'hust'      # 'hust' | 'uthar' | 'ntufi'",
    "MODE    = 'proc'      # 'proc' | 'raw'",
    "SEEDS   = [0, 4, 8, 17, 42]   # S4.1 = 5 seeds; dùng [0,4,8] nếu muốn gọn compute",
    "NUM_EPOCHS = 80       # S4.1 = 80ep  (NTU-Fi nhỏ → giảm 60 nếu thấy overfit)",
    "",
    "# Mount path của packed bench đã build & upload lên Kaggle.",
    "# Cấu trúc kỳ vọng: <DATA_ROOT>/<DIR>/bench/<mode>/{wavmamba/X_*.npy, y_*.npy, stats.json}",
    "DIRMAP    = {'hust': 'HUST-HAR', 'uthar': 'UT_HAR', 'ntufi': 'NTU-Fi_HAR'}",
    "DATA_ROOT = Path('/kaggle/input/datasets/imhoangt/s41-multidata')   # <- chỉnh theo dataset bạn upload",
    "BENCH_DIR = DATA_ROOT / DIRMAP[DATASET] / 'bench' / MODE",
    "",
    "# Đọc cấu hình tự-mô-tả từ stats.json (dataset/classes/class_names/split/dims/fs).",
    "meta        = json.load(open(BENCH_DIR / 'stats.json'))['meta']",
    "NUM_CLASSES = meta['classes']",
    "CLASS_NAMES = meta['class_names']",
    "SPLIT_DESC  = meta['split']",
    "C, T2, F2   = 3 * meta['n_per_sub'], meta['T2'], meta['F2']",
    "MODEL_KWARGS = {'n_links': 1, 'n_antennas': meta['n_per_sub'], 'f2': F2,",
    "                'subbands': ('LL', 'HL', 'LH'), 'pool': 'attnstat'}",
    "OUTPUT_DIR  = Path(f'/kaggle/working/outputs/s41_{DATASET}_{MODE}_p02')",
    "",
    "print(f'Dataset    : {DATASET} ({meta[\"dataset\"]})  mode={MODE}  fs={meta[\"fs\"]}')",
    "print(f'Classes    : {NUM_CLASSES}  {CLASS_NAMES}')",
    "print(f'Split      : {SPLIT_DESC}')",
    "print(f'Packed dims: C={C}  T2={T2}  F2={F2}')",
    "print(f'Model kwargs: {MODEL_KWARGS}')",
    "print(f'Seeds={SEEDS}  epochs={NUM_EPOCHS}')",
    "print(f'Output dir : {OUTPUT_DIR}')",
    "for f in ['stats.json', 'y_train.npy', 'y_test.npy',",
    "          'wavmamba/X_train.npy', 'wavmamba/X_test.npy']:",
    "    p = BENCH_DIR / f",
    "    print(f'  [{\"OK\" if p.exists() else \"MISSING\"}] {p}')",
))

cells.append(md(
    "### (Tùy chọn) Build trong notebook từ raw",
    "Chỉ dùng nếu bạn upload **raw** thay vì packed bench. Cần `PyWavelets` và đặt raw "
    "đúng `dataset/<DIR>/` trong repo. Mặc định dùng packed đã upload ở Cell 3 (khuyến nghị — "
    "đã verify ở local).",
))

cells.append(code(
    "# Cell 3b (TÙY CHỌN) — build packed bench từ raw trong notebook",
    "# !pip install -q PyWavelets",
    "# import subprocess, sys",
    "# subprocess.run([sys.executable, str(CODE_PATH / 'xrf55_bench/scripts/10_build_multi.py'),",
    "#                 '--dataset', DATASET, '--mode', MODE], check=True)",
    "# BENCH_DIR = CODE_PATH / 'dataset' / DIRMAP[DATASET] / 'bench' / MODE  # rồi chạy lại Cell 3 phần đọc meta",
))

cells.append(code(
    "# Cell 4 — Smoke-build + forward (kiểm Mamba/GPU + dims TRƯỚC khi train dài)",
    "import torch, gc",
    "from xrf55_bench.models.wavdualmamba.model import WavDualMamba",
    "",
    "dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "_m  = WavDualMamba(num_classes=NUM_CLASSES, **MODEL_KWARGS).to(dev)",
    "_x  = torch.randn(2, C, T2, F2, device=dev)",
    "_m.eval()",
    "with torch.no_grad():",
    "    _o = _m(_x)",
    "assert _o.shape == (2, NUM_CLASSES), f'bad output {tuple(_o.shape)}'",
    "assert _m.C_IN == C, f'C_IN {_m.C_IN} != packed C {C}'",
    "print(f'SMOKE OK: in {tuple(_x.shape)} -> out {tuple(_o.shape)}  '",
    "      f'params={sum(p.numel() for p in _m.parameters())/1e6:.3f}M')",
    "del _m, _x, _o; gc.collect()",
    "if torch.cuda.is_available():",
    "    torch.cuda.empty_cache()",
))

cells.append(code(
    "# Cell 5 — Train S4.1 (protocol giống ablation: 02, 80ep, betas 0.9/0.95, gc=1, lr 5e-4->1e-6)",
    "from xrf55_bench.config import TrainCfg_for_protocol",
    "",
    "cfg = TrainCfg_for_protocol('02', seeds=tuple(SEEDS), num_epochs=NUM_EPOCHS,",
    "                            betas=(0.9, 0.95), grad_clip=1.0,",
    "                            lr=5e-4, floor_lr=1e-6)",
    "",
    "run(model_name='wavdualmamba', bench_dir=BENCH_DIR, output_dir=OUTPUT_DIR,",
    "    train_cfg=cfg, num_workers=4, model_kwargs=MODEL_KWARGS,",
    "    num_classes=NUM_CLASSES, class_names=CLASS_NAMES,",
    "    dataset_name=DATASET, split_desc=SPLIT_DESC)",
))

cells.append(code(
    "# Cell 6 — Results",
    "import json",
    "mp = OUTPUT_DIR / 'metrics.json'",
    "if mp.exists():",
    "    m = json.load(open(mp))",
    "    s = m['summary']; seeds = m['config']['seeds']",
    "    print('=' * 55)",
    "    print(f\"  S4.1 — {m['dataset']} ({MODE})   split: {m['split']}\")",
    "    print(f\"  Seeds    : {seeds}\")",
    "    print(f\"  Accuracy : {s['test_accuracy_mean']*100:.2f}% ± {s['test_accuracy_std']*100:.2f}%\")",
    "    print(f\"  F1 Macro : {s['test_f1_macro_mean']*100:.2f}% ± {s['test_f1_macro_std']*100:.2f}%\")",
    "    print(f\"  Params   : {s['params_M']}M  |  MACs: {s.get('macs_G')}G\")",
    "    print(f\"  Time     : {s['total_time_s']}s   Best epochs: {s['best_epochs']}\")",
    "    print('=' * 55)",
    "    if len(seeds) == 1:",
    "        ps = m['per_seed'].get(str(seeds[0]), {})",
    "        if ps.get('test_f1_per_class'):",
    "            print('\\n  Per-class F1:')",
    "            for nm, v in zip(CLASS_NAMES, ps['test_f1_per_class']):",
    "                print(f'    {nm:<12}: {v*100:.2f}%')",
    "else:",
    "    print('metrics.json not found — training may not have completed.')",
))

cells.append(code(
    "# Cell 7 — Plots + Download zip",
    "import shutil",
    "from IPython.display import Image, display, FileLink",
    "",
    "for fname in ['training_curve.png', 'confusion_matrix.png', 'seed_comparison.png']:",
    "    p = OUTPUT_DIR / 'plots' / fname",
    "    if p.exists():",
    "        display(Image(str(p)))",
    "",
    "print('\\n--- Download ---')",
    "zips = sorted(OUTPUT_DIR.glob('*.zip'))",
    "for src in zips:",
    "    dst = Path('/kaggle/working') / src.name",
    "    shutil.copy2(src, dst)",
    "    print(f'{src.name}  ({dst.stat().st_size/1e6:.1f} MB)')",
    "    display(FileLink(src.name))",
    "if not zips:",
    "    print('[MISSING] no zip — run Cell 5 first.')",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent.parent / 'notebooks' / 's41_multidataset.ipynb'
json.dump(nb, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'wrote {out}  ({len(cells)} cells)')
