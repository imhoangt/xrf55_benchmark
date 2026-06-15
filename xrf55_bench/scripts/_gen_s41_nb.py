"""Generate xrf55_bench/notebooks/s41_multidataset.ipynb (nbformat v4).

S4.1 (WavDualMamba on Haar-3 LL|HL|LH + AttnStatPool) applied to HUST / UT-HAR /
NTU-Fi, in ONE run. Builds the packed Haar-3 bench in-notebook from the mounted
RAW Kaggle datasets, then trains each (sweep loop, like the xrf55_bench notebooks).
Everything dataset-specific is read from the self-describing stats.json.
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
    "Áp dụng mô hình tốt nhất **S4.1** (93.99% trên XRF55) cho **HUST / UT-HAR / NTU-Fi** "
    "trong **một lần chạy**. Notebook build packed Haar-3 ngay trong notebook từ dataset RAW "
    "đã mount, rồi train lần lượt (sweep), mỗi dataset 1 zip riêng.",
    "",
    "| Dataset | classes | packed (C,T2,F2) | split |",
    "|---|---|---|---|",
    "| HUST | 6 | (27,500,15) | random 80/20 seed 42 |",
    "| UT-HAR | 7 | (9,125,15) | official train; test=test+val |",
    "| NTU-Fi | 6 | (9,250,57) | official train/test (::4) |",
    "",
    "**Protocol** = S4.1: AdamW lr 5e-4→1e-6 warmup+cosine, betas (0.9,0.95), grad_clip 1.0, "
    "80 epoch, eval=last_model. **raw**=chỉ Haar (trung thành benchmark); **proc**=+Hampel+LPF.",
    "",
    "**Trước khi chạy:** Add Input cả 3 dataset RAW (`hust_dataset`, `ut_har_dataset`, "
    "`ntu_fi_dataset`) + bật **GPU**. ~2.5h cho cả 3 raw (≈7s/epoch).",
))

cells.append(code(
    "# Cell 1 — Install mamba-ssm (WavDualMamba) + PyWavelets (build Haar)",
    "!pip install -q ninja packaging wheel",
    "!pip install -q triton",
    "!pip install -q causal-conv1d>=1.2.0 --no-build-isolation",
    "!pip install -q mamba-ssm --no-build-isolation",
    "!pip install -q PyWavelets",
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
    "# Cell 3 — Configuration (chạy cả 3 dataset trong 1 lần)",
    "from pathlib import Path",
    "",
    "DATASETS   = ['hust', 'uthar', 'ntufi']   # bỏ bớt nếu chỉ muốn 1-2 cái",
    "MODE       = 'raw'      # 'raw' (chỉ Haar) | 'proc' (+Hampel+LPF) — đổi rồi chạy lại để có proc",
    "SEEDS      = [0, 4, 8, 17, 42]            # S4.1 = 5 seeds; [0,4,8] nếu muốn gọn",
    "NUM_EPOCHS = 80                           # đồng nhất cả 3 (protocol S4.1)",
    "OUT_ROOT   = '/kaggle/working'            # bench/ + outputs/ build vào đây",
    "",
    "DIRMAP  = {'hust': 'HUST-HAR', 'uthar': 'UT_HAR', 'ntufi': 'NTU-Fi_HAR'}",
    "# marker nhận diện mount của từng dataset trong /kaggle/input/<slug bất kỳ>/",
    "_MARKER = {'hust': 'HUST_HAR_labels.pt', 'uthar': 'X_train.csv', 'ntufi': 'train_amp'}",
    "build_py = CODE_PATH / 'xrf55_bench' / 'scripts' / '10_build_multi.py'",
    "",
    "def resolve_mount(ds):",
    "    \"\"\"Tự dò /kaggle/input/* chứa marker của ds (khỏi đoán slug).\"\"\"",
    "    base = Path('/kaggle/input')",
    "    for c in (sorted(base.iterdir()) if base.is_dir() else []):",
    "        if next(c.rglob(_MARKER[ds]), None) is not None:",
    "            return str(c)",
    "    raise FileNotFoundError(f'Không thấy /kaggle/input/* chứa {_MARKER[ds]} cho {ds}')",
    "",
    "print(f'DATASETS={DATASETS}  MODE={MODE}  SEEDS={SEEDS}  EPOCHS={NUM_EPOCHS}')",
    "for ds in DATASETS:",
    "    try:    print(f'  {ds:6s} mount: {resolve_mount(ds)}')",
    "    except Exception as e: print(f'  {ds:6s} !! {e}')",
))

cells.append(code(
    "# Cell 4 — Smoke 1 lần: chắc mamba_ssm + model chạy được TRƯỚC khi build/train dài",
    "import torch",
    "from xrf55_bench.models.wavdualmamba.model import WavDualMamba",
    "dev = 'cuda' if torch.cuda.is_available() else 'cpu'",
    "_m  = WavDualMamba(num_classes=6, n_links=1, n_antennas=9, f2=15,",
    "                   subbands=('LL', 'HL', 'LH'), pool='attnstat').to(dev)",
    "with torch.no_grad():",
    "    _o = _m(torch.randn(2, 27, 16, 15, device=dev))",
    "assert _o.shape == (2, 6), f'bad output {tuple(_o.shape)}'",
    "del _m, _o",
    "if dev == 'cuda':",
    "    torch.cuda.empty_cache()",
    "print(f'SMOKE OK ({dev}) — mamba/model chạy được')",
))

cells.append(code(
    "# Cell 5 — Sweep: build + train từng dataset (mỗi cái 1 OUTPUT_DIR / zip riêng)",
    "import subprocess, sys, time, json, gc, torch",
    "from xrf55_bench.config import TrainCfg_for_protocol",
    "",
    "def run_one(ds):",
    "    raw   = resolve_mount(ds)",
    "    bench = Path(OUT_ROOT) / DIRMAP[ds] / 'bench' / MODE",
    "    out   = Path(f'{OUT_ROOT}/outputs/s41_{ds}_{MODE}_p02')",
    "    # 1) build packed Haar-3 (raw: chỉ DWT; proc: +Hampel+LPF) -> bench/",
    "    subprocess.run([sys.executable, str(build_py), '--dataset', ds, '--mode', MODE,",
    "                    '--raw-root', raw, '--out-root', OUT_ROOT], check=True)",
    "    meta = json.load(open(bench / 'stats.json'))['meta']",
    "    mk = {'n_links': 1, 'n_antennas': meta['n_per_sub'], 'f2': meta['F2'],",
    "          'subbands': ('LL', 'HL', 'LH'), 'pool': 'attnstat'}",
    "    # 2) train S4.1 (protocol 02)",
    "    cfg = TrainCfg_for_protocol('02', seeds=tuple(SEEDS), num_epochs=NUM_EPOCHS,",
    "                                betas=(0.9, 0.95), grad_clip=1.0, lr=5e-4, floor_lr=1e-6)",
    "    run(model_name='wavdualmamba', bench_dir=bench, output_dir=out, train_cfg=cfg,",
    "        num_workers=4, model_kwargs=mk, num_classes=meta['classes'],",
    "        class_names=meta['class_names'], dataset_name=ds, split_desc=meta['split'])",
    "",
    "results = {}",
    "for ds in DATASETS:",
    "    t0 = time.time()",
    "    print(f\"\\n{'#'*64}\\n#  {ds} / {MODE}\\n{'#'*64}\")",
    "    try:",
    "        run_one(ds)",
    "        results[ds] = 'OK'",
    "    except Exception as e:",
    "        results[ds] = f'FAILED: {type(e).__name__}: {e}'",
    "        print(f'!! {ds} FAILED:', e)",
    "    gc.collect()",
    "    if torch.cuda.is_available():",
    "        torch.cuda.empty_cache()",
    "    print(f\"== {ds}: {results[ds]}  ({(time.time()-t0)/60:.1f} phút)\")",
    "print('\\n=== SWEEP SUMMARY ===')",
    "for k, v in results.items():",
    "    print(f'  {k:6s}: {v}')",
))

cells.append(code(
    "# Cell 6 — Gom tất cả zip + bảng tổng hợp + (tùy) plots dataset cuối",
    "import shutil, subprocess, sys",
    "from pathlib import Path",
    "from IPython.display import Image, display, FileLink",
    "",
    "print('--- Zips ---')",
    "for z in sorted(Path('/kaggle/working/outputs').rglob('*.zip')):",
    "    shutil.copy2(z, Path('/kaggle/working') / z.name)",
    "    print(f'{z.name}  ({z.stat().st_size/1e6:.1f} MB)')",
    "    display(FileLink(z.name))",
    "",
    "print('\\n--- Bảng tổng hợp ---')",
    "subprocess.run([sys.executable,",
    "                str(CODE_PATH / 'xrf55_bench/scripts/11_aggregate_multi.py'),",
    "                '--root', '/kaggle/working/outputs',",
    "                '--out', '/kaggle/working/summary.md'], check=True)",
    "print(open('/kaggle/working/summary.md').read())",
    "",
    "# (tùy) hiển thị confusion matrix của từng dataset",
    "for d in sorted(Path('/kaggle/working/outputs').glob('s41_*')):",
    "    cm = d / 'plots' / 'confusion_matrix.png'",
    "    if cm.exists():",
    "        print(d.name); display(Image(str(cm)))",
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
