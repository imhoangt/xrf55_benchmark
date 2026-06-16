"""Generate xrf55_bench/notebooks/s41_multidataset.ipynb (nbformat v4).

Applies a chosen MODEL (TF-Mamba original | S4.1 WavDualMamba) under a chosen
PROTOCOL (theirs = TF-Mamba paper | mine = 02*) to HUST / UT-HAR / NTU-Fi, in one
run. Builds the matching packed bench in-notebook from the mounted RAW Kaggle
datasets, then sweeps DATASETS x MODES. Dataset-specifics read from stats.json.

MODEL='tfmamba' + PROTOCOL='theirs' reproduces the TF-Mamba paper numbers
(validation that the original algorithm runs correctly).
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
    "# Multi-dataset — TF-Mamba (gốc) / S4.1 WavDualMamba on HUST / UT-HAR / NTU-Fi",
    "",
    "Chọn **MODEL** + **PROTOCOL** ở Cell 3, chạy 1 lần cho cả 3 dataset (sweep).",
    "Notebook build packed bench ngay trong notebook từ dataset RAW đã mount.",
    "",
    "| MODEL | Mô tả |",
    "|---|---|",
    "| `tfmamba` | **TF-Mamba gốc** (paper Liu 2025): Linear embed+PE, uni-Mamba×3, AdaptiveFusion, proj_s3, GAP |",
    "| `s41` | WavDualMamba Haar-3 {LL,HL,LH} + AttnStatPool (mô hình tốt nhất của ta) |",
    "",
    "| PROTOCOL | optimizer | lr | epochs | eval (so sánh) |",
    "|---|---|---|---|---|",
    "| `theirs` | AdamW (wd=0, betas .9/.999, eps 1e-8) | 1e-4 hằng số (no scheduler) | 40 | **best** (= early-stopping giữ checkpoint tốt nhất) |",
    "| `mine` | AdamW (wd=1e-3) | 5e-4→1e-6 warmup+cosine | 80 | last_model |",
    "",
    "`theirs` khớp **chính xác** mọi tham số paper Mục IV-B nêu (AdamW, lr 1e-4, 40ep, bs 32, "
    "CE, betas Adam). Paper KHÔNG nêu: weight-decay (→ 0, 'follow Adam'), tiêu chí early-stop "
    "(→ chạy đủ 40ep, report **best** = early-stopping chuẩn). Model = đúng S0 paper "
    "(d_model 64, 3 layers, uni-Mamba d_state 16, Linear embed+PE, GAP, proj_s3).",
    "",
    "**`tfmamba` + `theirs` = tái lập paper** (UT-HAR 99.00 / NTU-Fi 98.86 / HUST 99.72) "
    "→ so cột **best** với paper để chứng minh chạy đúng thuật toán gốc.",
    "",
    "**Trước khi chạy:** Add Input 3 dataset RAW (`hust_dataset`, `ut_har_dataset`, "
    "`ntu_fi_dataset`) + bật **GPU**.",
))

cells.append(code(
    "# Cell 1 — Install mamba-ssm + PyWavelets",
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
    "# Cell 3 — Configuration",
    "from pathlib import Path",
    "",
    "MODEL    = 'tfmamba'   # 'tfmamba' (gốc) | 's41' (WavDualMamba Haar-3)",
    "PROTOCOL = 'theirs'    # 'theirs' (TF-Mamba paper) | 'mine' (02*)",
    "DATASETS = ['hust', 'uthar', 'ntufi']",
    "MODES    = ['raw']     # ['raw'] | ['proc'] | ['raw','proc']",
    "SEEDS    = [0, 4, 8, 17, 42]",
    "OUT_ROOT = '/kaggle/working'",
    "",
    "DIRMAP  = {'hust': 'HUST-HAR', 'uthar': 'UT_HAR', 'ntufi': 'NTU-Fi_HAR'}",
    "_MARKER = {'hust': 'HUST_HAR_labels.pt', 'uthar': 'X_train.csv', 'ntufi': 'train_amp'}",
    "FORMAT  = 'tfmamba' if MODEL == 'tfmamba' else 'wavmamba'   # build layout per model",
    "build_py = CODE_PATH / 'xrf55_bench' / 'scripts' / '10_build_multi.py'",
    "",
    "def resolve_mount(ds):",
    "    base = Path('/kaggle/input')",
    "    for c in (sorted(base.iterdir()) if base.is_dir() else []):",
    "        if next(c.rglob(_MARKER[ds]), None) is not None:",
    "            return str(c)",
    "    raise FileNotFoundError(f'Không thấy /kaggle/input/* chứa {_MARKER[ds]} cho {ds}')",
    "",
    "print(f'MODEL={MODEL}  PROTOCOL={PROTOCOL}  FORMAT={FORMAT}')",
    "print(f'DATASETS={DATASETS}  MODES={MODES}  SEEDS={SEEDS}')",
    "for ds in DATASETS:",
    "    try:    print(f'  {ds:6s} mount: {resolve_mount(ds)}')",
    "    except Exception as e: print(f'  {ds:6s} !! {e}')",
))

cells.append(code(
    "# Cell 4 — Smoke 1 lần: model đã chọn chạy được trên GPU (fail-fast trước khi build/train)",
    "import torch",
    "dev = 'cuda' if torch.cuda.is_available() else 'cpu'",
    "if MODEL == 'tfmamba':",
    "    from xrf55_bench.models.tf_mamba.model import TFMamba",
    "    _m = TFMamba(num_features=135, d_model=64, num_layers=3, num_classes=6, max_len=500).to(dev)",
    "    with torch.no_grad():",
    "        _o = _m(torch.randn(2, 500, 135, device=dev), torch.randn(2, 500, 135, device=dev))",
    "else:",
    "    from xrf55_bench.models.wavdualmamba.model import WavDualMamba",
    "    _m = WavDualMamba(num_classes=6, n_links=1, n_antennas=9, f2=15,",
    "                      subbands=('LL', 'HL', 'LH'), pool='attnstat').to(dev)",
    "    with torch.no_grad():",
    "        _o = _m(torch.randn(2, 27, 16, 15, device=dev))",
    "assert _o.shape == (2, 6), f'bad output {tuple(_o.shape)}'",
    "del _m, _o",
    "if dev == 'cuda':",
    "    torch.cuda.empty_cache()",
    "print(f'SMOKE OK ({dev}, {MODEL}) — model chạy được')",
))

cells.append(code(
    "# Cell 5 — Sweep: build + train từng dataset (mỗi cái OUTPUT_DIR/zip riêng)",
    "import subprocess, sys, time, json, gc, torch",
    "from xrf55_bench.config import TrainCfg_for_protocol",
    "",
    "def make_cfg():",
    "    if PROTOCOL == 'theirs':   # TF-Mamba paper (Sec IV-B) — KHỚP CHÍNH XÁC phần paper nêu:",
    "        # AdamW, lr=1e-4 hằng số (không scheduler), 40 epochs, bs=32, CE,",
    "        # betas=(0.9,0.999)+eps=1e-8 (theo Adam paper họ trích). wd=0 (paper không nêu,",
    "        # 'follow Adam paper' => không weight decay). Early stopping 'when necessary'",
    "        # KHÔNG có tiêu chí trong paper => chạy đủ 40ep rồi report BEST checkpoint",
    "        # (đúng nghĩa early-stopping = giữ checkpoint tốt nhất).",
    "        return TrainCfg_for_protocol('02', seeds=tuple(SEEDS), optimizer='adamw',",
    "                                     lr=1e-4, weight_decay=0.0, betas=(0.9, 0.999),",
    "                                     eps=1e-8, num_epochs=40, batch_size=32,",
    "                                     scheduler=None, warmup_epochs=0, grad_clip=None,",
    "                                     criterion='ce', label_smoothing=0.0)",
    "    return TrainCfg_for_protocol('02', seeds=tuple(SEEDS), num_epochs=80,",
    "                                 betas=(0.9, 0.95), grad_clip=1.0, lr=5e-4, floor_lr=1e-6)",
    "",
    "def model_setup(meta):",
    "    F2, nps, T2 = meta['F2'], meta['n_per_sub'], meta['T2']",
    "    if MODEL == 'tfmamba':",
    "        return 'tfmamba', {'num_features': nps * F2, 'max_len': T2}",
    "    return 'wavdualmamba', {'n_links': 1, 'n_antennas': nps, 'f2': F2,",
    "                            'subbands': ('LL', 'HL', 'LH'), 'pool': 'attnstat'}",
    "",
    "def run_one(ds, md):",
    "    raw   = resolve_mount(ds)",
    "    bench = Path(OUT_ROOT) / DIRMAP[ds] / 'bench' / md",
    "    out   = Path(f'{OUT_ROOT}/outputs/{MODEL}_{ds}_{md}_{PROTOCOL}')",
    "    subprocess.run([sys.executable, str(build_py), '--dataset', ds, '--mode', md,",
    "                    '--raw-root', raw, '--out-root', OUT_ROOT, '--format', FORMAT], check=True)",
    "    meta = json.load(open(bench / 'stats.json'))['meta']",
    "    mname, mk = model_setup(meta)",
    "    run(model_name=mname, bench_dir=bench, output_dir=out, train_cfg=make_cfg(),",
    "        num_workers=4, model_kwargs=mk, num_classes=meta['classes'],",
    "        class_names=meta['class_names'], dataset_name=ds, split_desc=meta['split'])",
    "",
    "results = {}",
    "for ds in DATASETS:",
    "    for md in MODES:",
    "        t0 = time.time()",
    "        print(f\"\\n{'#'*64}\\n#  {MODEL} / {ds} / {md} / {PROTOCOL}\\n{'#'*64}\")",
    "        try:",
    "            run_one(ds, md); results[f'{ds}/{md}'] = 'OK'",
    "        except Exception as e:",
    "            results[f'{ds}/{md}'] = f'FAILED: {type(e).__name__}: {e}'; print('!!', e)",
    "        gc.collect()",
    "        if torch.cuda.is_available():",
    "            torch.cuda.empty_cache()",
    "        print(f\"== {ds}/{md}: {results[f'{ds}/{md}']}  ({(time.time()-t0)/60:.1f} phút)\")",
    "print('\\n=== SWEEP SUMMARY ===')",
    "for k, v in results.items():",
    "    print(f'  {k:14s}: {v}')",
))

cells.append(code(
    "# Cell 6 — Kết quả + gom zip (last_model = headline; best_epoch = chẩn đoán)",
    "import json, shutil",
    "from pathlib import Path",
    "from IPython.display import FileLink, display",
    "",
    "print('--- Results ---')",
    "for d in sorted(Path('/kaggle/working/outputs').glob(f'{MODEL}_*_{PROTOCOL}')):",
    "    mp = d / 'metrics.json'",
    "    if not mp.exists():",
    "        continue",
    "    m = json.load(open(mp)); s = m['summary']",
    "    last = f\"{s['test_accuracy_mean']*100:.2f}±{s['test_accuracy_std']*100:.2f}\"",
    "    best = f\"{s.get('best_test_acc_mean',0)*100:.2f}±{s.get('best_test_acc_std',0)*100:.2f}\"",
    "    print(f\"  {d.name:<28} last={last}  best={best}  F1={s['test_f1_macro_mean']*100:.2f}\")",
    "",
    "print('\\n--- Zips ---')",
    "for z in sorted(Path('/kaggle/working/outputs').rglob('*.zip')):",
    "    shutil.copy2(z, Path('/kaggle/working') / z.name)",
    "    print(z.name); display(FileLink(z.name))",
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
