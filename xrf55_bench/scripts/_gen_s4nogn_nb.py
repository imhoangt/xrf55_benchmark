"""Generate xrf55_bench/notebooks/s4nogn_multidataset.ipynb (nbformat v4).

Applies a chosen MODEL (TF-Mamba original | S4.nogn | S4.nogn_gate WavDualMamba Haar 2-subband) under a chosen
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
    "# Multi-dataset — TF-Mamba (gốc) / S4.nogn / S4.nogn_gate WavDualMamba (Haar 2 băng) on HUST / UT-HAR / NTU-Fi",
    "",
    "Chọn **MODEL** + **PROTOCOL** ở Cell 3, chạy 1 lần cho cả 3 dataset (sweep).",
    "Notebook build packed bench ngay trong notebook từ dataset RAW đã mount.",
    "",
    "| MODEL | Mô tả |",
    "|---|---|",
    "| `tfmamba` | **TF-Mamba gốc** (paper Liu 2025): Linear embed+PE, uni-Mamba×3, AdaptiveFusion, proj_s3, GAP |",
    "| `s4.nogn` | WavDualMamba Haar 2 băng {HL,LH} + AttnStatPool, **bỏ GroupNorm ở SubbandStem** (= ablation S4.nogn) |",
    "| `s4.nogn_gate` | `s4.nogn` nhưng **fusion='gate'** (per-channel) thay convex scalar (= ablation S4.nogn_gate) |",
    "",
    "| PROTOCOL | optimizer | lr | wd | epochs | grad-clip | early-stop |",
    "|---|---|---|---|---|---|---|",
    "| `theirs` (TF-Mamba gốc) | AdamW | **1e-3** | 0.01 | 40 | 1.0 | `if loss<0.01: break` |",
    "| `mine` | AdamW | 5e-4→1e-6 warmup+cosine | 1e-3 | 30 | 1.0 | tắt (MINE_EARLY_STOP=None) |",
    "",
    "`theirs` = protocol TF-Mamba (Liu 2025): **`lr=1e-3` theo CODE release**, "
    "`CrossEntropyLoss`, 40 epoch, bs 32, không scheduler, "
    "`clip_grad_norm_(max_norm=1.0)`, early-stop `if average_loss<0.01: break`, random 80/20. "
    "Các chi tiết wd=0.01 / betas / grad-clip / early-stop lấy từ **code gốc** `Mamba_HUST-HAR.py` "
    "(paper không ghi rõ). **LƯU Ý: mặc định dùng lr=1e-3 theo CODE release `Mamba_HUST-HAR.py`; paper ghi 1e-4 (đổi `lr` trong make_cfg nếu muốn bản theo paper).**",
    "",
    "⚠️ **Repo public chỉ có Mamba 1-stream** (`MambaSimple.py`) — KHÔNG phải TF-Mamba "
    "dual-stream của paper (không DWT/AdaptiveFusion/proj_s3/PE/GAP trong code public). "
    "`MODEL='tfmamba'` của ta = **kiến trúc paper** (dual-stream, đầy đủ hơn code public).",
    "",
    "**Chia dataset (giống git họ):** HUST random 80/20 (seed 42); UT-HAR official "
    "train=X_train / test=X_test (val KHÔNG dùng; `MERGE_VAL=True` để gộp val vào test); "
    "NTU-Fi train_amp/test_amp. "
    "**Chuẩn hoá 2 tầng, 2 cờ trực giao `PRENORM` × `Z_GRAN`:** "
    "`PRENORM='sensefi'` áp pre-norm raw (UT-HAR min-max / NTU-Fi (x−42.32)/4.98; HUST=none) hoặc "
    "`'none'` (KHÔNG, giong XRF55). `Z_GRAN='perpos'` = z per-position (C,T2,F2) (TF-Mamba) hoặc "
    "`'pcb'` = z per-channel-bin (C,F2) gop thời gian (XRF55). z-norm sau DWT LUÔN áp. "
    "4 tổ hợp: `sensefi+perpos` (=TF-Mamba) | `none+pcb` (=XRF55) | `sensefi+pcb` (lai) | `none+perpos`.",
    "",
    "**So sánh CÔNG BẰNG giữa các MODEL:** `make_cfg` chỉ phụ thuộc `PROTOCOL` (KHÔNG phụ "
    "thuộc MODEL); pre-norm + split áp ở tầng build (độc lập model); z-norm áp cho tất cả. "
    "→ chạy nhiều lần cùng `PROTOCOL`/`PRENORM`/`Z_GRAN`/`MERGE_VAL`/`SEEDS`, chỉ đổi `MODEL` "
    "(`tfmamba` / `s4.nogn` / `s4.nogn_gate`) = **protocol + chia dataset + chuẩn hoá GIỐNG HỆT**, "
    "khác duy nhất là **kiến trúc model** (giữa hai s4 chỉ khác `fusion` convex↔gate). "
    "Mỗi run lưu thư mục riêng theo `MODEL`+`RUN_TAG` nên không ghi đè.",
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
    "MODEL    = 's4.nogn'   # 'tfmamba' (gốc) | 's4.nogn' (S4 bỏ stem GroupNorm) | 's4.nogn_gate' (S4.nogn + fusion gate per-channel)",
    "PROTOCOL = 'theirs'    # 'theirs' (TF-Mamba paper) | 'mine' (02*)",
    "PRENORM  = 'sensefi'   # 'sensefi' = UT-HAR min-max / NTU-Fi (x-42.32)/4.98 (HUST=none) | 'none' = KHÔNG pre-norm (giong XRF55)",
    "Z_GRAN   = 'perpos'    # 'perpos' = z per-position (C,T2,F2) (TF-Mamba) | 'pcb' = z per-channel-bin (C,F2) gop thoi gian (XRF55)",
    "MERGE_VAL = True       # CHI UT-HAR: False=git SenseFi (test=X_test) | True=gộp val vào test",
    "MINE_EARLY_STOP = None # 'mine': mặc định TẮT early-stop. Đặt số (vd 0.01) để bật. 'theirs' cố định 0.01 (tác giả)",
    "WARMUP_EPOCHS = 5      # 'mine': số epoch warmup cho warmup_cosine. 'theirs' không scheduler nên không dùng",
    "DATASETS = ['hust', 'uthar', 'ntufi']",
    "MODES    = ['raw']     # ['raw'] | ['proc'] | ['raw','proc']",
    "SEEDS    = [0, 4, 8, 17, 42]",
    "OUT_ROOT = '/kaggle/working'",
    "",
    "DIRMAP  = {'hust': 'HUST-HAR', 'uthar': 'UT_HAR', 'ntufi': 'NTU-Fi_HAR'}",
    "_MARKER = {'hust': 'HUST_HAR_labels.pt', 'uthar': 'X_train.csv', 'ntufi': 'train_amp'}",
    "assert MODEL in ('tfmamba', 's4.nogn', 's4.nogn_gate'), f'MODEL khong hop le: {MODEL!r}'",
    "assert PRENORM in ('none','sensefi') and Z_GRAN in ('perpos','pcb'), 'PRENORM/Z_GRAN khong hop le'",
    "FORMAT  = 'tfmamba' if MODEL == 'tfmamba' else 'wavmamba'   # build layout per model",
    "WAV_SUBS = 'HL,LH'   # ca s4.nogn lan s4.nogn_gate deu Haar 2 bang {HL,LH} (no LL); chi dung khi FORMAT=wavmamba",
    "RUN_TAG = f'{PROTOCOL}_{PRENORM}_{Z_GRAN}' + ('_mv' if MERGE_VAL else '')   # phan biet run, tranh ghi de",
    "build_py = CODE_PATH / 'xrf55_bench' / 'scripts' / '10_build_multi.py'",
    "",
    "def resolve_mount(ds):",
    "    base = Path('/kaggle/input')",
    "    for c in (sorted(base.iterdir()) if base.is_dir() else []):",
    "        if next(c.rglob(_MARKER[ds]), None) is not None:",
    "            return str(c)",
    "    raise FileNotFoundError(f'Không thấy /kaggle/input/* chứa {_MARKER[ds]} cho {ds}')",
    "",
    "print(f'MODEL={MODEL}  PROTOCOL={PROTOCOL}  PRENORM={PRENORM}  Z_GRAN={Z_GRAN}  MERGE_VAL={MERGE_VAL}  FORMAT={FORMAT}')",
    "print(f'DATASETS={DATASETS}  MODES={MODES}  SEEDS={SEEDS}  RUN_TAG={RUN_TAG}')",
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
    "else:   # s4.nogn / s4.nogn_gate = WavDualMamba Haar 2 bang {HL,LH}, AttnStat, bo stem GroupNorm -> packed C = 18",
    "    from xrf55_bench.models.wavdualmamba.model import WavDualMamba",
    "    _mk = dict(num_classes=6, n_links=1, n_antennas=9, f2=15,",
    "               subbands=('HL', 'LH'), pool='attnstat', stem_norm=False,",
    "               fusion='convex')   # s4.nogn = nogn+convex; pin (default became 'gate' 2026-06)",
    "    if MODEL == 's4.nogn_gate':",
    "        _mk['fusion'] = 'gate'",
    "    _m = WavDualMamba(**_mk).to(dev)",
    "    with torch.no_grad():",
    "        _o = _m(torch.randn(2, 18, 16, 15, device=dev))",
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
    "    if PROTOCOL == 'theirs':   # Protocol TF-Mamba (Liu 2025) — lr + chi tiết theo CODE gốc:",
    "        # lr=1e-3 (theo code release Mamba_HUST-HAR.py; PAPER ghi 1e-4 — mac dinh dung gia tri CODE.",
    "        #  Doi lr=1e-4 neu muon ban theo paper).",
    "        # wd=0.01, betas=(0.9,0.999), eps=1e-8 (mặc định AdamW trong code); CrossEntropyLoss;",
    "        # 40 epochs; bs=32; no scheduler; clip_grad_norm_(max_norm=1.0) (code dòng 140);",
    "        # early stopping: if average_loss < 0.01: break -> early_stop_loss=0.01; report last_model.",
    "        # wd_exclude_norm_bias=False: AdamW(model.parameters()) cua ho decay MOI param",
    "        #   (KHONG loai tru norm/bias/A_log/D/pos_emb) -> ap cho ca tfmamba lan s4.nogn*.",
    "        return TrainCfg_for_protocol('02', seeds=tuple(SEEDS), optimizer='adamw',",
    "                                     lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999),",
    "                                     eps=1e-8, num_epochs=40, batch_size=32,",
    "                                     scheduler=None, warmup_epochs=0, grad_clip=1.0,",
    "                                     criterion='ce', label_smoothing=0.0, early_stop_loss=0.01,",
    "                                     wd_exclude_norm_bias=False)",
    "    # 'mine' (02*): them early_stop_loss=MINE_EARLY_STOP -> dung som khi train-loss",
    "    # < nguong (giong co che 'theirs', nhung nguong tu chinh). None de tat.",
    "    return TrainCfg_for_protocol('02', seeds=tuple(SEEDS), num_epochs=30,",
    "                                 betas=(0.9, 0.95), grad_clip=1.0, lr=5e-4, floor_lr=1e-6,",
    "                                 warmup_epochs=WARMUP_EPOCHS, early_stop_loss=MINE_EARLY_STOP)",
    "",
    "def model_setup(meta):",
    "    F2, nps, T2 = meta['F2'], meta['n_per_sub'], meta['T2']",
    "    if MODEL == 'tfmamba':",
    "        return 'tfmamba', {'num_features': nps * F2, 'max_len': T2}",
    "    mk = {'n_links': 1, 'n_antennas': nps, 'f2': F2,",
    "          'subbands': ('HL', 'LH'), 'pool': 'attnstat', 'stem_norm': False,",
    "          'fusion': 'convex'}   # s4.nogn = nogn+convex; pin (default became 'gate' 2026-06)",
    "    if MODEL == 's4.nogn_gate':",
    "        mk['fusion'] = 'gate'   # per-channel gate thay convex scalar (= S4.nogn_gate)",
    "    return 'wavdualmamba', mk",
    "",
    "def run_one(ds, md):",
    "    raw   = resolve_mount(ds)",
    "    md_sub = f'{md}_{PRENORM}_{Z_GRAN}'   # khop bench dir o build script (4 to hop khong de)",
    "    bench = Path(OUT_ROOT) / DIRMAP[ds] / 'bench' / md_sub",
    "    out   = Path(f'{OUT_ROOT}/outputs/{MODEL}_{ds}_{md}_{RUN_TAG}')",
    "    cmd = [sys.executable, str(build_py), '--dataset', ds, '--mode', md,",
    "           '--raw-root', raw, '--out-root', OUT_ROOT, '--format', FORMAT]",
    "    if FORMAT == 'wavmamba': cmd += ['--wav-subbands', WAV_SUBS]   # s4.nogn* -> 'HL,LH'",
    "    if MERGE_VAL: cmd.append('--merge-val')",
    "    cmd += ['--prenorm', PRENORM, '--z-gran', Z_GRAN]   # 2 co truc giao",
    "    subprocess.run(cmd, check=True)",
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
    "        print(f\"\\n{'#'*64}\\n#  {MODEL} / {ds} / {md} / {RUN_TAG}\\n{'#'*64}\")",
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
    "# Duyet ĐUNG cac thu muc cua MODEL nay (path chinh xac, tranh glob bat nham",
    "# 's4.nogn_gate' khi MODEL='s4.nogn' — 's4.nogn' la tien to cua 's4.nogn_gate').",
    "run_dirs = [Path(f'{OUT_ROOT}/outputs/{MODEL}_{ds}_{md}_{RUN_TAG}')",
    "            for ds in DATASETS for md in MODES]",
    "",
    "print('--- Results ---')",
    "for d in run_dirs:",
    "    mp = d / 'metrics.json'",
    "    if not mp.exists():",
    "        continue",
    "    m = json.load(open(mp)); s = m['summary']",
    "    last = f\"{s['test_accuracy_mean']*100:.2f}±{s['test_accuracy_std']*100:.2f}\"",
    "    best = f\"{s.get('best_test_acc_mean',0)*100:.2f}±{s.get('best_test_acc_std',0)*100:.2f}\"",
    "    print(f\"  {d.name:<34} last={last}  best={best}  F1={s['test_f1_macro_mean']*100:.2f}\")",
    "",
    "print('\\n--- Zips ---')",
    "for d in run_dirs:",
    "    for z in sorted(d.glob('*.zip')):",
    "        shutil.copy2(z, Path('/kaggle/working') / z.name)",
    "        print(z.name); display(FileLink(z.name))",
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

out = Path(__file__).parent.parent / 'notebooks' / 's4nogn_multidataset.ipynb'
json.dump(nb, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'wrote {out}  ({len(cells)} cells)')
