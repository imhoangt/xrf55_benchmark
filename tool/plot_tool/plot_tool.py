"""
plot_tool.py — XRF55 CSI signal visualization (interactive)
===========================================================

Tool tương tác vẽ đồ thị tín hiệu CSI cho dataset XRF55, sao y phong cách của
xrf55_bench (scripts/03_plot_amplitude.py + legacy/src/evaluation/plots.py).

Nguồn dữ liệu: CSI PHỨC gốc trong dataset/XRF55/raw/scene_01, đọc qua parser của
xrf55_bench. Cả biên độ (|csi|) lẫn pha (angle(csi)) đều dẫn xuất từ một nguồn phức
này — vì raw_npy_nosc (270,1000) chỉ chứa biên độ, không có pha.

Người dùng chọn:
    1. Hành động (01–11)   → label 0–10 → action_id 31–41 (ACTION_NAMES)
    2. Người    (01–30)   → vol_id
    3. Lần lặp  (01–20)   → rep_id
    4. RX       (01–03)   → 1 trong 3 thiết bị RX (dùng antenna 0 của RX đó)
    5. Tiền xử lý biên độ? (y/n) — bật/tắt overlay Hampel + Butterworth LPF.
       Pha LUÔN ở dạng thô (không tiền xử lý).

Xuất 4 file PNG vào tool/plot_tool/outputs/ (subcarrier 10/30 cho đồ thị 1D):
    {base}_amp_1d.png        biên độ 1D — raw (tắt) / raw+proc (bật)
    {base}_amp_heatmap.png   biên độ heatmap 30 subcarrier — raw (tắt) / raw|proc (bật)
    {base}_phase_1d.png      pha thô 1D
    {base}_phase_heatmap.png pha thô heatmap 30 subcarrier
trong đó base = "{vol:02d}_{action_id:02d}_{rep:02d}_rx{rx:02d}".

Usage:
    cd har_csi
    python tool/plot_tool/plot_tool.py
"""
import sys
from pathlib import Path

# Console Windows mặc định cp1252 không in được tiếng Việt → ép UTF-8.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

# tool/plot_tool/plot_tool.py → har_csi/ (đi lên 2 cấp) để import được xrf55_bench.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from xrf55_bench.preprocessing.parser import (
    ACTION_NAMES, ACTION_ID_TO_LABEL, load_xrf55_sample,
)
from xrf55_bench.preprocessing.amplitude import hampel_vectorized


# ── Config ──────────────────────────────────────────────────────────────────────
SCENE_DIR_DEFAULT = PROJECT_ROOT / 'dataset' / 'XRF55' / 'raw' / 'scene_01'
OUT_DIR_DEFAULT   = Path(__file__).resolve().parent / 'outputs'

VIZ_SUBCARRIER = 9        # 0-indexed → "Subcarrier 10/30" cho đồ thị 1D
_DUR_S = 5.0              # 1000 frame @ 200 Hz → 0–5 s
_N_SC  = 30
_FS    = 200.0

# Kích thước figure đồng nhất: mọi ảnh 1-panel (1D & 1 heatmap) cùng (12, 4);
# ảnh 2-panel (raw|proc) rộng gấp đôi nhưng cùng chiều cao 4.
_FIG_1P = (12, 4)
_FIG_2P = (16, 4)

# Butterworth bậc 4, cutoff 20 Hz — y hệt 03_plot_amplitude.py.
_SOS = butter(4, 20.0, btype='low', fs=_FS, output='sos')

# label (0–10) → action_id (31–41). ACTION_ID_TO_LABEL: {31:0, …, 41:10}.
_LABEL_TO_ACTION_ID = {lbl: aid for aid, lbl in ACTION_ID_TO_LABEL.items()}


# ── Nạp + xử lý dữ liệu ───────────────────────────────────────────────────────────

def _load_channel(scene_dir: Path, vol: int, action_id: int, rep: int, rx: int):
    """Trả (raw_amp, raw_phase) đều (1000, 30) cho RX `rx`, antenna 0.

    csi phức (1000, 30, M=3, A=3) = (time, subcarrier, device, antenna);
    chọn device rx-1, antenna 0 → (1000, 30) phức.
    """
    csi  = load_xrf55_sample(scene_dir, vol, action_id, rep)   # (1000,30,3,3) complex
    chan = csi[:, :, rx - 1, 0]                                 # (1000, 30) complex
    return np.abs(chan).astype(np.float32), np.angle(chan).astype(np.float32)


def _process_amp(raw_amp: np.ndarray) -> np.ndarray:
    """Hampel (window=8, n_sigma=3) + Butterworth LPF — y hệt 03_plot_amplitude.py.

    hampel_vectorized lọc theo axis=1 (time), nên đưa về (1, 1000, 30) rồi bóc ra.
    """
    x = raw_amp[None, :, :].astype(np.float32)                 # (1, 1000, 30)
    x = hampel_vectorized(x, window=8, n_sigma=3.0)
    x = sosfiltfilt(_SOS, x, axis=1).astype(np.float32)
    return x[0]                                                # (1000, 30)


def _suptitle(action_name: str, vol: int, rx: int, rep: int,
              subcarrier: int = None, all_sc: bool = False) -> str:
    parts = [action_name, f"Subject {vol:02d}", f"Rx {rx:02d}"]
    if all_sc:
        parts.append("All 30 Subcarriers")
    elif subcarrier is not None:
        parts.append(f"Subcarrier {subcarrier + 1}/30")
    parts.append(f"Rep {rep:02d}")
    return "  |  ".join(parts)


# ── Đồ thị ──────────────────────────────────────────────────────────────────────

def plot_amp_1d(raw, proc, info, out_path, with_proc: bool):
    """Biên độ 1D, 1 subcarrier. with_proc=False → chỉ raw (mục 1); True → raw+proc (mục 2)."""
    sc = VIZ_SUBCARRIER
    t  = np.linspace(0, _DUR_S, raw.shape[0], endpoint=False)
    fig, ax = plt.subplots(figsize=_FIG_1P)
    ax.plot(t, raw[:, sc], color='steelblue', lw=0.8, alpha=0.8, label='Raw CSI Amplitude')
    if with_proc:
        ax.plot(t, proc[:, sc], color='darkorange', lw=1.2,
                label='After Hampel + Butterworth LPF')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('CSI Amplitude raw and after preprocessing' if with_proc
                 else 'Raw CSI Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.suptitle(_suptitle(**info, subcarrier=sc), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_amp_heatmap(raw, proc, info, out_path, with_proc: bool):
    """Biên độ heatmap 30 subcarrier. False → 1 panel raw (mục 3); True → raw|proc cạnh nhau (mục 3+4)."""
    if with_proc:
        fig, axes = plt.subplots(1, 2, figsize=_FIG_2P)
        panels = [
            (axes[0], raw,  'Raw CSI Amplitude'),
            (axes[1], proc, 'CSI Amplitude after Hampel + Butterworth LPF'),
        ]
    else:
        fig, ax = plt.subplots(figsize=_FIG_1P)
        panels = [(ax, raw, 'Raw CSI Amplitude')]
    for ax, data, title in panels:
        vmin, vmax = np.percentile(data, [1, 99])
        im = ax.imshow(data.T, aspect='auto', origin='lower', cmap='viridis',
                       vmin=vmin, vmax=vmax, extent=[0, _DUR_S, 0, _N_SC - 1])
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Subcarrier')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(_suptitle(**info, all_sc=True), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_phase_1d(raw_ph, info, out_path):
    """Pha thô 1D, 1 subcarrier (mục 5)."""
    sc = VIZ_SUBCARRIER
    t  = np.linspace(0, _DUR_S, raw_ph.shape[0], endpoint=False)
    fig, ax = plt.subplots(figsize=_FIG_1P)
    ax.plot(t, raw_ph[:, sc], color='steelblue', lw=0.8, alpha=0.8,
            label='Raw CSI Phase')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Raw CSI Phase')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.suptitle(_suptitle(**info, subcarrier=sc), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_phase_heatmap(raw_ph, info, out_path):
    """Pha thô heatmap 30 subcarrier — panel trái của phase_raw_vs_proc (mục 6)."""
    fig, ax = plt.subplots(figsize=_FIG_1P)
    vmax = float(np.percentile(np.abs(raw_ph), 99))
    im = ax.imshow(raw_ph.T, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, extent=[0, _DUR_S, 0, _N_SC - 1])
    ax.set_title('Raw CSI Phase')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Subcarrier')
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(_suptitle(**info, all_sc=True), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Nhập liệu ─────────────────────────────────────────────────────────────────────

def _ask_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
        except ValueError:
            print(f"   -> Lỗi: nhập một số nguyên trong [{lo}, {hi}].")
            continue
        if lo <= v <= hi:
            return v
        print(f"   -> Ngoài khoảng: phải trong [{lo}, {hi}].")


def _ask_yesno(prompt: str) -> bool:
    while True:
        s = input(prompt).strip().lower()
        if s in ('y', 'yes', 'c', 'co', 'có', '1'):
            return True
        if s in ('n', 'no', 'k', 'khong', 'không', '0'):
            return False
        print("   -> Nhập y/n.")


def main():
    print("=" * 62)
    print(" PLOT TOOL — Trực quan hóa tín hiệu CSI (XRF55)")
    print("=" * 62)
    print(" Danh mục hành động (01–11):")
    for lbl, name in enumerate(ACTION_NAMES):
        col2 = f"   {lbl + 7:02d}. {ACTION_NAMES[lbl + 6]}" if lbl < 5 else ""
        if lbl < 6:
            print(f"   {lbl + 1:02d}. {name:<20}{col2}")
    print("-" * 62)

    action_choice = _ask_int("1. Chọn hành động (01–11): ", 1, 11)
    vol           = _ask_int("2. Chọn người    (01–30): ", 1, 30)
    rep           = _ask_int("3. Chọn lần lặp  (01–20): ", 1, 20)
    rx            = _ask_int("4. Chọn RX       (01–03): ", 1, 3)
    with_proc     = _ask_yesno("5. Tiền xử lý biên độ? (y/n) [pha luôn thô]: ")

    label       = action_choice - 1
    action_id   = _LABEL_TO_ACTION_ID[label]
    action_name = ACTION_NAMES[label]

    print("\n" + "-" * 62)
    print(f" Mẫu: {action_name} | Subject {vol:02d} | Rx {rx:02d} | Rep {rep:02d}"
          f" | action_id={action_id}")
    print(f" Tiền xử lý biên độ: {'BẬT (Hampel + LPF)' if with_proc else 'TẮT (raw)'}")
    print("-" * 62)

    if not SCENE_DIR_DEFAULT.exists():
        print(f"[Lỗi] Không tìm thấy thư mục dữ liệu: {SCENE_DIR_DEFAULT}")
        input("Nhấn Enter để thoát...")
        return

    try:
        raw_amp, raw_phase = _load_channel(SCENE_DIR_DEFAULT, vol, action_id, rep, rx)
    except FileNotFoundError as e:
        print(f"[Lỗi] Không tìm thấy mẫu: {e}")
        input("Nhấn Enter để thoát...")
        return

    proc_amp = _process_amp(raw_amp) if with_proc else None
    print(f"[Nạp xong] amplitude/phase shape {raw_amp.shape}  "
          f"(amp: {raw_amp.min():.2f}…{raw_amp.max():.2f})")

    OUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
    base = f"{vol:02d}_{action_id:02d}_{rep:02d}_rx{rx:02d}"
    info = dict(action_name=action_name, vol=vol, rx=rx, rep=rep)

    p_amp_1d  = OUT_DIR_DEFAULT / f"{base}_amp_1d.png"
    p_amp_hm  = OUT_DIR_DEFAULT / f"{base}_amp_heatmap.png"
    p_ph_1d   = OUT_DIR_DEFAULT / f"{base}_phase_1d.png"
    p_ph_hm   = OUT_DIR_DEFAULT / f"{base}_phase_heatmap.png"

    plot_amp_1d(raw_amp, proc_amp, info, p_amp_1d, with_proc)
    print(f"  ✓ {p_amp_1d.name}   (biên độ 1D, {'raw+proc' if with_proc else 'raw'})")
    plot_amp_heatmap(raw_amp, proc_amp, info, p_amp_hm, with_proc)
    print(f"  ✓ {p_amp_hm.name}   (biên độ heatmap, {'raw|proc' if with_proc else 'raw'})")
    plot_phase_1d(raw_phase, info, p_ph_1d)
    print(f"  ✓ {p_ph_1d.name}   (pha thô 1D)")
    plot_phase_heatmap(raw_phase, info, p_ph_hm)
    print(f"  ✓ {p_ph_hm.name}   (pha thô heatmap)")

    print(f"\n[Thành công] 4 đồ thị đã lưu vào {OUT_DIR_DEFAULT}")
    input("Nhấn Enter để thoát...")


if __name__ == '__main__':
    main()
