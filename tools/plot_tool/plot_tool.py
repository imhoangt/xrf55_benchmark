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
    4. RX       (01–03)   → CÓ THỂ chọn nhiều RX, cách nhau bởi dấu phẩy (vd: 01, 02, 03).
                            Nhiều RX → mỗi đồ thị xếp DỌC, mỗi RX một hàng.
    5. Antenna  (01–03)   → antenna trong mỗi RX (dùng chung cho mọi RX đã chọn).
    6. Tiền xử lý biên độ? (y/n) — bật/tắt overlay Hampel + Butterworth LPF.
       Pha LUÔN ở dạng thô (không tiền xử lý).

Xuất 4 file PNG vào tool/plot_tool/outputs/ (subcarrier 10/30 cho đồ thị 1D):
    {base}_amp_1d.png        biên độ 1D — raw (tắt) / raw+proc (bật)
    {base}_amp_heatmap.png   biên độ heatmap 30 subcarrier — raw (tắt) / raw|proc (bật)
    {base}_phase_1d.png      pha thô 1D
    {base}_phase_heatmap.png pha thô heatmap 30 subcarrier
trong đó base = "{vol:02d}_{action_id:02d}_{rep:02d}_rx{rx-list}_ant{ant:02d}".

Usage:
    cd har_csi
    python tool/plot_tool/plot_tool.py
"""
import re
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

# Kích thước figure đồng nhất: mỗi HÀNG (1 RX) cao 4; ảnh 1-panel rộng 12,
# ảnh 2-panel (raw|proc) rộng 16. Nhiều RX → chiều cao nhân theo số RX.
_FIG_W_1P = 12
_FIG_W_2P = 16
_FIG_H_ROW = 4

# Butterworth bậc 4, cutoff 20 Hz — y hệt 03_plot_amplitude.py.
_SOS = butter(4, 20.0, btype='low', fs=_FS, output='sos')

# label (0–10) → action_id (31–41). ACTION_ID_TO_LABEL: {31:0, …, 41:10}.
_LABEL_TO_ACTION_ID = {lbl: aid for aid, lbl in ACTION_ID_TO_LABEL.items()}


# ── Nạp + xử lý dữ liệu ───────────────────────────────────────────────────────────

def _channel(csi: np.ndarray, rx: int, ant: int):
    """Trả (raw_amp, raw_phase) đều (1000, 30) cho RX `rx`, antenna `ant`.

    csi phức (1000, 30, M=3, A=3) = (time, subcarrier, device, antenna);
    chọn device rx-1, antenna ant-1 → (1000, 30) phức.
    """
    chan = csi[:, :, rx - 1, ant - 1]                          # (1000, 30) complex
    return np.abs(chan).astype(np.float32), np.angle(chan).astype(np.float32)


def _process_amp(raw_amp: np.ndarray) -> np.ndarray:
    """Hampel (window=8, n_sigma=3) + Butterworth LPF — y hệt 03_plot_amplitude.py.

    hampel_vectorized lọc theo axis=1 (time), nên đưa về (1, 1000, 30) rồi bóc ra.
    """
    x = raw_amp[None, :, :].astype(np.float32)                 # (1, 1000, 30)
    x = hampel_vectorized(x, window=8, n_sigma=3.0)
    x = sosfiltfilt(_SOS, x, axis=1).astype(np.float32)
    return x[0]                                                # (1000, 30)


def _suptitle(action_name: str, vol: int, rx_label: str, ant: int, rep: int,
              subcarrier: int = None, all_sc: bool = False) -> str:
    parts = [action_name, f"Subject {vol:02d}", f"Rx {rx_label}", f"Ant {ant:02d}"]
    if all_sc:
        parts.append("All 30 Subcarriers")
    elif subcarrier is not None:
        parts.append(f"Subcarrier {subcarrier + 1}/30")
    parts.append(f"Rep {rep:02d}")
    return "  |  ".join(parts)


def _row_title(base_title: str, rx: int, n_rx: int) -> str:
    """Tiền tố 'Rx 0X — ' cho từng hàng khi vẽ nhiều RX (1 RX thì giữ nguyên)."""
    return f"Rx {rx:02d} — {base_title}" if n_rx > 1 else base_title


# ── Đồ thị (mỗi RX = 1 hàng, xếp dọc) ─────────────────────────────────────────────

def plot_amp_1d(chans, info, out_path, with_proc: bool):
    """Biên độ 1D, 1 subcarrier. with_proc=False → chỉ raw; True → raw+proc."""
    sc, n = VIZ_SUBCARRIER, len(chans)
    t = np.linspace(0, _DUR_S, chans[0]['raw_amp'].shape[0], endpoint=False)
    base_title = ('CSI Amplitude raw and after preprocessing' if with_proc
                  else 'Raw CSI Amplitude')
    fig, axes = plt.subplots(n, 1, figsize=(_FIG_W_1P, _FIG_H_ROW * n),
                             squeeze=False, sharex=True)
    for i, ch in enumerate(chans):
        ax = axes[i, 0]
        ax.plot(t, ch['raw_amp'][:, sc], color='steelblue', lw=0.8, alpha=0.8,
                label='Raw CSI Amplitude')
        if with_proc:
            ax.plot(t, ch['proc_amp'][:, sc], color='darkorange', lw=1.2,
                    label='After Hampel + Butterworth LPF')
        ax.set_ylabel('Amplitude')
        ax.set_title(_row_title(base_title, ch['rx'], n))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('Time (s)')
    fig.suptitle(_suptitle(**info, subcarrier=sc), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_amp_heatmap(chans, info, out_path, with_proc: bool):
    """Biên độ heatmap 30 subcarrier. n hàng (RX) × (1 cột raw | 2 cột raw,proc)."""
    n = len(chans)
    ncol = 2 if with_proc else 1
    width = _FIG_W_2P if with_proc else _FIG_W_1P
    fig, axes = plt.subplots(n, ncol, figsize=(width, _FIG_H_ROW * n), squeeze=False)
    for i, ch in enumerate(chans):
        cells = [(ch['raw_amp'], 'Raw CSI Amplitude')]
        if with_proc:
            cells.append((ch['proc_amp'], 'CSI Amplitude after Hampel + Butterworth LPF'))
        for j, (data, title) in enumerate(cells):
            ax = axes[i, j]
            vmin, vmax = np.percentile(data, [1, 99])
            im = ax.imshow(data.T, aspect='auto', origin='lower', cmap='viridis',
                           vmin=vmin, vmax=vmax, extent=[0, _DUR_S, 0, _N_SC - 1])
            ax.set_title(_row_title(title, ch['rx'], n))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Subcarrier')
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(_suptitle(**info, all_sc=True), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_phase_1d(chans, info, out_path):
    """Pha thô 1D, 1 subcarrier."""
    sc, n = VIZ_SUBCARRIER, len(chans)
    t = np.linspace(0, _DUR_S, chans[0]['raw_ph'].shape[0], endpoint=False)
    fig, axes = plt.subplots(n, 1, figsize=(_FIG_W_1P, _FIG_H_ROW * n),
                             squeeze=False, sharex=True)
    for i, ch in enumerate(chans):
        ax = axes[i, 0]
        ax.plot(t, ch['raw_ph'][:, sc], color='steelblue', lw=0.8, alpha=0.8,
                label='Raw CSI Phase')
        ax.set_ylabel('Phase (rad)')
        ax.set_title(_row_title('Raw CSI Phase', ch['rx'], n))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('Time (s)')
    fig.suptitle(_suptitle(**info, subcarrier=sc), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_phase_heatmap(chans, info, out_path):
    """Pha thô heatmap 30 subcarrier."""
    n = len(chans)
    fig, axes = plt.subplots(n, 1, figsize=(_FIG_W_1P, _FIG_H_ROW * n), squeeze=False)
    for i, ch in enumerate(chans):
        ax = axes[i, 0]
        data = ch['raw_ph']
        vmax = float(np.percentile(np.abs(data), 99))
        im = ax.imshow(data.T, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, extent=[0, _DUR_S, 0, _N_SC - 1])
        ax.set_title(_row_title('Raw CSI Phase', ch['rx'], n))
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


def _ask_int_list(prompt: str, lo: int, hi: int) -> list:
    """Nhận một hoặc nhiều số nguyên, cách nhau bởi dấu phẩy/khoảng trắng.
    Trả danh sách đã sắp xếp tăng + khử trùng lặp."""
    while True:
        s = input(prompt).strip()
        toks = [t for t in re.split(r'[,\s]+', s) if t]
        try:
            vals = [int(t) for t in toks]
        except ValueError:
            print(f"   -> Lỗi: nhập các số nguyên trong [{lo}, {hi}], cách nhau bởi dấu phẩy.")
            continue
        if vals and all(lo <= v <= hi for v in vals):
            return sorted(set(vals))
        print(f"   -> Mỗi giá trị phải trong [{lo}, {hi}] và không được rỗng.")


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
    for lbl in range(6):
        col2 = f"   {lbl + 7:02d}. {ACTION_NAMES[lbl + 6]}" if lbl + 6 < len(ACTION_NAMES) else ""
        print(f"   {lbl + 1:02d}. {ACTION_NAMES[lbl]:<20}{col2}")
    print("-" * 62)

    action_choice = _ask_int("1. Chọn hành động (01–11): ", 1, 11)
    vol           = _ask_int("2. Chọn người    (01–30): ", 1, 30)
    rep           = _ask_int("3. Chọn lần lặp  (01–20): ", 1, 20)
    rx_list       = _ask_int_list("4. Chọn RX (01–03, nhiều RX cách bởi dấu phẩy, vd 01, 02, 03): ", 1, 3)
    ant           = _ask_int("5. Chọn antenna  (01–03): ", 1, 3)
    with_proc     = _ask_yesno("6. Tiền xử lý biên độ? (y/n) [pha luôn thô]: ")

    label       = action_choice - 1
    action_id   = _LABEL_TO_ACTION_ID[label]
    action_name = ACTION_NAMES[label]
    rx_label    = ", ".join(f"{r:02d}" for r in rx_list)

    print("\n" + "-" * 62)
    print(f" Mẫu: {action_name} | Subject {vol:02d} | Rx {rx_label} | Ant {ant:02d}"
          f" | Rep {rep:02d} | action_id={action_id}")
    print(f" Tiền xử lý biên độ: {'BẬT (Hampel + LPF)' if with_proc else 'TẮT (raw)'}")
    print("-" * 62)

    if not SCENE_DIR_DEFAULT.exists():
        print(f"[Lỗi] Không tìm thấy thư mục dữ liệu: {SCENE_DIR_DEFAULT}")
        input("Nhấn Enter để thoát...")
        return

    try:
        csi = load_xrf55_sample(SCENE_DIR_DEFAULT, vol, action_id, rep)  # (1000,30,3,3)
    except FileNotFoundError as e:
        print(f"[Lỗi] Không tìm thấy mẫu: {e}")
        input("Nhấn Enter để thoát...")
        return

    # Một channel (raw_amp, proc_amp, raw_ph) cho mỗi RX đã chọn (cùng antenna).
    chans = []
    for rx in rx_list:
        raw_amp, raw_ph = _channel(csi, rx, ant)
        chans.append(dict(
            rx=rx,
            raw_amp=raw_amp,
            proc_amp=_process_amp(raw_amp) if with_proc else None,
            raw_ph=raw_ph,
        ))
    print(f"[Nạp xong] {len(chans)} RX × (1000, 30)  |  Ant {ant:02d}")

    OUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
    rx_tag = "-".join(f"{r:02d}" for r in rx_list)
    base   = f"{vol:02d}_{action_id:02d}_{rep:02d}_rx{rx_tag}_ant{ant:02d}"
    info   = dict(action_name=action_name, vol=vol, rx_label=rx_label, ant=ant, rep=rep)

    p_amp_1d = OUT_DIR_DEFAULT / f"{base}_amp_1d.png"
    p_amp_hm = OUT_DIR_DEFAULT / f"{base}_amp_heatmap.png"
    p_ph_1d  = OUT_DIR_DEFAULT / f"{base}_phase_1d.png"
    p_ph_hm  = OUT_DIR_DEFAULT / f"{base}_phase_heatmap.png"

    plot_amp_1d(chans, info, p_amp_1d, with_proc)
    print(f"  ✓ {p_amp_1d.name}   (biên độ 1D, {'raw+proc' if with_proc else 'raw'})")
    plot_amp_heatmap(chans, info, p_amp_hm, with_proc)
    print(f"  ✓ {p_amp_hm.name}   (biên độ heatmap, {'raw|proc' if with_proc else 'raw'})")
    plot_phase_1d(chans, info, p_ph_1d)
    print(f"  ✓ {p_ph_1d.name}   (pha thô 1D)")
    plot_phase_heatmap(chans, info, p_ph_hm)
    print(f"  ✓ {p_ph_hm.name}   (pha thô heatmap)")

    print(f"\n[Thành công] 4 đồ thị đã lưu vào {OUT_DIR_DEFAULT}")
    input("Nhấn Enter để thoát...")


if __name__ == '__main__':
    main()
