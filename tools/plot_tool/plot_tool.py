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
                            Nhiều RX → mỗi đồ thị xếp NGANG, mỗi RX một cột.
    5. Antenna  (01–03)   → CÓ THỂ chọn nhiều antenna, cách nhau bởi dấu phẩy (vd: 01, 02, 03).
                            Nhiều antenna → xếp DỌC, mỗi antenna một hàng.
                            Layout tổng quát: rows = antenna, columns = RX.
    6. Subcarrier (01–30)  → subcarrier hiển thị trong đồ thị 1D (heatmap luôn hiện đủ 30).
    7. Vẽ đồ thị pha? (y/n) — bật/tắt 2 file pha (phase_1d + phase_heatmap).
       Pha LUÔN ở dạng thô (không tiền xử lý).
    8. Tiền xử lý biên độ? (y/n) — bật/tắt overlay Hampel + Butterworth LPF.

Xuất 2–4 file PNG vào tools/plot_tool/outputs/:
    {base}_amp_1d.png        biên độ 1D tại subcarrier đã chọn — raw / raw+proc  [luôn xuất]
    {base}_amp_heatmap.png   biên độ heatmap 30 subcarrier — raw / raw|proc       [luôn xuất]
    {base}_phase_1d.png      pha thô 1D tại subcarrier đã chọn                    [nếu chọn pha]
    {base}_phase_heatmap.png pha thô heatmap 30 subcarrier                         [nếu chọn pha]
trong đó base = "{vol:02d}_{action_id:02d}_{rep:02d}_rx{rx-list}_ant{ant-list}_sc{sc:02d}".

Amp heatmap với tiền xử lý: mỗi RX chiếm 2 cột liền nhau (raw | proc).

Usage:
    cd har_csi
    python tools/plot_tool/plot_tool.py
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

# tools/plot_tool/plot_tool.py → har_csi/ (đi lên 3 cấp) để import được xrf55_bench.
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

_DUR_S = 5.0              # 1000 frame @ 200 Hz → 0–5 s
_N_SC  = 30
_FS    = 200.0

# Kích thước figure: mỗi cột (1 RX) rộng _FIG_W_1P; mỗi hàng (1 antenna) cao _FIG_H_ROW.
# Amp heatmap với proc: mỗi RX chiếm 2 cột (raw|proc) → độ rộng nhân _FIG_W_2P mỗi RX.
_FIG_W_1P  = 12   # rộng mỗi RX-column cho đồ thị 1-panel (1D, phase, heatmap raw-only)
_FIG_W_2P  = 16   # rộng mỗi RX-column cho amp heatmap với proc (2 panels: raw + proc)
_FIG_H_ROW = 4    # cao mỗi antenna-row

# Font sizes — đồng nhất cho mọi đồ thị.
_FS_SUPTITLE = 13   # tiêu đề tổng (fig.suptitle)
_FS_TITLE    = 11   # tiêu đề mỗi ô (ax.set_title)
_FS_LEGEND   = 9    # chú thích (ax.legend)
_FS_LABEL    = 9    # nhãn trục (xlabel, ylabel)

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


def _suptitle(action_name: str, vol: int, rx_label: str, ant_label: str, rep: int,
              subcarrier: int = None, all_sc: bool = False) -> str:
    parts = [action_name, f"Subject {vol:02d}", f"Rx {rx_label}", f"Ant {ant_label}"]
    if all_sc:
        parts.append("All 30 Subcarriers")
    elif subcarrier is not None:
        parts.append(f"Subcarrier {subcarrier + 1}/30")
    parts.append(f"Rep {rep:02d}")
    return "  |  ".join(parts)


# ── Đồ thị (rows = antenna, cols = RX) ─────────────────────────────────────────────

def plot_amp_1d(chans_grid, info, out_path, with_proc: bool, sc: int):
    """Biên độ 1D tại subcarrier `sc` (0-indexed). with_proc=False → raw; True → raw+proc.

    chans_grid[i_ant][i_rx] = dict(rx, ant, raw_amp, proc_amp, raw_ph).
    """
    n_ant = len(chans_grid)
    n_rx  = len(chans_grid[0])
    t  = np.linspace(0, _DUR_S, chans_grid[0][0]['raw_amp'].shape[0], endpoint=False)
    fig, axes = plt.subplots(n_ant, n_rx,
                             figsize=(_FIG_W_1P * n_rx, _FIG_H_ROW * n_ant),
                             squeeze=False, sharex=True)
    for i, row in enumerate(chans_grid):
        for j, ch in enumerate(row):
            ax = axes[i, j]
            ax.plot(t, ch['raw_amp'][:, sc], color='steelblue', lw=0.8, alpha=0.8,
                    label='Raw CSI Amplitude')
            if with_proc:
                ax.plot(t, ch['proc_amp'][:, sc], color='darkorange', lw=1.2,
                        label='After Hampel + Butterworth LPF')
            ax.set_title(f"Rx {ch['rx']:02d} / Ant {ch['ant']:02d}", fontsize=_FS_TITLE)
            ax.legend(loc='upper right', fontsize=_FS_LEGEND)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel('Amplitude', fontsize=_FS_LABEL)
            if i == n_ant - 1:
                ax.set_xlabel('Time (s)', fontsize=_FS_LABEL)
    fig.suptitle(_suptitle(**info, subcarrier=sc), fontsize=_FS_SUPTITLE)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_amp_heatmap(chans_grid, info, out_path, with_proc: bool):
    """Biên độ heatmap 30 subcarrier.

    Layout: rows = antenna, cols = RX × (1 nếu raw-only | 2 nếu raw+proc).
    Khi with_proc=True, cột j*2 = raw, cột j*2+1 = proc cho RX thứ j.
    """
    n_ant = len(chans_grid)
    n_rx  = len(chans_grid[0])
    ncol_per_rx = 2 if with_proc else 1
    total_cols  = n_rx * ncol_per_rx
    fig_w = _FIG_W_2P * n_rx if with_proc else _FIG_W_1P * n_rx
    fig, axes = plt.subplots(n_ant, total_cols,
                             figsize=(fig_w, _FIG_H_ROW * n_ant),
                             squeeze=False)
    for i, row in enumerate(chans_grid):
        for j, ch in enumerate(row):
            cells = [(ch['raw_amp'], f"Rx {ch['rx']:02d} / Ant {ch['ant']:02d} — Raw")]
            if with_proc:
                cells.append((ch['proc_amp'],
                               f"Rx {ch['rx']:02d} / Ant {ch['ant']:02d} — Hampel+LPF"))
            for k, (data, title) in enumerate(cells):
                ax = axes[i, j * ncol_per_rx + k]
                vmin, vmax = np.percentile(data, [1, 99])
                im = ax.imshow(data.T, aspect='auto', origin='lower', cmap='viridis',
                               vmin=vmin, vmax=vmax,
                               extent=[0, _DUR_S, 0, _N_SC - 1])
                ax.set_title(title, fontsize=_FS_TITLE)
                ax.set_xlabel('Time (s)', fontsize=_FS_LABEL)
                ax.set_ylabel('Subcarrier', fontsize=_FS_LABEL)
                plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(_suptitle(**info, all_sc=True), fontsize=_FS_SUPTITLE)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_phase_1d(chans_grid, info, out_path, sc: int):
    """Pha thô 1D tại subcarrier `sc` (0-indexed). rows = antenna, cols = RX."""
    n_ant = len(chans_grid)
    n_rx  = len(chans_grid[0])
    t  = np.linspace(0, _DUR_S, chans_grid[0][0]['raw_ph'].shape[0], endpoint=False)
    fig, axes = plt.subplots(n_ant, n_rx,
                             figsize=(_FIG_W_1P * n_rx, _FIG_H_ROW * n_ant),
                             squeeze=False, sharex=True)
    for i, row in enumerate(chans_grid):
        for j, ch in enumerate(row):
            ax = axes[i, j]
            ax.plot(t, ch['raw_ph'][:, sc], color='steelblue', lw=0.8, alpha=0.8,
                    label='Raw CSI Phase')
            ax.set_title(f"Rx {ch['rx']:02d} / Ant {ch['ant']:02d}", fontsize=_FS_TITLE)
            ax.legend(loc='upper right', fontsize=_FS_LEGEND)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel('Phase (rad)', fontsize=_FS_LABEL)
            if i == n_ant - 1:
                ax.set_xlabel('Time (s)', fontsize=_FS_LABEL)
    fig.suptitle(_suptitle(**info, subcarrier=sc), fontsize=_FS_SUPTITLE)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_phase_heatmap(chans_grid, info, out_path):
    """Pha thô heatmap 30 subcarrier. rows = antenna, cols = RX."""
    n_ant = len(chans_grid)
    n_rx  = len(chans_grid[0])
    fig, axes = plt.subplots(n_ant, n_rx,
                             figsize=(_FIG_W_1P * n_rx, _FIG_H_ROW * n_ant),
                             squeeze=False)
    for i, row in enumerate(chans_grid):
        for j, ch in enumerate(row):
            ax = axes[i, j]
            data = ch['raw_ph']
            vmax = float(np.percentile(np.abs(data), 99))
            im = ax.imshow(data.T, aspect='auto', origin='lower', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax,
                           extent=[0, _DUR_S, 0, _N_SC - 1])
            ax.set_title(f"Rx {ch['rx']:02d} / Ant {ch['ant']:02d}", fontsize=_FS_TITLE)
            ax.set_xlabel('Time (s)', fontsize=_FS_LABEL)
            ax.set_ylabel('Subcarrier', fontsize=_FS_LABEL)
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(_suptitle(**info, all_sc=True), fontsize=_FS_SUPTITLE)
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
    rx_list       = _ask_int_list(
        "4. Chọn RX      (01–03, nhiều RX cách bởi dấu phẩy, vd 1 2 3): ", 1, 3)
    ant_list      = _ask_int_list(
        "5. Chọn antenna (01–03, nhiều ant cách bởi dấu phẩy, vd 1 2 3): ", 1, 3)
    sc_user       = _ask_int("6. Chọn subcarrier 1D   (01–30): ", 1, 30)
    sc            = sc_user - 1                    # 0-indexed dùng nội bộ
    plot_phase    = _ask_yesno("7. Vẽ đồ thị pha?       (y/n): ")
    with_proc     = _ask_yesno("8. Tiền xử lý biên độ?  (y/n) [pha luôn thô]: ")

    label       = action_choice - 1
    action_id   = _LABEL_TO_ACTION_ID[label]
    action_name = ACTION_NAMES[label]
    rx_label    = ", ".join(f"{r:02d}" for r in rx_list)
    ant_label   = ", ".join(f"{a:02d}" for a in ant_list)

    print("\n" + "-" * 62)
    print(f" Mẫu: {action_name} | Subject {vol:02d} | Rx {rx_label}"
          f" | Ant {ant_label} | Rep {rep:02d} | action_id={action_id}")
    print(f" Layout: {len(ant_list)} hàng (antenna) × {len(rx_list)} cột (RX)"
          f" = {len(ant_list) * len(rx_list)} ô")
    print(f" Subcarrier 1D:      {sc_user:02d}/30")
    print(f" Vẽ pha:             {'CÓ' if plot_phase else 'KHÔNG'}")
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

    # chans_grid[i_ant][i_rx]: outer loop = antenna (rows), inner loop = RX (cols).
    chans_grid = []
    for ant in ant_list:
        row = []
        for rx in rx_list:
            raw_amp, raw_ph = _channel(csi, rx, ant)
            row.append(dict(
                rx=rx,
                ant=ant,
                raw_amp=raw_amp,
                proc_amp=_process_amp(raw_amp) if with_proc else None,
                raw_ph=raw_ph if plot_phase else None,
            ))
        chans_grid.append(row)
    print(f"[Nạp xong] {len(ant_list)} ant × {len(rx_list)} RX"
          f" = {len(ant_list) * len(rx_list)} kênh | (1000, 30) mỗi kênh")

    OUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
    rx_tag  = "-".join(f"{r:02d}" for r in rx_list)
    ant_tag = "-".join(f"{a:02d}" for a in ant_list)
    base    = f"{vol:02d}_{action_id:02d}_{rep:02d}_rx{rx_tag}_ant{ant_tag}_sc{sc_user:02d}"
    info    = dict(action_name=action_name, vol=vol, rx_label=rx_label,
                   ant_label=ant_label, rep=rep)

    p_amp_1d = OUT_DIR_DEFAULT / f"{base}_amp_1d.png"
    p_amp_hm = OUT_DIR_DEFAULT / f"{base}_amp_heatmap.png"
    p_ph_1d  = OUT_DIR_DEFAULT / f"{base}_phase_1d.png"
    p_ph_hm  = OUT_DIR_DEFAULT / f"{base}_phase_heatmap.png"

    n_saved = 0
    plot_amp_1d(chans_grid, info, p_amp_1d, with_proc, sc)
    print(f"  ✓ {p_amp_1d.name}   (biên độ 1D sc{sc_user:02d}, {'raw+proc' if with_proc else 'raw'})")
    n_saved += 1
    plot_amp_heatmap(chans_grid, info, p_amp_hm, with_proc)
    print(f"  ✓ {p_amp_hm.name}   (biên độ heatmap, {'raw|proc' if with_proc else 'raw'})")
    n_saved += 1
    if plot_phase:
        plot_phase_1d(chans_grid, info, p_ph_1d, sc)
        print(f"  ✓ {p_ph_1d.name}   (pha thô 1D sc{sc_user:02d})")
        n_saved += 1
        plot_phase_heatmap(chans_grid, info, p_ph_hm)
        print(f"  ✓ {p_ph_hm.name}   (pha thô heatmap)")
        n_saved += 1

    print(f"\n[Thành công] {n_saved} đồ thị đã lưu vào {OUT_DIR_DEFAULT}")
    input("Nhấn Enter để thoát...")


if __name__ == '__main__':
    main()
