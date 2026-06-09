import struct
import os
import sys

import numpy as np
import pandas as pd

# Console Windows mặc định cp1252 không in được tiếng Việt → ép UTF-8.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass


PACKET_COUNT = 1000
SEQ_MAX = 4096
DEVICE_CONFIG = {
    'esp': {
        'packet_size': 144,
        # header 16 bytes: seq(2), ts(8), channel(2), agc(1), fft(1), noise(1), rssi(1)
    },
    'asus': {
        'packet_size': 1044,
        # header 20 bytes: seq(2), ts(8), channel(2), agc(4), rssi(4)
    }
}

# ── Tuning của find_anchor_offset / cut_and_pad (đặt tên thay cho hằng số ma thuật) ──
PRESCAN_BACK_US = 600_000       # binary-search lùi 0.6s trước target để chừa biên rớt gói
DEFAULT_TOL_US  = 1_000_000     # 1s: dung sai timestamp khi khớp điểm neo
MAX_FWD_SKIP    = 100           # nếu gói target_seq mất, chấp nhận gói vượt trước ≤ MAX_FWD_SKIP seq
MAX_FWD_GAP     = SEQ_MAX // 2  # ranh giới tròn: seq_diff ≤ ngưỡng ⇒ vượt trước (mất gói);
                                # lớn hơn ⇒ lùi sau (gói trùng / đảo thứ tự)

# ── Cấu hình cho chế độ PLOT (chức năng 3) ────────────────────────────────────────
# Thư mục dữ liệu mặc định (chứa các folder phiên). Sửa 1 dòng này nếu đổi chỗ.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

DEVICE_ANTENNAS = {'esp': 1, 'asus': 4}   # số antenna mỗi thiết bị
N_SUBCARRIER    = 64
VIZ_SUBCARRIER  = 10                       # subcarrier (0-index) cho đồ thị 1D
_DUR_S = 5.0                               # 1000 gói @ 200 Hz → 0–5 s
_FS    = 200.0

# Danh mục hành động (menu) → action_name trong action_events.csv.
# *** Bảng map tạm (đoán) — sửa cột phải khi biết tên CSV chính xác. ***
ACTION_CATALOG = [
    ('sitdown',      'ngoi'),
    ('standup',      'dung'),
    ('pickup',       'cui_nhat'),
    ('fall',         'nga'),
    ('liestill',     'nam_yen'),
    ('walk',         'di'),
    ('run',          'chay'),
    ('emptyroom',    'phong_trong'),
    ('chuan_bi_lai', 'chuan_bi_lai'),
]

# Kích thước figure: mỗi hàng (1 RX) cao 4; 1-panel rộng 12, 2-panel rộng 16.
_FIG_W_1P, _FIG_W_2P, _FIG_H_ROW = 12, 16, 4


def _packet_count(file_path: str, packet_size: int) -> int:
    """Số gói nguyên trong file. Cảnh báo nếu kích thước không chia hết packet_size
    — dấu hiệu chọn nhầm loại thiết bị (esp 144B vs asus 1044B) hoặc file hỏng."""
    file_size = os.path.getsize(file_path)
    if file_size % packet_size != 0:
        print(f"[Cảnh báo] {os.path.basename(file_path)}: kích thước {file_size}B không "
              f"chia hết cho packet_size={packet_size}B — có thể chọn nhầm thiết bị.")
    return file_size // packet_size


def get_time_from_event(event_file: str, action_name: str, repeat_idx: int) -> int:
    if not os.path.exists(event_file):
        print(f"[Lỗi] không tìm thấy file event: {event_file}")
        return None

    try:
        if event_file.endswith('.csv'):
            df = pd.read_csv(event_file)
        else:
            df = pd.read_excel(event_file)
            
        required_columns = ['action_name', 'repeat_index', 'start_unix_us']
        for col in required_columns:
            if col not in df.columns:
                print(f"[Lỗi] File event thiếu cột quan trọng: '{col}'")
                return None
                
        condition = (df['action_name'] == action_name) & (df['repeat_index'] == repeat_idx)
        filtered_df = df[condition]
        
        if filtered_df.empty:
            print(f"[Cảnh báo] Không tìm thấy '{action_name}' với lần lặp {repeat_idx}.")
            return None
            
        raw_time = filtered_df['start_unix_us'].iloc[0]
        target_time = int(float(raw_time))
        
        print(f"[Event Log] đã nạp Timestamp: {target_time} cho {action_name}_rep{repeat_idx}")
        return target_time
        
    except Exception as e:
        print(f"[Lỗi xử lý Dataframe] {e}")
        return None

def find_first_packet_fast(file_path: str, target_time: int, packet_size: int, excel_output: str = "output_packet.xlsx") -> tuple:
    """Gói đầu tiên có ts >= target_time, qua binary-search.

    Giả định (precondition): timestamp các gói KHÔNG giảm dọc theo file (đúng với
    file ghi nối tiếp theo thứ tự tới). Nếu ts đảo lộn, kết quả lower-bound có thể
    lệch — nhưng điểm neo này chỉ dùng làm mốc cho find_anchor_offset (quét tuyến
    có dung sai), nên sai số nhỏ vẫn được hấp thụ ở bước sau.
    """
    if not os.path.exists(file_path):
        print(f"[Lỗi] Không tìm thấy file: {file_path}")
        return None, None

    total_packets = _packet_count(file_path, packet_size)

    if total_packets == 0:
        return None, None

    print(f"[{os.path.basename(file_path)}] Đang quét {total_packets} gói tin...")

    best_seq, best_ts = None, None
    best_index = -1
    
    with open(file_path, 'rb') as f:
        low = 0
        high = total_packets - 1
        
        while low <= high:
            mid = (low + high) // 2
            f.seek(mid * packet_size)
            
            header_data = f.read(10)
            if len(header_data) < 10:
                break
                
            seq, ts = struct.unpack('<HQ', header_data)
            
            if ts >= target_time:
                best_seq, best_ts = seq, ts
                best_index = mid
                high = mid - 1
            else:
                low = mid + 1
                
    if best_index != -1:
        if excel_output is not None:
            data_to_save = {
                "Sequence": [best_seq],
                "Timestamp": [best_ts],
                "Target_Time_Input": [target_time],
                "Time_Difference": [best_ts - target_time]
            }
            try:
                pd.DataFrame(data_to_save).to_excel(excel_output, index=False)
            except Exception as e:
                print(f" -> [Lỗi] Không thể ghi file Excel: {e}")
        print(f" -> Seq: {best_seq} | TS: {best_ts}")
        return best_seq, best_ts
            
    print(f" -> [Thất bại] Không có gói tin nào phù hợp.")
    return None, None

def find_anchor_offset(file_path: str, target_seq: int, target_ts: int, packet_size: int, tolerance: int = DEFAULT_TOL_US) -> int:
    """Offset byte của gói neo (seq == target_seq, hoặc gói vượt trước ≤ MAX_FWD_SKIP
    nếu gói neo bị mất), trong dung sai timestamp `tolerance`.

    Cùng giả định ts không giảm như find_first_packet_fast: binary-search định vị
    sơ bộ điểm bắt đầu (lùi PRESCAN_BACK_US để chừa biên), rồi quét tuyến tới.
    """
    if not os.path.exists(file_path):
        print(f"[Lỗi] Không tìm thấy file: {file_path}")
        return None

    total_packets = _packet_count(file_path, packet_size)
    if total_packets == 0:
        return None

    search_time = target_ts - PRESCAN_BACK_US
    low, high = 0, total_packets - 1
    start_index = 0

    with open(file_path, 'rb') as f:
        while low <= high:
            mid = (low + high) // 2
            f.seek(mid * packet_size)
            header = f.read(10)
            if len(header) < 10:
                break

            _, ts = struct.unpack('<HQ', header)

            if ts >= search_time:
                start_index = mid
                high = mid - 1
            else:
                low = mid + 1

    with open(file_path, 'rb') as f:
        f.seek(start_index * packet_size)
        while True:
            current_offset = f.tell()
            header = f.read(10)             # chỉ cần seq(2)+ts(8), không đọc cả payload
            if len(header) < 10:
                break

            seq, ts = struct.unpack('<HQ', header)

            if ts > target_ts + tolerance:
                print(f" -> [Cảnh báo] Rớt mạng quá lâu. Mốc gần nhất cách xa hơn {tolerance/1_000_000}s.")
                return None

            seq_diff = (seq - target_seq) % SEQ_MAX

            # Khớp đúng gói neo (seq == target_seq).
            if seq_diff == 0 and abs(ts - target_ts) <= tolerance:
                return current_offset

            # Gói neo bị mất: chấp nhận gói vượt trước vài seq (≤ MAX_FWD_SKIP) ở/sau target_ts.
            if ts >= target_ts and 0 < seq_diff <= MAX_FWD_SKIP:
                return current_offset

            f.seek(current_offset + packet_size)   # nhảy tới gói kế (bỏ qua payload)
    return None

def _cut_and_pad_bytes(file_in: str, start_offset: int, target_seq: int, packet_size: int,
                       max_forward_gap: int = MAX_FWD_GAP) -> bytes:
    """Lõi cắt: trả về đúng PACKET_COUNT gói (dạng bytes) căn theo lưới seq bắt đầu
    tại target_seq; đệm gói 0 cho mỗi seq bị mất để 3 receiver thẳng hàng.

    Phân biệt 2 ca không khớp seq (mấu chốt — đừng gộp chung):
      • seq VƯỢT TRƯỚC expected (mất gói): đệm 0 cho ô thiếu, giữ lại gói hiện tại
        để soi vòng sau cho tới khi expected_seq đuổi kịp.
      • seq LÙI SAU expected (gói trùng / đảo thứ tự): BỎ gói, đọc tiếp — KHÔNG đệm,
        KHÔNG seek lùi. (Nếu seek lùi, con trỏ kẹt tại gói này còn expected_seq cứ
        tăng → toàn bộ phần còn lại thành gói 0.)
    """
    output_data = bytearray()
    packets_collected = 0
    expected_seq = target_seq

    with open(file_in, 'rb') as f:
        f.seek(start_offset)

        while packets_collected < PACKET_COUNT:
            current_pos = f.tell()
            packet = f.read(packet_size)

            if not packet or len(packet) < packet_size:
                # Hết file / gói cụt → đệm 0 cho phần còn thiếu.
                output_data.extend(b'\x00' * packet_size)
                expected_seq = (expected_seq + 1) % SEQ_MAX
                packets_collected += 1
                continue

            seq, _ = struct.unpack('<HQ', packet[:10])

            if seq == expected_seq:
                output_data.extend(packet)
                expected_seq = (expected_seq + 1) % SEQ_MAX
                packets_collected += 1
                continue

            seq_diff = (seq - expected_seq) % SEQ_MAX
            if seq_diff <= max_forward_gap:
                # Mất gói (seq vượt trước): đệm 0 cho ô thiếu, giữ gói này lại (seek lùi).
                output_data.extend(b'\x00' * packet_size)
                expected_seq = (expected_seq + 1) % SEQ_MAX
                packets_collected += 1
                f.seek(current_pos)
            else:
                # Gói trùng / đảo thứ tự (seq lùi sau): bỏ, đọc tiếp — KHÔNG seek lùi.
                continue

    return bytes(output_data)


def cut_and_pad_1000(file_in: str, start_offset: int, target_seq: int, packet_size: int,
                     out_filename: str, max_forward_gap: int = MAX_FWD_GAP):
    """Cắt PACKET_COUNT gói rồi GHI ra file (bọc quanh _cut_and_pad_bytes)."""
    data = _cut_and_pad_bytes(file_in, start_offset, target_seq, packet_size, max_forward_gap)
    with open(out_filename, 'wb') as f_out:
        f_out.write(data)
    print(f"[Thành công] Đã cắt {PACKET_COUNT} gói -> {out_filename}")

def sync_and_cut_3_files(file1, file2, file3,target_seq: int, target_ts: int, packet_size: int, dev_type: str, base_out_name: str, output_dir: str):
    """
    đồng bộ 3 file và lưu vào đúng folder đích 
    """
    
    files = [file1, file2, file3]
    
    for i, file_path in enumerate(files):
        # xác định folder đích 
        rx_folder = f"{dev_type}{i+1}" 
        
        # đường dẫn lưu file hoàn chỉnh
        out_name = os.path.join(output_dir, rx_folder, base_out_name)
        
        print(f"\nĐang xử lý {rx_folder} ({os.path.basename(file_path)})...")

        anchor_offset= find_anchor_offset(file_path, target_seq, target_ts, packet_size)
        
        if anchor_offset is not None:
            cut_and_pad_1000(file_path, anchor_offset, target_seq, packet_size, out_name)
        else:
            print(f"[Thất bại] Không tìm thấy neo cho {rx_folder}. Xuất file mảng 0: {out_name}")
            with open(out_name, 'wb') as f_out:
                f_out.write(b'\x00' * packet_size * PACKET_COUNT)

def _dtype(dev_type: str) -> np.dtype:
    """dtype của 1 gói theo thiết bị (header + payload)."""
    if dev_type == 'esp':
        return np.dtype([
            ('seq', '<u2'), ('timestamp', '<u8'), ('channel', '<u2'),
            ('agc', 'u1'), ('fft', 'u1'), ('noise', 'i1'), ('rssi', 'i1'),
            ('payload', 'i1', (128,))      # 64 subcarrier × (q, i)
        ])
    return np.dtype([
        ('seq', '<u2'), ('timestamp', '<u8'), ('channel', '<u2'),
        ('agc_gain', 'u1', (4,)), ('rssi', 'i1', (4,)),
        ('payload', '<u4', (256,))         # 4 anten × 64 sub × 4 byte
    ])


def _decode_amp_phase(data: np.ndarray, dev_type: str):
    """Giải mã biên độ + pha từ mảng cấu trúc đã đọc.

    ESP : payload i8 [q0,i0,q1,i1,…] → amp/phase (N, 64).
    ASUS: payload u32 (Nexmon, design_note mục 2.1) → amp/phase (N, 4, 64).
          bit 29=sign I, 28:18=mantissa I, 17=sign Q, 16:6=mantissa Q,
          5:0=exponent (bù-2 6-bit); E=e+10; I=(-1)^sI·MI·2^E; underflow E<-12→0.
    """
    if dev_type == 'esp':
        Q = data['payload'][:, 0::2].astype(np.float32)        # chẵn = q (ảo)
        I = data['payload'][:, 1::2].astype(np.float32)        # lẻ  = i (thực)
        amplitude = np.sqrt(I**2 + Q**2)
        phase = np.arctan2(Q, I)
        return amplitude, phase                                # (N, 64)

    csi_raw = data['payload']                                  # (N, 256) uint32
    e_raw = csi_raw & 0x3F
    e = np.where(e_raw >= 32, e_raw.astype(np.int32) - 64, e_raw.astype(np.int32))
    E = e + 10
    m_i = ((csi_raw >> 18) & 0x7FF).astype(np.int64)          # bit 28:18
    s_i = ((csi_raw >> 29) & 0x1).astype(np.int32)            # bit 29
    m_q = ((csi_raw >>  6) & 0x7FF).astype(np.int64)          # bit 16:6
    s_q = ((csi_raw >> 17) & 0x1).astype(np.int32)            # bit 17
    scale = np.power(2.0, E.astype(np.float32))
    I = (1 - 2 * s_i) * m_i * scale                           # M·2^E (KHÔNG +1)
    Q = (1 - 2 * s_q) * m_q * scale
    under = E < -12                                            # underflow → 0
    I = np.where(under, 0.0, I)
    Q = np.where(under, 0.0, Q)
    amplitude = np.sqrt(I**2 + Q**2).reshape(-1, 4, 64)       # ant-major
    phase = np.arctan2(Q, I).reshape(-1, 4, 64)
    return amplitude, phase


def extract_csi_matrix(file_paths: list, dev_type: str):
    """Trích xuất ma trận biên độ và pha từ các file đã cắt (ESP hoặc ASUS)."""
    results = []
    for rx_idx, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"[Lỗi] Không tìm thấy file đã cắt: {file_path}")
            results.append(None)
            continue
        print(f"Đang phân tích ({dev_type.upper()}) cho Rx{rx_idx + 1}...")
        data = np.fromfile(file_path, dtype=_dtype(dev_type))
        amplitude, phase = _decode_amp_phase(data, dev_type)
        results.append({
            'rx_index': rx_idx,
            'timestamp': data['timestamp'],
            'amplitude': amplitude,    # ESP (1000,64) | ASUS (1000,4,64)
            'phase': phase,
        })
    print("[Thành công] Đã trích xuất xong mảng đa chiều!")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CHỨC NĂNG 3 — PLOT (gộp: chọn mẫu → cắt+giải mã trong RAM → vẽ, kiểu plot_tool)
# ══════════════════════════════════════════════════════════════════════════════

def _scan_sessions(data_dir: str) -> list:
    """Quét data_dir, trả list các phiên {folder, name, cfg} đọc từ session_config.json."""
    import json
    out = []
    if not os.path.isdir(data_dir):
        return out
    for name in sorted(os.listdir(data_dir)):
        cfg_path = os.path.join(data_dir, name, 'session_config.json')
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            continue
        out.append({'folder': os.path.join(data_dir, name), 'name': name, 'cfg': cfg})
    return out


def _process_amp(amp: np.ndarray) -> np.ndarray:
    """Tiền xử lý biên độ: Hampel (window=8, n_sigma=3) + Butterworth LPF (4, 20Hz).

    amp: (T, 64), lọc dọc trục thời gian (axis 0).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.signal import butter, sosfiltfilt
    w = 8
    x = amp.astype(np.float32)
    xpad = np.pad(x, ((w, w), (0, 0)), mode='reflect')
    xw = sliding_window_view(xpad, 2 * w + 1, axis=0)          # (T, 64, 2w+1)
    med = np.median(xw, axis=-1)
    mad = np.maximum(np.median(np.abs(xw - med[..., None]), axis=-1), 1e-6)
    x = np.where(np.abs(x - med) > 3.0 * 1.4826 * mad, med, x).astype(np.float32)
    sos = butter(4, 20.0, btype='low', fs=_FS, output='sos')
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


def _load_action_csi(folder, dev_type, rx_list, ant, action_name, repeat, with_proc):
    """Cắt + giải mã 1 hành động (đồng bộ theo seq) cho các RX đã chọn, đã chọn antenna.

    Trả list dict {rx, amp (1000,64), ph (1000,64), proc (1000,64)|None}, hoặc None nếu lỗi.
    """
    pkt = DEVICE_CONFIG[dev_type]['packet_size']
    event_file = os.path.join(folder, 'action_events.csv')
    ts = get_time_from_event(event_file, action_name, repeat)
    if ts is None:
        return None
    # Neo đồng bộ lấy từ RX đầu tiên được chọn (seq dùng chung mọi RX).
    ref_file = os.path.join(folder, f'raw_{dev_type}{rx_list[0]}.bin')
    seq0, ts0 = find_first_packet_fast(ref_file, ts, pkt, excel_output=None)
    if seq0 is None:
        return None
    chans = []
    for rx in rx_list:
        fpath = os.path.join(folder, f'raw_{dev_type}{rx}.bin')
        off = find_anchor_offset(fpath, seq0, ts0, pkt) if os.path.exists(fpath) else None
        raw = _cut_and_pad_bytes(fpath, off, seq0, pkt) if off is not None \
            else b'\x00' * pkt * PACKET_COUNT
        data = np.frombuffer(raw, dtype=_dtype(dev_type))
        amp, ph = _decode_amp_phase(data, dev_type)
        if dev_type == 'asus':                                # (N,4,64) → chọn antenna
            amp, ph = amp[:, ant - 1, :], ph[:, ant - 1, :]
        chans.append({'rx': rx, 'amp': amp, 'ph': ph,
                      'proc': _process_amp(amp) if with_proc else None})
    return chans


def _suptitle(meta, subcarrier=None, all_sc=False):
    parts = [
        meta['action_label'], meta['dev'].upper(),
        (f"Room {meta['room']:02d} · Setup {meta['setup']:02d} · "
         f"Session {meta['session']:02d} · User {meta['user']:02d} · Pos {meta['pos']:02d}"),
        f"Rx {meta['rx_label']}", f"Ant {meta['ant']:02d}",
    ]
    if all_sc:
        parts.append(f"All {N_SUBCARRIER} Subcarriers")
    elif subcarrier is not None:
        parts.append(f"Subcarrier {subcarrier + 1}/{N_SUBCARRIER}")
    parts.append(f"Rep {meta['repeat']:02d}")
    return "  |  ".join(parts)


def _row_title(base, rx, n):
    return f"Rx {rx:02d} — {base}" if n > 1 else base


def _plot_amp_1d(chans, meta, out_path, with_proc):
    import matplotlib.pyplot as plt
    sc, n = VIZ_SUBCARRIER, len(chans)
    t = np.linspace(0, _DUR_S, chans[0]['amp'].shape[0], endpoint=False)
    base = ('CSI Amplitude raw and after preprocessing' if with_proc else 'Raw CSI Amplitude')
    fig, axes = plt.subplots(n, 1, figsize=(_FIG_W_1P, _FIG_H_ROW * n),
                             squeeze=False, sharex=True)
    for i, ch in enumerate(chans):
        ax = axes[i, 0]
        ax.plot(t, ch['amp'][:, sc], color='steelblue', lw=0.8, alpha=0.8,
                label='Raw CSI Amplitude')
        if with_proc:
            ax.plot(t, ch['proc'][:, sc], color='darkorange', lw=1.2,
                    label='After Hampel + Butterworth LPF')
        ax.set_ylabel('Amplitude')
        ax.set_title(_row_title(base, ch['rx'], n))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('Time (s)')
    fig.suptitle(_suptitle(meta, subcarrier=sc), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _plot_amp_heatmap(chans, meta, out_path, with_proc):
    import matplotlib.pyplot as plt
    n = len(chans)
    ncol = 2 if with_proc else 1
    width = _FIG_W_2P if with_proc else _FIG_W_1P
    fig, axes = plt.subplots(n, ncol, figsize=(width, _FIG_H_ROW * n), squeeze=False)
    for i, ch in enumerate(chans):
        cells = [(ch['amp'], 'Raw CSI Amplitude')]
        if with_proc:
            cells.append((ch['proc'], 'CSI Amplitude after Hampel + Butterworth LPF'))
        for j, (mat, title) in enumerate(cells):
            ax = axes[i, j]
            # Thang màu bền với vài subcarrier rác (vd DC/null trung tâm của ASUS có
            # giá trị khổng lồ): 99th-pct theo TỪNG subcarrier rồi median → 1 subcarrier
            # outlier không kéo lệch toàn bộ thang.
            vmin = float(mat.min())
            vmax = float(np.median(np.percentile(mat, 99, axis=0)))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = float(np.percentile(mat, 99))
            im = ax.imshow(mat.T, aspect='auto', origin='lower', cmap='viridis',
                           vmin=vmin, vmax=vmax, extent=[0, _DUR_S, 0, N_SUBCARRIER - 1])
            ax.set_title(_row_title(title, ch['rx'], n))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Subcarrier')
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(_suptitle(meta, all_sc=True), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _plot_phase_1d(chans, meta, out_path):
    import matplotlib.pyplot as plt
    sc, n = VIZ_SUBCARRIER, len(chans)
    t = np.linspace(0, _DUR_S, chans[0]['ph'].shape[0], endpoint=False)
    fig, axes = plt.subplots(n, 1, figsize=(_FIG_W_1P, _FIG_H_ROW * n),
                             squeeze=False, sharex=True)
    for i, ch in enumerate(chans):
        ax = axes[i, 0]
        ax.plot(t, ch['ph'][:, sc], color='steelblue', lw=0.8, alpha=0.8,
                label='Raw CSI Phase')
        ax.set_ylabel('Phase (rad)')
        ax.set_title(_row_title('Raw CSI Phase', ch['rx'], n))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('Time (s)')
    fig.suptitle(_suptitle(meta, subcarrier=sc), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _plot_phase_heatmap(chans, meta, out_path):
    import matplotlib.pyplot as plt
    n = len(chans)
    fig, axes = plt.subplots(n, 1, figsize=(_FIG_W_1P, _FIG_H_ROW * n), squeeze=False)
    for i, ch in enumerate(chans):
        ax = axes[i, 0]
        mat = ch['ph']
        vmax = float(np.percentile(np.abs(mat), 99))
        im = ax.imshow(mat.T, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, extent=[0, _DUR_S, 0, N_SUBCARRIER - 1])
        ax.set_title(_row_title('Raw CSI Phase', ch['rx'], n))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Subcarrier')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(_suptitle(meta, all_sc=True), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _ask_int(prompt, lo, hi):
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
        except ValueError:
            print(f"   -> Nhập số nguyên trong [{lo}, {hi}].")
            continue
        if lo <= v <= hi:
            return v
        print(f"   -> Ngoài khoảng [{lo}, {hi}].")


def _ask_int_list(prompt, lo, hi):
    import re
    while True:
        toks = [t for t in re.split(r'[,\s]+', input(prompt).strip()) if t]
        try:
            vals = [int(t) for t in toks]
        except ValueError:
            print(f"   -> Nhập các số trong [{lo}, {hi}], cách bởi dấu phẩy."); continue
        if vals and all(lo <= v <= hi for v in vals):
            return sorted(set(vals))
        print(f"   -> Mỗi giá trị phải trong [{lo}, {hi}] và không rỗng.")


def _ask_yesno(prompt):
    while True:
        s = input(prompt).strip().lower()
        if s in ('y', 'yes', 'c', 'co', 'có', '1'):
            return True
        if s in ('n', 'no', 'k', 'khong', 'không', '0'):
            return False
        print("   -> Nhập y/n.")


def run_plot_mode(dev_type: str):
    """Chế độ vẽ: chọn mẫu theo chỉ số → cắt+giải mã trong RAM → vẽ 4 đồ thị."""
    import matplotlib
    matplotlib.use('Agg')

    sessions = _scan_sessions(DATA_DIR)
    if not sessions:
        print(f"[Lỗi] Không thấy phiên nào (session_config.json) trong: {DATA_DIR}")
        return

    room    = _ask_int("3. Chọn phòng    (room): ", 1, 99)
    setup   = _ask_int("4. Chọn setup           : ", 1, 99)
    session = _ask_int("5. Chọn phiên   (session): ", 1, 99)
    user    = _ask_int("6. Chọn người     (user): ", 1, 99)
    pos     = _ask_int("7. Chọn vị trí     (pos): ", 1, 99)

    matches = [s for s in sessions if (
        s['cfg'].get('room_id') == room and s['cfg'].get('setup_id') == setup and
        s['cfg'].get('session_no') == session and s['cfg'].get('person_id') == user and
        s['cfg'].get('position_id') == pos)]
    if not matches:
        print(f"[Lỗi] Không có phiên khớp (room={room}, setup={setup}, session={session}, "
              f"user={user}, pos={pos}).")
        return
    if len(matches) == 1:
        sel = matches[0]
    else:
        print("   Nhiều phiên khớp — chọn một:")
        for i, s in enumerate(matches, 1):
            print(f"     {i:02d}. {s['cfg'].get('scenario', '?')}   ({s['name']})")
        sel = matches[_ask_int("   -> Chọn phiên: ", 1, len(matches)) - 1]
    folder = sel['folder']
    print(f"   -> Phiên: {sel['name']}  (scenario={sel['cfg'].get('scenario', '?')})")

    repeat_count = int(sel['cfg'].get('repeat_count', 10)) or 10
    repeat = _ask_int(f"8. Chọn lần lặp  (01–{repeat_count:02d}): ", 1, repeat_count)

    print(" Danh mục hành động (01–09):")
    half = (len(ACTION_CATALOG) + 1) // 2
    for i in range(half):
        left = f"   {i + 1:02d}. {ACTION_CATALOG[i][0]:<14}"
        right = f"{i + half + 1:02d}. {ACTION_CATALOG[i + half][0]}" if i + half < len(ACTION_CATALOG) else ""
        print(left + right)
    a_idx = _ask_int("9. Chọn hành động (01–09): ", 1, len(ACTION_CATALOG))
    action_label, action_name = ACTION_CATALOG[a_idx - 1]

    rx_list = _ask_int_list("10. Chọn RX (01–03, nhiều thì cách bởi dấu phẩy): ", 1, 3)

    n_ant = DEVICE_ANTENNAS[dev_type]
    if n_ant > 1:
        ant = _ask_int(f"11. Chọn antenna (01–{n_ant:02d}): ", 1, n_ant)
    else:
        ant = 1
        print(f"11. Antenna: thiết bị {dev_type} chỉ 1 antenna → tự chọn 01.")
    with_proc = _ask_yesno("12. Tiền xử lý biên độ? (y/n) [pha luôn thô]: ")

    rx_label = ", ".join(f"{r:02d}" for r in rx_list)
    print("\n" + "-" * 62)
    print(f" Vẽ: {action_label} ({action_name}) | {dev_type} | Rx {rx_label} | Ant {ant:02d} | Rep {repeat:02d}")
    print(f" Tiền xử lý biên độ: {'BẬT (Hampel + LPF)' if with_proc else 'TẮT (raw)'}")
    print("-" * 62)

    chans = _load_action_csi(folder, dev_type, rx_list, ant, action_name, repeat, with_proc)
    if not chans:
        print("[Lỗi] Không nạp được dữ liệu (kiểm tra action_name trong action_events.csv "
              f"có khớp map '{action_name}' không).")
        return

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_output')
    os.makedirs(out_dir, exist_ok=True)
    rx_tag = "-".join(f"{r:02d}" for r in rx_list)
    base = f"{dev_type}_{action_label}_rep{repeat:02d}_rx{rx_tag}_ant{ant:02d}"
    meta = dict(action_label=action_label, dev=dev_type, room=room, setup=setup,
                session=session, user=user, pos=pos, rx_label=rx_label, ant=ant, repeat=repeat)

    _plot_amp_1d(chans, meta, os.path.join(out_dir, f"{base}_amp_1d.png"), with_proc)
    print(f"  ✓ {base}_amp_1d.png")
    _plot_amp_heatmap(chans, meta, os.path.join(out_dir, f"{base}_amp_heatmap.png"), with_proc)
    print(f"  ✓ {base}_amp_heatmap.png")
    _plot_phase_1d(chans, meta, os.path.join(out_dir, f"{base}_phase_1d.png"))
    print(f"  ✓ {base}_phase_1d.png")
    _plot_phase_heatmap(chans, meta, os.path.join(out_dir, f"{base}_phase_heatmap.png"))
    print(f"  ✓ {base}_phase_heatmap.png")
    print(f"\n[Thành công] 4 đồ thị đã lưu vào {out_dir}")


def main():

    # 1. nhập chức năng
    while True:
        func = input("1. Chọn chức năng (exfile / exarray / plot): ").strip().lower()
        if func in ['exfile', 'exarray', 'plot']:
            break
        else:
            print("   -> Lựa chọn không hợp lệ, gõ 'exfile' / 'exarray' / 'plot'.")


    while True:
        dev_type = input("2. Chọn loại thiết bị (esp / asus): ").strip().lower()

        # cho phép cả 'esp' và 'asus' thoát khỏi vòng lặp để chạy tiếp
        if dev_type in ['esp', 'asus']:
            break
        else:
            print("   -> Thiết bị không hợp lệ, vui lòng gõ 'esp' hoặc 'asus'.")

    pkt_size = DEVICE_CONFIG[dev_type]['packet_size']

    # ── Chức năng 3: PLOT (gộp, chọn theo chỉ số, vẽ kiểu plot_tool) ──
    if func == 'plot':
        run_plot_mode(dev_type)
        input("\nNhấn Enter để thoát...")
        return
    # 3. nhập đường dẫn
    path = input("3. Nhập đường dẫn thư mục database: ").strip().strip('"\'')

    # 4. nhập tên hành động
    action = input("4. Nhập tên hành động (VD: cui, nga,...): ").strip()

    # 5. Nhập chỉ số lặp lại
    while True:
        try:
            repeat_input = input("5. Nhập chỉ số lần lặp thứ (VD: 1, 2, 3...): ").strip()
            repeat = int(repeat_input)
            break
        except ValueError:
            print("   -> Lỗi: Vui lòng nhập một số nguyên.")

    print("\n" + "-"*60)
    print(f" ĐANG CHẠY CHỨC NĂNG: {func.upper()}")
    print("-"*60)

    output_dir = "Cut_Data" # Tên thư mục tổng chứa kết quả
    for d_type in ['esp', 'asus']:
        for i in range(1, 4):
            # Lệnh makedirs với exist_ok=True sẽ tự tạo folder nếu chưa có, 
            # và bỏ qua một cách an toàn nếu folder đã tồn tại từ những lần chạy trước.
            os.makedirs(os.path.join(output_dir, f"{d_type}{i}"), exist_ok=True)


    folder_name = os.path.basename(os.path.normpath(path))
    parts = folder_name.split('_')
    
    if len(parts) >= 7:
        prefix_5 = "_".join(parts[:5])     # -> "1_1_1_1_1"
        time_suffix = "_".join(parts[-2:]) # -> "0605_164421"
    else:
        prefix_5 = "1_1_1_1_1"
        time_suffix = "0000_000000"

    base_name = f"{prefix_5}_{repeat}_{action}_{time_suffix}"
    base_out_name = f"{prefix_5}_{repeat}_{action}_{time_suffix}.bin"
    print(f"[*] Đã sinh tên file chuẩn: {base_out_name}")

    
    event_file = os.path.join(path, "action_events.csv")
    f1 = os.path.join(path, f"raw_{dev_type}1.bin")
    f2 = os.path.join(path, f"raw_{dev_type}2.bin")
    f3 = os.path.join(path, f"raw_{dev_type}3.bin")

    time_input = get_time_from_event(event_file, action, repeat)
    if time_input is None:
        print("\n[Kết thúc] Dừng chương trình do không tìm thấy mốc thời gian.")
        input("Nhấn Enter để thoát...")
        return
    log_file_name =  os.path.join(output_dir, f"{base_name}_log.xlsx")
    found_seq, found_ts = find_first_packet_fast(f1, time_input, pkt_size, log_file_name)
    
    if found_seq is None:
        print("\n[Kết thúc] Dừng chương trình do không tìm thấy điểm neo đồng bộ.")
        input("Nhấn Enter để thoát...")
        return
    # Gọi hàm cắt và truyền các thông số định tuyến folder
    sync_and_cut_3_files(f1, f2, f3,found_seq, found_ts, pkt_size, dev_type, base_out_name, output_dir)

    # Xử lý mảng nếu chọn exarray
    if func == 'exarray':
        # Trỏ đường dẫn đến đúng 3 file vừa được cắt nằm trong 3 folder
        cut_files = [
            os.path.join(output_dir, f"{dev_type}1", base_out_name),
            os.path.join(output_dir, f"{dev_type}2", base_out_name),
            os.path.join(output_dir, f"{dev_type}3", base_out_name)
        ]
        csi_matrices = extract_csi_matrix(cut_files, dev_type)
        
        if csi_matrices and csi_matrices[0] is not None:
            print(f"\n[Hoàn tất EXARRAY] Dữ liệu đã sẵn sàng trong RAM.")
            print(f" -> Kích thước mảng: {csi_matrices[0]['amplitude'].shape}")

    print("\n[Thành công] Đã hoàn tất toàn bộ quy trình!")
    input("Nhấn Enter để thoát chương trình...")

if __name__ == "__main__":
      main()