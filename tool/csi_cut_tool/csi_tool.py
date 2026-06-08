import struct
import numpy as np
import pandas as pd
import os


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
        data_to_save = {
            "Sequence": [best_seq],
            "Timestamp": [best_ts],
            "Target_Time_Input": [target_time],
            "Time_Difference": [best_ts - target_time]
        }
        
        try:
            pd.DataFrame(data_to_save).to_excel(excel_output, index=False)
            print(f" -> Seq: {best_seq} | TS: {best_ts}")
        except Exception as e:
            print(f" -> [Lỗi] Không thể ghi file Excel: {e}")
            
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

def cut_and_pad_1000(file_in: str, start_offset: int, target_seq: int, packet_size: int,
                     out_filename: str, max_forward_gap: int = MAX_FWD_GAP):
    """Cắt đúng PACKET_COUNT gói liên tiếp từ start_offset, căn theo lưới seq bắt đầu
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

    with open(out_filename, 'wb') as f_out:
        f_out.write(output_data)

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

def extract_csi_matrix(file_paths: list, dev_type: str):
    """
    Trích xuất ma trận biên độ và pha tùy theo loại thiết bị (ESP hoặc ASUS).
    """
    results = []

    for rx_idx, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"[Lỗi] Không tìm thấy file đã cắt: {file_path}")
            results.append(None)
            continue
            
        print(f"Đang phân tích ({dev_type.upper()}) cho Rx{rx_idx + 1}...")
        
        if dev_type == 'esp':
            # --- ESP ---
            esp_dtype = np.dtype([
                ('seq', '<u2'), ('timestamp', '<u8'), ('channel', '<u2'),
                ('agc', 'u1'), ('fft', 'u1'), ('noise', 'i1'), ('rssi', 'i1'),
                ('payload', 'i1', (128,)) # 64 subcarriers * 2
            ])
            data = np.fromfile(file_path, dtype=esp_dtype)
            Q_float = data['payload'][:, 0::2].astype(np.float32)
            I_float = data['payload'][:, 1::2].astype(np.float32)
            
            amplitude = np.sqrt(I_float**2 + Q_float**2)
            phase = np.arctan2(Q_float, I_float)
            
            rx_data = {
                'rx_index': rx_idx,
                'timestamp': data['timestamp'], 
                'amplitude': amplitude, # Shape: (1000, 64)
                'phase': phase          # Shape: (1000, 64)
            }
            results.append(rx_data)

        elif dev_type == 'asus':
            # --- ASUS ---
            asus_dtype = np.dtype([
                ('seq', '<u2'), ('timestamp', '<u8'), ('channel', '<u2'),
                ('agc_gain', 'u1', (4,)), ('rssi', 'i1', (4,)),
                ('payload', '<u4', (256,)) # 4 Anten * 64 Sub * 4 Byte = 256 phần tử UInt32
            ])
            data = np.fromfile(file_path, dtype=asus_dtype)
            csi_raw = data['payload'] # Shape: (1000, 256)
            
            # 1. Trích xuất bit cho Q
            s_q = (csi_raw >> 29) & 0x01
            m_q = (csi_raw >> 18) & 0x07ff
            e   = csi_raw & 0x3f  # Dùng chung số mũ cho cả I và Q
            
            # 2. Trích xuất bit cho I
            s_i = (csi_raw >> 17) & 0x01
            m_i = (csi_raw >> 6) & 0x07ff
            
            # 3. Tính toán giá trị thực (Vectorized)
            # Tối ưu hóa (-1)^s thành (1 - 2*s) để Numpy tính nhanh hơn
            sign_q = 1 - 2 * s_q 
            sign_i = 1 - 2 * s_i 
            
            # Ép kiểu e sang float32 trước khi dùng phép lũy thừa số âm
            exponent = 2.0 ** (e.astype(np.float32) - 127)
            
            Q_float = sign_q * (1 + m_q) * exponent
            I_float = sign_i * (1 + m_i) * exponent
            
            # 4. Tính Biên độ và Pha
            amplitude = np.sqrt(I_float**2 + Q_float**2)
            phase = np.arctan2(Q_float, I_float)
            
            # 5. Định hình lại mảng (Reshape) để phân tách rõ 4 Anten
            # Biến mảng (1000, 256) thành (1000, 4, 64)
            amplitude = amplitude.reshape(-1, 4, 64)
            phase = phase.reshape(-1, 4, 64)
            
            rx_data = {
                'rx_index': rx_idx,
                'timestamp': data['timestamp'], 
                'amplitude': amplitude, # Shape: (1000, 4, 64)
                'phase': phase          # Shape: (1000, 4, 64)
            }
            results.append(rx_data)
            
    print("[Thành công] Đã trích xuất xong mảng đa chiều!")
    return results


def main():

    # 1. nhập chức năng
    while True:
        func = input("1. Chọn chức năng (exfile / exarray): ").strip().lower()
        if func in ['exfile', 'exarray']:
            break
        else:
            print("   -> Lựa chọn không hợp lệ, vui lòng gõ 'exfile' hoặc 'exarray'.")

   
    while True:
        dev_type = input("2. Chọn loại thiết bị (esp / asus): ").strip().lower()
        
        # cho phép cả 'esp' và 'asus' thoát khỏi vòng lặp để chạy tiếp
        if dev_type in ['esp', 'asus']:
            break
        else:
            print("   -> Thiết bị không hợp lệ, vui lòng gõ 'esp' hoặc 'asus'.")
    
    pkt_size = DEVICE_CONFIG[dev_type]['packet_size']
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