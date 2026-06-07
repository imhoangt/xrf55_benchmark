"""Test tái hiện lỗi cut_and_pad_1000 (gói lùi/trùng → output toàn 0) và xác minh bản vá.

Chạy:  python csi_cut_tool/test_cut_and_pad.py
Không cần pytest. In PASS/FAIL cho từng kiểm tra.
"""
import os
import struct
import tempfile

import csi_tool

PSIZE = 144  # esp packet size


def make_packet(seq: int, ts: int, packet_size: int = PSIZE) -> bytes:
    """Gói giả: header seq(<H)+ts(<Q) rồi đệm bằng (seq & 0xFF) (khác 0 để phân biệt
    với gói đệm toàn 0). Gói thật trong test có seq>=100, ts>0 nên không bao giờ là 0."""
    body = struct.pack('<HQ', seq, ts)
    return body + bytes([seq & 0xFF]) * (packet_size - len(body))


def build_stream(seqs, base_ts=1_000_000) -> bytes:
    """Nối các gói theo danh sách seq, ts tăng đều."""
    return b''.join(make_packet(s, base_ts + i) for i, s in enumerate(seqs))


def read_records(blob: bytes, packet_size: int = PSIZE):
    """Tách blob thành list (seq, is_zero) cho từng gói output."""
    out = []
    for i in range(len(blob) // packet_size):
        rec = blob[i * packet_size:(i + 1) * packet_size]
        is_zero = rec == b'\x00' * packet_size
        seq = None if is_zero else struct.unpack('<H', rec[:2])[0]
        out.append((seq, is_zero))
    return out


def cut_and_pad_OLD(file_in, start_offset, target_seq, packet_size, out_filename):
    """Bản logic CŨ (có lỗi) — chỉ để đối chứng: mọi mismatch đều pad + seek lùi."""
    output_data = bytearray()
    packets_collected = 0
    expected_seq = target_seq
    with open(file_in, 'rb') as f:
        f.seek(start_offset)
        while packets_collected < csi_tool.PACKET_COUNT:
            current_pos = f.tell()
            packet = f.read(packet_size)
            if not packet or len(packet) < packet_size:
                output_data.extend(b'\x00' * packet_size)
                expected_seq = (expected_seq + 1) % csi_tool.SEQ_MAX
                packets_collected += 1
                continue
            seq, _ = struct.unpack('<HQ', packet[:10])
            if seq == expected_seq:
                output_data.extend(packet)
                expected_seq = (expected_seq + 1) % csi_tool.SEQ_MAX
                packets_collected += 1
            else:
                output_data.extend(b'\x00' * packet_size)
                expected_seq = (expected_seq + 1) % csi_tool.SEQ_MAX
                packets_collected += 1
                f.seek(current_pos)              # ← lỗi: seek lùi cả khi gói lùi sau
    with open(out_filename, 'wb') as f_out:
        f_out.write(output_data)


def run_case(cut_fn, seqs, target_seq, n_out):
    """Ghi stream ra file tạm, chạy cut_fn, trả list (seq, is_zero)."""
    csi_tool.PACKET_COUNT = n_out
    f_in = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    f_in.write(build_stream(seqs)); f_in.close()
    f_out = tempfile.NamedTemporaryFile(suffix='.bin', delete=False); f_out.close()
    try:
        cut_fn(f_in.name, 0, target_seq, PSIZE, f_out.name)
        with open(f_out.name, 'rb') as fh:
            return read_records(fh.read())
    finally:
        os.unlink(f_in.name)
        os.unlink(f_out.name)


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    # Lưới mong muốn (8 ô): seq 100..107.
    # Stream thật: thiếu 102 (mất gói) và có 1 gói TRÙNG 103 (lùi sau).
    seqs       = [100, 101, 103, 103, 104, 105, 106, 107]
    target_seq = 100
    n_out      = 8
    all_ok = True

    print("== Bản vá MỚI ==")
    new = run_case(csi_tool.cut_and_pad_1000, seqs, target_seq, n_out)
    new_seqs = [s for s, _ in new]
    # Kỳ vọng: [100, 101, ZERO(102 mất), 103, 104, 105, 106, 107]; gói trùng 103 bị bỏ.
    all_ok &= check("đủ 8 gói output",              len(new) == 8)
    all_ok &= check("ô 102 (index 2) là gói đệm 0", new[2][1] is True)
    all_ok &= check("index 3 giữ gói thật seq=103", new_seqs[3] == 103)
    all_ok &= check("đuôi 104..107 là DỮ LIỆU THẬT, không phải 0",
                    new_seqs[4:] == [104, 105, 106, 107])
    all_ok &= check("chỉ đúng 1 gói đệm 0 trong toàn output",
                    sum(1 for _, z in new if z) == 1)

    print("== Bản CŨ (đối chứng — phải dính lỗi) ==")
    old = run_case(cut_and_pad_OLD, seqs, target_seq, n_out)
    old_seqs = [s for s, _ in old]
    n_zero_old = sum(1 for _, z in old if z)
    print(f"   output cũ: {old_seqs}  (số gói 0 = {n_zero_old})")
    # Bản cũ: gói trùng 103 làm con trỏ kẹt → đuôi 104..107 mất, thành toàn 0.
    bug_reproduced = old_seqs[4:] == [None, None, None, None]
    all_ok &= check("bản cũ ĐÃ tái hiện lỗi (đuôi thành toàn 0)", bug_reproduced)
    all_ok &= check("bản mới giữ được nhiều dữ liệu thật hơn bản cũ",
                    sum(1 for s in new_seqs if s is not None) >
                    sum(1 for s in old_seqs if s is not None))

    print("\n" + ("TẤT CẢ PASS ✅" if all_ok else "CÓ KIỂM TRA FAIL ❌"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
