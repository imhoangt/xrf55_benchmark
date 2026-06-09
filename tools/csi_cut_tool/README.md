# csi_cut_tool

Cắt/đồng bộ dữ liệu CSI thô thu từ thiết bị (ESP / ASUS Nexmon) và giải mã biên độ/pha.

Thu CSI từ 3 receiver chạy độc lập → lệch mốc thời gian. Tool này neo theo mốc sự kiện,
cắt đúng **1000 gói** (5 giây @ 200 Hz) đồng bộ theo `sequence` 12-bit dùng chung giữa các
receiver, đệm gói 0 cho chỗ mất gói. Xem [ARCHITECTURE.md](ARCHITECTURE.md) cho luồng chi tiết.

## Định dạng gói
- **ESP**: 144 B/gói = 16 header + 128 payload (64 subcarrier × q,i int8).
- **ASUS**: 1044 B/gói = 20 header + 256 uint32 (4 anten × 64 subcarrier), giải mã theo
  Nexmon (design note 2.1).

## Chế độ chạy (`func`)
- `exfile` — chỉ cắt/đồng bộ, xuất file `.bin` đã cắt.
- `exarray` — cắt + giải mã, trả mảng amplitude/phase trong bộ nhớ.
- `plot`   — vẽ đồ thị biên độ/pha cho bộ dữ liệu tự thu (bố cục giống plot_tool).

## Chạy
```bash
python tools/csi_cut_tool/csi_tool.py          # tương tác chọn chế độ
python -m tools.csi_cut_tool.test_cut_and_pad  # chạy test cut/pad (từ gốc repo)
```

## Cài độc lập (tùy chọn)
```bash
cd tools/csi_cut_tool && pip install -e .
```
Phụ thuộc: `numpy`, `pandas`, `openpyxl`.
