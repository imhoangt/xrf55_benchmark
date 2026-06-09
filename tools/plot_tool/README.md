# plot_tool

Trình xem tín hiệu CSI của XRF55 **theo kiểu tương tác** — bố cục đồ thị giống
`xrf55_bench/scripts/03_plot_amplitude.py` nhưng cho phép chọn mẫu tùy ý.

Đọc CSI phức từ `dataset/XRF55/raw/scene_01/rx_{01,02,03}/...` qua
`xrf55_bench.preprocessing.parser.load_xrf55_sample` (raw_npy chỉ có biên độ, không có pha).

## Chọn được
- Hành động (01–11), người (01–30), lần lặp (01–20).
- Nhiều RX cùng lúc (xếp dọc, nhập kiểu `01, 02, 03`), chọn anten.
- Tiền xử lý: Hampel + Butterworth LPF (giống `03_plot_amplitude.py`).

## Đồ thị
Biên độ 1D / heatmap, pha 1D / heatmap; họ biên độ + DWT.

## Chạy
```bash
python tools/plot_tool/plot_tool.py   # tương tác
```
Ảnh xuất ra `tools/plot_tool/outputs/` (đã gitignore).
Phụ thuộc: `numpy`, `scipy`, `matplotlib`, `csiread` (qua xrf55_bench parser).
