main()  [Hàm Điều Phối Trung Tâm]
 │
 ├── 1. Giao tiếp UI & Thiết lập
 │      └── Nhận input người dùng, khởi tạo thư mục `Cut_Data/...`
 │
 ├── 2. get_time_from_event() 
 │      └── Tra cứu file log để lấy mốc thời gian (target_time)
 │
 ├── 3. find_first_packet_fast() 
 │      └── Quét nhị phân trên Rx1 để chốt điểm neo chuẩn (found_seq, found_ts)
 │
 ├── 4. sync_and_cut_3_files() 
 │      │   [Lặp qua 3 thiết bị Rx1, Rx2, Rx3]
 │      ├───> find_anchor_offset()
 │      │       └── Định vị tọa độ byte xuất phát (hỗ trợ nhảy cóc nếu rớt gói)
 │      └───> cut_and_pad_1000()
 │              └── Cắt đúng 1000 gói tin, tự động chèn Byte 0 nếu đứt gãy Sequence
 │
 └── 5. extract_csi_matrix()  [Chỉ chạy nếu chọn 'exarray']
        └── Giải mã Toán học I/Q, xuất mảng Amplitude & Phase (Hỗ trợ phân luồng ESP / ASUS)



csi_matrices[index của Rx][thông tin cần trích xuất][index gói tin,index subcarrier]                  // esp
csi_matrices[index của Rx][thông tin cần trích xuất][index gói tin,index anten,index subcarrier]      //asus

thông tin cần trích xuất: 'timestamp','amplitude','phase'
index : xuất phát từ 0
