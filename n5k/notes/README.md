# Ghi chép kết luận — task Nutrition5k + TiMePReSt

Mỗi kết luận rút ra từ thí nghiệm có một file riêng trong thư mục này, để cuối dự án tổng hợp lại.
- Tên file: `YYYY-MM-DD_HHMM_<chủ-đề>.md`, theo thời điểm viết ghi chú.
- Đầu mỗi file ghi: thời điểm viết (giờ máy UTC+8 và UTC), ngày chạy thí nghiệm, nguồn số liệu (thư mục kết quả, kernel).
- Chi tiết đầy đủ và quyết định thiết kế vẫn nằm ở `n5k/PLAN.md` (mục 10: quyết định, mục 11: kết quả). Ghi chú ở đây là bản tóm tắt kết luận.
- Các ghi chú viết lúc 2026-10-15 12:05 tóm tắt lại những thí nghiệm chạy trước đó (ngày 2026-10-14).

## Mục lục (theo thời gian)

| Thời điểm viết | File | Nội dung |
|---|---|---|
| 2026-10-15 12:05 (04:05 UTC) | [2026-10-15_1205_tong-hop-B1-B5.md](2026-10-15_1205_tong-hop-B1-B5.md) | Dữ liệu, checks, partition, loss, nhánh tổng trực tiếp, lr, ảnh 224/320, DDP (thí nghiệm ngày 2026-10-14) |
| 2026-10-15 12:05 (04:05 UTC) | [2026-10-15_1205_B6-chay-dai.md](2026-10-15_1205_B6-chay-dai.md) | Kết quả chính: 5 hệ × 3 seed × 30 epoch + bench_comm; TiMePReSt có lợi gì |
| 2026-10-15 12:05 (04:05 UTC) | [2026-10-15_1205_run7-timing.md](2026-10-15_1205_run7-timing.md) | Vì sao thời gian single dao động |
| 2026-10-15 12:05 (04:05 UTC) | [2026-10-15_1205_so-sanh-tai-lieu.md](2026-10-15_1205_so-sanh-tai-lieu.md) | So sánh độ chính xác với các nghiên cứu khác trên Nutrition5k |
| 2026-10-15 12:05 (04:05 UTC) | [2026-10-15_1205_two-stream-quyet-dinh.md](2026-10-15_1205_two-stream-quyet-dinh.md) | Quyết định thử mô hình two-stream R50 + R50, quy tắc chọn |
| 2026-10-15 17:30 (09:30 UTC) | [2026-10-15_1730_two-stream-ket-qua.md](2026-10-15_1730_two-stream-ket-qua.md) | Kết quả run8_2s: two-stream đạt quy tắc (điểm −0,091), chậm 2,15 lần; pipeline cần chia lại |
| 2026-10-15 18:04 (10:04 UTC) | [2026-10-15_1804_two-stream-partition.md](2026-10-15_1804_two-stream-partition.md) | run9: chỉ `[0,4,18]` vừa bộ nhớ; pipeline two-stream không nhanh hơn single, lệch tải 0,55 / 0,94 |
| 2026-10-16 02:15 (18:15 UTC ngày 15) | [2026-10-16_0215_two-stream-sweep-thoi-gian.md](2026-10-16_0215_two-stream-sweep-thoi-gian.md) | run10: lr giữ như một nhánh; pipeline 2 GPU chậm hơn single 10%, DDP nhanh 2 lần; TiMePReSt kém PipeDream 13% kcal sau 8 epoch |
| 2026-10-16 02:20 (18:20 UTC ngày 15) | [2026-10-16_0220_chot-mo-hinh-chinh.md](2026-10-16_0220_chot-mo-hinh-chinh.md) | Người dùng chọn: mô hình một nhánh là kết quả chính; two-stream là thí nghiệm phụ; chuyển sang B7 |
