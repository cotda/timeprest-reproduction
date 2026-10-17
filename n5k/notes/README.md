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
| 2026-10-17 00:45 (16:45 UTC ngày 16) | [2026-10-17_0045_B7-ban-do-calo.md](2026-10-17_0045_B7-ban-do-calo.md) | B7 (run11): huấn luyện lại khớp B6; cả 4 loại bản đồ trung thực với mô hình ở TiMePReSt và single; CAM chỉ riêng một phần cho từng nguyên liệu |
| 2026-10-17 00:50 (16:50 UTC ngày 16) | [2026-10-17_0050_mae-theo-thoi-gian.md](2026-10-17_0050_mae-theo-thoi-gian.md) | Log B6: TiMePReSt chỉ nhanh hơn PipeDream ở đầu; dao động mạnh epoch 6–18; DDP tốt nhất mọi mốc thời gian |
| 2026-10-17 00:57 (16:57 UTC ngày 16) | [2026-10-17_0057_them-epoch.md](2026-10-17_0057_them-epoch.md) | Thêm epoch: single/PipeDream gần bão hòa ở 30 epoch; TiMePReSt đang underfit (train MAE 45 so với 35), có thể lợi từ lịch dài hơn |
| 2026-10-17 11:45 (03:45 UTC) | [2026-10-17_1145_lr-30-epoch.md](2026-10-17_1145_lr-30-epoch.md) | run12/13: lr tốt nhất ở 30 epoch là 0.02 cho cả hai; TiMePReSt 54,7 so với PipeDream 52,0 (kém 5%); dao động là do cơ chế; B7 PipeDream cũng trung thực |
| 2026-10-17 12:43 (04:43 UTC) | [2026-10-17_1243_60-epoch.md](2026-10-17_1243_60-epoch.md) | run14/15 (1 seed): 60 epoch giúp TiMePReSt bắt kịp PipeDream (test 53,7 so với 54,8), PipeDream không lợi (overfit); PipeDream 30 epoch đã đạt mức đó |
| 2026-10-17 14:03 (06:03 UTC) | [2026-10-17_1403_60-epoch-3-seed.md](2026-10-17_1403_60-epoch-3-seed.md) | run14/15/16 (3 seed): 60 epoch chỉ giúp TiMePReSt lr 0.01 (−2,9); TiMePReSt 60 epoch bắt kịp PipeDream 60 epoch (kém 0,9) vì PipeDream overfit; PipeDream 30 epoch (52,0) vẫn tốt nhất |
