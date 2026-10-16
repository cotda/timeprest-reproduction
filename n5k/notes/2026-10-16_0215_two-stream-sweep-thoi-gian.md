# Sweep lr và thời gian 4 hệ với two-stream (run10_2s_sweep)

- Viết lúc: 2026-10-16 02:15 (UTC+8) = 2026-10-15 18:15 UTC.
- Ngày chạy: 2026-10-15. Kernel `kaggleolap/n5k-run10-2s-sweep`: đẩy 16:47 UTC, chạy đến khoảng 18:00 UTC (80 phút), code hash `1ffd8b2f03bd`.
- Nguồn: `n5k/results/kaggle/run10_2s_sweep/`; chi tiết ở `n5k/PLAN.md` mục 10–11.
- Cách chạy: two-stream R50 + R50, seed 0, 8 epoch, warmup 1, lr {0.02, 0.01}. PipeDream/TiMePReSt chạy pipeline thật trên 2 GPU (`[0,4,18]`); single chỉ dùng GPU0; DDP chỉ chạy lr 0.02. Các lần chạy nối tiếp nhau trên cùng máy, thứ tự đối xứng: single, DDP, PD, TP (0.02) rồi TP, PD, single (0.01).
- Quy tắc chọn lr (đặt trước): mỗi hệ lấy lr có val MAE kcal nhỏ nhất; nếu chênh dưới 1% thì lấy lr có mAP cao hơn.

## Số liệu (val ở epoch có MAE kcal tốt nhất)

| Hệ | lr | MAE kcal | mAP | s/epoch | so với single | Peak MB |
|---|---|---|---|---|---|---|
| single | 0.02 | 62,11 | 71,8 | 54,7 | 1,00 | 13803 |
| DDP | 0.02 | 60,37 | 73,9 | 27,9 | 0,51 | 7372 / 7380 |
| PipeDream | 0.02 | 60,33 | 75,1 | 61,0 | 1,12 | 13442 / 9412 |
| TiMePReSt | 0.01 | 68,18 | 63,1 | 59,8 | 1,10 | 13341 / 9328 |

Lần chạy còn lại: single 0.01 cho 63,00; PipeDream 0.01 cho 65,10; TiMePReSt 0.02 cho 69,19.

## Kết luận
1. **lr giữ như mô hình một nhánh:** single / DDP / PipeDream 0.02, TiMePReSt 0.01.
2. **Pipeline bất đồng bộ trên 2 GPU chậm hơn single GPU khoảng 10%** và chậm hơn DDP khoảng 2,2 lần. Nguyên nhân là lệch tải: GPU bận 0,50 / 0,95, vì chỉ cắt được ở khối 4.
3. **Pipeline không giảm bộ nhớ:** GPU0 vẫn dùng 13,4 GB, gần bằng single (13,8 GB). DDP chỉ dùng 7,4 GB mỗi GPU.
4. **Lợi thế thời gian của TiMePReSt gần như mất:** TiMePReSt / PipeDream = 0,986, so với 0,95 ở mô hình một nhánh (B6).
5. **TiMePReSt kém về chất lượng:** sau 8 epoch kém PipeDream 13% kcal và 12 điểm mAP. Single, DDP và PipeDream đều tốt hơn nhờ two-stream, còn TiMePReSt gần như không đổi so với mô hình một nhánh (68,6 → 68,2).
6. Hạn chế: 1 seed, 8 epoch. Biến động giữa các lần chạy khoảng ±3 kcal. PipeDream và TiMePReSt vẫn đang tiến bộ ở epoch 8.
7. Ý nghĩa cho bài: với mô hình two-stream trên 2×T4, pipeline bất đồng bộ không có lợi về thời gian, bộ nhớ hay chất lượng so với DDP. Đây là kết quả âm tính cho câu hỏi "khi nào TiMePReSt có lợi": mô hình nặng hơn nhưng vẫn vừa 1 GPU thì không đủ để pipeline có lợi.
8. Phụ: `sweep.py --summary-only` không đọc được metrics của `dist_train` (thiếu cột `wall_time_s`); đã sửa để đọc cả `epoch_time_s`.
