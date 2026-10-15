# Kết quả chạy thử two-stream R50 + R50 (run8_2s)

- Viết lúc: 2026-10-15 17:30 (UTC+8) = 09:30 UTC.
- Ngày chạy: 2026-10-15. Kernel `kaggleolap/n5k-run8-two-stream`: đẩy 04:06 UTC, chạy từ 04:06 đến 04:45 UTC (38 phút), code `180ea68-dirty`.
- Nguồn: `n5k/results/kaggle/run8_2s/`; chi tiết ở `n5k/PLAN.md` mục 11. Quyết định và quy tắc: `2026-10-15_1205_two-stream-quyet-dinh.md`.
- Cách chạy: single GPU, chỉ GPU0, lr 0.02, 8 epoch, seed 0 và 1. Chạy xen kẽ một nhánh / two-stream trên cùng máy.

## Số liệu (val ở epoch có MAE kcal tốt nhất)

| Mô hình | seed | MAE kcal | mAP | Test kcal | Test mAP | s/epoch | Peak MB |
|---|---|---|---|---|---|---|---|
| Một nhánh | 0 | 65,53 | 67,81 | 66,82 | 69,37 | 27,6 | 5642 |
| Two-stream | 0 | 61,75 | 75,42 | 59,26 | 74,24 | 59,1 | 13822 |
| Một nhánh | 1 | 64,21 | 72,66 | 61,36 | 71,81 | 27,3 | 5643 |
| Two-stream | 1 | 62,73 | 72,11 | 58,90 | 73,28 | 59,3 | 13821 |

Trung bình: một nhánh 64,87 kcal / mAP 70,24; two-stream 62,24 kcal / mAP 73,76.

## Kết luận
1. **Two-stream đạt quy tắc chọn:** điểm −0,091 < −0,05 (kcal −4,1%, mAP +3,5 điểm). Đánh giá ở epoch cuối (−0,077) và trên test (kcal −7,8%, mAP +3,2 điểm) cho cùng chiều.
2. Mức cải thiện là vừa phải và chỉ có 2 seed × 8 epoch. kcal tốt hơn ở cả 2 seed; mAP chủ yếu tốt hơn ở seed 0. Cần chạy dài 30 epoch × 3 seed để khẳng định.
3. **Chi phí lớn:** chậm hơn 2,15 lần, tốn bộ nhớ gấp 2,45 lần (13,8 GB, gần chạm trần 14,56 GB của T4 khi M=64 trên 1 GPU).
4. **Pipeline 2 GPU với cách chia cũ `[0,8,18]` bị thiếu bộ nhớ trên GPU0** (check D2). Phải đo lại cách chia trước khi chạy pipeline. D0, D1 PASS: pipeline two-stream tính đúng như engine 1 GPU.
5. Đây đúng là trường hợp làm pipeline có ý nghĩa hơn: mô hình gần chạm trần bộ nhớ 1 GPU.
6. Kcal tính từ gram từng nguyên liệu không cải thiện (khoảng 115 kcal); two-stream chỉ cải thiện nhánh đoán tổng và nhận diện.
