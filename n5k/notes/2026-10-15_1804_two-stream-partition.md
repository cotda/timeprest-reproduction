# Cách chia pipeline cho two-stream (run9_2s_part)

- Viết lúc: 2026-10-15 18:04 (UTC+8) = 10:04 UTC.
- Ngày chạy: 2026-10-15. Kernel `kaggleolap/n5k-run9-2s-partition`: đẩy 09:34 UTC, chạy 09:34–10:01 UTC (26 phút), code `180ea68-dirty`.
- Nguồn: `n5k/results/kaggle/run9_2s_part/`; chi tiết ở `n5k/PLAN.md` mục 10–11.
- Cách chạy: checks D0–D4 của two-stream với 3 cách chia; D4 chạy 4 epoch trên toàn tập train cho PipeDream và TiMePReSt.
- Quy tắc (đặt trước): trong các cách chia có D0–D3 PASS và cả hai GPU dưới 14 GB, lấy cách có PipeDream nhanh nhất.

## Số liệu

| Cách chia | Kết quả |
|---|---|
| `[0,5,18]` | Thiếu bộ nhớ trên GPU0 (check D2) |
| `[0,6,18]` | Thiếu bộ nhớ trên GPU0 (check D2) |
| `[0,4,18]` | D0–D4 PASS. PipeDream 58,4 s/epoch, TiMePReSt 58,1 s; peak 13,5 / 9,4 GB; GPU bận 0,55 / 0,94; truyền 13,7 GB/epoch |

## Kết luận
1. **Chọn `[0,4,18]`**, cách chia duy nhất vừa bộ nhớ. GPU0 chỉ còn dư khoảng 0,5 GB.
2. Pipeline two-stream tính đúng (D1 khớp engine 1 GPU, chênh lệch 0,0; D2 đúng version trọng số; D3 resume khớp).
3. **Pipeline 2 GPU không nhanh hơn single GPU** với mô hình này: khoảng 58 s/epoch so với 59 s ở run8 (khác máy, nhưng không có dấu hiệu nhanh hơn rõ).
4. Nguyên nhân: GPU0 phải giữ activation của 2 mini-batch đang chạy (đặc điểm của pipeline bất đồng bộ PipeDream/TiMePReSt với 2 GPU), nên chỉ cắt được rất sớm. Kết quả là lệch tải (GPU1 bận 94%, GPU0 chỉ 55%) và truyền nhiều gấp 4 lần.
5. Pipeline bất đồng bộ cũng **không giảm bộ nhớ lớn nhất**: GPU0 dùng 13,4 GB, gần bằng single GPU (13,8 GB).
6. TiMePReSt và PipeDream ngang nhau về thời gian ở cấu hình này (tỉ lệ 0,995).
7. Ý nghĩa cho bài: mô hình nặng hơn chưa chắc làm pipeline bất đồng bộ có lợi. Bộ nhớ ở stage đầu là giới hạn chính. Một pipeline đồng bộ (GPipe/DeepSpeed, chỉ giữ 1 mini-batch) có thể cắt muộn hơn; điều này chưa kiểm chứng.
