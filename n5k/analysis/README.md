# Script phân tích kết quả Nutrition5k

Các script tính lại các bảng và kết luận ghi trong `n5k/PLAN.md` và `n5k/notes/` từ log đã tải về `n5k/results/kaggle/`.
Chỉ đọc kết quả, không chạy huấn luyện. Chạy từ gốc repo, ví dụ `python n5k/analysis/analyze_60ep_3s.py n5k/results/kaggle`.

| Script | Tham số | Dùng cho |
|---|---|---|
| `analyze_run5.py` | `<repo>` | run5a/run5b: partition ở ảnh 320, chọn kích thước ảnh (PLAN mục 10) |
| `agg_run6.py` | `<repo>` | bảng B6: trung bình ± độ lệch chuẩn 3 seed của mọi hệ |
| `analyze_run9.py` | `<repo>` | run9_2s_part: partition của two-stream |
| `analyze_run10.py` | `n5k/results/kaggle/run10_2s_sweep` | run10: sweep lr two-stream, thời gian 4 hệ |
| `time_mae_b6.py` | `n5k/results/kaggle` | B6: val MAE kcal theo epoch và theo thời gian cộng dồn |
| `analyze_lr30.py` | `n5k/results/kaggle` | run12/run13 + B6: sweep lr 30 epoch, dao động |
| `analyze_60ep.py` | `n5k/results/kaggle` | run14/run15: 60 epoch, seed 0 |
| `analyze_60ep_3s.py` | `n5k/results/kaggle` | run14/15/16: 60 epoch, 3 seed, quy tắc (a)(b)(c) |
| `analyze_run17.py` | `n5k/results/kaggle/run17_ddp_slow` | run17: DDP so với TiMePReSt / PipeDream khi mạng chậm, quy tắc 5% |
| `analyze_run18.py` | `n5k/results/kaggle` | run18: B6 với w_total = 3 (3 seed) so với B6 cũ, kiểm 5 dự đoán |

Phần nghiên cứu thêm có script riêng ở `n5k/added/analysis/` (và `python -m timeprest.n5k.added.balance` cho phân tích cân bằng tải).
