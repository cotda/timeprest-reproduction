# GĐ2 — pipeline thật 2 stage, Kaggle T4 ×2 (VGG-16-BN / CIFAR-100, 160 epoch, seed 0)

Mỗi thư mục `runN_...` là một lần chạy dài (TiMePReSt + PipeDream), đã qua check D1–D4 trước khi chạy.
Mỗi run con có `metrics.csv` (mỗi epoch), `summary.json`, `config.json`, `checks.json`, `op_trace_epoch1.json`.
Biểu đồ và bảng so sánh: `compare_kaggle.png`, `compare_kaggle_summary.json` (`python -m timeprest.plot`).

| Thư mục | Ngày | Code | lr | TiMePReSt backward | Ghi chú |
|---|---|---|---|---|---|
| `run1_lr0.05_timeprest-recompute/` | 2026-09-25 | `573f49af60d8` | 0.05 | `recompute` (chạy lại forward của stage lúc backward) | Nay là ablation `timeprest_recompute`. PipeDream lr 0.05 dùng làm tham khảo. paper_notes §22 |
| `run2_lr0.02_timeprest-graph/` | 2026-09-25 | `bdf5b8b3ebbc` | 0.02 | `graph` (cơ chế PipeDream, không recompute) | Cấu hình chính hiện tại (lr tốt nhất của mỗi hệ theo sweep §22.8–22.9). paper_notes §22.10 |

Tóm tắt (top-1 cuối / thời gian mỗi epoch / peak GPU0|GPU1):

| | TiMePReSt | PipeDream |
|---|---|---|
| run1 | 73.99 % / 19.1 s / 537\|441 MB | 73.46 % / 15.0 s / 905\|353 MB |
| run2 | 72.44 % / 16.4 s / 853\|441 MB | 72.25 % / 15.1 s / 905\|353 MB |

Run mới: tạo `runN_<lr>_<điểm khác biệt>/` và thêm một dòng vào bảng trên.
