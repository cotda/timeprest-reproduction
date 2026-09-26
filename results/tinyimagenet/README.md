# VGG-16-BN / Tiny-ImageNet-200 trên Kaggle T4 ×2 (seed 0)

Thiết kế: paper_notes §24. Kết quả và đánh giá: §24.1–24.2. Bảng tổng hợp: `results/phase3/report.md` (`python -m timeprest.report`).

| Thư mục | Ngày | Code | Nội dung |
|---|---|---|---|
| `run1_80ep_lr0.02/runs/` | 2026-09-26 | `e9ffd9ef30f7` | 80 epoch, lr 0.02, `graph`: `timeprest`, `variant1` (tài khoản A); `pipedream`, `pipedream_vsync`, `variant2` (tài khoản B). Biểu đồ: `compare_tin.png` (trục thời gian không so được giữa 2 tài khoản) |
| `run1_80ep_lr0.02/bench_comm/` | 2026-09-26 | `e9ffd9ef30f7` | Thời gian/epoch trên cùng máy (tài khoản A): TiMePReSt, PipeDream, PipeDream-vsync × {cùng máy, 2, 1 Gbit/s}, tập con 20k ảnh |

Run mới: tạo `runN_<điểm khác biệt>/` và thêm một dòng vào bảng trên.
