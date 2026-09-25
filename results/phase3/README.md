# GĐ3 — Đánh giá trên Kaggle T4 ×2 (VGG-16-BN / CIFAR-100, seed 0)

Thiết kế: paper_notes §23. Kết quả và đánh giá: paper_notes §23.1. Bảng tổng hợp mọi run GĐ2 + GĐ3: `report.md`
(tạo lại bằng `python -m timeprest.report`).

| Thư mục | Ngày | Code | Nội dung |
|---|---|---|---|
| `run1_gd3/runs/` | 2026-09-25 | `ae7d8321017d` | E1: `variant1` (nF1B + stashing), `variant2` (1F1B, bỏ stashing); E2: `timeprest_n2` (N=2, M=192), `timeprest_n2_m384` (N=2, M=384). 160 epoch, lr 0.02, `graph` |
| `run1_gd3/bench_comm/` | 2026-09-25 | `ae7d8321017d` | E3: thời gian/epoch của TiMePReSt và PipeDream theo băng thông giả lập (∞, 10, 5, 2, 1, 0.5 Gbit/s; latency 0.1 ms) |

Baseline dùng để so sánh: `results/phase2/run2_lr0.02_timeprest-graph/`.
Run mới: tạo `runN_<điểm khác biệt>/` và thêm một dòng vào bảng trên.
