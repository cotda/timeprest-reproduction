# VGG-16-BN / Tiny-ImageNet-200 trên Kaggle T4 ×2 (seed 0)

Thiết kế: paper_notes §24. Kết quả và đánh giá: §24.1–24.2. Bảng tổng hợp: `results/phase3/report.md` (`python -m timeprest.report`).

| Thư mục | Ngày | Code | Nội dung |
|---|---|---|---|
| `run1_80ep_lr0.02/runs/` | 2026-09-26 | `e9ffd9ef30f7` | 80 epoch, lr 0.02, `graph`: `timeprest`, `variant1` (tài khoản A); `pipedream`, `pipedream_vsync`, `variant2` (tài khoản B). Biểu đồ: `compare_tin.png` (trục thời gian không so được giữa 2 tài khoản) |
| `run1_80ep_lr0.02/bench_comm/` | 2026-09-26 | `e9ffd9ef30f7` | Thời gian/epoch trên cùng máy (tài khoản A): TiMePReSt, PipeDream, PipeDream-vsync × {cùng máy, 2, 1 Gbit/s}, tập con 20k ảnh |

| `run2_seeds_bn/runs/` | 2026-09-26 | `406efe51593a` | Seed 1 (tài khoản A) và seed 2 (tài khoản B): `timeprest`, `pipedream`, 80 epoch, trace mỗi 10 epoch. Kiểm tra BatchNorm `tin_bn_seq_n3` / `tin_bn_seq_n1` (1 GPU, 1 mini-batch trong pipeline, 30k ảnh, 20 epoch). paper_notes §24.5 |
| `run3_resnet50/` | 2026-09-27 | `ac27eff678b1` | ResNet-50 (stem 3×3 stride 1 + max-pool, zero-init residual), 80 epoch, lr 0.1: `timeprest`, `pipedream`; bench cùng máy (∞/2/1 Gbit/s); `compare_r50.png`. paper_notes §25 |
| `run4_deepspeed/` | 2026-09-27 | `e330dd91519e` | DeepSpeed pipeline (1F1B đồng bộ, DeepSpeed 0.19.7), VGG-16, 80 epoch, seed 0 (cùng khởi tạo/thứ tự dữ liệu với TiMePReSt/PipeDream seed 0); bản quick 3 epoch; `compare_deepspeed.png`. paper_notes §26 |
| `run5_pipedream_m64/` | 2026-09-27 | `303cae94d915` | PipeDream M=64 (1F1B theo micro-batch; cách hiểu thứ hai về baseline của paper), VGG-16, 80 epoch, lr 0.02; bench cùng máy TiMePReSt / PipeDream M=192 / M=64 (∞/2/1 Gbit/s); `compare_m64.png`. paper_notes §27 |

Run mới: tạo `runN_<điểm khác biệt>/` và thêm một dòng vào bảng trên.
