# report/: báo cáo và hình so sánh với paper

- `report_2026-09-27.md`: báo cáo tiến độ (24/09 → 27/09/2026).
- `make_figures.py`: vẽ lại toàn bộ hình từ `results/` → `figures/*.png`. Chạy: `python report/make_figures.py` (chỉ cần matplotlib, chạy trên CPU).

Quy ước chung của các hình:
- Mỗi hệ luôn có cùng màu và kiểu nét ở mọi hình: TiMePReSt xanh dương nét liền, PipeDream cam nét đứt, còn lại có nhãn trực tiếp.
- **Marker rỗng kèm thanh khoảng** là giá trị đọc bằng mắt từ hình của paper (xấp xỉ, xem paper_notes §11), đặt ở bên phải dữ liệu của mình.
- Tên file `figNN_...` dùng số hình của paper để tiện đặt cạnh nhau so sánh.

## Hình của mình ↔ hình của paper

| Hình của mình | Hình paper | Điểm cần nhìn khi so sánh |
|---|---|---|
| `fig04_vgg16_cifar100.png` | Fig.4 (Cluster A, W=2) | Theo epoch hai đường trùng nhau, giống paper. Theo thời gian: paper thấy PipeDream bị bỏ xa (chậm 5–6 lần/epoch), của mình hai đường gần trùng. |
| `fig08_vgg16_tinyimagenet.png` | Fig.8 (Cluster B, W=3) | Paper: TiMePReSt cần **nhiều** epoch hơn. Mình: TiMePReSt đi **trước** ở giữa quá trình và bằng nhau ở cuối. Paper chạy ~140 epoch, mình chạy 80. |
| `fig08b_vgg16_tinyimagenet_3seeds.png` | Fig.8 | Như trên, trung bình ± sd 3 seed: khác biệt giữa quá trình lớn hơn nhiễu giữa các seed. |
| `fig05_resnet50_tinyimagenet.png` | Fig.5 | Cùng xu hướng: bằng nhau theo epoch; hai hệ gần nhau theo thời gian. |
| `fig09_vgg16_tinyimagenet_systems.png` | Fig.9 (Cluster C, W=4) | Paper: TiMePReSt 69–73% còn các hệ khác chỉ 24–35%. Mình: cả 5 hệ về cùng mức ~58–61%, DeepSpeed (đồng bộ) không có đoạn học nhanh giữa quá trình. |
| `fig13_ablation_vgg16_tinyimagenet.png` | Fig.13 | Paper: Variant 1 > TiMePReSt > Variant 2. Mình: cả ba ~59%, Variant 1 bám sát TiMePReSt. |
| `fig11_microbatches_vgg16_cifar100.png` | Fig.11 | Giống paper: N=2 với mini-batch lớn hơn (M=384) kém nhất. Khác: của mình là CIFAR-100 W=2, paper là Tiny-ImageNet W=4. |
| `fig15_memory.png` | Fig.15 (Cluster C, W=4) | Paper: TiMePReSt tiết kiệm ~30%. Mình: stage 1 giảm 0–6%, stage 2 tăng, tổng gần bằng. PipeDream M=64 và DeepSpeed dùng ít bộ nhớ nhất. |
| `fig16_epoch_time.png` | Fig.16a / 17a | **Hai panel có thang y khác nhau.** Paper: PipeDream chậm hơn 1.7–5.7 lần. Mình (cùng máy): 0.89–1.05 lần. |
| `figX_bandwidth.png` | (không có trong paper) | Lợi thế thời gian chỉ xuất hiện khi truyền đắt (1–2 Gbit/s: nhanh hơn 9–16%). Khi cùng máy thì không có lợi thế. |

## Bảng số liệu (cùng dữ liệu với các hình)

### Top-1 cuối cùng (%)

| Workload | TiMePReSt | PipeDream | Khác | Paper (đọc từ hình) |
|---|---|---|---|---|
| VGG-16 / CIFAR-100, 160 ep | 72.44 | 72.25 | Variant 1 72.10, Variant 2 72.46, N=2 71.78, N=2 M=384 69.67 | TiMePReSt ~72–75, PipeDream ~73–77 (Fig.4b) |
| VGG-16 / Tiny-ImageNet, 80 ep, seed 0 / 1 / 2 | 59.07 / 58.82 / 58.67 | 58.94 / 58.50 / 59.21 | PipeDream M=64 60.94, DeepSpeed 58.33, Variant 1 59.05, Variant 2 58.84, PipeDream+vsync 59.14 | TiMePReSt ~68–72, PipeDream ~73–77 (Fig.8b, ~140 ep) |
| ResNet-50 / Tiny-ImageNet, 80 ep | 66.49 | 65.44 | — | không đọc được số |

### Peak memory mỗi GPU, stage 1 / stage 2 (MB, run dài)

| Workload | TiMePReSt | PipeDream | PipeDream M=64 | DeepSpeed |
|---|---|---|---|---|
| VGG-16 / CIFAR-100 | 853 / 441 | 905 / 353 | — | — |
| VGG-16 / Tiny-ImageNet | 3344 / 833 | 3529 / 867 | 1198 / 420 | 1182 / 446 |
| ResNet-50 / Tiny-ImageNet | 8433 / 1580 | 8515 / 1470 | — | — |

### Thời gian mỗi epoch theo băng thông giả lập (s, bench_comm)

| Workload | Băng thông (Gbit/s) | TiMePReSt | PipeDream | PipeDream M=64 | PipeDream / TiMePReSt |
|---|---|---|---|---|---|
| VGG-16 / CIFAR-100 | cùng máy | 17.6 | 16.6 | — | 0.94 |
| | 10 | 18.1 | 17.0 | — | 0.94 |
| | 5 | 18.3 | 17.9 | — | 0.98 |
| | 2 | 23.9 | 27.5 | — | 1.15 |
| | 1 | 37.7 | 41.5 | — | 1.10 |
| | 0.5 | 67.5 | 69.3 | — | 1.03 |
| VGG-16 / Tiny-ImageNet | cùng máy | 21.6 | 23.6 | 19.9 | 1.09 |
| | 2 | 35.7 | 42.9 | 39.0 | 1.20 |
| | 1 | 57.9 | 64.8 | 63.5 | 1.12 |
| ResNet-50 / Tiny-ImageNet | cùng máy | 49.0 | 45.2 | — | 0.92 |
| | 2 | 63.5 | 62.7 | — | 0.99 |
| | 1 | 72.6 | 83.8 | — | 1.15 |

Nguồn: VGG-16/CIFAR-100 từ `results/phase3/run1_gd3/bench_comm`; VGG-16/Tiny-ImageNet từ `results/tinyimagenet/run5_pipedream_m64/bench_comm` (lần đo trước ở `run1_80ep_lr0.02/bench_comm` cho cùng xu hướng: 23.1 / 22.7 s khi cùng máy); ResNet-50 từ `results/tinyimagenet/run3_resnet50/bench_comm`.
Giải thích chi tiết và sai khác so với paper: `paper_notes.md` §19–§27.
