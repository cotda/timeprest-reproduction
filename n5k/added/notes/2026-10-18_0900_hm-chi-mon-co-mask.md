# HM: huấn luyện chỉ trên các món có HandMasks (thử nghiệm, chưa phải đóng góp)

- Viết lúc: 2026-10-18 09:00 (UTC+8) = 01:00 UTC.
- Ngày chạy: 2026-10-17 16:35–17:07 UTC, `ktanvn/n5k-run20-hm` (version 3; mask tải từ GitHub ở commit 0997586). Mọi bước OK, checks PASS.
- Dữ liệu: Nutrition5k, chỉ giữ các món có mask, chia theo split của mình (741 train / 157 val / 154 test).
  - HM1: nhãn 195 nguyên liệu.
  - HM2: nhãn 40 nhóm của HandMasks.
  - Cấu hình chính: w_total = 3, lr 0.02, 30 epoch, seed 0.
- Mốc so sánh: mô hình chính run18 seed 0, chấm bằng run19 trên **cùng 154 món test**.
- Nguồn: `n5k/results/added/kaggle/run20_hm/`. Script: `n5k/added/analysis/analyze_hm.py`.
- Quy tắc (đặt trước, README dòng 22:25; chỉ là đề xuất). HM được đề xuất đưa vào phần chính chỉ khi thỏa cả hai:
  - (a) MAE kcal không kém mô hình chính quá 2 kcal ở cả 3 hệ;
  - (b) vị trí tốt hơn rõ (M2 rel_gain cao hơn 0,05 hoặc M3 hit cao hơn 0,05) ở ít nhất 2/3 hệ.

## Số liệu (154 món test)

| hệ | mô hình | MAE kcal | M1 kcal trên thức ăn | M2 MAE tỉ lệ vùng (gain so với diện tích) | M3 trúng (mốc 0,50) |
|---|---|---|---|---|---|
| single | chính | **43,2** | 0,52 | 0,159 (+0,16) | 0,85 |
| single | HM1 | 54,7 | 0,53 | 0,187 (+0,02) | 0,83 |
| single | HM2 | 55,1 | 0,56 | **0,146 (+0,23)** | **0,87** |
| PipeDream | chính | **46,4** | 0,52 | 0,147 (+0,22) | 0,83 |
| PipeDream | HM1 | 52,7 | 0,53 | 0,191 (+0,00) | 0,82 |
| PipeDream | HM2 | 54,2 | 0,56 | 0,145 (+0,24) | 0,85 |
| TiMePReSt | chính | **48,6** | 0,51 | 0,151 (+0,21) | 0,85 |
| TiMePReSt | HM1 | 107,6 (đầu ra kcal chết) | 0,58 | 0,224 (−0,14) | 0,69 |
| TiMePReSt | HM2 | 57,4 | 0,55 | 0,147 (+0,23) | 0,87 |

- mAP nguyên liệu của HM1 chỉ 62–64, so với khoảng 86 của mô hình chính. Nhiều nguyên liệu hiếm không xuất hiện đủ trong 741 món train. HM2 có mAP 76–78 trên 40 nhóm; bộ nhãn khác nên không so trực tiếp được.
- **TiMePReSt HM1: đầu ra kcal chết** từ epoch 10. Val MAE kcal đứng yên ở 201–216 tới hết, trong khi mass vẫn học tiếp. Đây là cùng kiểu với B3 (mass chết ở w_total = 10): TiMePReSt cộng với nhánh tổng có trọng số lớn, ở đây thêm ít dữ liệu và ít bước (12 bước mỗi epoch). Checkpoint tốt nhất lấy ở epoch 6.

## Kết luận
1. Theo quy tắc, **cả HM1 và HM2 đều không đạt**.
   - (a) kcal kém mô hình chính 6–12 kcal ở mọi hệ; TiMePReSt HM1 kém 59 kcal.
   - (b) chỉ HM2 single cải thiện vị trí đủ ngưỡng (M2 +0,07), nên là 1/3 hệ.
   Đề xuất: **không đưa HM vào phần chính**. Người dùng quyết định.
2. Dự đoán đúng phần (a): kcal kém 5–15 kcal vì dữ liệu chỉ bằng 1/3; riêng TiMePReSt HM1 tệ hơn nhiều vì đầu ra chết. Phần (b) gần đúng: HM2 định vị nhỉnh hơn một chút (M3 +1–3 điểm, M2 +0,01–0,07, M1 +0,04), nhưng không vượt ngưỡng.
3. Ý nghĩa: huấn luyện trên nhãn khớp với mask (HM2) chỉ giúp vị trí rất ít, mà mất nhiều kcal và mAP. Mô hình chính, học trên toàn bộ dữ liệu, đã định vị gần như tốt bằng.
4. Thêm một lần đầu ra chết ở TiMePReSt (sau B3), củng cố phát hiện: TiMePReSt kém ổn định khi gradient của nhánh tổng lớn so với dữ liệu.
