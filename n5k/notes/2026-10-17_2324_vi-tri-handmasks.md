# B7b: vị trí của bản đồ so với mask nguyên liệu thật (HandMasks, run19)

- Viết lúc: 2026-10-17 23:24 (UTC+8) = 15:24 UTC.
- Ngày chạy: 2026-10-17 15:18–15:22 UTC, `ktanvn/n5k-run19-handmasks` (CPU), cả 3 bước OK.
- Dùng checkpoint B7 của run18_s0 (w_total = 3, seed 0) cho single, PipeDream, TiMePReSt.
- Mask: HandMasks (Hand và cộng sự 2026, CC BY 4.0), mask nguyên liệu vẽ trên chính ảnh Nutrition5k. Dùng 154 món test và 157 món val của mình; 741 món thuộc train bị bỏ.
- Code: `timeprest/n5k/handmasks.py`. Nguồn: `n5k/results/kaggle/run19_handmasks/`.
- Quy tắc (đặt trước, PLAN mục 10 dòng 21:32):
  - M1: phần kcal của bản đồ nằm trên thức ăn, trừ tỉ lệ diện tích thức ăn, phải lớn hơn 0,05, Wilcoxon p < 0,01.
  - M2: MAE của tỉ lệ kcal theo vùng phải tốt hơn mốc diện tích hơn 10%, p < 0,01.
  - M3: tỉ lệ trúng phải cao hơn mốc ngẫu nhiên hơn 10 điểm.

## Số liệu (test, 154 món, lưới 7×7)

| hệ | M1 bản đồ calories (mốc 0,18) | M2 MAE tỉ lệ vùng (mốc 0,19) | M3 trúng (mốc 0,50) |
|---|---|---|---|
| single | 0,52 ✓ | 0,159 (−16%) ✓ | 0,85 ✓ |
| PipeDream | 0,52 ✓ | 0,147 (−22%) ✓ | 0,83 ✓ |
| TiMePReSt | 0,51 ✓ | 0,151 (−21%) ✓ | 0,85 ✓ |

- Bản đồ kcal trực tiếp cũng đạt cả 3 metric ở cả 3 hệ, nhưng yếu hơn: M1 0,32–0,39.
- Val cho kết quả tương tự.
- M3 theo nhóm: nhóm lớn trúng 90–100%; alium_vegetable (hành, tỏi) gần mức ngẫu nhiên; rice, grain_other, tomato, berry ở mức 0,6–0,8.

## Kết luận
1. **Bản đồ calories định vị được thức ăn thật, tốt hơn rõ so với mốc**, ở cả 3 hệ. Đây là lần đầu task này có kết luận về vị trí thật; trước đây chỉ có kết luận "trung thực với mô hình" từ deletion test.
2. Dự đoán M2 "yếu hoặc không đạt" sai: bản đồ còn chia kcal giữa các vùng đúng hơn chia theo diện tích, khoảng 20%.
3. Giới hạn: chỉ khoảng một nửa kcal của bản đồ nằm trên thức ăn, vì lưới 7×7 thô. Nguyên liệu nhỏ (hành, tỏi) gần như không định vị được.
4. Cách huấn luyện song song không ảnh hưởng tới vị trí: TiMePReSt ngang PipeDream và single.
