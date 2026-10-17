# MAE theo epoch và tại cùng mốc thời gian (phân tích log B6)

- Viết lúc: 2026-10-17 00:50 (UTC+8) = 2026-10-16 16:50 UTC.
- Không chạy thêm; phân tích `metrics.csv` của `n5k/results/kaggle/run6_s{0,1,2}/` (B6, chạy 2026-10-14). Bảng đầy đủ ở `n5k/PLAN.md` mục 11 ("B8 (phân tích)").
- Script: scratchpad `time_mae_b6.py` (val MAE kcal; thời gian = thời gian train cộng dồn).

## Số liệu chính (val MAE kcal, trung bình 3 seed)
- Epoch 5: PipeDream 111,1, TiMePReSt 92,4. Epoch 15: PipeDream 57,9, TiMePReSt 100,1. Epoch 30: PipeDream 49,4, TiMePReSt 55,2.
- Thời gian để đạt best ≤ 60 kcal: DDP 205 s, PipeDream 292 s, single 318 s, TiMePReSt 358 s.

## Kết luận
1. TiMePReSt chỉ học nhanh hơn PipeDream ở đầu (epoch ≤ 5, thời gian ≤ 180 s). Sau đó nó kém hơn tại mọi mốc thời gian.
2. **Phát hiện mới:** TiMePReSt dao động mạnh ở giữa quá trình (epoch 6–18) ở cả 3 seed (val MAE nhảy 70 → 115, 76 → 144). PipeDream và single ổn định dần từ epoch 11. Dao động hết khi lr cosine giảm. Nhiều khả năng do trọng số khác version giữa forward và backward khi lr còn lớn; lr nhỏ hơn có thể giúp. Chưa kiểm chứng.
3. DDP tốt nhất tại mọi mốc thời gian.
4. Đối chiếu dự đoán ở PLAN mục 8: "MAE cuối ≈ PipeDream" sai; "học nhanh hơn ở giữa" chỉ đúng ở đầu; "DDP nhanh nhất" đúng; "nhanh hơn PipeDream khi mạng 1–2 Gbit/s" đúng.
