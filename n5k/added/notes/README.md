# Ghi chú kết luận của phần nghiên cứu thêm (thử nghiệm, chưa phải đóng góp)

| Thời điểm (UTC+8) | File | Tóm tắt |
|---|---|---|
| 2026-10-17 14:38 (06:38 UTC) | [2026-10-17_1438_can-bang-tai.md](2026-10-17_1438_can-bang-tai.md) | C: lệch tải 2 GPU chỉ khoảng 1,1; vấn đề chính là pipeline làm thêm 23–41% việc GPU so với single, nên chỉ nhanh hơn 1,2–1,4 lần (DDP 1,9) |
| 2026-10-17 16:29 (08:29 UTC) | [2026-10-17_1629_ema-va-w-total.md](2026-10-17_1629_ema-va-w-total.md) | A: EMA không cải thiện test (best ở cuối lịch cosine), chỉ làm phẳng giữa quá trình; B: dao động TiMePReSt tỉ lệ với w_total, w_total=3 tốt hơn cả hai hệ (1 seed); TiMePReSt huấn luyện lại lệch test 2,8 kcal |
| 2026-10-17 17:57 (09:57 UTC) | [2026-10-17_1757_w-total-3-ba-seed.md](2026-10-17_1757_w-total-3-ba-seed.md) | B2 (3 seed): w_total=3 tốt hơn ở cả hai hệ (TiMePReSt 57,5 → 55,0; PipeDream 51,0 → 49,3, EMA 48,7); khoảng cách 6,5 → 5,7; dao động tăng ở cả hai |
