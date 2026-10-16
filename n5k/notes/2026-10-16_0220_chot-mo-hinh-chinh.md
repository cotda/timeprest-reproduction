# Chốt mô hình chính: một nhánh; two-stream là thí nghiệm phụ

- Viết lúc: 2026-10-16 02:20 (UTC+8) = 2026-10-15 18:20 UTC.
- Quyết định của người dùng ("chọn hướng 2"), sau kết quả run8–run10. Đã ghi vào `n5k/PLAN.md` mục 10.
- Căn cứ: `2026-10-15_1730_two-stream-ket-qua.md`, `2026-10-15_1804_two-stream-partition.md`, `2026-10-16_0215_two-stream-sweep-thoi-gian.md`.

## Quyết định
1. Kết quả chính của bài là mô hình **một nhánh** `n5k_resnet50` (B6: 5 hệ × 3 seed × 30 epoch). Config `n5k_base.yaml` giữ nguyên.
2. Two-stream R50 + R50 là **thí nghiệm phụ**, không chạy B6 với two-stream (tiết kiệm khoảng 7 giờ GPU).
3. Cách trình bày two-stream trong báo cáo: chất lượng tốt hơn khi chạy single/DDP/PipeDream (kcal −4%, mAP +3,5 ở run8), nhưng trên 2×T4 pipeline bất đồng bộ chậm hơn single 10% và chậm hơn DDP 2,2 lần, không giảm bộ nhớ, và TiMePReSt kém PipeDream rõ rệt. Đây là ví dụ "mô hình nặng hơn chưa chắc làm TiMePReSt có lợi".
4. Bước tiếp theo: B7 (bản đồ calories, deletion test), đề xuất kế hoạch trước khi code.
