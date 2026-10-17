# Nghiên cứu thêm cho Nutrition5k (THỬ NGHIỆM, CHƯA PHẢI ĐÓNG GÓP)

Mọi thứ trong thư mục này là chạy thử để xem kết quả. Không gọi là đóng góp của nghiên cứu, và không đưa vào
kết luận của `n5k/PLAN.md` hay slides chính, cho đến khi người dùng nói rõ mục nào là đóng góp.

- Code: `timeprest/n5k/added/` (test: `timeprest/n5k/added/tests/`). Code cũ không sửa; chỉ dùng lớp con / file mới.
- Config: `n5k/added/configs/`. Kernel Kaggle: `n5k/added/kaggle/`. Kết quả: `n5k/results/added/`.
- Ghi chú kết luận có ghi giờ: `n5k/added/notes/` (tên `YYYY-MM-DD_HHMM_<chủ-đề>.md`, giờ máy UTC+8 và UTC).
- Bối cảnh: câu hỏi chính của nghiên cứu là huấn luyện song song, nên mỗi thử nghiệm ghi cả khía cạnh nhiều GPU
  (cân bằng tải, tăng tốc), không chỉ độ chính xác.

## Nhật ký quyết định

- 2026-10-17 14:30 (06:30 UTC; người dùng: "làm 1 và 3 nhưng tạo 1 folder khác"; "làm theo thứ tự C, A, B"). Ba thử nghiệm:
  - **C. Cân bằng tải giữa 2 GPU** (không tốn GPU). Đọc `op_trace_epoch*.json` có sẵn (thời điểm bắt đầu/kết thúc trên GPU
    của từng thao tác F/B ở mỗi stage), tách thời gian bận F/B, khoảng trống đầu/giữa/cuối epoch, chu kỳ mỗi mini-batch
    của từng stage, và so tổng thời gian bận của 2 GPU với thời gian của single. Câu hỏi: vì sao cả hai GPU bận trên 80%
    mà pipeline chỉ nhanh hơn single khoảng 1,1–1,3 lần (DDP khoảng 1,9 lần). Lưu ý: khi gửi qua gloo, `t.cpu()` chạy
    đồng bộ trong thao tác nên thời gian chép GPU→CPU nằm trong thời gian "bận".
  - **A. EMA trọng số khi đánh giá.** Lớp con của `N5kDistTrainer`: sau mỗi `opt.step` của từng stage cập nhật EMA
    (tham số và buffer BN); mỗi epoch đánh giá cả trọng số thường và EMA với decay {0,98; 0,99; 0,995} (khoảng 1,5 / 3 / 6
    epoch); mỗi loại có checkpoint best riêng và test riêng. EMA không ảnh hưởng huấn luyện, nên so sánh theo cặp trên
    cùng quá trình. TiMePReSt và PipeDream, lr 0.02, 30 epoch, seed 0/1/2.
    - Quy tắc (đặt trước): decay chọn theo trung bình val 3 seed. "EMA tốt hơn" với một hệ nếu trung bình test MAE kcal
      giảm hơn 1,5 kcal so với trọng số thường cùng lần chạy. Ghi thêm dao động của đường val EMA và khoảng cách
      TiMePReSt − PipeDream khi dùng EMA.
  - **B. Trọng số loss của nhánh tổng** `loss.w_total` ∈ {0,3; 3} (mặc định 1), TiMePReSt và PipeDream, lr 0.02, 30 epoch,
    seed 0, có ghi EMA như A.
    - Quy tắc (đặt trước, 1 seed nên chỉ là dấu hiệu): "w_total ảnh hưởng dao động" nếu dao động của TiMePReSt (bước tăng
      lớn nhất của val MAE kcal giữa hai epoch liền nhau, epoch 6–18; mốc lr 0.02 3 seed: 59,6) đổi hơn 50% ở một trong
      hai giá trị, trong khi PipeDream vẫn dưới 15. Ghi thêm test MAE và `gshare_total`.
  - Cơ sở (log có sẵn, `run15_tp60_lr02` so với `run14_60ep` PipeDream, epoch 21): train loss của TiMePReSt vẫn giảm đều,
    kcal cộng từ nguyên liệu gần PipeDream (val 104 so với 97), nhưng kcal của nhánh tổng nhảy mạnh cả trên train (121 so
    với 43) và val (188 so với 57). Nhánh tổng chỉ nhận khoảng 10–20% gradient (`gshare_total`).
- 2026-10-17 14:50 (06:50 UTC): code A xong (`timeprest/n5k/added/ema.py`, `ema_train.py`; 6 test CPU PASS, 57 test n5k
  cũ PASS). Trên 2 tiến trình CPU, trainer EMA cho train/val **trùng** `dist_train` sau khi khôi phục trạng thái RNG
  toàn cục quanh các lần đánh giá thêm (mỗi DataLoader đánh giá lấy một base seed từ RNG toàn cục; không khôi phục thì
  augmentation các epoch sau khác đi). Đã đẩy: `kaggleolap/n5k-adda-ema` (A: 6 lần chạy, khoảng 105 phút) và
  `ktanvn/n5k-addb-wtotal` (B: 4 lần chạy, khoảng 72 phút). Kernel ở `n5k/added/kaggle/`.
- 2026-10-17 16:50 (08:50 UTC; người dùng: "chạy w_total 3 seed 1 và 2"): **B2: `loss.w_total` = 3, seed 1 và 2**, TiMePReSt và
  PipeDream, lr 0.02, 30 epoch, cùng code và trainer EMA như A/B (không đẩy code mới). Seed 1 trên kaggleolap
  (`n5k-addb2-wtot3-s1`), seed 2 trên ktanvn (`n5k-addb2-wtot3-s2`), thứ tự hệ xen kẽ giữa hai seed.
  - Quy tắc (đặt trước): dùng trung bình 3 seed (seed 0 từ B). "w_total = 3 tốt hơn w_total = 1" với một hệ nếu trung bình
    test MAE kcal (trọng số thường) giảm hơn 1,5 kcal so với A (cùng code, 3 seed: TiMePReSt 57,45; PipeDream 50,99).
    Ghi thêm khoảng cách TiMePReSt − PipeDream ở w_total = 3 (so với 6,5 ở A), dao động, và EMA.
- 2026-10-17 18:05 (10:05 UTC; người dùng: "chạy thử w_total 10"): **B3: `loss.w_total` = 10**, TiMePReSt và PipeDream,
  lr 0.02, 30 epoch, seed 0, cùng code/trainer (`kaggleolap/n5k-addb3-wtot10`).
  - Quy tắc (đặt trước; chọn theo val, không theo test): với mỗi hệ, "w_total = 10 tốt hơn 3" nếu best val MAE kcal giảm hơn
    1,5 kcal so với w_total = 3 cùng seed 0 (TiMePReSt 54,5; PipeDream 44,7) **và** val mAP nguyên liệu ở epoch đó không
    thấp hơn w_total = 1 seed 0 quá 2 điểm (TiMePReSt 84,4; PipeDream 86,5). Nếu cả hai hệ đạt: chạy thêm seed 1/2 trước khi
    chọn 10. Nếu không: giữ đề xuất w_total = 3. Ghi thêm test, dao động, gshare_total.
  - 19:45 (11:45 UTC): kernel trên kaggleolap vẫn QUEUED sau 1 giờ 40 phút (có thể hết quota GPU hoặc kẹt phiên), nên
    đẩy cùng kernel sang `ktanvn/n5k-addb3-wtot10`. Bản nào xong trước thì dùng bản đó. Nếu cả hai đều chạy thì bản sau
    là lần chạy lại cùng config, dùng để xem nhiễu khi huấn luyện lại.

## Kết quả
- **C (2026-10-17 14:38, note `notes/2026-10-17_1438_can-bang-tai.md`):** từ op_trace của B6 (3 seed) và run16. Lệch tải
  giữa 2 GPU chỉ 1,06–1,12 (stage 0 là nút cổ chai; chia lại partition hoàn hảo lợi khoảng 7%). Tổng thời gian bận của
  2 GPU bằng 1,23–1,41 lần single, nên pipeline chỉ nhanh hơn single 1,2–1,4 lần (DDP 1,9). Nguồn của việc làm thêm
  (chép qua gloo, chép trọng số, hay tính) chưa tách được; đề xuất C2 (đo riêng) chưa làm.
- **A (2026-10-17 16:29, note `notes/2026-10-17_1629_ema-va-w-total.md`):** theo quy tắc, **EMA không tốt hơn** (test
  TiMePReSt 57,45 → 57,45–57,55; PipeDream 50,99 → 50,98–51,00; 3 seed). Checkpoint tốt nhất ở epoch 27–30, lúc lr gần 0
  nên EMA trùng trọng số thường. EMA chỉ làm phẳng giữa quá trình (dao động TiMePReSt 60 → 5). Lỗi thiết kế: warmup làm
  ba decay gần như trùng nhau (khoảng 0,975) tới epoch 12,6 / 25,4 / 51.
- **B (cùng note):** theo quy tắc, **w_total ảnh hưởng dao động của TiMePReSt** (0,3: 13,9; 1: khoảng 60; 3: 137),
  PipeDream vẫn dưới 15. w_total = 3 cho test tốt hơn ở cả hai hệ (TiMePReSt 55,0; PipeDream 49,4; 1 seed); 0,3 kém hơn.
- Phát hiện phụ: cùng config với run12, TiMePReSt huấn luyện lại cho test kém hơn 2,8 kcal (trung bình 3 seed) dù val gần
  như nhau; PipeDream tốt hơn 1,0. Gộp 6 lần: khoảng cách TiMePReSt − PipeDream khoảng 4,6 kcal (9%) thay vì 2,7 (5%).
- **B2 (2026-10-17 17:57, note `notes/2026-10-17_1757_w-total-3-ba-seed.md`):** theo quy tắc, **w_total = 3 tốt hơn ở cả
  hai hệ** (3 seed, trọng số thường): TiMePReSt 57,45 → 55,01 (−2,44), PipeDream 50,99 → 49,33 (−1,66; EMA 48,71); cả 6
  cặp seed đều giảm. Khoảng cách TiMePReSt − PipeDream 6,5 → 5,7. Dao động tăng ở cả hai (TiMePReSt 60 → 93; PipeDream
  10 → 35, chủ yếu do seed 1).
- **B3 (2026-10-17 20:26, note `notes/2026-10-17_2026_w-total-10.md`):** w_total = 10, seed 0 (`ktanvn/n5k-addb3-wtot10`).
  Theo quy tắc, cả hai hệ đạt: best val TiMePReSt 54,5 → 47,9, PipeDream 44,7 → 41,6; val mAP thấp hơn w_total = 1 khoảng
  1 điểm. Test kcal: TiMePReSt 51,8 (w_total = 3: 55,0), PipeDream 49,6 (49,4, không đổi). **Nhưng đầu ra mass của TiMePReSt
  chết**: val MAE mass cố định 214 g từ epoch 8, test 223 g (các w_total khác khoảng 35 g); nhiều khả năng softplus của
  nhánh tổng bão hòa ở 0 sau đoạn dao động mạnh. Quy tắc không có điều kiện cho mass nên không bắt được. Đề xuất: giữ
  w_total = 3; chưa chạy seed 1/2 (chờ người dùng). Bản kaggleolap (cùng seed) chạy từ 12:22 UTC, là lần chạy lại.
