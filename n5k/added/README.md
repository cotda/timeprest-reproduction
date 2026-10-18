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

- 2026-10-17 22:25 (14:25 UTC; người dùng: "chỉ dùng 1.056 món có mask của Hand ... coi như làm 1 experiment khác cũng
  được, nếu oke thì làm chính còn không thì thôi"): **KẾ HOẠCH (chờ xác nhận) — HM: huấn luyện chỉ trên các món có
  HandMasks.** Dữ liệu vẫn là Nutrition5k: cùng ảnh, cùng cache, chỉ giữ các món có mask, chia theo split của mình.
  Còn 741 món train, 157 val, 154 test; 4 món không thuộc split nào bị bỏ. Không sửa `data.py`, cache, split và trainer
  chính: launcher mới trong `timeprest/n5k/added/` nạp dataset như cũ, lọc bằng `N5kImages.subset`, rồi truyền vào
  trainer qua `data=` (mọi trainer đều nhận tham số này).
  - **HM1 (nhãn 195 nguyên liệu như phần chính).** Chỉ đổi lượng dữ liệu, không cần code nhãn mới.
  - **HM2 (nhãn 40 nhóm của HandMasks).** Đây là bản "đồng bộ" thực sự: mô hình dự đoán đúng 40 nhóm của mask, nên M3
    không cần bảng ánh xạ.
    - Nhãn presence của nhóm: món có vùng thuộc nhóm đó.
    - Gram của nhóm: tổng khối lượng các vùng. Riêng 98 vùng không có khối lượng thì lấy từ gram của Nutrition5k qua bảng
      ánh xạ (bảng này khớp 99,9%).
    - kcal mỗi gram của nhóm: trung vị kcal/mass của các vùng train.
    - Tổng của món (kcal, mass, fat, carb, protein): giữ nguyên của Nutrition5k.
    - Hạn chế đã biết: dầu và đường không có nhóm, nên kcal cộng từ nguyên liệu sẽ thấp hơn thật. Nhánh tổng trực tiếp,
      vốn cho metric kcal chính, không bị ảnh hưởng.
  - Hệ và lịch:
    - Single, PipeDream, TiMePReSt; seed 0 trước. Cùng config chính: w_total = 3, lr 0.02, 30 epoch.
    - Mỗi epoch chỉ còn khoảng 12 bước, so với 35 bước ở phần chính. Tổng số bước vì vậy chỉ bằng khoảng 1/3. Đây là
      một khác biệt có ghi lại.
    - Sau đó chấm vị trí bằng `handmasks.py` trên cùng 154 món test. HM2 dùng ánh xạ đồng nhất (mỗi nhóm là chính nó).
  - Quy tắc "đưa vào phần chính" (đặt trước; so với mô hình chính run18 cùng seed, trên **cùng 154 món test** có mask).
    HM được đưa vào phần chính chỉ khi thỏa **cả hai** điều kiện:
    - (a) test MAE kcal không kém mô hình chính quá 2 kcal;
    - (b) vị trí tốt hơn rõ: M2 rel_gain cao hơn 0,05, hoặc M3 hit cao hơn 5 điểm, ở ít nhất 2 trong 3 hệ.
    Nếu đạt seed 0 thì chạy thêm seed 1 và 2 trước khi quyết định. Nếu không đạt thì dừng và ghi kết quả.
  - Dự đoán: (a) không đạt, kcal kém khoảng 5–15 kcal vì dữ liệu chỉ bằng 1/3; (b) HM2 có thể tốt hơn ở M3, HM1 thì không.
  - Chi phí: mỗi kernel khoảng 25–30 phút (3 hệ × khoảng 5 phút, cộng checks và handmasks). Làm sau run19, để đã có mốc
    vị trí của mô hình chính.
  - 22:31 (14:31 UTC; người dùng: "oke làm cả 2, quy tắc vậy được (lúc nào tôi nói nó là chính thì nó mới là chính)"):
    **xác nhận HM1 + HM2 và quy tắc.** Quy tắc (a)(b) chỉ là **đề xuất**. HM chỉ thành phần chính khi người dùng nói.
  - 22:40 (14:40 UTC): code xong: `timeprest/n5k/added/hm.py`, config `n5k/added/configs/hm_{base,single,pipedream,timeprest}.yaml`,
    test `timeprest/n5k/added/tests/test_hm.py` (5 PASS, gồm chạy pipeline 2 tiến trình). Toàn bộ test n5k: 74 PASS.
    `handmasks.py` thêm 3 thứ: fat/carb/protein của vùng, ánh xạ `identity` cho HM2, và `mae_kcal` trên đúng các món
    được chấm để so mô hình trên cùng món. Dataset `kaggleolap/n5k-handmasks` là bản sao của ktanvn. Kernel
    `kaggleolap/n5k-run20-hm` (`n5k/added/kaggle/run20_hm/`) đã đẩy, chạy song song với run18_s0 / run19; mốc so sánh
    lấy sau.
  - 23:32 (15:32 UTC): kernel trên kaggleolap vẫn QUEUED sau 52 phút, nên đẩy cùng kernel sang `ktanvn/n5k-run20-hm`. Bản nào xong trước thì dùng bản đó.
  - 2026-10-18 00:34 (16:34 UTC): cả hai bản đã bị hủy (CANCEL_ACKNOWLEDGED, không phải do agent). Thử bằng các kernel nhỏ
    trên kaggleolap (in GPU rồi thoát):
    - không gắn gì: chạy ngay;
    - chỉ dataset mask: chạy sau 2,5 phút;
    - code + cache: chạy sau 1 phút;
    - **code + mask + cache như run20: vẫn QUEUED sau hơn 13 phút**.
    Ngoài ra, kernel đang queue cũng bị tính vào giới hạn 2 phiên GPU (lỗi "Maximum batch GPU session count of 2 reached").
    Cách sửa: run20 bỏ dataset mask, tự tải mask từ GitHub ở commit 0997586 bằng sparse checkout `data/` (không có
    cutouts), rồi kiểm tra đủ 1056 mask; thử trên máy thì giống hệt bản đang có. `ktanvn/n5k-run20-hm` version 3:
    queue 3 phút, rồi RUNNING lúc 16:34 UTC.

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
  - **Chạy lại cùng seed 0 (kaggleolap, 21:00):** mass **không** chết (test 35,7 g). Vậy việc mass chết là ngẫu nhiên
    (1/2 lần ở w_total = 10; 0/6 ở w_total = 3). TiMePReSt test kcal 49,9 (lần đầu 51,8; w_total = 3: 55,0). PipeDream
    lần này không đạt quy tắc (best val chỉ giảm 0,8) và test 49,3. Kết luận: w_total = 10 có lợi rõ cho TiMePReSt, gần
    bằng PipeDream, nhưng có nguy cơ chết đầu ra; với PipeDream thì không có lợi ổn định. Cấu hình chính giữ w_total = 3.
- **HM (2026-10-18 09:00, note `notes/2026-10-18_0900_hm-chi-mon-co-mask.md`):** chỉ 741 món có mask, seed 0, so với mô hình
  chính trên cùng 154 món test. kcal: HM1 54,7 / 52,7 / 107,6 và HM2 55,1 / 54,2 / 57,4 (single / PipeDream / TiMePReSt),
  so với chính 43,2 / 46,4 / 48,6. TiMePReSt HM1 chết đầu ra kcal từ epoch 10. Vị trí: HM2 nhỉnh hơn rất ít (M3 +1–3 điểm,
  M2 +0,01–0,07); HM1 kém hơn. **Quy tắc (a)(b) không đạt ở cả HM1 và HM2** → đề xuất không đưa vào chính (người dùng quyết định).
