# Nhật ký phiên bản: luồng auto bounding box món ăn

Mỗi phiên bản ghi lại: đã làm gì, cấu hình, kết quả, lỗi phát hiện và cách sửa. Phiên bản mới thêm **lên đầu** mục "Các phiên bản".
Quy ước tên:
- `vN`: một lần chạy kernel Kaggle mới (ghi số version của kernel trên Kaggle).
- `vN.M`: thay đổi tính lại trên máy từ dự đoán đã tải về của `vN` (không chạy lại GPU).

Số liệu để đối chiếu nằm trong `results/<kernel>/output/` (`results.csv` = bảng của chính kernel; `results_<tag>.csv` = tính lại
bằng `tools/reeval_01.py`). Git do người dùng tự commit: khi commit, ghi tag phiên bản (vd. `[foodbox v1.1]`) vào message.

## Tóm tắt nhanh (cập nhật mỗi phiên bản)

| Phiên bản | Ngày | Việc chính | Model tốt nhất (AP@0.5 trên UEC mini) | Trạng thái |
|---|---|---|---|---|
| v2.1 | 2026-11-01 | Dựng lại 02: conf 0.35 + `other_conf` 0.45; chấm tay toàn bộ 194 ảnh có box | (GDINO) | **Đạt tiêu chí**: 97% ảnh có box, 89.5% box đúng (box chính 94.8%) |
| v2 | 2026-11-01 | Kernel 02: gán nhãn 200 ảnh Food-101 tiny bằng GDINO conf 0.45 | (GDINO, chọn bởi người dùng) | Xong; 78.5% ảnh có box, conf 0.45 quá chặt cho món tráng miệng |
| v1.1 | 2026-11-01 | Bỏ 2 bộ lọc làm hỏng kết quả (max_area 0.98, drop_container) | YOLO-World 0.735 ≈ GDINO 0.732 | Chờ người dùng chọn model cho 02 |
| v1 | 2026-11-01 | Kernel 01: so sánh 3 model ± SAM trên 200 ảnh UEC | YOLO-World 0.647, GDINO 0.618 | Xong, có lỗi hậu xử lý (sửa ở v1.1) |
| v0 | 2026-11-01 | Chọn dataset, tạo UEC mini, viết thư viện + test, chạy thử CPU | — | Xong |

---

## Các phiên bản

### v2.1 — Dựng lại bước 02 với ngưỡng mới + chấm tay toàn bộ (2026-11-01, chỉ chạy trên máy)

**Tiêu chí "ổn" (người dùng đồng ý):** ≥ 90% ảnh có box **và** ≥ 85% box đúng khi chấm bằng mắt.

**Thay đổi:** `conf` 0.45 → **0.35** (box món chính); thêm tham số `other_conf` = **0.45** (`foodbox/autolabel.py`): box
`food_other` cần điểm ≥ 0.45 mới giữ. Dùng lại đúng dự đoán thô của v2 (`tmp_preds/`), không chạy lại GPU.
Hai giá trị này thành mặc định mới trong CFG của notebook 02.

**Lệnh tái tạo:** `python tools/rebuild_02.py results/02_auto_label --tag v2.1 --conf 0.35 --other-conf 0.45`
(tái tạo v2 đúng từng số: `--tag v2_repro --conf 0.45`). Kết quả: `results/02_auto_label/output_v2.1/`.

**Kết quả tự động:** 194/200 ảnh có box (97.0%), tổng 257 box (194 món chính + 63 `food_other`), 1,28 box/ảnh;
cần xem lại 80 ảnh (`box_near_full_image` 39, `low_conf` 37, `no_box` 6, `many_boxes` 5); failed 6
(cannoli/1417238, ice_cream/155035, tiramisu/1039481, tiramisu/1060127, tiramisu/1380802, tiramisu/1644287).

**Chấm tay** (người chấm: Claude, xem 13 lưới ảnh `review_v2.1/grid_*.jpg`, mỗi ảnh ~380 px; từng ảnh ở `review_v2.1/review_v2.1.csv`):

| Chỉ số | Kết quả | Tiêu chí |
|---|---|---|
| Ảnh có box | 194/200 = **97.0%** | ≥ 90% → đạt |
| Tổng box đúng | 230/257 = **89.5%** | ≥ 85% → đạt |
| Box món chính đúng | 184/194 = 94.8% (thiếu 8, sai 2) | |
| Box `food_other` đúng | 46/63 = 73.0% | điểm yếu |
| Ảnh hoàn toàn đúng (box chính đúng, không có `food_other` sai) | 178/194 = 91.8% (89.0% trên 200 ảnh) | |

Quy ước chấm: "đúng" = box ôm trọn phần thức ăn của món (ôm cả phần đồ ăn kèm trên cùng đĩa vẫn tính đúng; ảnh chụp cận
box gần cả ảnh vẫn đúng); "thiếu" = bỏ sót phần lớn món (vd. chỉ 1 trong 2 lát bánh); "sai" = ôm nhầm vật khác.
`food_other` ôm **một phần ăn khác cùng loại** (bát bibimbap thứ hai, ly kem khác) tính là đúng theo quy tắc hiện tại.

**Các lỗi còn lại (từ review):**
- Box chính sai (2): #44 cannoli/125648 (tủ kính, ôm khay bánh khác); #97 falafel/1104277 (ôm hummus, falafel bị gán `food_other`).
- Box chính thiếu (8): món gồm nhiều miếng rời: french toast (#107, #108), wrap (#96, #98), kem 2 viên (#131), sushi/sashimi
  (#158, #166, #170).
- `food_other` sai (17 box ở 10 ảnh): **12/17 là từng miếng của chính món chính** (falafel #88, #94; sushi #166, #170; #97),
  còn lại: cốc trà (#68), đốm đèn (#158), box trùng trong ảnh tủ kính/nhiều nồi (#122, #135, #149).

**Kết luận v2.1:** đạt tiêu chí, dùng được làm nhãn nháp cho dataset thật. Ngưỡng 0.35 không làm tăng box sai: các ảnh mới có
box nhờ hạ ngưỡng (tiramisu #178/#182/#185, apple_pie #10, cannoli #54...) đều đúng.
Điểm yếu chính là **món nhiều miếng**: model khi thì ôm cả đĩa, khi thì ôm từng miếng, làm sinh `food_other` sai hoặc box chính thiếu.

**Lưu ý về độ tin cậy:** chỉ một người chấm (Claude), xem ảnh thu nhỏ, có vài ca không chắc (#135). Nên để một thành viên
nhóm chấm lại khoảng 30 ảnh ngẫu nhiên để đối chiếu.

---

### v2 — Kernel `kaggleolap/foodbox-02-auto-label`, version 2 (2026-11-01)

**Đã làm:** tự gán box cho 200 ảnh Food-101 tiny (10 món × 20 ảnh, seed 0; gộp `train/` + `valid/`), xuất bộ nhãn YOLO đầy đủ.
Version 1 của kernel lỗi ở bước dò dữ liệu (E6), sửa rồi đẩy lại thành version 2.

**Cấu hình (người dùng chọn):** model `gdino` (Grounding DINO base), `conf=0.45`, prompt `["food"]`, `sam=False`, NMS 0.5,
`min_area` 0.002, `max_area` 1.0, `drop_container` False, `main_rule="largest"`, `review_conf` = 0.55, `max_boxes` 4, `val_frac` 0.2.
Dataset `msarmi9/food101tiny`. 2 GPU, mỗi GPU 100 ảnh (shard). Môi trường giống v1.

**Thời gian:** notebook ~3 phút (detect 2,1 phút, nạp model 25,5 s/GPU), VRAM đỉnh 2,26 GB/GPU. Output 22 MB (E5 đã hết).

**Kết quả** (`results/02_auto_label/output/`):
- 157/200 ảnh (78.5%) có box, tổng 213 box: 157 box món chính + 56 box `food_other`; trung bình 1,06 box/ảnh;
  confidence cao nhất trung bình 0,62.
- `failed.csv`: 43 ảnh không có box (tiramisu 11, cannoli 9, ice_cream 5, sushi 5, falafel 4, french_toast 4, ramen 3, apple_pie 2).
- `report.csv`: 117 ảnh cần xem lại: `low_conf` 49, `no_box` 43, `box_near_full_image` 32, `many_boxes` 4.
- `train.txt` 126 ảnh, `val.txt` 31 ảnh (không gồm ảnh không box).

| món | ảnh | box/ảnh | conf cao nhất TB | cần xem lại |
|---|---|---|---|---|
| apple_pie | 20 | 1.00 | 0.65 | 8 |
| bibimbap | 20 | 1.35 | 0.62 | 7 |
| cannoli | 20 | 0.65 | 0.59 | 15 |
| edamame | 20 | 1.10 | 0.70 | 5 |
| falafel | 20 | 1.00 | 0.56 | 15 |
| french_toast | 20 | 0.85 | 0.61 | 12 |
| ice_cream | 20 | 1.70 | 0.58 | 14 |
| ramen | 20 | 1.45 | 0.72 | 9 |
| sushi | 20 | 1.00 | 0.53 | 17 |
| tiramisu | 20 | 0.55 | 0.59 | 15 |

**Phân tích**
1. **Box có được thì tốt:** ôm sát thức ăn (edamame, kem trong ly, chả falafel trong giấy bạc); ảnh bibimbap tách đúng món chính
   và các đĩa phụ (`food_other`). Xem `preview_grid.jpg`, `output/visualized/`.
2. **conf 0.45 (chọn theo UEC) quá chặt cho Food-101**, nhất là món tráng miệng: cả 43 ảnh failed đều có box ứng viên điểm
   0,31–0,45 (tính lại từ `tmp_preds/`). Vẽ box tốt nhất của 20 ảnh failed: ~14 box đúng (cannoli, tiramisu, ramen, falafel...),
   ~6 sai/thiếu (french toast chỉ ôm 1 lát, tiramisu chỉ ôm quả mâm xôi, tủ kính cannoli).
   Số ảnh có box theo ngưỡng: conf 0.25 → 200, 0.30 → 200, 0.35 → 194, 0.40 → 183, 0.45 → 157.
3. **`food_other` hơi nhiều** (56 box, kem 1,70 box/ảnh): một phần là đồ trang trí (dâu, sốt) hoặc một phần của món, không phải
   món khác. Cần xem trong `visualized/` trước khi train.
4. Đề xuất v2.1 (không tốn GPU, dựng lại từ `tmp_preds/`): conf 0.35 (194/200 ảnh có box; ảnh 0.35–0.45 vẫn bị đánh dấu
   `low_conf` để sửa tay) và chỉ giữ box `food_other` khi điểm ≥ conf món chính. **Chờ người dùng đồng ý.**

**Lỗi phát hiện:** E6, E7.

---

### v1.1 — Tính lại kết quả 01 với hậu xử lý đã sửa (2026-11-01, chỉ chạy trên máy)

**Thay đổi** (mặc định mới trong `foodbox/postprocess.py`, `autolabel.py`, CFG của 2 notebook):
- `max_area`: 0.98 → **1.0**. Lý do: ảnh chụp cận (món chiếm cả khung, vd. chả giò `fried_spring_rolls/162684`) có box đúng
  chiếm ~99% ảnh (GDINO 0.62) và bị bộ lọc xoá → ảnh không còn box nào. Box gần cả ảnh giờ chỉ bị đánh dấu xem lại
  (`box_near_full_image` trong report.csv), không bị xoá.
- `drop_container`: True → **False**. Lý do: món nhiều miếng (sushi, chả giò, khoai tây chiên) được model trả cả box "cả đĩa"
  lẫn box từng miếng; bộ lọc coi box cả đĩa là "khay" và xoá nó, trong khi box thật của UEC (và yêu cầu "mỗi món một box")
  là box cả đĩa. Vẫn giữ tuỳ chọn để thử cho ảnh khay cơm nhiều món.
- `launch(..., cwd="/tmp/foodbox_models")`: trọng số model không còn bị ghi vào output (xem lỗi E5).

**Lệnh tái tạo:** `python tools/reeval_01.py results/01_compare/output --tag v1.1`
(bảng v1 tái tạo đúng từng số: `--tag v1_repro --max-area 0.98 --drop-container`).

**Kết quả** (`results/01_compare/output/results_v1.1.csv`; P/R/F1/IoU ở ngưỡng `conf` cho F1 cao nhất của từng model):

| model | AP@0.5 | AP@0.3 | Precision | Recall | F1 | IoU TB | area_ratio | chọn đúng món chính (box lớn nhất) | conf | s/ảnh (T4) | VRAM (MB) | ảnh không box |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yoloworld | **0.735** | 0.834 | 0.674 | 0.683 | 0.679 | 0.596 | 0.835 | **0.715** | 0.218 | **0.057** | 1387 | **9** |
| gdino | 0.732 | **0.836** | 0.676 | **0.717** | **0.696** | **0.611** | 0.815 | 0.700 | 0.468 | 0.582 | 2261 | 17 |
| yoloworld+sam | 0.710 | 0.832 | 0.728 | 0.617 | 0.668 | 0.540 | 0.827 | 0.675 | 0.289 | 0.376 | 1950 | 15 |
| gdino+sam | 0.709 | 0.834 | 0.664 | 0.703 | 0.683 | 0.598 | 0.803 | 0.680 | 0.468 | 0.919 | 2261 | 17 |
| owlv2 | 0.386 | 0.475 | 0.447 | 0.503 | 0.473 | 0.427 | 0.821 | 0.460 | 0.186 | 0.379 | 780 | 53 |
| owlv2+sam | 0.384 | 0.478 | 0.454 | 0.493 | 0.473 | 0.415 | 0.811 | 0.455 | 0.188 | 0.713 | 1685 | 54 |

AP@0.5 theo món (v1.1):

| món | gdino | gdino+sam | owlv2 | yoloworld | yoloworld+sam |
|---|---|---|---|---|---|
| bibimbap | 0.79 | 0.79 | 0.57 | 0.79 | 0.79 |
| french_fries | 0.68 | 0.68 | 0.37 | 0.77 | 0.77 |
| fried_spring_rolls | 0.67 | 0.64 | 0.70 | 0.49 | 0.39 |
| green_salad | 0.77 | 0.77 | 0.53 | 0.74 | 0.73 |
| miso_soup | 0.64 | 0.59 | 0.40 | 0.71 | 0.64 |
| pho | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 |
| pizza | 0.84 | 0.84 | 0.28 | 0.86 | 0.86 |
| ramen_noodle | 0.85 | 0.74 | 0.52 | 0.74 | 0.71 |
| rice | 0.56 | 0.52 | 0.31 | 0.70 | 0.67 |
| sushi | 0.75 | 0.75 | 0.35 | 0.72 | 0.72 |

**Kết luận v1.1**
1. **YOLO-World và Grounding DINO ngang nhau về độ chính xác** (AP@0.5 0.735 vs 0.732; AP@0.3 0.834 vs 0.836). GDINO có
   recall/IoU cao hơn một chút; YOLO-World **nhanh gấp ~10 lần** (0.057 vs 0.58 s/ảnh), ít ảnh trống hơn (9 vs 17), nhưng yếu ở
   món nhiều miếng (chả giò 0.49 vs 0.67). OWLv2 kém rõ rệt (0.39), loại.
2. **SAM 2.1 không giúp** so với box thật của UEC (AP giảm 0.02–0.03) và chậm thêm 0.3 s/ảnh. Nguyên nhân chính: box thật UEC
   ôm cả bát/đĩa, còn detector với prompt "food" vốn đã ôm sát phần thức ăn (`area_ratio` ≈ 0.8 cho mọi model, nghĩa là box dự
   đoán nhỏ hơn box thật ~20%). Hình `compare/rice/11201.jpg`: mọi model ôm đúng phần cơm, box thật ôm cả bát → IoU < 0.5 → bị
   tính sai. Vậy điểm số trên UEC **phạt đúng hành vi mình muốn**; điểm thực tế cho "ôm sát thức ăn" cao hơn con số này.
3. Món khó: cơm (rice, ôm cơm thay vì cả bát), súp miso (bát nhỏ, nhiều món trong khay), chả giò (nhiều miếng).
4. Đề xuất cho bước 02: **Grounding DINO, conf ≈ 0.45** (độ chính xác/recall tốt nhất, 200 ảnh mất ~2 phút trên 2 GPU),
   hoặc **YOLO-World, conf ≈ 0.22** nếu cần nhanh (dataset thật ~1.500 ảnh: ~1,5 phút). SAM tắt mặc định, chỉ bật khi thấy box
   dính đĩa trên dữ liệu thật.

---

### v1 — Kernel `kaggleolap/foodbox-01-compare`, version 1 (2026-11-01)

**Đã làm:** chạy Grounding DINO base, OWLv2 base, YOLO-World v2-x, mỗi model ± SAM 2.1, trên 200 ảnh UEC mini; so với box
thật; xuất `results.csv`, `results_per_class.csv`, 200 hình so sánh (box thật xanh lá, model đỏ), `compare_grid.jpg`.

**Môi trường (log kernel):** Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, ultralytics 8.4.170 (pip, cố định),
Pillow 13.0.0, 2 × Tesla T4. Đẩy lúc 05:07 UTC, notebook chạy ~6 phút (detect 4,9 phút), tổng < 15 phút quota GPU.

**Cấu hình:** dataset `kaggleolap/uecfood256-mini` (v0); prompt `["food"]`; GPU 0: gdino; GPU 1: owlv2 → yoloworld; SAM 2.1 base
(`sam2.1_b.pt`) cho top 20 box mỗi ảnh; hậu xử lý NMS 0.5, `min_area` 0.002, `max_area` 0.98, `drop_container` True; IoU 0.5.
Thời gian nạp model: gdino 24,5 s, owlv2 24,4 s, yoloworld 13,3 s.

**Kết quả** (`results/01_compare/output/results.csv`):

| model | AP@0.5 | AP@0.3 | P | R | F1 | IoU TB | conf | s/ảnh | ảnh không box |
|---|---|---|---|---|---|---|---|---|---|
| yoloworld | 0.647 | 0.767 | 0.660 | 0.627 | 0.643 | 0.540 | 0.218 | 0.057 | 21 |
| yoloworld+sam | 0.621 | 0.763 | 0.711 | 0.567 | 0.631 | 0.490 | 0.289 | 0.376 | 28 |
| gdino | 0.618 | 0.739 | 0.764 | 0.583 | 0.662 | 0.489 | 0.535 | 0.582 | 48 |
| gdino+sam | 0.593 | 0.738 | 0.747 | 0.570 | 0.647 | 0.478 | 0.535 | 0.919 | 48 |
| owlv2 | 0.290 | 0.404 | 0.410 | 0.397 | 0.403 | 0.348 | 0.186 | 0.379 | 63 |
| owlv2+sam | 0.285 | 0.400 | 0.417 | 0.387 | 0.401 | 0.334 | 0.189 | 0.713 | 67 |

**Lỗi phát hiện ở v1:** E4, E5 (bên dưới); sửa ở v1.1.

---

### v0 — Chuẩn bị (2026-11-01)

**Dataset** (người dùng chọn):
- UECFOOD-256: bản gốc `rkuo2000/uecfood256` (4,2 GB) quá nặng để gắn vào kernel → tải về máy một lần (vài phút), cắt bản mini
  bằng `tools/make_uec_mini.py` → dataset riêng tư **`kaggleolap/uecfood256-mini`** (18,6 MB, 458 file): 10 món
  `1,7,11,18,23,36,87,98,206,209` (rice, sushi, bibimbap, pizza, ramen noodle, miso soup, green salad, french fries, pho,
  fried spring rolls) × 20 ảnh (seed 0) + toàn bộ 256 file `bb_info.txt` + `category.txt`.
  Thống kê: 200 ảnh, 58 ảnh có ≥ 2 box (ảnh nhiều món), 1 ảnh có tag xoay EXIF.
- Food-101: **`msarmi9/food101tiny`** (96 MB; 10 món × 150 train + 50 valid; `data/food-101-tiny/{train,valid}/<món>/*.jpg`).
  Bản đầy đủ `kmader/food41` (5,7 GB) để dành khi cần nhiều món hơn.

**Code:** thư viện `foodbox/` (dò dữ liệu, chuyển UEC → YOLO, 3 detector + SAM, hậu xử lý, chỉ số, xuất bộ nhãn), notebook
viết dạng `notebook.py` rồi ghép `.ipynb` có nhúng code (`tools/build_notebook.py`), `run_kaggle.py` (push → chờ → tải),
12 test CPU (`python -m pytest`), `survey.md`, `README.md`.

**Chạy thử trên CPU máy local** (venv riêng, transformers 4.49, ultralytics 8.4.170): cả 3 model + SAM chạy được; chạy trọn
notebook 01 (YOLO-World, 20 ảnh) và 02 (20 ảnh Food-101 tiny); train thử YOLO11n 1 epoch trên nhãn sinh ra → đọc được, 0 file lỗi.

**Lỗi phát hiện và sửa trước khi lên Kaggle:** E1, E2, E3.

---

## Danh sách lỗi và bài học

| Mã | Phát hiện ở | Lỗi | Nguyên nhân | Cách sửa | Trạng thái |
|---|---|---|---|---|---|
| E1 | v0 (chạy thử CPU) | GDINO gần như không ra box (recall 0.17 trên 4 ảnh) dù điểm thô tốt (bát phở 0.77) | `drop_containers` xoá box bát phở vì bên trong có vài box rác điểm 0.1 (miếng thịt, hành) | Chỉ tính box con có điểm ≥ 0.5 × điểm box cha | Đã sửa (AP thử 0.17 → 0.93) |
| E2 | v0 | Tiến trình con `run_detect` không import được `foodbox` | Tiến trình con không kế thừa `sys.path` của notebook | `launch` tự thêm thư mục thư viện vào `PYTHONPATH` | Đã sửa |
| E3 | v0 (train thử YOLO) | Ảnh không tìm được box vẫn nằm trong train.txt → YOLO coi là "ảnh nền không có thức ăn" | Thiếu điều kiện | Ảnh không box chỉ ghi vào failed.csv, không vào train/val | Đã sửa |
| E4 | v1 | `max_area` 0.98 xoá box đúng của ảnh chụp cận; `drop_container` xoá món nhiều miếng → AP thấp hơn thực tế ~0.09, ảnh trống GDINO 48/200 | Giả định "box gần cả ảnh = cả bàn" sai với ảnh món ăn chụp cận (Food-101 phần lớn là kiểu này) | Mặc định `max_area=1.0`, `drop_container=False` (v1.1) | Đã sửa |
| E5 | v1 | Output tải về 665 MB thay vì 33 MB | Ultralytics tải `sam2.1_b.pt`, `yolov8x-worldv2.pt`, CLIP vào thư mục hiện tại `/kaggle/working` | Tiến trình con chạy với `cwd=/tmp/foodbox_models` | Đã sửa (chưa kiểm chứng trên Kaggle) |
| E6 | v2 (kernel version 1) | Notebook 02 dừng ở bước dò dữ liệu: "Không tìm thấy thư mục dạng <tên_món>/*.jpg" | Kaggle gắn dataset ở `/kaggle/input/datasets/<owner>/<slug>/...`; thư mục món của food101tiny sâu 6 cấp, hàm dò chỉ tìm 5 cấp (chạy thử trên máy thư mục nông hơn nên không lộ) | `max_depth` 5 → 8 cho `find_class_roots`, `find_uec_root`; thêm test cấu trúc sâu kiểu Kaggle | Đã sửa (version 2 chạy được) |
| E7 | v2 | Kết quả tải về thiếu file: `visualized/` 76/200, không có `tmp_preds/`, `preview_grid.jpg` | `kaggle kernels output` chỉ tải trang đầu (500 file) rồi in "Next page token"; `--file-pattern` cũng chỉ lọc trong trang đầu | `run_kaggle.py` tự tải mọi trang qua Kaggle API (`download_output`); thêm `--download-only` | Đã sửa (tải lại đủ 631 file) |
| L1 | v1 | Box thật UEC ôm cả bát/đĩa, ngược với mục tiêu "ôm sát thức ăn" | Cách gán nhãn của UEC | Báo thêm AP@0.3, `area_ratio`, xem hình so sánh; nếu cần đo đúng "ôm sát" thì dùng dataset có mask thức ăn (UECFoodPix Complete, FoodSeg103) | Ghi nhận |
| L2 | v0 | Kaggle CLI trên Windows: Git Bash đổi `/kaggle/...` thành `C:\Program Files\Git\kaggle\...` khi truyền làm đối số | Chuyển đường dẫn tự động của MSYS | Đặt `MSYS_NO_PATHCONV=1` khi chạy script có đối số dạng `/kaggle/...` | Ghi nhận |

## Việc tiếp theo

1. ~~Chọn model cho 02~~ (GDINO conf 0.45, v2).
2. ~~v2.1: hạ conf 0.35 + lọc `food_other`~~ (đạt tiêu chí, đã thành mặc định).
3. Sửa tay bằng X-AnyLabeling: ưu tiên 6 ảnh failed + 16 ảnh có lỗi trong `review_v2.1.csv` (`image_ok = 0`).
4. Món nhiều miếng: thử gộp các box `food_other` nằm trong box món chính, hoặc prompt theo tên món ("a plate of sushi").
5. Khi có dataset thật: upload, đổi `dataset_sources`, chạy thử `per_class=5` để ước lượng thời gian.
6. Thử thêm (nếu cần): prompt khác ("food on the plate", tên món), SAM 3, ngưỡng theo từng món.
7. Nhờ một thành viên nhóm chấm lại ~30 ảnh ngẫu nhiên của v2.1 để kiểm tra kết quả chấm của Claude.
