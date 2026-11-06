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
| v3 | 2026-11-01 | Kernel 03: thử 4 prompt ("food", tên món, thêm "plate. bowl. cup. tray.") trên UEC + Food-101 | "food" vẫn tốt nhất (0.732) | Giữ "food"; đề xuất dự phòng bằng tên món cho ảnh không có box (chờ người dùng) |
| v2.2 | 2026-11-01 | Mỗi box một màu; lưu kết quả detector theo đợt + chạy tiếp | — | Xong (số liệu v2.1 không đổi) |
| v5 | 2026-11-06 | Gộp các miếng cùng đĩa/thớt thành 1 khung (lượt 2 tìm vật chứa); kernel 04 kiểm tra | GDINO + gộp: ảnh 1 món recall 0.901→0.937, precision 0.744→0.821 | Bật mặc định trong module + notebook gán nhãn |
| v4.1 | 2026-11-06 | Notebook `kaggle/02_food2001` cho dataset thật của nhóm (300 món × 5 ảnh); sửa lỗi tên món có dấu trên Windows | — | Chạy thử trên máy 3 ảnh thật OK; chưa chạy trên Kaggle |
| v4 | 2026-11-06 | Đóng gói module `foodbox`: nhập 1 ảnh → ra box (ảnh có khung, ảnh cắt, JSON); lệnh `foodbox-detect`; zip | — | Xong, chạy thử thật trên CPU 3 ảnh |
| v2.3 | 2026-11-05 | Cắt từng box thành ảnh riêng (từ ảnh gốc, không khung) | — | Xong: 257 ảnh cắt (Food-101 v2.1), 192 ảnh cắt (58 ảnh nhiều món UEC) |
| v1.2 | 2026-11-05 | Tách kết quả 01 theo ảnh 1 món / nhiều món (UEC); hình so sánh 58 ảnh nhiều món | GDINO: 1 món 0.875, nhiều món 0.571 | Ảnh nhiều món yếu: món chính đúng 22/58, món phụ 40/99 (ghép 1-1) |
| v2.1 | 2026-11-01 | Dựng lại 02: conf 0.35 + `other_conf` 0.45; chấm tay toàn bộ 194 ảnh có box | (GDINO) | **Đạt tiêu chí**: 97% ảnh có box, 89.5% box đúng (box chính 94.8%) |
| v2 | 2026-11-01 | Kernel 02: gán nhãn 200 ảnh Food-101 tiny bằng GDINO conf 0.45 | (GDINO, chọn bởi người dùng) | Xong; 78.5% ảnh có box, conf 0.45 quá chặt cho món tráng miệng |
| v1.1 | 2026-11-01 | Bỏ 2 bộ lọc làm hỏng kết quả (max_area 0.98, drop_container) | YOLO-World 0.735 ≈ GDINO 0.732 | Chờ người dùng chọn model cho 02 |
| v1 | 2026-11-01 | Kernel 01: so sánh 3 model ± SAM trên 200 ảnh UEC | YOLO-World 0.647, GDINO 0.618 | Xong, có lỗi hậu xử lý (sửa ở v1.1) |
| v0 | 2026-11-01 | Chọn dataset, tạo UEC mini, viết thư viện + test, chạy thử CPU | — | Xong |

---

## Các phiên bản

### v3 — Kernel `kaggleolap/foodbox-03-prompts`, version 1 (2026-11-01): thử text prompt khác

**Mục đích:** yêu cầu 3 của đề ("box ôm cả đĩa → thử text prompt khác"). Bốn biến thể, cùng Grounding DINO base:
`_food` ("food."), `_food_pb` ("food. plate. bowl. cup. tray.", bỏ box mà điểm token đĩa/bát/cốc/khay cao hơn điểm "food"),
`_class` (tên món theo thư mục, vd. "french toast."), `_class_pb` (tên món + "plate. bowl. cup. tray.").
Cài đặt: `GroundingDINO.predict(img, prompts=, extras=)` tính điểm box = điểm cao nhất trên token của prompt chính
(`foodbox/detectors.py`); `run_detect --prompt-sets`. Kiểm chứng: `_food` cho kết quả trùng đúng v1.1 (AP@0.5 0.732, cùng conf 0.468).

**Chạy:** GPU 0 = 200 ảnh UEC mini, GPU 1 = đúng 200 ảnh Food-101 tiny của bước 02; detect 9,8 phút, ~0,63 s/ảnh/biến thể;
output 8,6 MB (`results/03_prompts/output/`: `prompts_uec.csv`, `prompts_food.csv`, `compare_uec/`, `compare_food/`, `preds_*/`).

**UEC** (P/R/F1 ở ngưỡng F1 tốt nhất của từng biến thể):

| biến thể | AP@0.5 | AP@0.3 | P | R | F1 | IoU TB | area_ratio | conf | ảnh không box |
|---|---|---|---|---|---|---|---|---|---|
| `_food` | **0.732** | **0.836** | 0.676 | **0.717** | **0.696** | **0.611** | 0.815 | 0.468 | **17** |
| `_food_pb` | 0.632 | 0.760 | 0.654 | 0.667 | 0.660 | 0.570 | 0.798 | 0.486 | 23 |
| `_class` | 0.590 | 0.663 | 0.736 | 0.530 | 0.616 | 0.482 | 0.934 | 0.358 | 28 |
| `_class_pb` | 0.412 | 0.512 | 0.549 | 0.390 | 0.456 | 0.368 | 0.888 | 0.374 | 52 |

**Food-101** (200 ảnh; `main_area` = diện tích box món chính / ảnh, nhỏ hơn = ôm sát hơn; `near_full` = box chính > 90% ảnh):

| biến thể | ngưỡng | ảnh có box | box/ảnh | main_area | near_full |
|---|---|---|---|---|---|
| `_food` | 0.35 | **194** | 1.80 | 0.641 | 39 |
| `_food_pb` | 0.35 | 195 | 1.94 | 0.620 | 39 |
| `_class` | 0.35 | 172 | 1.25 | 0.515 | 25 |
| `_class_pb` | 0.35 | 105 | 0.94 | 0.388 | 4 |

**Xem hình** (`compare_uec/`, `compare_food/`):
- `_food_pb` (thêm "plate. bowl."): box gần như y hệt `_food`, không ôm sát hơn (main_area 0.62 so với 0.64), nhưng **mất các món
  đựng trong bát** (súp miso, edamame, cơm): model gán cả bát + thức ăn cho từ "bowl" → bị bỏ. Có hại.
- `_class` (tên món): chỉ tìm món theo tên thư mục, **bỏ qua món khác trong khay** (recall UEC 0.53); với tên ít phổ biến
  (phở, bibimbap ảnh cận) điểm thấp; french toast bị tách từng lát. Nhưng với món tráng miệng mà "food" yếu thì lại tốt
  (tiramisu 0.73 khi "food" < 0.47). Box nhỏ hơn (`main_area` 0.52) một phần vì hay ôm từng miếng.
- `_class_pb`: mất quá nhiều (105/200 ảnh có box). Loại.
- Box của `_food` vốn đã ôm phần thức ăn chứ không ôm cả đĩa (đúng như 01: area_ratio 0.8 so với box thật UEC ôm cả bát),
  nên vấn đề "ôm cả đĩa" chủ yếu nằm ở box thật UEC, không phải ở prompt.

**Phương án kết hợp (tính trên máy từ `preds_*`):** dùng "food"; ảnh nào không có box ≥ 0.35 thì lấy kết quả prompt tên món.
Food-101: cứu 5/6 ảnh failed của v2.1 ở ngưỡng 0.35 (6/6 ở 0.30): cannoli/1417238, tiramisu/1039481, /1060127, /1380802,
/1644287 (+ ice_cream/155035 ở 0.32); xem hình: **cả 6 box đúng**. UEC: không đổi gì (không ảnh nào trống ở 0.35).
→ Nếu áp dụng: 199/200 ảnh có box. Chi phí: chạy thêm prompt tên món chỉ cho ảnh không có box (~0,6 s/ảnh).

**Kết luận v3:** giữ prompt **"food"** làm mặc định. Không dùng "plate/bowl" (hại món đựng trong bát).
Đề xuất v3.1: thêm bước dự phòng bằng tên món cho ảnh không có box. **Chờ người dùng đồng ý** (cần sửa notebook 02 và chạy lại trên Kaggle).

---

### v5 — Gộp các miếng cùng đĩa/thớt thành 1 khung (2026-11-06)

**Yêu cầu người dùng** (gửi 2 ảnh bruschetta 10 và 13 khung): một đĩa/thớt nhiều miếng của cùng món = 1 món → chỉ cần 1 khung
to; khung nhỏ chỉ dành cho món khác nhau ở đĩa khác nhau. Người dùng chọn: khung gộp ôm sát phần thức ăn (hợp các khung nhỏ,
không ôm cả đĩa); cùng đĩa nhưng khác thứ (bít tết + khoai) cũng gộp 1 khung; làm luôn, có kiểm tra trên Kaggle.

**Thử trước (CPU, 5 ảnh bruschetta của dataset 2001):** hỏi thêm "plate. bowl. tray. cutting board." → model khoanh đúng
thớt/đĩa ở cả 5 ảnh (điểm 0.45–0.67); ly nước, chén muối cũng được khoanh nhưng không chứa miếng nào nên không ảnh hưởng.

**Cài đặt:** `postprocess.group_dishes` (1) box to bao ≥ 2 box con (≥ 80% diện tích) → giữ box to; (2) mỗi box thuộc vật chứa
nhỏ nhất bao nó (≥ 70%), vật chứa điểm ≥ 0.30 và ≥ 1% ảnh; box cùng vật chứa → hợp, điểm = max; vật chứa bao ≥ 2 vật chứa
khác (khay) không dùng để gộp; (3) NMS lại. Gộp làm TRƯỚC luật box phụ ≥ 0.45 (miếng điểm 0.35–0.45 được gộp vào món).
`detectors.CONTAINER_PROMPTS`, `run_detect --containers` (lưu c_boxes/c_scores), `build_dataset(group=)`, `FoodBoxDetector(group=True)`
mặc định, `foodbox-detect --no-group`; notebook 02 / 02_food2001 `group=True`. Chậm gấp ~2 (2 lượt model).
Test: 3 test luật gộp + 1 test module (27 test PASS).

**Kiểm tra: kernel `kaggleolap/foodbox-04-group` version 1** (2×T4, 7,2 phút; 200 ảnh UEC + 200 ảnh Food-101 như bước 01/02,
mỗi ảnh 2 lượt). Đánh giá trên máy: `python tools/eval_group.py results/04_group/output`
(cùng dự đoán, chỉ đổi hậu xử lý; box ≥ 0.35, box phụ ≥ 0.45; ghép 1-1 IoU ≥ 0.5 với box thật UEC):

| nhóm ảnh UEC | phương án | recall | precision | box/ảnh | đúng số món | ảnh có 1 khung bao ≥ 2 món thật |
|---|---|---|---|---|---|---|
| 1 món (142) | không gộp (v2.1) | 0.901 | 0.744 | 1.21 | 85.9% | 0 |
| 1 món (142) | **gộp** | **0.937** | **0.821** | 1.14 | **89.4%** | 0 |
| 1 món (142) | chỉ theo vật chứa | 0.937 | 0.816 | 1.15 | 88.7% | 0 |
| 1 món (142) | chỉ luật box to | 0.901 | 0.753 | 1.20 | 87.3% | 0 |
| nhiều món (58) | không gộp | 0.677 | 0.554 | 3.33 | 24.1% | 10 |
| nhiều món (58) | **gộp** | 0.652 | 0.566 | 3.14 | 24.1% | 17 |

Xem hình 17 ảnh UEC có số khung thay đổi (`results/04_group/output/group_uec_1.jpg`, `_2.jpg`): các lần "1 khung bao ≥ 2 món
thật" đều là các món trên **cùng 1 đĩa** (salad + cốt lết, đĩa hamburg kèm bắp + khoai, 2 con cá, nhiều miếng sushi/chả giò) =
gộp đúng theo quy tắc người dùng (box thật UEC chấm riêng từng món trên cùng đĩa, nên recall nhiều món giảm 0.025). Không thấy
ca nào gộp 2 bát/đĩa khác nhau; cơm, súp miso, chén nước chấm vẫn tách. Lưng chừng: fried_spring_rolls/266882 (chén nước chấm
đặt trên cùng mâm chả giò → gộp); sushi/699 chỉ gộp được một phần (1 miếng nằm ngoài khung đĩa).
Food-101: 5/200 ảnh thay đổi, cả 5 đúng các lỗi đã chấm tay ở v2.1: falafel #88, #94, #97 (từng viên / hummus), sushi #166, #170
(khung chỉ 1–2 miếng) → nay 1 khung cả đĩa. Ảnh ≥ 3 khung: 14 → 10.

**Dataset 2001 (CPU, module `foodbox-detect` bản gộp):** bruschetta 001–005: 2, 1, 1, 1, 1 khung (trước: 10–13 khung; khung
thứ 2 ở 001 là chén muối); bánh_cannoli 001–005: 1, 2, 1, 1, 3 khung (003 trước 11 khung; 005: 2 đĩa riêng + 1 miếng ở mép ảnh).
Tốc độ CPU bản gộp 27–42 s/ảnh. (So sánh đủ 30 ảnh có/không gộp: đang chạy.)

**Đóng gói:** module lên **0.2.0** (gộp theo đĩa bật mặc định; 0.1.0 = mỗi miếng 1 khung, nay là `--no-group`).
`dist/foodbox_module_v0.2.0.zip` (34 KB, 15 file; bản 0.1.0 trong dist/ đã thay). Kiểm tra: giải nén ra thư mục mới → venv →
`pip install -e .` → 4 test PASS → `foodbox-detect` bruschetta/002 (trước 13 khung) → 1 khung, CPU 64 s/ảnh (CPU đang chạy song
song việc khác), nạp model 35 s.

---

### v4.1 — Notebook cho dataset thật 2001_food_dataset + sửa lỗi tiếng Việt (2026-11-06, chạy trên máy)

**Dataset của nhóm:** `D:/University/Research/2001_food_dataset/2001_food_dataset` (469 MB): `images/<tên_món>/001..005.jpg`
(300 món × 5 = 1.500 ảnh; 1.490 jpg, 6 png, 4 jpeg) + `food_dataset.csv` (food_name, food_img). Tên món có dấu tiếng Việt.

**Notebook `kaggle/02_food2001/`** (người dùng hỏi cách chạy trên Kaggle; chưa upload dataset, chưa chạy trên Kaggle):
chạy hết 1.500 ảnh (Grounding DINO, tham số v2.1), xuất bộ nhãn YOLO + ảnh cắt (`yolo/`), thư mục theo ảnh
`theo_anh/<món>/001.jpg, 001_bbox.jpg, 001_1.jpg...` (giống đầu ra foodbox-detect), nén thành 1 file `ketqua_2001.zip`
(ZIP_STORED) rồi xoá thư mục đã nén (output tải về 1 file thay vì ~10.000 file). Không cần pip install trên Kaggle.
Hướng dẫn từng bước (web Kaggle / dòng lệnh): `kaggle/02_food2001/HUONG_DAN.md`. `foodbox/crop.write_image_folder`
(dùng chung với `tools/make_crop_folders.py`, kiểm tra cho ra file giống hệt bản cũ trên 58 thư mục UEC).

**Lỗi tìm được khi chạy thử với dataset thật trên Windows (E8):** `build_dataset` ghi `train.txt` không chỉ định mã hoá →
`UnicodeEncodeError` với tên món có dấu (cp1252). Trên Kaggle (Linux, UTF-8) không lộ ra. Sửa: mọi `open()` trong `foodbox/`
dùng `encoding="utf-8"`; tiến trình con `run_detect` chạy với `PYTHONIOENCODING=utf-8`; notebook đọc log bằng UTF-8.
Thêm test tên món có dấu (`test_build_dataset_vietnamese_class_names`). 23 test PASS. Module zip đóng gói lại.

**Chạy thử notebook trên máy** (CPU, dataset thật, 3 món × 1 ảnh): 3/3 ảnh có box (1, 1, 11 box), 13 ảnh cắt, zip 24 MB, 1,5 phút.
Ảnh `bánh_cannoli/003` có 11 box: xem ảnh, đúng 11 chiếc cannoli, mỗi chiếc 1 box (bị gắn `many_boxes` để xem lại; trong nhãn
YOLO 1 chiếc là `bánh_cannoli`, 10 chiếc là `food_other` theo quy tắc box lớn nhất).
Ước tính trên Kaggle 2×T4: ~0,6 s/ảnh/GPU → 1.500 ảnh ~8 phút detect; cộng nạp model, xuất file, nén → ~15 phút.

---

### v4 — Đóng gói module: nhập 1 ảnh → ra bounding box (2026-11-06)

**Yêu cầu (anh hướng dẫn):** "đóng gói thành module, nhập một bức ảnh, ra bounding" + báo cáo, up code lên task.
Người dùng chọn: up code bằng GitHub + zip; chạy trên máy cá nhân bằng Python; đầu ra = ảnh có khung + ảnh cắt + JSON.

**Cài đặt:**
- `foodbox/api.py`: `FoodBoxDetector(device="auto")`, `.detect(ảnh)` → list box {id, x1, y1, x2, y2, score, largest}
  (sắp theo điểm, id từ 1), `.detect_and_save(ảnh, thư_mục)` → `<tên>_bbox.jpg` (khung đánh số), `<tên>_<i>.jpg` (cắt đúng
  box i, từ ảnh gốc), `<tên>.json`. Tham số = pipeline v2.1 (prompt "food", 0.35 / 0.45, NMS 0.5, min_area 0.002, tối đa 20 box).
  Không đặt tên món (1 ảnh lẻ không có tên thư mục).
- `foodbox/cli.py` + `pyproject.toml`: `pip install -e .` → lệnh `foodbox-detect <ảnh|thư mục>... --out ...`
  (`--device`, `--conf`, `--other-conf`, `--prompt`, `--no-crops`, `--no-vis`, `--pad`); ảnh lỗi thì báo rồi chạy tiếp.
  Phụ thuộc chỉ torch, transformers ≥ 4.40, pillow, numpy (ultralytics/pandas là tuỳ chọn `[compare]`).
- `import foodbox` không nạp torch (FoodBoxDetector nạp khi cần).
- `MODULE_README.md` (hướng dẫn cho người dùng module), `tools/make_module_zip.py` → `dist/foodbox_module_v0.1.0.zip`
  (31 KB, 15 file: foodbox/, pyproject.toml, MODULE_README.md, tests/test_api.py).
- `tests/test_api.py`: 3 test với model giả (luật lọc box, file ghi ra, CLI với thư mục con). Tổng 22 test PASS.

**Chạy thử thật** (venv riêng, `pip install -e .`, CPU laptop, transformers 4.49): nạp model 30 s; 3 ảnh
(bibimbap_13995, miso_soup_38 của UEC; ice_cream_1609293 của Food-101): 4 / 6 / 5 box, 9–15 s/ảnh. So với kết quả pipeline
trên GPU Kaggle (transformers 5.16): cùng số box; bibimbap và ice cream trùng hết (IoU ≥ 0.976, lệch ≤ 2,6 px);
miso_soup_38: 5/6 box trùng, box thứ 6 nằm sát ngưỡng 0.45: GPU giữ bát súp miso (0.4506), CPU giữ gói gia vị (0.4512)
(điểm lệch ~0.01–0.03 giữa CPU/GPU và giữa 2 phiên bản transformers). Ghi chú này đã đưa vào MODULE_README.
Zip đã kiểm tra: giải nén ra thư mục mới → `pip install -e .` → 3 test PASS.

---

### v2.3 — Cắt box thành ảnh riêng (2026-11-05, chỉ chạy trên máy)

**Yêu cầu người dùng:** cắt vùng trong khung bounding box thành các file ảnh riêng, không có khung.

**Cài đặt:**
- `foodbox/crop.py`: `crop_box` (cắt từ ảnh gốc đã xoay EXIF, nới `pad` x cạnh box mỗi phía, kẹp trong ảnh), `save_crops`
  (bỏ ảnh cắt có cạnh ngắn < 16 px), `crops.csv` (ảnh cắt, ảnh nguồn, thứ tự box k, class, món chính hay không, điểm, toạ độ).
- Bước 02: `build_dataset(..., crops=True, crop_pad=0.05)`; CFG notebook 02 mặc định `crops=True`, `crop_pad=0.05`
  → `output/crops/<món>/<ảnh>_<k>.jpg` (k = 0 là box món chính khi box được sắp theo điểm; xem cột `is_main`).
- `tools/crop_from_labels.py`: cắt theo nhãn YOLO có sẵn (dùng sau khi sửa tay bằng X-AnyLabeling).
- `tools/make_multi_dish_pairs.py --crops-dir`: cắt box pipeline của 58 ảnh nhiều món UEC.
- `tools/crops_preview.py`: hình xem trước "ảnh gốc → các ảnh cắt" (8 ảnh gốc ngẫu nhiên, seed 0).
- 2 test mới (cắt có nới + kẹp mép; build_dataset cắt đúng và cắt lại từ nhãn ra cùng kết quả): 18 test PASS.

**Kết quả:**
- Food-101 (`results/02_auto_label/output_v2.1/crops/`): 194 ảnh → 257 ảnh cắt (194 món chính + 63 `food_other`), 13 MB.
- 58 ảnh nhiều món UEC (`report/multi_dish_crops/`): 192 ảnh cắt (1 box < 16 px bị bỏ), 3,5 MB; `preview.jpg`.
- Xem `preview.jpg`: mỗi ảnh cắt là một món, không có khung. Chất lượng ảnh cắt phụ thuộc box: box sai/thiếu (xem v1.2, v2.1)
  thì ảnh cắt cũng sai/thiếu (vd. french_fries/15283: khoai tây chiên bị bỏ sót nên không có ảnh cắt). Ảnh UEC độ phân giải thấp
  nên ảnh cắt nhỏ (vài trăm pixel).
- Thư mục xem trước theo yêu cầu người dùng (`tools/make_crop_folders.py`): `report/multi_dish_preview/<món>_<ảnh>/` gồm ảnh gốc
  `<món>_<ảnh>.jpg` (sạch) + `<món>_<ảnh>_bbox.jpg` (khung đánh số 1..n, khung i = vùng của file `_i`) + ảnh cắt
  `<món>_<ảnh>_1.jpg`, `_2`, ... (58 thư mục, 192 ảnh cắt; `index.csv`). Ảnh cắt lấy từ ảnh gốc theo toạ độ box, không
  cắt từ ảnh đã vẽ khung.
  Số ảnh cắt/thư mục: 1 → 11 thư mục (pipeline gộp cả khay/đĩa thành 1 box hoặc bỏ sót món), 2 → 10, 3 → 9, 4 → 15, 5 → 8, 6–8 → 5.
- **Sửa (cùng ngày, theo người dùng: "phải cắt theo bounding mà nó khoanh được"):** bản đầu nới 5% mỗi phía
  (`pad=0.05`) nên ảnh cắt lớn hơn box và ảnh `_bbox.jpg` vẽ vùng đã nới. Nay mặc định `pad=0`: cắt đúng box model khoanh;
  `crops.csv` thêm cột `box_x1..box_y2` (box model) bên cạnh `x1..y2` (vùng cắt); `_bbox.jpg` vẽ đúng box model.
  Kiểm tra 449 ảnh cắt: vùng cắt = box làm tròn (lệch ≤ 0,5 px), trừ 22 chỗ box vượt mép ảnh tối đa 1,7 px (bị kẹp ở mép).
  Đã cắt lại toàn bộ (`output_v2.1/crops/`, `report/multi_dish_crops/`, `report/multi_dish_preview/`). 19 test PASS.
- Chưa chạy lại kernel 02 trên Kaggle với `crops=True` (code giống hệt bản đã chạy trên máy; sẽ chạy khi có dataset thật).

---

### v1.2 — Tách kết quả bước 01 theo ảnh 1 món / nhiều món (2026-11-05, chỉ chạy trên máy)

**Câu hỏi của người dùng:** đã test ảnh nhiều món chưa? Có: 58/200 ảnh UEC mini có ≥ 2 món (27 ảnh 2 món, 20 ảnh 3 món,
11 ảnh 4 món; 158 box thật), nằm sẵn trong tập đánh giá của v1/v1.1 nhưng trước đây chỉ báo số chung.
Tính lại từ `results/01_compare/output/preds/` với hậu xử lý v1.1, ở ngưỡng F1 tốt nhất trên cả 200 ảnh (GDINO 0.47, YOLO-World 0.22).
"Món phụ" = box thật của món khác món theo thư mục (sẽ thành `food_other`).

| model | nhóm ảnh | AP@0.5 | AP@0.3 | P | R | IoU TB | chọn đúng món chính (box lớn nhất) | tìm thấy món phụ (IoU ≥ 0.5) |
|---|---|---|---|---|---|---|---|---|
| GDINO | 1 món (142) | 0.875 | 0.941 | 0.768 | 0.817 | 0.694 | 80% | — |
| GDINO | nhiều món (58) | 0.571 | 0.712 | 0.593 | 0.627 | 0.537 | **47%** | 61/99 = 62% |
| YOLO-World | 1 món (142) | 0.858 | 0.924 | 0.685 | 0.859 | 0.736 | 85% | — |
| YOLO-World | nhiều món (58) | 0.629 | 0.767 | 0.659 | 0.525 | 0.471 | 40% | 57/99 = 58% |

**Sửa số liệu "tìm thấy món phụ" (cùng ngày, khi làm hình so sánh):** cột "tìm thấy món phụ" ở bảng trên đếm DƯ: một box
dự đoán bất kỳ (kể cả box đã bị gán làm món chính) trùng box thật món phụ là tính "tìm thấy", nên một box được tính cho hai việc
(vd. french_fries/15414: box ôm bát cơm bị gán nhầm là french_fries vẫn được tính là tìm thấy món phụ "rice").
Chấm lại đúng theo đầu ra pipeline (`tools/make_multi_dish_pairs.py`: món chính ≥ 0.35, `food_other` ≥ 0.45, box lớn nhất = món
chính; món phụ ghép 1-1 chỉ với box `food_other`, IoU ≥ 0.5) trên 58 ảnh nhiều món, GDINO:
- món chính đúng **22/58 (38%)**; món phụ tìm đúng **40/99 (40%)**; ảnh đúng hoàn toàn (món chính đúng + đủ món phụ) **7/58 (12%)**.
- Hình: `report/multi_dish/pages_01..10.jpg` (6 ảnh/trang: ảnh gốc | pipeline | box thật UEC) + `report/multi_dish/images/` (từng ảnh)
  + `multi_dish_list.csv` (từng ảnh: số món, món chính đúng/sai, món phụ tìm thấy, box food_other thừa/lệch, tên các món thật).
  Bản gửi thầy (không dòng chấm điểm, không chữ trên box, chỉ khung box): `report/multi_dish_clean/` (`--no-notes --no-box-labels`).
- **Chỉ xét vẽ box, không xét tên món** (người dùng chỉ cần bounding, 2026-11-05; cùng tham số pipeline, IoU ≥ 0.5 với box thật):
  ảnh 1 món: tìm được 128/142 món (90%), box đúng 128/172 (74%); ảnh nhiều món: tìm được 107/158 món (68%), box đúng 107/193 (55%).
  "Box đúng" bị ước lượng thấp vì box thật UEC thiếu các món ngoài 256 món và box thật ôm cả bát/đĩa.
- Lưu ý khi đọc hình: box thật UEC **không đầy đủ** (chỉ gán nhãn món thuộc 256 món của UEC, vd. bánh socola trong khay bibimbap/13995,
  gà viên trong french_fries/15283 không có nhãn) và đôi khi sai (french_fries/11365 có box "samul" cả ảnh; french_fries/12952 có box
  "twice cooked pork" chồng lên khoai tây). Vì vậy một phần "box thừa" là thức ăn thật và số liệu nhiều món là ước lượng thấp.

**Kết luận v1.2:** ảnh nhiều món (khay cơm) yếu hơn rõ: AP@0.5 giảm ~0.3, chỉ tìm được ~60% món phụ. Điểm yếu lớn nhất là
**chọn món chính**: quy tắc "box lớn nhất = món theo thư mục" chỉ đúng 47% (trong khay, bát cơm/đĩa phụ thường to hơn món chính).
Food-101 (bước 02) chủ yếu là ảnh 1 món nên chưa phản ánh tình huống khay cơm; ảnh mẫu người dùng gửi (khay 4 món) giống nhóm "nhiều món" của UEC.
Hướng cải thiện (chưa làm): chọn món chính bằng prompt tên món (v3 cho thấy prompt tên món tìm đúng món theo thư mục,
bỏ qua món khác); nếu dataset thật có nhiều ảnh khay thì cần đánh giá riêng trên ảnh khay.

---

### v2.2 — Mỗi box một màu + lưu theo đợt (2026-11-01, sửa code, không chạy GPU riêng)

- **Mỗi box một màu** (đề bài: "mỗi box một màu"): `autolabel.box_colors`: trong một ảnh box món chính màu đầu, các box
  `food_other` các màu tiếp theo, không trùng. Trước đây mỗi *class* một màu nên các box `food_other` trùng màu.
  Đã dựng lại `output_v2.1/visualized/` và `report/pairs_*.jpg` bằng màu mới (số liệu v2.1 không đổi: 194/200, 257 box).
- **Lưu theo đợt (checkpoint):** `run_detect` ghi `partial_<biến thể>_shard<i>.json` mỗi `--ckpt-every` ảnh (mặc định 50,
  ghi file tạm rồi đổi tên); chạy lại cùng lệnh thì bỏ qua ảnh đã xong; `load_preds` đọc được phần đã lưu khi kernel bị dừng.
  Có test giả lập bị dừng giữa chừng rồi chạy tiếp. Kernel 03 (v3) đã chạy với bản này.
- 16 test CPU PASS.

---

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

**Báo cáo (2026-11-01):** `report/TONG_HOP.md` + `report/pairs_{1,2,3}.jpg` (30 cặp ảnh gốc | đã vẽ box, chọn ngẫu nhiên 3 ảnh/món, seed 0, từ `output_v2.1`; 28/30 đúng theo chấm tay). Tạo bằng `tools/make_report_pairs.py`.

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
| E8 | v4.1 | `UnicodeEncodeError` khi ghi train.txt với tên món có dấu (Windows) | `open()` không chỉ định encoding → cp1252 | Mọi file I/O trong foodbox dùng UTF-8; subprocess PYTHONIOENCODING=utf-8; test mới | Đã sửa |
| L1 | v1 | Box thật UEC ôm cả bát/đĩa, ngược với mục tiêu "ôm sát thức ăn" | Cách gán nhãn của UEC | Báo thêm AP@0.3, `area_ratio`, xem hình so sánh; nếu cần đo đúng "ôm sát" thì dùng dataset có mask thức ăn (UECFoodPix Complete, FoodSeg103) | Ghi nhận |
| L2 | v0 | Kaggle CLI trên Windows: Git Bash đổi `/kaggle/...` thành `C:\Program Files\Git\kaggle\...` khi truyền làm đối số | Chuyển đường dẫn tự động của MSYS | Đặt `MSYS_NO_PATHCONV=1` khi chạy script có đối số dạng `/kaggle/...` | Ghi nhận |

## Việc tiếp theo

0. Người dùng quyết định v3.1 (dự phòng bằng tên món cho ảnh không có box).

1. ~~Chọn model cho 02~~ (GDINO conf 0.45, v2).
2. ~~v2.1: hạ conf 0.35 + lọc `food_other`~~ (đạt tiêu chí, đã thành mặc định).
3. Sửa tay bằng X-AnyLabeling: ưu tiên 6 ảnh failed + 16 ảnh có lỗi trong `review_v2.1.csv` (`image_ok = 0`).
4. Món nhiều miếng: thử gộp các box `food_other` nằm trong box món chính, hoặc prompt theo tên món ("a plate of sushi").
5. Khi có dataset thật: upload, đổi `dataset_sources`, chạy thử `per_class=5` để ước lượng thời gian.
6. Thử thêm (nếu cần): prompt khác ("food on the plate", tên món), SAM 3, ngưỡng theo từng món.
7. Nhờ một thành viên nhóm chấm lại ~30 ảnh ngẫu nhiên của v2.1 để kiểm tra kết quả chấm của Claude.
