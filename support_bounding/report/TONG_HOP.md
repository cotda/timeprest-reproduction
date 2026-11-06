# Tổng hợp: luồng tự động vẽ bounding box cho món ăn (FoodAI)

Ngày 2026-11-01. Chi tiết từng lần chạy, cấu hình và lỗi: `../NOTES.md` (phiên bản mới nhất: v2.1).

## 1. Mục tiêu
Từ ảnh món ăn đã chia thư mục theo tên món (`<tên_món>/*.jpg`), **tự động vẽ bounding box** ôm phần thức ăn, mỗi món một box,
xuất nhãn YOLO để sau này train model nhận diện món. Dataset thật của nhóm (~200 món châu Á + ~100 món Tây) chưa có, nên dùng
dataset công khai để xây và đánh giá luồng trước.

## 2. Dataset

| Dataset | Dùng để | Bản trên Kaggle | Tập con dùng |
|---|---|---|---|
| **UECFOOD-256** (Kawano & Yanai, 2014): 256 món, chủ yếu châu Á, **có box thật** | Đánh giá, so sánh model (có đáp án) | `rkuo2000/uecfood256` (4,2 GB) → tự cắt bản mini `kaggleolap/uecfood256-mini` (18 MB) | 10 món × 20 ảnh = 200 ảnh (cơm, sushi, bibimbap, pizza, ramen, súp miso, salad, khoai tây chiên, phở, chả giò), 58 ảnh có nhiều món |
| **Food-101** (Bossard et al., 2014): 101 món Tây, **không có box** | Thử tình huống giống dataset thật của nhóm (chỉ có ảnh + tên món) | `msarmi9/food101tiny` (96 MB) | 10 món × 20 ảnh = 200 ảnh (apple pie, bibimbap, cannoli, edamame, falafel, french toast, ice cream, ramen, sushi, tiramisu) |

Bản mini được cắt vì gắn dataset 4–6 GB vào kernel Kaggle rất chậm. Chọn ảnh ngẫu nhiên với seed cố định để chạy lại ra cùng kết quả.

## 3. Model dùng để bounding

**Model được chọn: Grounding DINO (bản base)**, chạy kiểu *zero-shot*: đưa vào ảnh + chữ "food", model trả về box + điểm tin cậy.
**Không cần train**: dùng trọng số có sẵn `IDEA-Research/grounding-dino-base` (license Apache-2.0).

Đã so sánh 3 model trên 200 ảnh UECFOOD-256 có box thật (GPU Tesla T4, Kaggle):

| Model | AP@0.5 | AP@0.3 | Recall | IoU TB | Thời gian/ảnh | VRAM |
|---|---|---|---|---|---|---|
| **Grounding DINO base** (chọn) | 0.732 | 0.836 | **0.717** | **0.611** | 0.58 s | 2.3 GB |
| YOLO-World v2-x | **0.735** | 0.834 | 0.683 | 0.596 | **0.06 s** | 1.4 GB |
| OWLv2 base | 0.386 | 0.475 | 0.503 | 0.427 | 0.38 s | 0.8 GB |
| (mỗi model + SAM 2.1 để siết box theo mask) | thấp hơn 0.02–0.03 | | | | +0.3 s | |

- Grounding DINO và YOLO-World ngang nhau. Chọn Grounding DINO vì recall và IoU cao hơn; YOLO-World nhanh gấp 10 lần, là
  phương án thay thế khi dataset lớn.
- SAM 2.1 không giúp nên tắt.
- Lưu ý: box thật của UECFOOD-256 ôm cả bát/đĩa, còn model ôm sát thức ăn (đúng mục tiêu), nên điểm ở IoU 0.5 thấp hơn thực tế.
- Khảo sát các model khác (Florence-2, SAM 3, YOLOE, autodistill, ...): `../survey.md`.

## 4. Luồng xử lý

1. **Dò dữ liệu:** tự tìm các thư mục dạng `<tên_món>/*.jpg` (mỗi bản upload có cấu trúc khác nhau), chọn tập con.
2. **Phát hiện:** Grounding DINO với prompt "food", chạy song song trên 2 GPU T4 (mỗi GPU một nửa số ảnh).
3. **Hậu xử lý:**
   - NMS (IoU 0.5): bỏ box trùng nhau.
   - Lọc box quá nhỏ (< 0,2% diện tích ảnh).
   - Ngưỡng tin cậy: box món chính ≥ **0.35**; box món khác (`food_other`) ≥ **0.45**.
4. **Gán class:** box lớn nhất = món theo tên thư mục; các box còn lại (món khác trong ảnh) = `food_other`.
5. **Xuất kết quả:** ảnh, nhãn YOLO (`class x_center y_center width height`, chuẩn hóa 0–1), ảnh đã vẽ box (mỗi class một màu),
   `classes.txt`, `data.yaml`, `train.txt`/`val.txt` (chia 80/20 theo món).
6. **Báo lỗi để sửa tay:** `failed.csv` (ảnh không có box), `report.csv` (số box, độ tin cậy, cờ cần xem lại: điểm thấp,
   quá nhiều box, box gần cả ảnh). Sửa bằng X-AnyLabeling (hướng dẫn trong `../README.md`).

Toàn bộ chạy trên Kaggle bằng một lệnh `python run_kaggle.py kaggle/02_auto_label` (đẩy kernel → chờ → tải kết quả về).
Mỗi lần chạy 200 ảnh mất ~3 phút GPU.

## 5. Kết quả gán nhãn trên Food-101 (200 ảnh, bản v2.1)

Food-101 không có box thật nên **chấm bằng mắt toàn bộ 194 ảnh có box** (người chấm: Claude; từng ảnh trong
`results/02_auto_label/review_v2.1/review_v2.1.csv`):

| Chỉ số | Kết quả |
|---|---|
| Ảnh có box | 194/200 = **97%** |
| Tổng số box đúng | 230/257 = **89.5%** |
| Box món chính đúng | 184/194 = **94.8%** (8 box thiếu một phần món, 2 box sai) |
| Box `food_other` đúng | 46/63 = 73% |

Tiêu chí đặt ra (≥ 90% ảnh có box, ≥ 85% box đúng): **đạt**.
Lỗi còn lại chủ yếu ở **món gồm nhiều miếng rời** (sushi, falafel, french toast): model lúc ôm cả đĩa, lúc ôm từng miếng.

## 6. Đã train / test chưa?

| Hạng mục | Trạng thái |
|---|---|
| Train model bounding box | **Chưa train.** Grounding DINO dùng trọng số có sẵn (zero-shot), không cần train. |
| Test / đánh giá model bounding | **Đã test** trên 200 ảnh UECFOOD-256 có box thật (mục 3), và chấm tay 194 ảnh Food-101 (mục 5). |
| Train model nhận diện món (YOLOv9/v10/v11) từ nhãn tự động | **Chưa train.** Đã có sẵn `data.yaml` + `train.txt`/`val.txt` (155 ảnh train / 39 ảnh val) để train ngay; mới chỉ chạy thử 1 epoch trên 20 ảnh để kiểm tra định dạng nhãn đọc được, chưa có kết quả độ chính xác. Đây là việc của bước sau (tháng 11–12 theo kế hoạch nhóm). |

## 7. Hình minh họa (30 cặp ảnh gốc | ảnh đã vẽ box)

`pairs_1.jpg`, `pairs_2.jpg`, `pairs_3.jpg` (mỗi hình 10 cặp; danh sách trong `pairs_list.csv`).
Ảnh được **chọn ngẫu nhiên** (3 ảnh mỗi món, seed 0) trong số ảnh có box của v2.1, không chọn lọc ảnh đẹp. Theo kết quả chấm tay,
28/30 cặp đúng, 2 cặp có lỗi (ghi chữ đỏ dưới ảnh): falafel/1104277 (box chính ôm nhầm hummus) và french_toast/1299998
(chỉ ôm một lát bánh).

Tạo lại hình: `python tools/make_report_pairs.py results/02_auto_label/output_v2.1 report --review results/02_auto_label/review_v2.1/review_v2.1.csv`

## 8. Hạn chế và việc tiếp theo
- Kết quả chấm tay mới do một người chấm (Claude) → nên nhờ thành viên nhóm chấm lại ~30 ảnh để đối chiếu.
- Món nhiều miếng: thử gộp box con nằm trong box món chính, hoặc prompt theo tên món.
- Khi có dataset thật: upload lên Kaggle, đổi tên dataset trong `kernel-metadata.json`, chạy thử 5 ảnh/món rồi chạy hết.
- Sau đó: sửa tay ảnh bị đánh dấu → train YOLO nhận diện món → ghép với phần tính dinh dưỡng.
