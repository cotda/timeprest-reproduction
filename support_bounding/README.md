# FoodAI: luồng tự vẽ bounding box cho món ăn (chạy trên Kaggle)

Input: ảnh chia theo thư mục tên món (`<tên_món>/*.jpg`). Output: box ôm phần thức ăn, mỗi món một box và một màu,
nhãn YOLO train được ngay. Box không phải món chính (ảnh nhiều món) có class `food_other`.

## Dùng như module: nhập 1 ảnh → ra bounding box

```bash
pip install -e .                               # trong support_bounding/
foodbox-detect khay_com.jpg --out ketqua/      # -> khay_com_bbox.jpg, khay_com_1.jpg..., khay_com.json
```

Hoặc trong Python: `from foodbox import FoodBoxDetector; FoodBoxDetector().detect("khay_com.jpg")`.
Chi tiết: `MODULE_README.md`. Đóng gói zip để gửi: `python tools/make_module_zip.py` → `dist/foodbox_module_v0.2.0.zip`.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `NOTES.md` | **Nhật ký phiên bản**: mỗi lần chạy, cấu hình, kết quả, lỗi và cách sửa (đọc trước khi đối chiếu) |
| `survey.md` | Khảo sát model (Grounding DINO, OWLv2, YOLO-World, Florence-2, SAM 2/3, autodistill, ...) |
| `foodbox/` | Thư viện dùng chung: dò dữ liệu, chuyển nhãn UEC sang YOLO, model, hậu xử lý, đo độ chính xác, xuất bộ nhãn |
| `kaggle/01_compare/` | So sánh model trên UECFOOD-256 (có box thật) → `results.csv` + hình so sánh |
| `kaggle/02_auto_label/` | Tự gán nhãn tập ảnh (Food-101 tiny; sau này là dataset của nhóm) |
| `kaggle/03_prompts/` | Thử text prompt khác cho Grounding DINO ("food", tên món, thêm "plate. bowl. cup. tray.") |
| `kaggle/*/notebook.py` | **Nguồn** của notebook (sửa ở đây). File `.ipynb` được ghép tự động, có nhúng code `foodbox/` |
| `run_kaggle.py` | Một lệnh: build → đẩy lên Kaggle (2×T4) → chờ → tải output + log về `results/<tên>/` |
| `tools/make_uec_mini.py` | Cắt bản UECFOOD-256 mini (10 món × 20 ảnh + toàn bộ `bb_info.txt`) để kernel gắn nhanh |
| `tools/reeval_01.py` | Tính lại bảng 01 từ dự đoán đã tải về với tham số hậu xử lý khác (không cần GPU) |
| `tests/` | Test CPU: `python -m pytest` |

Dataset trên Kaggle: `kaggleolap/uecfood256-mini` (18 MB, riêng tư, cắt từ `rkuo2000/uecfood256`) và `msarmi9/food101tiny` (96 MB).

## Cách chạy

Cần Kaggle API đã cấu hình (`kaggle kernels list --mine` chạy được). Trong thư mục `support_bounding/`:

```bash
python run_kaggle.py kaggle/01_compare          # so sánh model (~20-30 phút GPU)
python run_kaggle.py kaggle/02_auto_label       # gán nhãn
python run_kaggle.py kaggle/02_auto_label --no-push   # chỉ chờ + tải kết quả lần đẩy trước
python run_kaggle.py kaggle/02_auto_label --download-only   # chỉ tải lại output (mọi trang)
```

`run_kaggle.py` tự tải đủ mọi trang output (lệnh `kaggle kernels output` chỉ tải 500 file đầu).
Kaggle API không trả log khi kernel đang chạy; xem log trực tiếp trên trang web của kernel
(`https://www.kaggle.com/code/<user>/<slug>`). API cũng không hủy được kernel: muốn hủy thì vào trang web.
Mỗi bước chạy model có giới hạn thời gian (`timeout_min`), nên kernel không treo vô hạn.
Kết quả detector được **lưu theo đợt** (mặc định mỗi 50 ảnh, `--ckpt-every`) vào `partial_*.json`: kernel bị dừng giữa
chừng thì phần đã chạy vẫn đọc được, chạy lại cùng lệnh thì chỉ làm tiếp ảnh còn thiếu.

Đẩy lên tài khoản khác: `--user <tên>`. Khi đó dataset riêng tư `kaggleolap/uecfood256-mini` không dùng được, phải tạo
lại bằng `tools/make_uec_mini.py` rồi `kaggle datasets create`.

## Tham số chính (ô `CFG` trong `kaggle/02_auto_label/notebook.py`)

| Tham số | Ý nghĩa |
|---|---|
| `input_dirs` | `None` = tự dò mọi thư mục dạng `<món>/*.jpg` dưới `/kaggle/input`; hoặc list đường dẫn |
| `classes`, `n_classes`, `per_class` | Chọn món / số món / số ảnh mỗi món (`None` = tất cả) |
| `model` | `gdino`, `owlv2` hoặc `yoloworld` |
| `prompts` | Câu chữ mô tả thứ cần tìm, vd. `["food"]` |
| `conf` | Ngưỡng confidence giữ box (lấy từ cột `conf` của bước 01) |
| `sam` | `True` = siết box theo mask SAM 2.1 (box chỉ co lại để bỏ phần đĩa) |
| `nms_iou`, `min_area`, `max_area`, `drop_container` | Bỏ box trùng / quá nhỏ / gần cả ảnh / box ôm cả khay chứa nhiều món |
| `main_rule` | Box nào là món chính: `largest` (lớn nhất) hoặc `score` (điểm cao nhất) |
| `review_conf`, `max_boxes` | Ngưỡng đánh dấu "cần xem lại" trong `report.csv` |

Khi có dataset thật: đổi `dataset_sources` trong `kaggle/02_auto_label/kernel-metadata.json` thành dataset của nhóm
(upload bằng `kaggle datasets create -p <thư_mục> --dir-mode zip`), đặt `n_classes=None`, `per_class=None`, rồi chạy lại.
Nên chạy thử với `per_class=5` trước để ước lượng thời gian.

## Output (`results/02_auto_label/output/`)

```
images/<món>/*.jpg        ảnh (chép nguyên bản; ảnh có tag xoay EXIF thì lưu bản đã xoay)
labels/<món>/*.txt        nhãn YOLO: class_id x_center y_center width height (0-1)
labels/classes.txt        bản sao classes.txt cho X-AnyLabeling
visualized/<món>/*.jpg    ảnh đã vẽ box, mỗi box một màu (box món chính luôn màu đầu), kèm tên class + confidence
classes.txt, data.yaml    train ngay: yolo detect train data=data.yaml model=yolo11n.pt
train.txt, val.txt        chia 80/20 theo từng món (không gồm ảnh không có box)
crops/<món>/<ảnh>_<k>.jpg  từng box cắt thành ảnh riêng từ ảnh gốc (không có khung), k = 0 là món chính; crops/crops.csv
report.csv                mọi ảnh: số box, confidence trung bình/cao nhất, needs_review + lý do
failed.csv                ảnh không có box hoặc lỗi → sửa tay
```

Lý do cần xem lại: `no_box`, `low_conf` (confidence cao nhất < `review_conf`), `many_boxes`,
`box_near_full_image` (box món chính > 90% ảnh, thường là ôm cả đĩa/bàn).

## Cắt box thành ảnh riêng

- Bước 02 tự cắt **đúng vùng box model khoanh được** (`crops=True`, `crop_pad=0`; vd. `crop_pad=0.05` nếu muốn nới 5% mỗi phía).
- Cắt lại sau khi sửa tay nhãn: `python tools/crop_from_labels.py <thư mục output>` (đọc `images/` + `labels/`, cắt đúng box đã sửa).
- Xem trước: `python tools/crops_preview.py <thư mục crops> --images-root <thư mục output>`.

## Sửa nhãn bằng X-AnyLabeling

1. Cài: `pip install x-anylabeling` (hoặc tải bản cài sẵn tại https://github.com/CVHub520/X-AnyLabeling/releases), chạy `xanylabeling`.
2. **Open Dir** → chọn `images/<món>/`.
3. Menu **Upload** (biểu tượng mũi tên lên) → **YOLO-Hbb Annotations** → chọn `classes.txt` → chọn thư mục `labels/<món>/`.
   Tên menu có thể khác một chút tùy phiên bản.
4. Sửa box: kéo góc, xoá box sai (Delete), vẽ box mới (phím `R`) và chọn class.
5. Xuất lại: **Export** → **YOLO-Hbb Annotations** → ghi đè `labels/<món>/`.

Nên sửa trước các ảnh trong `failed.csv`, rồi tới ảnh có `needs_review = 1` trong `report.csv`.

## Lưu ý khi đọc kết quả bước 01

Box thật của UECFOOD-256 thường **ôm cả bát/đĩa**, còn mục tiêu của mình là box ôm sát thức ăn. Vì vậy ở IoU 0.5, box ôm
sát đúng ý lại bị tính sai. Nên xem thêm `ap30` (IoU 0.3), `area_ratio` (< 1 nghĩa là box nhỏ hơn box thật) và hình so sánh.
