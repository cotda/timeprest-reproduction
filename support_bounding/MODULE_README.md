# foodbox: nhập ảnh món ăn → ra bounding box

Phiên bản 0.2.0 (2026-11-06): thêm **gộp theo đĩa** (các miếng cùng đĩa/thớt → 1 khung), bật mặc định.
Bản 0.1.0: mỗi miếng 1 khung (dùng `--no-group` để có hành vi cũ).

Module tự khoanh vùng từng món ăn trong ảnh. Model: **Grounding DINO base** (zero-shot, trọng số công khai
`IDEA-Research/grounding-dino-base`, license Apache-2.0), prompt `"food"`. Không cần train, không cần nhãn.

## Cài đặt

```bash
cd support_bounding          # thư mục chứa pyproject.toml
pip install -e .             # cài torch, transformers, pillow, numpy + lệnh foodbox-detect
```

- Lần chạy đầu tự tải trọng số model từ HuggingFace (~900 MB, cần mạng); các lần sau dùng lại bản đã tải.
- Có GPU NVIDIA thì tự dùng GPU (~1,2 s/ảnh trên T4); không có thì chạy CPU (đo trên laptop: 25–40 s/ảnh, nạp model ~30 s).
  Thêm `--no-group` thì nhanh gấp ~2 (bỏ bước gộp theo đĩa, xem bên dưới).
- Python ≥ 3.9. Nếu máy chưa có PyTorch bản GPU, cài theo hướng dẫn ở https://pytorch.org trước rồi mới `pip install -e .`.

## Cách 1: một lệnh

```bash
foodbox-detect khay_com.jpg --out ketqua/
foodbox-detect anh1.jpg anh2.jpg thu_muc_anh/ --out ketqua/     # nhiều ảnh / cả thư mục
python -m foodbox.cli khay_com.jpg --out ketqua/                # nếu lệnh foodbox-detect chưa có trong PATH
```

Kết quả cho mỗi ảnh (ví dụ `khay_com.jpg`):

| File | Nội dung |
|---|---|
| `khay_com_bbox.jpg` | Ảnh có khung, mỗi khung một màu, đánh số 1, 2, 3… (sắp theo điểm tin cậy giảm dần) |
| `khay_com_1.jpg`, `khay_com_2.jpg`, … | Khung số i cắt thành ảnh riêng: **đúng vùng khung**, cắt từ ảnh gốc (không dính nét khung) |
| `khay_com.json` | Toạ độ từng khung (pixel), điểm tin cậy, tham số đã dùng |

Tuỳ chọn: `--device cpu|cuda`, `--conf 0.35` (ngưỡng box), `--other-conf 0.45` (ngưỡng box phụ), `--no-group`, `--no-crops`,
`--no-vis`, `--pad 0.05` (nới ảnh cắt 5% mỗi phía), `--prompt "food"`.

## Cách 2: gọi trong Python

```python
from foodbox import FoodBoxDetector

det = FoodBoxDetector()                      # nạp model 1 lần (device="auto": GPU nếu có)
boxes = det.detect("khay_com.jpg")           # hoặc truyền ảnh PIL
for b in boxes:
    print(b["id"], b["x1"], b["y1"], b["x2"], b["y2"], b["score"])
# ví dụ thật (ảnh khay miso_soup_38, CPU):
# [{'id': 1, 'x1': 175.5, 'y1': 305.7, 'x2': 255.1, 'y2': 387.2, 'score': 0.6493, 'largest': False}, ...]

det.detect_and_save("khay_com.jpg", "ketqua/")   # ghi ảnh có khung + ảnh cắt + JSON như cách 1
```

## Cách module xử lý một ảnh

1. Đọc ảnh, xoay theo EXIF, đổi sang RGB.
2. Grounding DINO tìm các vùng khớp với chữ "food" → box + điểm tin cậy.
3. Hậu xử lý: bỏ box trùng (NMS, IoU 0,5), bỏ box quá nhỏ (< 0,2% ảnh), giữ box có điểm ≥ 0,35.
4. **Gộp theo đĩa** (mặc định bật, tắt bằng `--no-group` / `FoodBoxDetector(group=False)`): chạy thêm 1 lượt tìm vật chứa
   (plate, bowl, tray, cutting board). Các miếng nằm trong cùng một đĩa/thớt → **1 khung** ôm sát toàn bộ phần thức ăn
   (vd. 10 miếng bruschetta trên thớt → 1 khung). Món ở đĩa/bát khác nhau (khay cơm) vẫn tách khung riêng; khay chứa
   nhiều bát không dùng để gộp. Nếu đã có 1 khung to bao ≥ 2 khung nhỏ thì giữ khung to.
5. Box lớn nhất luôn giữ, các box còn lại cần điểm ≥ 0,45; tối đa 20 box. Vẽ khung, cắt từng khung, ghi JSON.

Tham số giống hệt luồng gán nhãn đã đánh giá (xem `NOTES.md`, bản v2.1): trên ảnh 1 món tìm được ~90% món, ảnh khay
nhiều món ~68% món (đánh giá trên UECFOOD-256). Module chỉ khoanh vùng, không đặt tên món.

Lưu ý: chạy CPU và GPU (hoặc khác phiên bản transformers) cho điểm tin cậy lệch nhau ~0,01–0,03, nên box nằm sát
ngưỡng (0,35 / 0,45) có thể có ở lần chạy này mà không có ở lần khác. Ví dụ ảnh khay miso_soup_38: 5/6 box trùng nhau
(IoU ≥ 0,97), box thứ 6 trên GPU là bát súp miso (0,4506), trên CPU là gói gia vị (0,4512).
