# Chạy gán bounding box cho 2001_food_dataset trên Kaggle (GPU T4 ×2)

Notebook: `02_food2001.ipynb` (đã nhúng sẵn code `foodbox`, không cần cài gì thêm). Thời gian: ~15 phút GPU cho 1.500 ảnh.
Kết quả: 1 file `ketqua_2001.zip`.

## Cách 1: làm trên trang web Kaggle (không cần cài gì trên máy)

### Bước 1: Upload dataset (làm 1 lần)
1. Trên máy: chuột phải thư mục `D:\University\Research\2001_food_dataset\2001_food_dataset` → **Send to → Compressed (zipped) folder**
   (tạo file zip bên cạnh, thư mục gốc không bị sửa).
2. Vào https://www.kaggle.com/datasets → **+ New Dataset** → kéo file zip vào → đặt tên `food-2001` → để **Private** → **Create**.
   Kaggle tự giải nén; chờ đến khi trang dataset hiện thư mục `images/` (vài phút).

### Bước 2: Tạo notebook
1. Vào https://www.kaggle.com/code → **+ New Notebook**.
2. Menu **File → Import Notebook** → chọn file `support_bounding/kaggle/02_food2001/02_food2001.ipynb` → Import.
3. Cột bên phải:
   - **Input → + Add Input → Your Datasets →** chọn `food-2001`.
   - **Session options → Accelerator → GPU T4 x2**.
   - **Internet → On** (cần để tải model ~900 MB từ HuggingFace; lần đầu Kaggle có thể yêu cầu xác minh số điện thoại).

### Bước 3: Chạy
- **Chạy thử trước (khuyên làm):** ở ô tham số `CFG`, sửa `per_class=None` thành `per_class=1` (300 ảnh, ~5 phút) → **Run All**.
  Kiểm tra ô cuối in ra `Xong sau ... phút: /kaggle/working/ketqua_2001.zip`, mở `preview_grid.jpg` ở tab Output xem khung có đúng không.
- **Chạy hết:** sửa lại `per_class=None` → bấm **Save Version → Save & Run All (Commit) → Save**. Notebook chạy ngầm,
  có thể đóng trình duyệt; xem tiến độ ở trang notebook → tab **Logs**.

### Bước 4: Lấy kết quả
Trang notebook → tab **Output** → tải `ketqua_2001.zip` (và `preview_grid.jpg`, `run_summary.json`).

## Cách 2: dòng lệnh (trên máy có repo `pipe_dream_pj` + Kaggle API đã cấu hình)

```bash
cd support_bounding
# 1. Upload dataset: chép sang thư mục tạm có dataset-metadata.json (không thêm file vào thư mục dataset gốc)
python -c "import shutil; shutil.copytree(r'D:\University\Research\2001_food_dataset\2001_food_dataset', r'_upload_food2001\2001_food_dataset')"
kaggle datasets init -p _upload_food2001          # sửa "title" = "food-2001", "id" = "<tên_kaggle>/food-2001" trong file vừa tạo
kaggle datasets create -p _upload_food2001 --dir-mode zip
# 2. Chạy notebook (tự build, đẩy lên 2xT4, chờ, tải kết quả về results/02_food2001/)
python run_kaggle.py kaggle/02_food2001
```
Nếu tài khoản Kaggle không phải `kaggleolap`: sửa `"dataset_sources"` trong `kernel-metadata.json` thành `<tên_kaggle>/food-2001`
và chạy `python run_kaggle.py kaggle/02_food2001 --user <tên_kaggle>`.

## Bên trong ketqua_2001.zip

```
theo_anh/<món>/001.jpg          ảnh gốc
theo_anh/<món>/001_bbox.jpg     ảnh có khung, đánh số 1, 2, ...
theo_anh/<món>/001_1.jpg ...    khung i cắt thành ảnh riêng (đúng vùng khung)
theo_anh/index.csv              ảnh cắt nào ở vị trí nào
yolo/                           bộ nhãn để train YOLO: images/, labels/, classes.txt (300 món + food_other), data.yaml,
                                train.txt / val.txt, visualized/, crops/
yolo/report.csv                 mọi ảnh: số box, điểm tin cậy, needs_review + lý do (low_conf, many_boxes, box_near_full_image)
yolo/failed.csv                 ảnh không có box → vẽ tay bằng X-AnyLabeling
run_summary.json                tham số, số ảnh, thời gian chạy
```

Lưu ý: ảnh có nhiều chiếc cùng một món (vd. 11 chiếc cannoli) cho ra nhiều khung, mỗi chiếc một khung; trong nhãn YOLO
khung lớn nhất mang tên món, các khung còn lại là `food_other`.
