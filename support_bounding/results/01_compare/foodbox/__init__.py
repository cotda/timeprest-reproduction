"""foodbox: luồng auto bounding box cho món ăn (dùng chung cho các notebook Kaggle).

Các module:
- data.py        dò cấu trúc thư mục, chọn tập con, đọc box thật UECFOOD-256 và chuyển sang YOLO
- detectors.py   Grounding DINO, OWLv2, YOLO-World và SAM 2.1 (lấy box từ mask)
- postprocess.py NMS, lọc diện tích, bỏ box ôm cả khay, gán class
- metrics.py     IoU, Precision, Recall, AP@0.5
- labels.py      ghi nhãn YOLO, classes.txt, data.yaml
- viz.py         vẽ box
- run_detect.py  chạy model trên 1 GPU (notebook gọi 2 tiến trình cho 2 GPU)
"""
