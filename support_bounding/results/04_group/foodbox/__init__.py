"""foodbox: luồng auto bounding box cho món ăn.

Dùng như module (nhập 1 ảnh → ra bounding box):
    from foodbox import FoodBoxDetector
    det = FoodBoxDetector()
    boxes = det.detect("khay_com.jpg")
    det.detect_and_save("khay_com.jpg", "ketqua/")
hoặc lệnh: foodbox-detect khay_com.jpg --out ketqua/

Các module:
- api.py         FoodBoxDetector: 1 ảnh → box (+ ảnh có khung, ảnh cắt, JSON)
- cli.py         lệnh foodbox-detect
- data.py        dò cấu trúc thư mục, chọn tập con, đọc box thật UECFOOD-256 và chuyển sang YOLO
- detectors.py   Grounding DINO, OWLv2, YOLO-World và SAM 2.1 (lấy box từ mask)
- postprocess.py NMS, lọc diện tích, bỏ box ôm cả khay, gán class
- metrics.py     IoU, Precision, Recall, AP@0.5
- labels.py      ghi nhãn YOLO, classes.txt, data.yaml
- crop.py        cắt từng box thành ảnh riêng
- autolabel.py   gán nhãn cả bộ ảnh <món>/*.jpg (bước 02)
- viz.py         vẽ box
- run_detect.py  chạy model trên 1 GPU (notebook gọi 2 tiến trình cho 2 GPU)
"""
__version__ = "0.1.0"


def __getattr__(name):
    # Nạp FoodBoxDetector khi cần: "import foodbox" không kéo theo torch/transformers
    if name == "FoodBoxDetector":
        from .api import FoodBoxDetector
        return FoodBoxDetector
    raise AttributeError(name)
