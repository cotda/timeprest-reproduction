"""Dùng như một module: nhập 1 ảnh → ra bounding box (+ ảnh có khung, ảnh cắt, JSON).

    from foodbox import FoodBoxDetector
    det = FoodBoxDetector()                         # tự dùng GPU nếu có, không thì CPU
    boxes = det.detect("khay_com.jpg")              # list box, sắp theo điểm giảm dần
    det.detect_and_save("khay_com.jpg", "ketqua/")  # ghi khay_com_bbox.jpg, khay_com_1.jpg..., khay_com.json

Cùng model và tham số với pipeline gán nhãn (bước 02, NOTES.md v2.1): Grounding DINO base, prompt "food",
box ≥ 0.35, box phụ ≥ 0.45 (box lớn nhất luôn giữ), NMS 0.5, bỏ box < 0.2% ảnh, tối đa 20 box. Chỉ trả box,
không đặt tên món.
"""
import json
import os
import time

from PIL import Image, ImageOps

from .crop import crop_box
from .postprocess import pick_main, postprocess
from .viz import color_for, draw_boxes

MODEL_ID = "IDEA-Research/grounding-dino-base"
DEFAULTS = {"prompts": ["food"], "conf": 0.35, "other_conf": 0.45, "nms_iou": 0.5, "min_area": 0.002, "max_det": 20}


def _to_image(image):
    """Đường dẫn hoặc ảnh PIL -> ảnh PIL RGB đã xoay theo EXIF (giống pipeline)."""
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    with Image.open(image) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


class FoodBoxDetector:
    """Phát hiện vùng thức ăn trong ảnh. Nạp model 1 lần (lần đầu tự tải ~900 MB từ HuggingFace), gọi nhiều lần."""

    def __init__(self, device="auto", model_id=MODEL_ID, prompts=None, conf=None, other_conf=None, nms_iou=None,
                 min_area=None, max_det=None):
        import torch
        from .detectors import GroundingDINO
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        p = dict(DEFAULTS)
        for k, v in dict(prompts=prompts, conf=conf, other_conf=other_conf, nms_iou=nms_iou, min_area=min_area,
                         max_det=max_det).items():
            if v is not None:
                p[k] = list(v) if k == "prompts" else v
        self.params, self.device, self.model_id = p, device, model_id
        self._det = GroundingDINO(prompts=p["prompts"], device=device, model_id=model_id)

    def detect(self, image):
        """Trả list box {id, x1, y1, x2, y2, score, largest}, sắp theo điểm giảm dần (id 1, 2, ...).

        Toạ độ pixel trên ảnh đã xoay EXIF; largest = box lớn nhất (box pipeline coi là món chính).
        """
        img = _to_image(image)
        return self._boxes(img)[0]

    def _boxes(self, img):
        p = self.params
        t = time.time()
        raw_b, raw_s = self._det.predict(img)
        b, s = postprocess(raw_b, raw_s, img.width, img.height, conf=p["conf"], nms_iou=p["nms_iou"],
                           min_area=p["min_area"], max_det=p["max_det"])
        main = pick_main(b, s, "largest")
        keep = [k for k in range(len(s)) if k == main or s[k] >= p["other_conf"]]
        out = [{"id": i, "x1": round(float(b[k][0]), 1), "y1": round(float(b[k][1]), 1),
                "x2": round(float(b[k][2]), 1), "y2": round(float(b[k][3]), 1),
                "score": round(float(s[k]), 4), "largest": k == main}
               for i, k in enumerate(keep, 1)]
        return out, time.time() - t

    def detect_and_save(self, image, out_dir, vis=True, crops=True, pad=0.0, name=None):
        """Chạy detect và ghi vào out_dir: <tên>_bbox.jpg (khung đánh số i), <tên>_i.jpg (cắt đúng box i, từ ảnh gốc),
        <tên>.json (toạ độ). pad > 0 thì nới ảnh cắt mỗi phía pad x cạnh box. Trả dict kết quả (như trong JSON)."""
        img = _to_image(image)
        if name is None:
            name = os.path.splitext(os.path.basename(image))[0] if isinstance(image, (str, os.PathLike)) else "image"
        boxes, sec = self._boxes(img)
        os.makedirs(out_dir, exist_ok=True)
        files = {}
        if vis:
            files["bbox"] = f"{name}_bbox.jpg"
            draw_boxes(img, [[x["x1"], x["y1"], x["x2"], x["y2"]] for x in boxes],
                       labels=[str(x["id"]) for x in boxes], colors=[color_for(i) for i in range(len(boxes))]
                       ).save(os.path.join(out_dir, files["bbox"]), quality=92)
        for x in boxes:
            if crops:
                c, _ = crop_box(img, (x["x1"], x["y1"], x["x2"], x["y2"]), pad)
                x["crop"] = f"{name}_{x['id']}.jpg"
                c.save(os.path.join(out_dir, x["crop"]), quality=95)
        result = {"image": str(image) if isinstance(image, (str, os.PathLike)) else name, "width": img.width,
                  "height": img.height, "n_boxes": len(boxes), "boxes": boxes, "files": files,
                  "model": self.model_id, "device": self.device, "params": self.params, "seconds": round(sec, 3)}
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        return result
