"""Các model phát hiện theo câu chữ (open-vocabulary) và SAM 2.1 để lấy box ôm sát từ mask.

Mọi detector có cùng giao diện: predict(img PIL RGB) -> (boxes Nx4 pixel xyxy, scores N).
Ngưỡng ở đây để thấp (low_thr) để tính được AP; ngưỡng thật áp dụng ở bước hậu xử lý.
"""
import numpy as np
import torch

# Tên model -> id checkpoint mặc định
DEFAULT_IDS = {
    "gdino": "IDEA-Research/grounding-dino-base",   # Apache-2.0, HF transformers
    "owlv2": "google/owlv2-base-patch16-ensemble",  # Apache-2.0, HF transformers
    "yoloworld": "yolov8x-worldv2.pt",              # Ultralytics (AGPL-3.0)
    "sam": "sam2.1_b.pt",                           # SAM 2.1 base qua Ultralytics (trọng số Apache-2.0)
}


def _empty():
    return np.zeros((0, 4)), np.zeros(0)


def _topk(boxes, scores, k):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float).reshape(-1)
    order = np.argsort(-scores, kind="stable")[:k]
    return boxes[order], scores[order]


class GroundingDINO:
    """Grounding DINO (Liu et al., ECCV 2024) qua HF transformers. Prompt dạng 'food.' (chữ thường, kết thúc bằng dấu chấm)."""

    def __init__(self, prompts=("food",), device="cuda", model_id=None, low_thr=0.05, max_det=100):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        mid = model_id or DEFAULT_IDS["gdino"]
        self.processor = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device).eval()
        self.text = " ".join(p.strip().lower().rstrip(".") + "." for p in prompts)
        self.device, self.low_thr, self.max_det = device, low_thr, max_det

    @torch.no_grad()
    def predict(self, img):
        inputs = self.processor(images=img, text=self.text, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        size = [(img.height, img.width)]
        post = self.processor.post_process_grounded_object_detection
        try:  # transformers >= 4.51 đổi box_threshold -> threshold
            r = post(out, inputs["input_ids"], threshold=self.low_thr, text_threshold=self.low_thr, target_sizes=size)[0]
        except TypeError:
            r = post(out, inputs["input_ids"], box_threshold=self.low_thr, text_threshold=self.low_thr, target_sizes=size)[0]
        if len(r["scores"]) == 0:
            return _empty()
        return _topk(r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(), self.max_det)


class OWLv2:
    """OWLv2 (Minderer et al., NeurIPS 2023) qua HF transformers. Mỗi prompt là một câu truy vấn."""

    def __init__(self, prompts=("food",), device="cuda", model_id=None, low_thr=0.02, max_det=100):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        mid = model_id or DEFAULT_IDS["owlv2"]
        self.processor = Owlv2Processor.from_pretrained(mid)
        self.model = Owlv2ForObjectDetection.from_pretrained(mid).to(device).eval()
        self.queries = [list(prompts)]
        self.device, self.low_thr, self.max_det = device, low_thr, max_det

    @torch.no_grad()
    def predict(self, img):
        inputs = self.processor(text=self.queries, images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        # OWLv2 đệm ảnh thành hình vuông (thêm phải/dưới) nên target size là cạnh dài x cạnh dài
        side = max(img.width, img.height)
        size = torch.tensor([[side, side]], device=self.device)
        try:
            r = self.processor.post_process_grounded_object_detection(out, threshold=self.low_thr, target_sizes=size)[0]
        except (AttributeError, TypeError):
            r = self.processor.post_process_object_detection(out, threshold=self.low_thr, target_sizes=size)[0]
        if len(r["scores"]) == 0:
            return _empty()
        b = r["boxes"].float().cpu().numpy()
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, img.width)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, img.height)
        return _topk(b, r["scores"].float().cpu().numpy(), self.max_det)


def _bgr(img):
    """PIL RGB -> numpy BGR (Ultralytics nhận mảng numpy theo thứ tự BGR như OpenCV)."""
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


class YOLOWorld:
    """YOLO-World v2 (Cheng et al., CVPR 2024) qua Ultralytics: rất nhanh, prompt = danh sách tên class."""

    def __init__(self, prompts=("food",), device="cuda", model_id=None, low_thr=0.01, max_det=100, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(model_id or DEFAULT_IDS["yoloworld"])
        self.model.set_classes(list(prompts))
        self.device, self.low_thr, self.max_det, self.imgsz = device, low_thr, max_det, imgsz

    def predict(self, img):
        r = self.model.predict(_bgr(img), conf=self.low_thr, iou=0.7, imgsz=self.imgsz, max_det=self.max_det,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return _empty()
        return _topk(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), self.max_det)


class SAMRefiner:
    """Đưa box của detector vào SAM 2.1 làm prompt, lấy box bao quanh mask -> box ôm sát phần thức ăn.

    Box mới chỉ được co lại trong box gốc. Giữ box gốc khi mask rỗng hoặc quá nhỏ so với box (min_fill),
    vì khi đó SAM thường chỉ bắt được một mẩu nhỏ.
    """

    def __init__(self, device="cuda", model_id=None, min_fill=0.15):
        from ultralytics import SAM
        self.model = SAM(model_id or DEFAULT_IDS["sam"])
        self.device, self.min_fill = device, min_fill

    def refine(self, img, boxes):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if len(boxes) == 0:
            return boxes.copy()
        r = self.model.predict(_bgr(img), bboxes=boxes.tolist(), device=self.device, verbose=False)[0]
        if r.masks is None:
            return boxes.copy()
        masks = r.masks.data.cpu().numpy() > 0.5
        out = boxes.copy()
        for k in range(min(len(masks), len(boxes))):
            m = masks[k]
            if m.shape != (img.height, img.width):  # phòng khi mask ở độ phân giải khác ảnh gốc
                sy, sx = m.shape[0] / img.height, m.shape[1] / img.width
            else:
                sy = sx = 1.0
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            nb = np.array([xs.min() / sx, ys.min() / sy, (xs.max() + 1) / sx, (ys.max() + 1) / sy])
            b = boxes[k]
            # Chỉ cho box co lại: mask tràn ra ngoài box gốc thường là đĩa/bàn bị dính vào
            nb = np.concatenate([np.maximum(nb[:2], b[:2]), np.minimum(nb[2:], b[2:])])
            if nb[2] - nb[0] < 1 or nb[3] - nb[1] < 1:
                continue
            box_area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
            if len(xs) / (sx * sy) >= self.min_fill * box_area:
                out[k] = nb
        return out


DETECTORS = {"gdino": GroundingDINO, "owlv2": OWLv2, "yoloworld": YOLOWorld}


def build_detector(name, prompts, device="cuda", model_id=None, low_thr=None):
    kw = {"prompts": prompts, "device": device, "model_id": model_id}
    if low_thr is not None:
        kw["low_thr"] = low_thr
    return DETECTORS[name](**kw)
