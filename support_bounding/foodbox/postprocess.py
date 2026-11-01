"""Hậu xử lý box: NMS (bỏ box trùng), lọc diện tích, bỏ box ôm cả khay, gán class."""
import numpy as np


def as_boxes(boxes):
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def area(b):
    b = as_boxes(b)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def intersection(a, b):
    """Ma trận diện tích giao nhau |a| x |b|."""
    a, b = as_boxes(a), as_boxes(b)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def iou_matrix(a, b):
    inter = intersection(a, b)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(boxes, scores, iou_thr=0.5):
    """NMS không phân biệt class: giữ box điểm cao, bỏ box trùng IoU > iou_thr. Trả chỉ số giữ lại."""
    boxes, scores = as_boxes(boxes), np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return keep


def drop_containers(boxes, scores=None, contain_thr=0.85, min_children=2, min_rel_score=0.5):
    """Bỏ box chứa >= min_children box khác (box ôm cả khay/bàn chứa nhiều món). Trả chỉ số giữ lại.

    Box con "nằm trong" box cha nếu phần giao chiếm >= contain_thr diện tích box con. Chỉ tính box con có
    điểm >= min_rel_score x điểm box cha: bát phở 0.8 chứa vài box rác 0.1 (miếng thịt, hành) vẫn được giữ.
    """
    boxes = as_boxes(boxes)
    if len(boxes) < min_children + 1:
        return list(range(len(boxes)))
    scores = np.ones(len(boxes)) if scores is None else np.asarray(scores, dtype=np.float64)
    inter = intersection(boxes, boxes)
    inside = inter / np.maximum(area(boxes)[None, :], 1e-9) >= contain_thr  # inside[i, j]: j nằm trong i
    np.fill_diagonal(inside, False)
    bigger = area(boxes)[:, None] > area(boxes)[None, :]
    strong = scores[None, :] >= min_rel_score * scores[:, None]
    n_children = (inside & bigger & strong).sum(axis=1)
    return [i for i in range(len(boxes)) if n_children[i] < min_children]


def postprocess(boxes, scores, w, h, conf=0.0, nms_iou=0.5, min_area=0.002, max_area=1.0,
                drop_container=False, max_det=20):
    """Chuỗi lọc chuẩn; trả (boxes, scores) đã sắp theo điểm giảm dần.

    min_area / max_area là tỉ lệ diện tích so với ảnh (box quá nhỏ = nhiễu).
    Mặc định max_area=1.0 và drop_container=False: kết quả 01 (v1.1 trong NOTES.md) cho thấy max_area=0.98 xoá box
    đúng của ảnh chụp cận (món chiếm cả khung) và drop_container xoá món nhiều miếng (sushi, chả giò);
    tắt cả hai thì AP@0.5 tăng ~0.09. Box gần cả ảnh được đánh dấu xem lại trong report.csv thay vì xoá.
    """
    boxes, scores = as_boxes(boxes), np.asarray(scores, dtype=np.float64)
    frac = area(boxes) / float(w * h)
    m = (scores >= conf) & (frac >= min_area) & (frac <= max_area)
    boxes, scores = boxes[m], scores[m]
    keep = nms(boxes, scores, nms_iou)
    boxes, scores = boxes[keep], scores[keep]
    if drop_container:
        keep = drop_containers(boxes, scores)
        boxes, scores = boxes[keep], scores[keep]
    return boxes[:max_det], scores[:max_det]


def pick_main(boxes, scores, rule="largest"):
    """Chỉ số box món chính: 'largest' (diện tích lớn nhất) hoặc 'score' (điểm cao nhất). -1 nếu không có box."""
    if len(scores) == 0:
        return -1
    if rule == "score":
        return int(np.argmax(scores))
    return int(np.argmax(area(boxes)))


def assign_classes(boxes, scores, main_id, other_id, rule="largest"):
    """Box món chính lấy class theo tên thư mục, các box còn lại là food_other."""
    i = pick_main(boxes, scores, rule)
    return [main_id if k == i else other_id for k in range(len(scores))]
