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


def _inside(a, b):
    """inside[i, j] = tỉ lệ diện tích box a[i] nằm trong box b[j]."""
    return intersection(a, b) / np.maximum(area(a)[:, None], 1e-9)


def group_dishes(boxes, scores, c_boxes=None, c_scores=None, w=None, h=None, c_conf=0.30, inside_thr=0.7,
                 c_min_area=0.01, contain_thr=0.8, nms_iou=0.5, keep_parent=True):
    """Gộp các miếng của CÙNG MỘT đĩa/thớt thành 1 box (yêu cầu người dùng, NOTES v5). Trả (boxes, scores) sắp theo điểm.

    1. Box to bao trọn >= 2 box khác (mỗi box con nằm >= contain_thr trong nó): giữ box to, bỏ các box con.
    2. Vật chứa (c_boxes: plate/bowl/tray/cutting board, điểm >= c_conf, >= c_min_area ảnh): mỗi box thức ăn thuộc vật chứa
       NHỎ NHẤT bao nó (>= inside_thr diện tích box nằm trong). Các box cùng vật chứa -> 1 box = hợp của chúng (ôm sát
       phần thức ăn, không ôm cả đĩa), điểm = điểm cao nhất. Vật chứa bao >= 2 vật chứa khác (khay chứa nhiều bát) không
       dùng để gộp, để các món khác bát trong khay vẫn tách riêng.
    3. Box không thuộc vật chứa nào giữ nguyên. Cuối cùng NMS lần nữa (box hợp có thể trùng box to sẵn có).
    keep_parent=False: bỏ bước 1 (để đánh giá riêng từng bước).
    """
    boxes, scores = as_boxes(boxes), np.asarray(scores, dtype=np.float64)
    if len(boxes) == 0:
        return boxes, scores
    # 1) box to chứa >= 2 box con -> bỏ box con
    ins = _inside(boxes, boxes)  # ins[i, j]: i nằm trong j
    np.fill_diagonal(ins, 0)
    bigger = area(boxes)[None, :] > area(boxes)[:, None]
    is_parent = ((ins >= contain_thr) & bigger).sum(axis=0) >= 2
    child = ((ins >= contain_thr) & bigger & is_parent[None, :]).any(axis=1) & keep_parent
    boxes, scores = boxes[~child], scores[~child]

    # 2) gộp theo vật chứa nhỏ nhất
    if c_boxes is not None and len(c_boxes) and len(boxes) >= 2:
        cb, cs = as_boxes(c_boxes), np.asarray(c_scores, dtype=np.float64)
        keep = cs >= c_conf
        if w and h:
            keep &= area(cb) / float(w * h) >= c_min_area
        cb = cb[keep]
        if len(cb):
            cin = _inside(cb, cb)
            np.fill_diagonal(cin, 0)
            tray = ((cin >= inside_thr) & (area(cb)[None, :] > area(cb)[:, None])).sum(axis=0) >= 2
            cb = cb[~tray]
        if len(cb):
            fin = _inside(boxes, cb)  # box thức ăn i nằm trong vật chứa j
            owner = np.full(len(boxes), -1)
            for i in range(len(boxes)):
                cand = np.where(fin[i] >= inside_thr)[0]
                if len(cand):
                    owner[i] = cand[np.argmin(area(cb[cand]))]
            out_b, out_s = [], []
            for j in sorted(set(owner[owner >= 0])):
                g = np.where(owner == j)[0]
                gb = boxes[g]
                out_b.append([gb[:, 0].min(), gb[:, 1].min(), gb[:, 2].max(), gb[:, 3].max()])
                out_s.append(scores[g].max())
            for i in np.where(owner < 0)[0]:
                out_b.append(boxes[i].tolist())
                out_s.append(scores[i])
            boxes, scores = as_boxes(out_b), np.asarray(out_s, dtype=np.float64)

    # 3) NMS lần nữa, sắp theo điểm
    keep = nms(boxes, scores, nms_iou)
    return boxes[keep], scores[keep]


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
