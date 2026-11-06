"""Đo độ chính xác box (không phân biệt món): IoU, Precision, Recall, F1, AP@0.5."""
import numpy as np

from .postprocess import area, as_boxes, iou_matrix, pick_main


def match_image(boxes, scores, gt_boxes, iou_thr=0.5):
    """Ghép tham lam theo điểm giảm dần (kiểu VOC/COCO): mỗi box thật ghép tối đa 1 box dự đoán.

    Trả (tp, iou, gt_idx) theo đúng thứ tự box dự đoán đầu vào.
    """
    boxes, gt_boxes = as_boxes(boxes), as_boxes(gt_boxes)
    n = len(boxes)
    tp, best_iou, gt_idx = np.zeros(n, bool), np.zeros(n), np.full(n, -1)
    if n == 0 or len(gt_boxes) == 0:
        return tp, best_iou, gt_idx
    ious = iou_matrix(boxes, gt_boxes)
    used = np.zeros(len(gt_boxes), bool)
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        cand = np.where(used, -1.0, ious[i])
        j = int(np.argmax(cand))
        best_iou[i] = ious[i].max()
        if cand[j] >= iou_thr:
            tp[i], gt_idx[i], used[j] = True, j, True
    return tp, best_iou, gt_idx


def average_precision(scores, tp, n_gt):
    """AP = diện tích dưới đường Precision-Recall (nội suy mọi điểm, như VOC2010+)."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tp, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(preds, gts, iou_thr=0.5, conf=None):
    """preds: {key: (boxes, scores)} (lọc ngưỡng thấp); gts: {key: [(is_main, cat, box), ...]}.

    - ap50: AP@0.5 trên toàn bộ box dự đoán (không phụ thuộc ngưỡng confidence).
    - P/R/F1, mean_iou, ...: tính ở ngưỡng `conf`; conf=None thì dùng ngưỡng cho F1 cao nhất (best_conf).
      Mỗi model có thang điểm khác nhau (OWLv2 điểm thấp, GDINO điểm cao) nên so ở ngưỡng tốt nhất của từng model.
    - mean_iou: mỗi box thật lấy IoU cao nhất với box dự đoán (0 nếu không có), rồi lấy trung bình.
    - area_ratio: trung vị (diện tích box dự đoán / box thật) của các cặp ghép đúng; > 1 = box rộng hơn box thật.
    - main_acc_*: tỉ lệ ảnh mà box được chọn làm món chính (lớn nhất / điểm cao nhất) trùng box món theo thư mục.
    """
    all_s, all_tp, all_iou, all_ratio, all_key = [], [], [], [], []
    n_gt = 0
    for key, gt in gts.items():
        gt_boxes = [g[2] for g in gt]
        boxes, scores = preds.get(key, (np.zeros((0, 4)), np.zeros(0)))
        boxes, scores = as_boxes(boxes), np.asarray(scores, float)
        tp, iou, gi = match_image(boxes, scores, gt_boxes, iou_thr)
        ratio = np.full(len(scores), np.nan)
        if len(gt_boxes):
            ga = area(gt_boxes)
            ratio[tp] = area(boxes[tp]) / np.maximum(ga[gi[tp]], 1e-9)
        n_gt += len(gt_boxes)
        all_s.append(scores), all_tp.append(tp), all_iou.append(iou), all_ratio.append(ratio)
        all_key += [key] * len(scores)
    s, tp = np.concatenate(all_s), np.concatenate(all_tp)
    ap = average_precision(s, tp, n_gt)

    if conf is None:  # ngưỡng cho F1 cao nhất
        order = np.argsort(-s, kind="stable")
        ctp = np.cumsum(tp[order])
        k = np.arange(1, len(s) + 1)
        f1 = 2 * ctp / np.maximum(k + n_gt, 1e-9)
        conf = float(s[order][int(np.argmax(f1))]) if len(s) else 0.0
    m = s >= conf
    n_tp, n_pred = int(tp[m].sum()), int(m.sum())
    prec = n_tp / n_pred if n_pred else 0.0
    rec = n_tp / n_gt if n_gt else 0.0
    ratio = np.concatenate(all_ratio)[m & tp]

    ious_gt, main_ok, n_main, no_box = [], {"largest": 0, "score": 0}, 0, 0
    for key, gt in gts.items():
        boxes, scores = preds.get(key, (np.zeros((0, 4)), np.zeros(0)))
        boxes, scores = as_boxes(boxes), np.asarray(scores, float)
        keep = scores >= conf
        boxes, scores = boxes[keep], scores[keep]
        no_box += int(len(scores) == 0)
        if gt:
            g = as_boxes([x[2] for x in gt])
            ious_gt += list(iou_matrix(g, boxes).max(axis=1)) if len(boxes) else [0.0] * len(g)
        mains = as_boxes([x[2] for x in gt if x[0]])
        if len(mains):
            n_main += 1
            for rule in main_ok:
                i = pick_main(boxes, scores, rule)
                if i >= 0 and iou_matrix(boxes[i:i + 1], mains).max() >= iou_thr:
                    main_ok[rule] += 1
    return {
        "ap50": ap, "conf": conf, "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "mean_iou": float(np.mean(ious_gt)) if ious_gt else 0.0,
        "area_ratio": float(np.median(ratio)) if len(ratio) else float("nan"),
        "main_acc_largest": main_ok["largest"] / n_main if n_main else float("nan"),
        "main_acc_score": main_ok["score"] / n_main if n_main else float("nan"),
        "n_images": len(gts), "n_gt": n_gt, "n_pred": n_pred, "n_tp": n_tp, "images_no_box": no_box,
    }
