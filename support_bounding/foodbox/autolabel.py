"""Biến kết quả detector thành bộ nhãn YOLO hoàn chỉnh: ảnh, nhãn, hình vẽ box, classes.txt, data.yaml, report/failed.csv."""
import csv
import os
import random
import shutil

import numpy as np

from . import labels, viz
from .data import exif_rotated, load_image
from .postprocess import assign_classes, postprocess

REPORT_FIELDS = ["image", "class", "n_boxes", "mean_conf", "max_conf", "main_conf", "needs_review", "reasons"]


def review_reasons(scores, n_boxes, main_area_frac, review_conf, max_boxes):
    """Lý do cần xem lại một ảnh (chuỗi rỗng = ổn)."""
    r = []
    if n_boxes == 0:
        r.append("no_box")
    else:
        if float(np.max(scores)) < review_conf:
            r.append("low_conf")
        if n_boxes > max_boxes:
            r.append("many_boxes")
        if main_area_frac > 0.9:
            r.append("box_near_full_image")
    return ";".join(r)


def build_dataset(items, results, out, conf, nms_iou=0.5, min_area=0.002, max_area=1.0, drop_container=False,
                  main_rule="largest", review_conf=None, max_boxes=4, val_frac=0.2, seed=0, max_det=20,
                  other_conf=None):
    """items: list {key, class, path, stem}; results: {key: {w, h, boxes, scores[, error]}} từ run_detect.

    Ghi vào out/: images/<món>/, labels/<món>/, visualized/<món>/, classes.txt, data.yaml, train.txt, val.txt,
    report.csv (mọi ảnh), failed.csv (ảnh không có box hoặc lỗi). Ảnh không có box vẫn được chép (để sửa tay)
    nhưng không nằm trong train.txt/val.txt. Trả (report_rows, failed_rows).

    other_conf: ngưỡng riêng cho box food_other (None = conf). Box món chính được dùng ngưỡng thấp hơn vì đã biết
    ảnh có món đó (tên thư mục); box thêm cần điểm cao hơn mới giữ, để bớt nhận nhầm đồ trang trí (02 v2: 56 box food_other).
    """
    review_conf = conf + 0.1 if review_conf is None else review_conf
    names = labels.class_list(sorted({it["class"] for it in items}))
    cid = {n: i for i, n in enumerate(names)}
    colors = {i: viz.color_for(i) for i in range(len(names))}
    report, failed, ok_images = [], [], []
    for it in items:
        r = results.get(it["key"])
        rel = f"{it['class']}/{it['stem']}"
        if r is None or "error" in r:
            failed.append({"image": rel, "class": it["class"], "reason": (r or {}).get("error", "missing")})
            continue
        boxes, scores = postprocess(r["boxes"], r["scores"], r["w"], r["h"], conf=conf, nms_iou=nms_iou,
                                    min_area=min_area, max_area=max_area, drop_container=drop_container,
                                    max_det=max_det)
        cls = assign_classes(boxes, scores, cid[it["class"]], cid[labels.OTHER_CLASS], main_rule)
        if other_conf is not None:
            keep = [k for k, c in enumerate(cls) if c != cid[labels.OTHER_CLASS] or scores[k] >= other_conf]
            boxes, scores, cls = boxes[keep], scores[keep], [cls[k] for k in keep]

        # Ảnh: chép nguyên bản; ảnh có tag EXIF xoay thì lưu bản đã xoay để khớp toạ độ nhãn trong mọi công cụ
        ext = os.path.splitext(it["path"])[1].lower()
        img_rel = f"images/{rel}{ext}"
        os.makedirs(os.path.join(out, "images", it["class"]), exist_ok=True)
        img = load_image(it["path"])
        if exif_rotated(it["path"]):
            img.save(os.path.join(out, img_rel), quality=95)
        else:
            shutil.copy2(it["path"], os.path.join(out, img_rel))
        labels.write_yolo_txt(os.path.join(out, "labels", rel + ".txt"), cls, boxes, r["w"], r["h"])
        vis = viz.draw_boxes(img, boxes, labels=[f"{names[c]} {s:.2f}" for c, s in zip(cls, scores)],
                             colors=[colors[c] for c in cls])
        os.makedirs(os.path.join(out, "visualized", it["class"]), exist_ok=True)
        vis.save(os.path.join(out, "visualized", rel + ".jpg"), quality=85)

        main = [k for k, c in enumerate(cls) if c == cid[it["class"]]]
        main_frac = 0.0
        if main:
            b = boxes[main[0]]
            main_frac = (b[2] - b[0]) * (b[3] - b[1]) / (r["w"] * r["h"])
        reasons = review_reasons(scores, len(scores), main_frac, review_conf, max_boxes)
        report.append({"image": rel, "class": it["class"], "n_boxes": len(scores),
                       "mean_conf": round(float(np.mean(scores)), 4) if len(scores) else "",
                       "max_conf": round(float(np.max(scores)), 4) if len(scores) else "",
                       "main_conf": round(float(scores[main[0]]), 4) if main else "",
                       "needs_review": int(bool(reasons)), "reasons": reasons})
        if len(scores) == 0:  # ảnh có thức ăn mà không có box: không đưa vào train (YOLO sẽ hiểu là ảnh nền)
            failed.append({"image": rel, "class": it["class"], "reason": "no_box"})
        else:
            ok_images.append((it["class"], "./" + img_rel))

    # Chia train/val theo từng món (val_frac mỗi món), để train YOLO được ngay
    rng = random.Random(seed)
    train, val = [], []
    for c in sorted({c for c, _ in ok_images}):
        imgs = sorted(p for cc, p in ok_images if cc == c)
        rng.shuffle(imgs)
        n_val = max(1, round(len(imgs) * val_frac)) if len(imgs) > 1 else 0
        val += imgs[:n_val]
        train += imgs[n_val:]
    for fname, lst in [("train.txt", train), ("val.txt", val)]:
        with open(os.path.join(out, fname), "w") as f:
            f.write("\n".join(sorted(lst)) + ("\n" if lst else ""))
    labels.write_classes(out, names)
    labels.write_data_yaml(out, names)
    with open(os.path.join(out, "classes.txt")) as f:  # X-AnyLabeling cần classes.txt cạnh thư mục nhãn
        content = f.read()
    with open(os.path.join(out, "labels", "classes.txt"), "w") as f:
        f.write(content)
    _write_csv(os.path.join(out, "report.csv"), REPORT_FIELDS, report)
    _write_csv(os.path.join(out, "failed.csv"), ["image", "class", "reason"], failed)
    return report, failed


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
