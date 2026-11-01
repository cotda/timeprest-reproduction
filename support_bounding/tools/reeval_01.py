"""Tính lại bảng kết quả bước 01 từ dự đoán đã tải về (preds/*.json), với tham số hậu xử lý khác. Không cần GPU.

Dùng (trong support_bounding/):
  python tools/reeval_01.py results/01_compare/output --tag v1.1                       # mặc định mới
  python tools/reeval_01.py results/01_compare/output --tag v1 --max-area 0.98 --drop-container   # tái tạo v1
Ghi results_<tag>.csv và results_per_class_<tag>.csv vào thư mục output.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foodbox import labels, metrics  # noqa: E402
from foodbox import postprocess as pp  # noqa: E402
from foodbox.run_detect import as_arrays, load_preds  # noqa: E402

COLS = ["model", "ap50", "ap30", "precision", "recall", "f1", "mean_iou", "area_ratio", "main_acc_largest",
        "main_acc_score", "conf", "sec_per_img", "peak_vram_mb", "images_no_box"]


def load_gts(out):
    """Box thật từ gt_labels/<món>/<ảnh>.txt; class khác food_other = món chính."""
    with open(os.path.join(out, "items.json")) as f:
        items = json.load(f)
    names = open(os.path.join(out, "gt_labels", "classes.txt"), encoding="utf-8").read().split()
    other = names.index(labels.OTHER_CLASS)
    gts = {}
    for it in items:
        cls, boxes = labels.read_yolo_txt(os.path.join(out, "gt_labels", it["class"], it["stem"] + ".txt"),
                                          it["w"], it["h"])
        gts[it["key"]] = [(c != other, c, tuple(b)) for c, b in zip(cls, boxes)]
    return items, gts


def evaluate_all(out, nms_iou=0.5, min_area=0.002, max_area=1.0, drop_container=False, iou_thr=0.5):
    items, gts = load_gts(out)
    classes = sorted({it["class"] for it in items})
    rows, per_class = [], []
    for variant, r in sorted(load_preds(os.path.join(out, "preds")).items()):
        P = {k: pp.postprocess(*as_arrays(x), max(x["w"], 1), max(x["h"], 1), conf=0.0, nms_iou=nms_iou,
                               min_area=min_area, max_area=max_area, drop_container=drop_container, max_det=100)
             for k, x in r["images"].items()}
        m = metrics.evaluate(P, gts, iou_thr)
        ok = [x for x in r["images"].values() if "error" not in x]
        rows.append({"model": variant, **m, "ap30": metrics.evaluate(P, gts, 0.3)["ap50"],
                     "sec_per_img": float(np.mean([x["time_s"] for x in ok])),
                     "peak_vram_mb": max(mm["peak_vram_mb"] for mm in r["meta"])})
        for c in classes:
            g = {k: v for k, v in gts.items() if k.startswith(c + "/")}
            mc = metrics.evaluate(P, g, iou_thr, conf=m["conf"])
            per_class.append({"model": variant, "class": c, "ap50": mc["ap50"], "recall": mc["recall"],
                              "mean_iou": mc["mean_iou"]})
    return pd.DataFrame(rows).sort_values("ap50", ascending=False), pd.DataFrame(per_class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="thư mục output đã tải về (có items.json, gt_labels/, preds/)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=0.002)
    ap.add_argument("--max-area", type=float, default=1.0)
    ap.add_argument("--drop-container", action="store_true")
    a = ap.parse_args()
    df, pc = evaluate_all(a.out, a.nms_iou, a.min_area, a.max_area, a.drop_container)
    df.to_csv(os.path.join(a.out, f"results_{a.tag}.csv"), index=False)
    pc.to_csv(os.path.join(a.out, f"results_per_class_{a.tag}.csv"), index=False)
    print(f"[{a.tag}] nms_iou={a.nms_iou} min_area={a.min_area} max_area={a.max_area} drop_container={a.drop_container}")
    print(df[COLS].round(3).to_string(index=False))
    print("\nAP@0.5 theo món:")
    print(pc.pivot(index="class", columns="model", values="ap50").round(2).to_string())


if __name__ == "__main__":
    main()
