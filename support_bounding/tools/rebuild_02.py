"""Dựng lại bộ nhãn bước 02 từ dự đoán thô đã tải về (tmp_preds/), với tham số hậu xử lý khác. Không cần GPU.

Ảnh lấy từ output/images/ của lần chạy gốc (bản đã chép về máy), nên không cần dataset gốc.
Dùng (trong support_bounding/):
  python tools/rebuild_02.py results/02_auto_label --tag v2.1 --conf 0.35 --other-conf 0.45
Ghi vào results/02_auto_label/output_<tag>/ + rebuild_params.json.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foodbox.autolabel import build_dataset  # noqa: E402
from foodbox.run_detect import load_preds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="thư mục kết quả đã tải về (có tmp_preds/, output/images/)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="gdino", help="biến thể trong tmp_preds (vd. gdino, gdino+sam)")
    ap.add_argument("--conf", type=float, required=True)
    ap.add_argument("--other-conf", type=float, default=None)
    ap.add_argument("--review-conf", type=float, default=None)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=0.002)
    ap.add_argument("--max-area", type=float, default=1.0)
    ap.add_argument("--max-boxes", type=int, default=4)
    ap.add_argument("--main-rule", default="largest")
    a = ap.parse_args()

    with open(os.path.join(a.run_dir, "tmp_preds", "items.json")) as f:
        items = json.load(f)
    src = os.path.join(a.run_dir, "output", "images")
    for it in items:  # đường dẫn Kaggle -> bản ảnh đã chép về máy
        ext = os.path.splitext(it["path"])[1].lower()
        it["path"] = os.path.join(src, it["class"], it["stem"] + ext)
    results = load_preds(os.path.join(a.run_dir, "tmp_preds", "preds"))[a.variant]["images"]
    out = os.path.join(a.run_dir, f"output_{a.tag}")
    report, failed = build_dataset(items, results, out, conf=a.conf, nms_iou=a.nms_iou, min_area=a.min_area,
                                   max_area=a.max_area, main_rule=a.main_rule, review_conf=a.review_conf,
                                   max_boxes=a.max_boxes, other_conf=a.other_conf)
    with open(os.path.join(a.run_dir, f"rebuild_params_{a.tag}.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    rep = pd.DataFrame(report)
    n_box = rep.n_boxes.sum()
    print(f"[{a.tag}] {vars(a)}")
    print(f"Ảnh có box: {(rep.n_boxes > 0).sum()}/{len(rep)} ({(rep.n_boxes > 0).mean():.1%}) | tổng box {n_box} | "
          f"box/ảnh {rep.n_boxes.mean():.2f} | cần xem lại {rep.needs_review.sum()} | failed {len(failed)}")
    print("Lý do xem lại:", rep.reasons[rep.reasons != ""].str.split(";").explode().value_counts().to_dict())
    print(rep.groupby("class").agg(box_per_img=("n_boxes", "mean"), no_box=("n_boxes", lambda s: (s == 0).sum()),
                                   review=("needs_review", "sum")).round(2).to_string())


if __name__ == "__main__":
    main()
