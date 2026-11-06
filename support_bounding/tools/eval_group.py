"""Đánh giá gộp món theo đĩa (NOTES v5) trên dự đoán của kernel 04 (food + vật chứa). Không cần GPU.

So sánh các phương án hậu xử lý trên CÙNG dự đoán:
  none        = pipeline v2.1 (không gộp)
  group       = gộp đầy đủ (box to chứa >= 2 box con + gộp theo vật chứa nhỏ nhất)
  group_cont  = chỉ gộp theo vật chứa
  big_only    = chỉ luật box to chứa box con
Mọi phương án dùng chung: box >= 0.35, box phụ >= 0.45 (box lớn nhất luôn giữ).

UEC (box thật = 1 box mỗi món): tách ảnh 1 món / nhiều món; recall = món thật được 1 box khớp (IoU >= 0.5, ghép 1-1),
precision = box dự đoán khớp 1 món thật, "đúng số món" = số box == số món thật, "gộp nhầm" = 1 box dự đoán bao
>= 2 món thật khác nhau (mỗi món nằm >= 70% trong box).
Food-101: không có box thật -> thống kê số box/ảnh, + lưới hình các ảnh có số box thay đổi để xem bằng mắt.

Dùng (trong support_bounding/):
  python tools/eval_group.py results/04_group/output
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foodbox import labels, viz  # noqa: E402
from foodbox import postprocess as pp  # noqa: E402
from foodbox.data import load_image  # noqa: E402
from foodbox.run_detect import as_arrays, load_preds  # noqa: E402

VARIANTS = {"none": None, "group": dict(keep_parent=True, use_cont=True),
            "group_cont": dict(keep_parent=False, use_cont=True), "big_only": dict(keep_parent=True, use_cont=False)}


def final_boxes(x, variant, conf=0.35, other_conf=0.45):
    b, s = pp.postprocess(*as_arrays(x), max(x["w"], 1), max(x["h"], 1), conf=conf)
    v = VARIANTS[variant]
    if v is not None:
        cb = np.asarray(x.get("c_boxes", []), float).reshape(-1, 4) if v["use_cont"] else None
        cs = np.asarray(x.get("c_scores", []), float) if v["use_cont"] else None
        b, s = pp.group_dishes(b, s, cb, cs, x["w"], x["h"], keep_parent=v["keep_parent"])
    main = pp.pick_main(b, s, "largest")
    keep = [k for k in range(len(s)) if k == main or s[k] >= other_conf]
    return b[keep], s[keep]


def match(pred, gt, thr=0.5):
    """Ghép 1-1 theo IoU giảm dần; trả số cặp khớp."""
    if len(pred) == 0 or len(gt) == 0:
        return 0
    M = pp.iou_matrix(pred, gt)
    used_p, used_g, n = set(), set(), 0
    for v, i, j in sorted(((M[i, j], i, j) for i in range(len(pred)) for j in range(len(gt))), reverse=True):
        if v < thr:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            n += 1
    return n


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output kernel 04 đã tải về")
    ap.add_argument("--uec-root", default="data/uecfood256-mini/UECFOOD256", help="ảnh UEC trên máy (vẽ hình)")
    ap.add_argument("--food-images", default="results/02_auto_label/output_v2.1/images", help="ảnh Food-101 trên máy")
    a = ap.parse_args()
    O = a.out

    # ---------- UEC ----------
    items = json.load(open(f"{O}/items_uec.json", encoding="utf-8"))
    P = load_preds(f"{O}/preds_uec")["gdino"]["images"]
    rows = []
    for it in items:
        x = P.get(it["key"])
        if x is None or "error" in x:
            continue
        _, g = labels.read_yolo_txt(f"{O}/gt_labels/{it['class']}/{it['stem']}.txt", it["w"], it["h"])
        g = pp.as_boxes(g)
        for v in VARIANTS:
            b, _ = final_boxes(x, v)
            m = match(b, g)
            merged = 0
            if len(b) and len(g) >= 2:
                ins = pp.intersection(g, b) / np.maximum(pp.area(g)[:, None], 1e-9)  # món thật i nằm trong box j
                merged = int(((ins >= 0.7).sum(axis=0) >= 2).sum())
            rows.append({"variant": v, "image": it["key"], "group": "1 món" if len(g) == 1 else "nhiều món",
                         "n_gt": len(g), "n_pred": len(b), "n_match": m, "merged": merged})
    df = pd.DataFrame(rows)
    df.to_csv(f"{O}/eval_group_uec_images.csv", index=False)
    summ = df.groupby(["group", "variant"]).agg(
        n_img=("image", "size"), recall=("n_match", "sum"), n_gt=("n_gt", "sum"), n_pred=("n_pred", "sum"),
        dung_so_mon=("n_gt", lambda s: 0.0), merged=("merged", "sum")).reset_index()
    for i, r in summ.iterrows():
        sub = df[(df.group == r.group) & (df.variant == r.variant)]
        summ.loc[i, "dung_so_mon"] = float((sub.n_pred == sub.n_gt).mean())
        summ.loc[i, "anh_gop_nham"] = int((sub.merged > 0).sum())
    summ["precision"] = summ.recall / summ.n_pred.clip(lower=1)
    summ["recall"] = summ.recall / summ.n_gt
    summ["box_per_img"] = summ.n_pred / summ.n_img
    order = list(VARIANTS)
    summ["o"] = summ.variant.map(order.index)
    summ = summ.sort_values(["group", "o"]).drop(columns="o")
    summ.to_csv(f"{O}/eval_group_uec.csv", index=False)
    print("UEC (IoU >= 0.5, ghép 1-1):")
    print(summ[["group", "variant", "n_img", "recall", "precision", "box_per_img", "dung_so_mon", "merged",
                "anh_gop_nham"]].round(3).to_string(index=False))

    # ---------- Food-101 ----------
    items_f = json.load(open(f"{O}/items_food.json", encoding="utf-8"))
    F = load_preds(f"{O}/preds_food")["gdino"]["images"]
    stats, changed = [], []
    for it in items_f:
        x = F.get(it["key"])
        if x is None or "error" in x:
            continue
        nb = {v: len(final_boxes(x, v)[0]) for v in VARIANTS}
        stats.append({"image": it["key"], "class": it["class"], **{f"n_{v}": n for v, n in nb.items()}})
        if nb["none"] != nb["group"]:
            changed.append(it)
    fs = pd.DataFrame(stats)
    fs.to_csv(f"{O}/eval_group_food_images.csv", index=False)
    print(f"\nFood-101 ({len(fs)} ảnh): số box mỗi ảnh")
    print(pd.DataFrame({v: {"box/ảnh": fs[f"n_{v}"].mean(), "ảnh 1 box": (fs[f"n_{v}"] == 1).mean(),
                            "ảnh có box": (fs[f"n_{v}"] > 0).mean(), "ảnh >= 3 box": (fs[f"n_{v}"] >= 3).sum()}
                        for v in VARIANTS}).round(3).to_string())
    print(f"Ảnh có số box thay đổi khi gộp: {len(changed)}")

    # Lưới hình: các ảnh Food-101 + UEC có số box thay đổi (trước | sau gộp)
    def local(it, name):  # đường dẫn Kaggle -> ảnh trên máy (cùng ảnh)
        tail = it["path"].replace("\\", "/").split("/")
        if name == "uec":
            return os.path.join(a.uec_root, tail[-2], tail[-1])
        return os.path.join(a.food_images, it["class"], it["stem"] + os.path.splitext(tail[-1])[1].lower())

    for name, its, PP, k in [("food", changed, F, 40), ("uec", [it for it in items if P.get(it["key"]) and
                                                             len(final_boxes(P[it["key"]], "none")[0]) !=
                                                             len(final_boxes(P[it["key"]], "group")[0])], P, 40)]:
        tiles = []
        for it in its[:k]:
            x = PP[it["key"]]
            src = local(it, name)
            if not os.path.exists(src):
                continue
            img = load_image(src)
            pair = []
            for v in ["none", "group"]:
                b, s = final_boxes(x, v)
                pair.append(viz.fit(viz.draw_boxes(img, b, labels=[str(i + 1) for i in range(len(b))],
                                                   colors=[viz.color_for(i) for i in range(len(b))]), 300))
            tiles.append(viz.titled(viz.grid(pair, 2), f"{it['key']}: không gộp {len(final_boxes(x, 'none')[0])} | "
                                                       f"gộp {len(final_boxes(x, 'group')[0])}", 14))
        if tiles:
            for p0 in range(0, len(tiles), 12):
                viz.grid(tiles[p0:p0 + 12], 3).save(f"{O}/group_{name}_{p0 // 12 + 1}.jpg", quality=85)
            print(f"Hình so sánh {name}: {len(tiles)} ảnh -> {O}/group_{name}_*.jpg")
        else:
            print(f"Hình so sánh {name}: không có ảnh nguồn trên máy (đường dẫn Kaggle) -> bỏ qua")


if __name__ == "__main__":
    main()
