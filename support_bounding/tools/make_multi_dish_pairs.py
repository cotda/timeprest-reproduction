"""Hình so sánh cho ảnh NHIỀU MÓN của UEC (bước 01): ảnh gốc | box pipeline | box thật, kèm đánh giá từng ảnh.

Box pipeline = dự đoán đã tải về của bước 01 + hậu xử lý và gán class y như bước 02 (conf món chính, other_conf food_other,
box lớn nhất = món theo thư mục). Không cần GPU.
Dùng (trong support_bounding/):
  python tools/make_multi_dish_pairs.py results/01_compare/output report/multi_dish --variant gdino --conf 0.35 --other-conf 0.45
Ghi: <out>/images/<món>__<ảnh>.jpg (từng ảnh), <out>/pages_<i>.jpg (6 ảnh/trang), <out>/multi_dish_list.csv.
"""
import argparse
import csv
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foodbox import data  # noqa: E402
from foodbox import postprocess as pp  # noqa: E402
from foodbox.autolabel import box_colors  # noqa: E402
from foodbox.crop import save_crops, write_crops_csv  # noqa: E402
from foodbox.labels import OTHER_CLASS  # noqa: E402
from foodbox.run_detect import as_arrays, load_preds  # noqa: E402
from tools.make_report_pairs import draw_cell, font  # noqa: E402
from tools.reeval_01 import load_gts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_out", help="output bước 01 đã tải về (items.json, gt_labels/, preds/)")
    ap.add_argument("out")
    ap.add_argument("--variant", default="gdino")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--other-conf", type=float, default=0.45)
    ap.add_argument("--uec-root", default="data", help="nơi có category.txt (để ghi tên món của box thật)")
    ap.add_argument("--side", type=int, default=380)
    ap.add_argument("--per-page", type=int, default=6)
    ap.add_argument("--no-notes", action="store_true", help="bỏ dòng chấm điểm dưới ảnh (bản gửi người xem)")
    ap.add_argument("--no-box-labels", action="store_true", help="chỉ vẽ khung box, không ghi tên món / điểm trên box")
    ap.add_argument("--crops-dir", default=None, help="cắt từng box của pipeline thành ảnh riêng vào thư mục này")
    ap.add_argument("--crop-pad", type=float, default=0.0, help="0 = cắt đúng box model khoanh")
    a = ap.parse_args()

    items, _ = load_gts(a.run_out)
    uec_root = data.find_uec_root(a.uec_root)
    cats = data.read_uec_categories(uec_root)
    for it in items:  # đường dẫn Kaggle -> bản UEC mini trên máy (cùng cấu trúc <id món>/<ảnh>.jpg)
        it["path"] = os.path.join(uec_root, *it["path"].replace("\\", "/").split("/")[-2:])
    # Box thật kèm id món gốc của UEC (để ghi tên món), đọc thẳng từ bb_info.txt như bước 01
    uec_boxes = data.read_uec_boxes(uec_root)
    cat_of_class = {data.safe_name(n): c for c, n in cats.items()}
    gts = {it["key"]: data.uec_gt_for_item(it, uec_boxes, cat_of_class, it["w"], it["h"]) for it in items}
    res = load_preds(os.path.join(a.run_out, "preds"))[a.variant]["images"]
    multi = [it for it in items if len(gts[it["key"]]) >= 2]
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    S, gap, cap_h = a.side, 10, 28 if a.no_notes else 52
    f_head, f_cap, f_box = font(20), font(15), font(14)
    rows, triplets, crop_rows = [], [], []
    for it in multi:
        x, g = res[it["key"]], gts[it["key"]]
        b, s = pp.postprocess(*as_arrays(x), x["w"], x["h"], conf=a.conf)
        main_i = pp.pick_main(b, s, "largest")
        cls = [it["class"] if k == main_i else OTHER_CLASS for k in range(len(s))]
        keep = [k for k in range(len(s)) if k == main_i or s[k] >= a.other_conf]  # giống build_dataset(other_conf=...)
        b, s, cls = b[keep], s[keep], [cls[k] for k in keep]
        # Đánh giá: món chính đúng nếu box món chính trùng (IoU >= 0.5) box thật của món theo thư mục
        g_main = [box for is_main, _, box in g if is_main]
        g_side = [box for is_main, _, box in g if not is_main]
        mi = cls.index(it["class"]) if it["class"] in cls else -1
        main_ok = mi >= 0 and len(g_main) > 0 and pp.iou_matrix(b[mi:mi + 1], g_main).max() >= 0.5
        # Món phụ: ghép 1-1 (IoU >= 0.5, cặp IoU cao trước) giữa box thật món phụ và các box food_other.
        # Box món chính không được tính là "tìm thấy món phụ" (tránh đếm một box cho hai việc).
        others = [k for k in range(len(s)) if k != mi]
        side_found, used = 0, set()
        if g_side and others:
            M = pp.iou_matrix(g_side, b[others])
            used_g = set()
            for v, i, j in sorted(((M[i, j], i, j) for i in range(len(g_side)) for j in range(len(others))), reverse=True):
                if v < 0.5:
                    break
                if i not in used_g and j not in used:
                    used_g.add(i)
                    used.add(j)
                    side_found += 1
        n_wrong = len(others) - len(used)  # box food_other không ghép được với món phụ nào: thừa hoặc lệch

        img = data.load_image(it["path"])
        if a.crops_dir and len(s):
            crop_rows += save_crops(img, b, cls, a.crops_dir, it["class"], it["stem"], f"{it['class']}/{it['stem']}",
                                    scores=s, main_flags=[c == it["class"] for c in cls], pad=a.crop_pad)
        ids = [0 if c == it["class"] else 1 for c in cls]
        pred_txt = [""] * len(s) if a.no_box_labels else [f"{c} {v:.2f}" for c, v in zip(cls, s)]
        pred = draw_cell(img, b, pred_txt, box_colors(ids, 0), S, f_box)
        g_names = [cats[c] for _, c, _ in g]
        g_ids = [0 if is_main else 1 for is_main, _, _ in g]
        truth = draw_cell(img, [box for *_, box in g], [""] * len(g) if a.no_box_labels else g_names,
                          box_colors(g_ids, 0), S, f_box)
        orig = draw_cell(img, [], [], [], S, f_box)
        note = (f"{len(g)} món thật | pipeline {len(s)} box | món chính: {'ĐÚNG' if main_ok else 'SAI'} | "
                f"món phụ tìm thấy {side_found}/{len(g_side)}" + (f" | {n_wrong} box food_other thừa/lệch" if n_wrong else ""))
        tri = Image.new("RGB", (3 * S + 2 * gap, S + cap_h), (255, 255, 255))
        for j, im in enumerate([orig, pred, truth]):
            tri.paste(im, (j * (S + gap), 0))
        d = ImageDraw.Draw(tri)
        d.text((0, S + 4), f"{it['class']}/{it['stem']}", fill=(0, 0, 0), font=f_cap)
        if not a.no_notes:
            d.text((0, S + 24), note, fill=(0, 120, 0) if main_ok and side_found == len(g_side) else (200, 0, 0), font=f_cap)
        name = f"{it['class']}__{it['stem']}.jpg"
        tri.save(os.path.join(a.out, "images", name), quality=90)
        triplets.append(tri)
        rows.append({"image": f"{it['class']}/{it['stem']}", "file": f"images/{name}", "n_dishes": len(g),
                     "n_boxes": len(s), "main_ok": int(main_ok), "side_found": side_found, "n_side": len(g_side),
                     "food_other_extra_or_off": n_wrong, "dishes": "; ".join(g_names)})

    W = 3 * S + 2 * gap
    for p in range(0, len(triplets), a.per_page):
        page = triplets[p:p + a.per_page]
        head = 34
        canvas = Image.new("RGB", (W + 2 * gap, head + len(page) * (S + cap_h + gap)), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for j, t in enumerate(["Ảnh gốc", "Pipeline tự vẽ box", "Box thật (đáp án UEC)"]):
            d.text((gap + j * (S + gap) + S // 2, 6), t, fill=(0, 0, 0), font=f_head, anchor="mt")
        for k, t in enumerate(page):
            canvas.paste(t, (gap, head + k * (S + cap_h + gap)))
        canvas.save(os.path.join(a.out, f"pages_{p // a.per_page + 1:02d}.jpg"), quality=85)
    with open(os.path.join(a.out, "multi_dish_list.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if a.crops_dir:
        write_crops_csv(a.crops_dir, crop_rows)
        print(f"[multi] {len(crop_rows)} ảnh cắt -> {a.crops_dir}")
    n = len(rows)
    print(f"[multi] {n} ảnh nhiều món -> {a.out} ({(n + a.per_page - 1) // a.per_page} trang)")
    print(f"[multi] món chính đúng {sum(r['main_ok'] for r in rows)}/{n}; món phụ tìm thấy "
          f"{sum(r['side_found'] for r in rows)}/{sum(r['n_side'] for r in rows)}; "
          f"ảnh đúng hoàn toàn (món chính đúng + đủ món phụ) "
          f"{sum(r['main_ok'] and r['side_found'] == r['n_side'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
