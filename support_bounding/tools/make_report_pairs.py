"""Ghép hình cho báo cáo: mỗi cặp = ảnh gốc | ảnh đã vẽ box, chọn ngẫu nhiên (seed cố định) k ảnh mỗi món.

Dùng (trong support_bounding/):
  python tools/make_report_pairs.py results/02_auto_label/output_v2.1 report --per-class 3 --pairs-per-page 10
Ghi report/pairs_<i>.jpg + report/pairs_list.csv (ảnh nào, kết quả chấm tay nếu có review_v2.1.csv).
"""
import argparse
import csv
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foodbox.data import load_image  # noqa: E402
from foodbox.labels import read_yolo_txt  # noqa: E402
from foodbox.autolabel import box_colors  # noqa: E402

VERDICT = {"ok": "", "partial": "box thiếu một phần món", "wrong": "box sai"}


def font(size):
    for f in ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:  # cần font có dấu tiếng Việt
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def cell(img, side, bg=(255, 255, 255)):
    """Thu ảnh vào ô vuông side x side, căn giữa."""
    s = side / max(img.size)
    im = img.resize((round(img.width * s), round(img.height * s)))
    out = Image.new("RGB", (side, side), bg)
    out.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return out


def draw_cell(img, boxes, texts, colors, side, f):
    """Vẽ box ở kích thước ô (nét dày, chữ to) thay vì thu nhỏ ảnh đã vẽ (nét mảnh khó nhìn trên slide)."""
    s = side / max(img.size)
    im = img.resize((round(img.width * s), round(img.height * s)))
    d = ImageDraw.Draw(im)
    for b, t, c in zip(boxes, texts, colors):
        x1, y1, x2, y2 = [v * s for v in b]
        d.rectangle([x1, y1, x2, y2], outline=c, width=4)
        if not t:  # không có chữ: chỉ vẽ khung
            continue
        tb = d.textbbox((x1, y1), t, font=f)
        ty = y1 - (tb[3] - tb[1]) - 6 if y1 - (tb[3] - tb[1]) - 6 > 0 else y1 + 2
        tb = d.textbbox((x1 + 3, ty), t, font=f)
        d.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=c)
        d.text((x1 + 3, ty), t, fill=(255, 255, 255), font=f)
    out = Image.new("RGB", (side, side), (255, 255, 255))
    out.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="bộ nhãn (có images/, visualized/, report.csv)")
    ap.add_argument("report_dir")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--pairs-per-page", type=int, default=10)
    ap.add_argument("--side", type=int, default=340)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--review", default=None, help="review csv (tuỳ chọn) để ghi kết quả chấm tay dưới mỗi cặp")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(os.path.join(a.out_dir, "report.csv"), encoding="utf-8"))
            if int(r["n_boxes"]) > 0]
    review = {}
    if a.review and os.path.exists(a.review):
        review = {r["image"]: r for r in csv.DictReader(open(a.review, encoding="utf-8"))}
    rng = random.Random(a.seed)
    picked = []
    for c in sorted({r["class"] for r in rows}):
        cand = sorted((r for r in rows if r["class"] == c), key=lambda r: r["image"])
        picked += sorted(rng.sample(cand, min(a.per_class, len(cand))), key=lambda r: r["image"])

    os.makedirs(a.report_dir, exist_ok=True)
    S, gap, cap_h, head_h = a.side, 12, 34, 40
    f_head, f_cap, f_box = font(22), font(17), font(16)
    names = open(os.path.join(a.out_dir, "classes.txt"), encoding="utf-8").read().split()
    pair_w = 2 * S + gap
    cols = 2  # 2 cặp mỗi hàng
    listing = []
    pages = [picked[i:i + a.pairs_per_page] for i in range(0, len(picked), a.pairs_per_page)]
    for pi, page in enumerate(pages):
        nrows = (len(page) + cols - 1) // cols
        W = cols * pair_w + (cols + 1) * gap * 2
        H = head_h + nrows * (S + cap_h + gap) + gap
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for ci in range(cols):  # tiêu đề cột
            x0 = gap * 2 + ci * (pair_w + gap * 2)
            d.text((x0 + S // 2, 10), "Ảnh gốc", fill=(0, 0, 0), font=f_head, anchor="mt")
            d.text((x0 + S + gap + S // 2, 10), "Sau khi tự vẽ box", fill=(0, 0, 0), font=f_head, anchor="mt")
        for k, r in enumerate(page):
            c, stem = r["image"].split("/")
            ext = next(e for e in [".jpg", ".jpeg", ".png"] if os.path.exists(os.path.join(a.out_dir, "images", c, stem + e)))
            orig = load_image(os.path.join(a.out_dir, "images", c, stem + ext))
            cls, boxes = read_yolo_txt(os.path.join(a.out_dir, "labels", c, stem + ".txt"), orig.width, orig.height)
            x0 = gap * 2 + (k % cols) * (pair_w + gap * 2)
            y0 = head_h + (k // cols) * (S + cap_h + gap)
            canvas.paste(cell(orig, S), (x0, y0))
            canvas.paste(draw_cell(orig, boxes, [names[k] for k in cls],
                                   box_colors(cls, names.index(c)), S, f_box),
                         (x0 + S + gap, y0))
            v = review.get(r["image"], {})
            note = VERDICT.get(v.get("main", "ok"), "")
            if v and int(v.get("bad_food_other", 0)):
                note = (note + "; " if note else "") + f"{v['bad_food_other']} box food_other sai"
            cap = f"{r['image']}  ({r['n_boxes']} box, conf {float(r['max_conf']):.2f})"
            d.text((x0, y0 + S + 4), cap, fill=(0, 0, 0), font=f_cap)
            if note:
                d.text((x0 + pair_w, y0 + S + 4), note, fill=(200, 0, 0), font=f_cap, anchor="ra")
            listing.append({"page": pi + 1, "image": r["image"], "n_boxes": r["n_boxes"], "max_conf": r["max_conf"],
                            "review": note or ("đúng" if v else "")})
        path = os.path.join(a.report_dir, f"pairs_{pi + 1}.jpg")
        canvas.save(path, quality=90)
        print(f"[pairs] {path}: {len(page)} cặp, {W}x{H}")
    with open(os.path.join(a.report_dir, "pairs_list.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(listing[0]))
        w.writeheader()
        w.writerows(listing)
    bad = sum(1 for x in listing if x["review"] not in ("", "đúng"))
    print(f"[pairs] {len(listing)} cặp, {bad} cặp có lỗi theo chấm tay")


if __name__ == "__main__":
    main()
