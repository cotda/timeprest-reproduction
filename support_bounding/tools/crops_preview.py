"""Hình xem trước: mỗi hàng = ảnh gốc | các ảnh cắt từ nó (đọc crops.csv). Không cần GPU.

Dùng (trong support_bounding/):
  python tools/crops_preview.py report/multi_dish_crops --images-root data --uec --n 8 --out report/multi_dish_crops/preview.jpg
  python tools/crops_preview.py results/02_auto_label/output_v2.1/crops --images-root results/02_auto_label/output_v2.1 --n 8
"""
import argparse
import csv
import os
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foodbox import data  # noqa: E402
from tools.make_report_pairs import font  # noqa: E402


def fit_h(img, h):
    return img.resize((max(1, round(img.width * h / img.height)), h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crops_dir", help="thư mục có crops.csv")
    ap.add_argument("--images-root", required=True, help="gốc để tìm ảnh nguồn (cột image của crops.csv)")
    ap.add_argument("--uec", action="store_true", help="ảnh nguồn là UEC mini: image = <món>/<id>, tìm theo id trong bb_info")
    ap.add_argument("--n", type=int, default=8, help="số ảnh gốc (chọn ngẫu nhiên, seed 0, ưu tiên ảnh có >= 2 ảnh cắt)")
    ap.add_argument("--h", type=int, default=220)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.crops_dir, "crops.csv"), encoding="utf-8")))
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    multi = sorted(k for k, v in by_img.items() if len(v) >= 2)
    pool = multi if len(multi) >= a.n else sorted(by_img)
    picked = sorted(random.Random(0).sample(pool, min(a.n, len(pool))))
    uec_root = data.find_uec_root(a.images_root) if a.uec else None
    f = font(16)
    lines = []
    for key in picked:
        if uec_root:  # UEC: ảnh nằm ở <id món>/<id ảnh>.jpg, tìm thư mục chứa ảnh đó
            stem = key.split("/")[1]
            src = next(os.path.join(uec_root, d, stem + ".jpg") for d in sorted(os.listdir(uec_root))
                       if os.path.exists(os.path.join(uec_root, d, stem + ".jpg")))
        else:
            src = os.path.join(a.images_root, key)
        tiles = [fit_h(data.load_image(src), a.h)]
        for r in sorted(by_img[key], key=lambda r: int(r["k"])):
            tiles.append(fit_h(Image.open(os.path.join(a.crops_dir, r["crop"])).convert("RGB"), a.h))
        gap = 8
        row = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) + 2) + 300, a.h + 24), (255, 255, 255))
        x = 0
        d = ImageDraw.Draw(row)
        for i, t in enumerate(tiles):
            row.paste(t, (x, 24))
            if i == 0:
                title = f"Ảnh gốc: {key}"
                d.text((x, 2), title, fill=(0, 0, 0), font=f)
                x += max(t.width, round(d.textlength(title, font=f))) + gap  # tên dài không đè lên ảnh cắt
                d.text((x, a.h // 2 + 14), "→", fill=(0, 0, 0), font=font(28))
                x += 3 * gap
            else:
                d.text((x, 2), f"ảnh cắt {i}", fill=(90, 90, 90), font=f)
                x += t.width + gap
        lines.append(row)
    W = max(r.width for r in lines)
    sheet = Image.new("RGB", (W, sum(r.height + 12 for r in lines)), (255, 255, 255))
    y = 0
    for r in lines:
        sheet.paste(r, (0, y))
        y += r.height + 12
    out = a.out or os.path.join(a.crops_dir, "preview.jpg")
    sheet.save(out, quality=90)
    print(f"[preview] {len(lines)} ảnh gốc -> {out} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
