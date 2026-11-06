"""Mỗi ảnh gốc một thư mục: ảnh gốc <tên>.jpg, ảnh có khung <tên>_bbox.jpg (khung số i = vùng của <tên>_i.jpg)
và các ảnh cắt <tên>_1.jpg, <tên>_2.jpg, ... (đọc crops.csv). Không cần GPU.
Ảnh cắt lấy từ ẢNH GỐC theo toạ độ box (không cắt từ ảnh đã vẽ khung), nên không dính nét khung.

Tên = "<món>_<ảnh>" (vd. ảnh miso_soup/38 -> miso_soup_38.jpg, miso_soup_38_1.jpg, ...). Thứ tự _1, _2... theo cột k
của crops.csv (box sắp theo điểm, _1 thường là món chính; xem cột is_main trong index.csv). Ảnh gốc được chép, không sửa.
Dùng (trong support_bounding/):
  python tools/make_crop_folders.py report/multi_dish_crops report/multi_dish_preview --images-root data --uec
  python tools/make_crop_folders.py results/02_auto_label/output_v2.1/crops <out> --images-root results/02_auto_label/output_v2.1
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foodbox import data  # noqa: E402
from foodbox.crop import write_image_folder  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):  # console Windows: in được tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("crops_dir", help="thư mục có crops.csv")
    ap.add_argument("out")
    ap.add_argument("--images-root", required=True, help="gốc để tìm ảnh nguồn (cột image của crops.csv)")
    ap.add_argument("--uec", action="store_true", help="ảnh nguồn là UEC mini: image = <món>/<id>, tìm theo id")
    ap.add_argument("--min-crops", type=int, default=1, help="chỉ làm cho ảnh có ít nhất N ảnh cắt")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.crops_dir, "crops.csv"), encoding="utf-8")))
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    uec_root = data.find_uec_root(a.images_root) if a.uec else None
    index, n_dir = [], 0
    for key in sorted(by_img):
        crops = sorted(by_img[key], key=lambda r: int(r["k"]))
        if len(crops) < a.min_crops:
            continue
        if uec_root:  # UEC: ảnh nằm ở <id món>/<id ảnh>.jpg
            stem = key.split("/")[1]
            src = next(os.path.join(uec_root, d, stem + ".jpg") for d in sorted(os.listdir(uec_root))
                       if os.path.exists(os.path.join(uec_root, d, stem + ".jpg")))
        else:
            src = os.path.join(a.images_root, key)
        base = os.path.splitext(key)[0].replace("/", "_")  # vd. miso_soup_38
        index += write_image_folder(src, crops, a.crops_dir, os.path.join(a.out, base), base)
        n_dir += 1
    with open(os.path.join(a.out, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"[folders] {n_dir} thư mục, {len(index)} ảnh cắt -> {a.out}")


if __name__ == "__main__":
    main()
