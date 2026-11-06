"""Cắt mọi box của một bộ nhãn YOLO (images/<món>/ + labels/<món>/ + classes.txt) thành file ảnh riêng.

Dùng được cho output của bước 02, kể cả SAU KHI sửa tay bằng X-AnyLabeling (cắt theo nhãn đã sửa). Không cần GPU.
Dùng (trong support_bounding/):
  python tools/crop_from_labels.py results/02_auto_label/output_v2.1          # cắt đúng box
Ghi <dataset>/crops/<món>/<ảnh>_<k>.jpg (k = thứ tự box trong file nhãn) + crops/crops.csv.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foodbox.crop import save_crops, write_crops_csv  # noqa: E402
from foodbox.data import IMG_EXT, load_image  # noqa: E402
from foodbox.labels import OTHER_CLASS, read_yolo_txt  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):  # console Windows: in được tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="thư mục có images/, labels/, classes.txt")
    ap.add_argument("--out", default=None, help="mặc định <dataset>/crops")
    ap.add_argument("--pad", type=float, default=0.0, help="0 = cắt đúng box; > 0 nới mỗi phía pad x cạnh box")
    ap.add_argument("--min-size", type=int, default=16)
    a = ap.parse_args()

    out = a.out or os.path.join(a.dataset, "crops")
    names = open(os.path.join(a.dataset, "classes.txt"), encoding="utf-8").read().split()
    rows, n_img = [], 0
    lab_root = os.path.join(a.dataset, "labels")
    for cls_folder in sorted(os.listdir(lab_root)):
        d = os.path.join(lab_root, cls_folder)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".txt"):
                continue
            stem = f[:-4]
            img_path = next((os.path.join(a.dataset, "images", cls_folder, stem + e) for e in sorted(IMG_EXT)
                             if os.path.exists(os.path.join(a.dataset, "images", cls_folder, stem + e))), None)
            if img_path is None:
                print(f"[crop] thiếu ảnh cho {cls_folder}/{f}")
                continue
            img = load_image(img_path)
            cls, boxes = read_yolo_txt(os.path.join(d, f), img.width, img.height)
            if not boxes:
                continue
            n_img += 1
            rows += save_crops(img, boxes, [names[c] for c in cls], out, cls_folder, stem,
                               os.path.relpath(img_path, a.dataset).replace(os.sep, "/"),
                               main_flags=[names[c] != OTHER_CLASS for c in cls], pad=a.pad, min_size=a.min_size)
    write_crops_csv(out, rows)
    n_main = sum(1 for r in rows if r["is_main"] == 1)
    print(f"[crop] {n_img} ảnh -> {len(rows)} ảnh cắt ({n_main} món chính, {len(rows) - n_main} food_other) vào {out}")


if __name__ == "__main__":
    main()
