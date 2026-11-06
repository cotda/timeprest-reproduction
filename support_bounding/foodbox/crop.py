"""Cắt từng box thành file ảnh riêng: cắt ĐÚNG vùng box model khoanh được, từ ảnh gốc (không có khung vẽ)."""
import csv
import os
import shutil

# box_*: box model khoanh (toạ độ pixel thật); x1..y2: vùng đã cắt (= box làm tròn, kẹp trong ảnh; nới thêm nếu pad > 0)
CROP_FIELDS = ["crop", "image", "k", "class", "is_main", "score", "box_x1", "box_y1", "box_x2", "box_y2",
               "x1", "y1", "x2", "y2", "crop_w", "crop_h"]


def crop_box(img, box, pad=0.0):
    """Cắt 1 box (x1, y1, x2, y2 pixel). Mặc định pad=0: cắt đúng box; pad > 0 nới mỗi phía pad x cạnh box. Kẹp trong ảnh."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    l, t = max(0, round(x1 - dx)), max(0, round(y1 - dy))
    r, b = min(img.width, round(x2 + dx)), min(img.height, round(y2 + dy))
    return img.crop((l, t, r, b)), (l, t, r, b)


def save_crops(img, boxes, class_names, out_dir, cls_folder, stem, image_rel, scores=None, main_flags=None,
               pad=0.0, min_size=16, quality=95):
    """Ghi <out_dir>/<cls_folder>/<stem>_<k>.jpg cho từng box; trả list dòng cho crops.csv.

    Box nhỏ hơn min_size pixel (cạnh ngắn) bị bỏ qua: ảnh cắt quá nhỏ không dùng được.
    """
    rows = []
    os.makedirs(os.path.join(out_dir, cls_folder), exist_ok=True)
    for k, b in enumerate(boxes):
        crop, (l, t, r, bt) = crop_box(img, b, pad)
        if min(crop.size) < min_size:
            continue
        name = f"{cls_folder}/{stem}_{k}.jpg"
        crop.save(os.path.join(out_dir, *name.split("/")), quality=quality)
        rows.append({"crop": name, "image": image_rel, "k": k, "class": class_names[k],
                     "is_main": int(main_flags[k]) if main_flags is not None else "",
                     "score": round(float(scores[k]), 4) if scores is not None else "",
                     "box_x1": round(float(b[0]), 1), "box_y1": round(float(b[1]), 1),
                     "box_x2": round(float(b[2]), 1), "box_y2": round(float(b[3]), 1),
                     "x1": l, "y1": t, "x2": r, "y2": bt, "crop_w": crop.width, "crop_h": crop.height})
    return rows


def write_crops_csv(out_dir, rows):
    with open(os.path.join(out_dir, "crops.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CROP_FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_image_folder(src, crops, crops_dir, dest_dir, base):
    """Ghi bộ file của 1 ảnh vào dest_dir: <base>.<đuôi> (ảnh gốc, chép nguyên), <base>_bbox.jpg (khung đúng box model,
    đánh số i), <base>_<i>.jpg (ảnh cắt của khung i, i từ 1). crops: các dòng crops.csv của ảnh đó. Trả list dòng index."""
    from .data import load_image
    from .viz import color_for, draw_boxes
    crops = sorted(crops, key=lambda r: int(r["k"]))
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dest_dir, base + os.path.splitext(src)[1].lower()))
    boxes = [[float(r[c]) for c in ("box_x1", "box_y1", "box_x2", "box_y2")] for r in crops]
    draw_boxes(load_image(src), boxes, labels=[str(i) for i in range(1, len(crops) + 1)],
               colors=[color_for(i) for i in range(len(crops))]).save(os.path.join(dest_dir, f"{base}_bbox.jpg"),
                                                                       quality=92)
    index = []
    for i, r in enumerate(crops, 1):
        name = f"{base}_{i}.jpg"
        shutil.copy2(os.path.join(crops_dir, *r["crop"].split("/")), os.path.join(dest_dir, name))
        index.append({"folder": os.path.basename(dest_dir), "file": name, "is_main": r["is_main"], "score": r["score"],
                      "x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"]})
    return index
