"""Dò cấu trúc dataset, chọn tập con, đọc box thật của UECFOOD-256 (bb_info.txt) và chuyển sang YOLO."""
import os
import random
import re
from collections import deque

from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_DIRS = {"__MACOSX", "__pycache__"}


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMG_EXT


def _subdirs(d):
    try:
        return sorted(e.path for e in os.scandir(d)
                      if e.is_dir() and e.name not in SKIP_DIRS and not e.name.startswith("."))
    except OSError:
        return []


def _has_image(d, max_scan=200):
    # Chỉ xem tối đa max_scan mục đầu: thư mục món có hàng nghìn ảnh, không cần liệt kê hết
    try:
        with os.scandir(d) as it:
            for i, e in enumerate(it):
                if i >= max_scan:
                    break
                if e.is_file() and is_image(e.name):
                    return True
    except OSError:
        pass
    return False


def print_tree(root, max_depth=3, max_children=6):
    """In cây thư mục (chỉ thư mục, kèm số ảnh trực tiếp bên trong) để xem cấu trúc bản upload."""
    def walk(d, depth):
        subs = _subdirs(d)
        try:
            n_img = sum(1 for n in os.listdir(d) if is_image(n))
        except OSError:
            n_img = 0
        print("  " * depth + f"{os.path.basename(d) or d}/  ({len(subs)} thư mục con, {n_img} ảnh)")
        if depth >= max_depth:
            return
        for s in subs[:max_children]:
            walk(s, depth + 1)
        if len(subs) > max_children:
            print("  " * (depth + 1) + f"... còn {len(subs) - max_children} thư mục nữa")
    walk(root, 0)


def find_class_roots(root, max_depth=8, min_classes=2):
    """Tìm mọi thư mục dạng <thư_mục>/<tên_món>/*.jpg (vd. train/ và valid/ của Food-101 tiny).

    Không đi vào bên trong thư mục món (chứa ảnh) để không phải liệt kê hàng nghìn file.
    Kaggle gắn dataset ở /kaggle/input/datasets/<owner>/<slug>/..., nên thư mục món thường sâu 5-7 cấp
    (food101tiny: .../data/food-101-tiny/train/<món>, sâu 6) -> max_depth mặc định 8.
    """
    found = []
    queue = deque([(root, 0)])
    while queue:
        d, depth = queue.popleft()
        subs = _subdirs(d)
        with_img = [s for s in subs if _has_image(s)]
        if len(with_img) >= min_classes:
            found.append(d)
            continue
        if depth < max_depth:
            queue.extend((s, depth + 1) for s in subs if s not in with_img)
    return found


def list_class_images(class_roots):
    """Gộp ảnh của các thư mục gốc: {tên_món: [đường dẫn ảnh, ...]} (sắp xếp để chạy lại cho cùng kết quả)."""
    out = {}
    for r in class_roots:
        for s in _subdirs(r):
            imgs = sorted(os.path.join(s, n) for n in os.listdir(s) if is_image(n))
            if imgs:
                out.setdefault(os.path.basename(s), []).extend(imgs)
    return {k: sorted(v) for k, v in sorted(out.items())}


def sample_images(class_images, classes=None, n_classes=10, per_class=20, seed=0):
    """Chọn tập con: danh sách món (hoặc n_classes món ngẫu nhiên), mỗi món per_class ảnh (None = tất cả).

    Trả về list dict {key, class, path, stem}; key = "<món>/<tên file không đuôi>" dùng làm id ảnh.
    """
    rng = random.Random(seed)
    names = sorted(class_images)
    if classes is None:
        classes = names if n_classes is None or n_classes >= len(names) else sorted(rng.sample(names, n_classes))
    items = []
    for c in classes:
        imgs = class_images[c]
        pick = imgs if per_class is None or per_class >= len(imgs) else sorted(rng.sample(imgs, per_class))
        for p in pick:
            stem = os.path.splitext(os.path.basename(p))[0]
            items.append({"key": f"{c}/{stem}", "class": c, "path": p, "stem": stem})
    return items


def load_image(path):
    """Mở ảnh RGB, xoay theo EXIF (giống cv2.imread mà YOLO/Ultralytics và X-AnyLabeling dùng)."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def exif_rotated(path):
    """True nếu ảnh có tag EXIF xoay (box thật có thể lệch nếu người gán nhãn dùng ảnh chưa xoay)."""
    try:
        with Image.open(path) as im:
            return im.getexif().get(0x0112, 1) not in (1, None)
    except OSError:
        return False


# ----------------------------- UECFOOD-256 -----------------------------

def safe_name(name):
    """Tên món -> tên thư mục/class an toàn: 'chicken-'n'-egg on rice' -> 'chicken-n-egg_on_rice'."""
    s = re.sub(r"[^\w\- ]+", "", name.strip().lower())
    return re.sub(r"\s+", "_", s)


def find_uec_root(root, max_depth=8):
    """Tìm thư mục chứa category.txt và các thư mục <id>/bb_info.txt."""
    queue = deque([(root, 0)])
    while queue:
        d, depth = queue.popleft()
        if os.path.isfile(os.path.join(d, "category.txt")) and os.path.isfile(os.path.join(d, "1", "bb_info.txt")):
            return d
        if depth < max_depth:
            queue.extend((s, depth + 1) for s in _subdirs(d) if not _has_image(s))
    return None


def read_uec_categories(uec_root):
    """category.txt ('id<TAB>name') -> {id: name}."""
    cats = {}
    with open(os.path.join(uec_root, "category.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip().isdigit():
                cats[int(parts[0])] = parts[1].strip()
    return cats


def read_uec_boxes(uec_root):
    """Đọc mọi <id>/bb_info.txt -> {img_id: [(cat_id, x1, y1, x2, y2), ...]}.

    Ảnh nhiều món nằm trong nhiều thư mục món, mỗi thư mục chỉ ghi box của món đó, nên phải gộp
    tất cả bb_info.txt mới có đủ box thật cho một ảnh.
    """
    boxes = {}
    for d in _subdirs(uec_root):
        cat = os.path.basename(d)
        f = os.path.join(d, "bb_info.txt")
        if not cat.isdigit() or not os.path.isfile(f):
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                p = line.split()
                if len(p) != 5 or not p[0].isdigit():
                    continue  # bỏ dòng tiêu đề "img x1 y1 x2 y2"
                b = (int(cat), *(float(v) for v in p[1:]))
                if b not in boxes.setdefault(p[0], []):
                    boxes[p[0]].append(b)
    return boxes


def uec_class_images(uec_root, cat_ids, categories=None):
    """{tên_món_an_toàn: [ảnh trong thư mục <cat_id>]} cho các món được chọn."""
    categories = categories or read_uec_categories(uec_root)
    out = {}
    for cid in cat_ids:
        d = os.path.join(uec_root, str(cid))
        out[safe_name(categories[cid])] = sorted(os.path.join(d, n) for n in os.listdir(d) if is_image(n))
    return out


def clip_box(box, w, h):
    """Kẹp box (x1, y1, x2, y2) vào trong ảnh; trả None nếu box suy biến."""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((min(max(x1, 0.0), w), min(max(x2, 0.0), w)))
    y1, y2 = sorted((min(max(y1, 0.0), h), min(max(y2, 0.0), h)))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (x1, y1, x2, y2)


def uec_gt_for_item(item, uec_boxes, cat_of_class, w, h):
    """Box thật của 1 ảnh: list (is_main, cat_id, (x1, y1, x2, y2)); is_main = box của món theo thư mục."""
    main_cat = cat_of_class[item["class"]]
    out = []
    for cat, *b in uec_boxes.get(item["stem"], []):
        cb = clip_box(b, w, h)
        if cb is not None:
            out.append((cat == main_cat, cat, cb))
    return out


def convert_uec_to_yolo(items, uec_boxes, cat_of_class, out_dir, categories=None):
    """Chuyển box thật UECFOOD-256 sang nhãn YOLO: <out_dir>/<món>/<ảnh>.txt + classes.txt.

    categories=None: class = các món trong tập con (theo thư mục) + 'food_other' cho box của món khác
    (cùng quy ước với nhãn tự động). categories={id: tên}: giữ đủ 256 class gốc (class_id = id - 1).
    Trả {key: [(is_main, cat_id, box), ...]} để đánh giá; ghi thêm it['w'], it['h'] vào từng item.
    """
    from .labels import OTHER_CLASS, class_list, write_classes, write_yolo_txt
    if categories is None:
        names = class_list(sorted({it["class"] for it in items}))
        cid = {n: i for i, n in enumerate(names)}
        to_id = lambda it, is_main, cat: cid[it["class"]] if is_main else cid[OTHER_CLASS]
    else:
        names = [categories[k] for k in sorted(categories)]
        to_id = lambda it, is_main, cat: cat - 1
    gts = {}
    for it in items:
        with Image.open(it["path"]) as im:
            w, h = ImageOps.exif_transpose(im).size
        it["w"], it["h"] = w, h
        g = uec_gt_for_item(it, uec_boxes, cat_of_class, w, h)
        gts[it["key"]] = g
        write_yolo_txt(os.path.join(out_dir, it["class"], it["stem"] + ".txt"),
                       [to_id(it, m, c) for m, c, _ in g], [b for *_, b in g], w, h)
    os.makedirs(out_dir, exist_ok=True)
    write_classes(out_dir, names)
    return gts
