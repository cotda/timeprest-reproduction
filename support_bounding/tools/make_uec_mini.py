"""Cắt bản UECFOOD-256 mini từ file zip gốc (không giải nén hết, không sửa file gốc), để upload lên Kaggle.

Giữ: category.txt, README.txt, TOÀN BỘ <id>/bb_info.txt (ảnh nhiều món cần box từ mọi thư mục món),
và per_class ảnh ngẫu nhiên (seed cố định) của mỗi món được chọn.

Dùng:
  kaggle datasets download rkuo2000/uecfood256 -p <thư_mục_tải>
  python tools/make_uec_mini.py <thư_mục_tải>/uecfood256.zip data/uecfood256-mini --classes 1,7,11,18,23,36,87,98,206,209
  kaggle datasets create -p data/uecfood256-mini      (lần sau: kaggle datasets version -p ... -m "...")
"""
import argparse
import json
import os
import random
import zipfile

DEFAULT_CLASSES = "1,7,11,18,23,36,87,98,206,209"  # rice, sushi, bibimbap, pizza, ramen, miso soup, green salad, fries, pho, spring rolls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("out")
    ap.add_argument("--classes", default=DEFAULT_CLASSES, help="id món (theo category.txt), cách nhau bởi dấu phẩy")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slug", default="uecfood256-mini")
    ap.add_argument("--owner", default="kaggleolap")
    a = ap.parse_args()

    z = zipfile.ZipFile(a.zip)
    names = z.namelist()
    prefix = names[0].split("/")[0] + "/"  # "UECFOOD256/"
    keep = [n for n in names if n.endswith(("bb_info.txt", "category.txt", "README.txt"))]
    rng = random.Random(a.seed)
    classes = [int(c) for c in a.classes.split(",")]
    for cid in classes:
        imgs = sorted(n for n in names if n.startswith(f"{prefix}{cid}/") and n.lower().endswith(".jpg"))
        keep += sorted(rng.sample(imgs, min(a.per_class, len(imgs))))

    total = 0
    for n in keep:
        dst = os.path.join(a.out, *n.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            data = z.read(n)
            f.write(data)
            total += len(data)
    meta = {"title": a.slug, "id": f"{a.owner}/{a.slug}", "licenses": [{"name": "other"}],
            "description": "Tập con UECFOOD-256 (Kawano & Yanai, 2014; http://foodcam.mobi/dataset256.html), "
                           f"món {a.classes}, {a.per_class} ảnh/món (seed {a.seed}) + toàn bộ bb_info.txt. "
                           "Chỉ dùng cho nghiên cứu, theo giấy phép của dataset gốc."}
    with open(os.path.join(a.out, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[mini] {len(keep)} file, {total / 2**20:.1f} MB -> {a.out}")


if __name__ == "__main__":
    main()
