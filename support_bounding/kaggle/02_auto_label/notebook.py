# %% [markdown]
# # 02 — Tự gán bounding box cho ảnh món ăn (tập con Food-101)
# - Input dạng `<tên_món>/*.jpg` (tự dò). Có dataset thật của nhóm: đổi `dataset_sources` trong `kernel-metadata.json`
#   (và `input_dirs` bên dưới nếu muốn chỉ định thư mục), không cần sửa gì khác.
# - Output trong `/kaggle/working/output/`: `images/`, `labels/` (YOLO), `visualized/`, `classes.txt`, `data.yaml`,
#   `train.txt`/`val.txt`, `report.csv` (mọi ảnh), `failed.csv` (ảnh không có box).
# - Class theo tên thư mục; box không phải món chính (ảnh nhiều món) là `food_other`.
# - File nguồn: `support_bounding/kaggle/02_auto_label/notebook.py` (build bằng `tools/build_notebook.py`).

# %%
# Cài Ultralytics (YOLO-World, SAM 2.1) với phiên bản cố định; torch/transformers dùng bản có sẵn của Kaggle
!pip install -q ultralytics==8.4.170
import sys, torch, transformers, ultralytics, PIL
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| ultralytics", ultralytics.__version__, "| pillow", PIL.__version__)
print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %% [foodbox]

# %% [markdown]
# ## Tham số
# `model`, `conf`, `prompts` lấy theo kết quả bước 01. Ảnh có `max_conf < review_conf` hoặc quá nhiều box sẽ được đánh dấu cần xem lại.

# %%
# Toàn bộ tham số của lần chạy
CFG = dict(
    data_root="/kaggle/input",   # nơi Kaggle gắn dataset (chỉ đọc)
    input_dirs=None,             # None = tự dò mọi thư mục dạng <món>/*.jpg; hoặc list đường dẫn cụ thể
    classes=None,                # None = chọn ngẫu nhiên n_classes món; hoặc list tên thư mục món
    n_classes=10,                # None = tất cả món
    per_class=20,                # None = tất cả ảnh của mỗi món
    seed=0,
    model="gdino",               # gdino | owlv2 | yoloworld
    prompts=["food"],
    conf=0.35,                   # ngưỡng giữ box món chính (v2.1: 97% ảnh có box, 94.8% box chính đúng; 0.45 chỉ 78.5% ảnh có box)
    other_conf=0.45,             # ngưỡng riêng cho box food_other (cao hơn để bớt nhận nhầm đồ trang trí)
    sam=False,                   # True = siết box bằng mask SAM 2.1 (01 v1.1: không tăng điểm, chậm thêm 0.3 s/ảnh)
    nms_iou=0.5,                 # NMS: bỏ box trùng IoU > nms_iou
    min_area=0.002, max_area=1.0,   # lọc box quá nhỏ (tỉ lệ so với ảnh); 1.0 = giữ box cả ảnh (ảnh chụp cận)
    drop_container=False,        # True = bỏ box chứa >= 2 box khác (v1: xoá nhầm sushi, chả giò; xem NOTES.md)
    main_rule="largest",         # box món chính: largest (lớn nhất) | score (điểm cao nhất)
    review_conf=None,            # None = conf + 0.1
    max_boxes=4,                 # nhiều box hơn thế thì đánh dấu xem lại
    val_frac=0.2,                # tỉ lệ ảnh val mỗi món trong train.txt / val.txt
    timeout_min=60,
    out="/kaggle/working/output",
)

# %%
# Nạp thư viện, tạo thư mục output
import json, os, time
import pandas as pd
from foodbox import data, viz
from foodbox.autolabel import build_dataset
from foodbox.run_detect import launch, load_preds
OUT = CFG["out"]
WORK = "/kaggle/working/tmp_preds"  # kết quả thô của detector (không nằm trong bộ nhãn)
for d in [OUT, f"{WORK}/logs"]:
    os.makedirs(d, exist_ok=True)
T0 = time.time()

# %% [markdown]
# ## 1. Dò dataset, chọn tập con

# %%
# In cấu trúc input, tìm các thư mục <món>/*.jpg và chọn ảnh
data.print_tree(CFG["data_root"], max_depth=4)
roots = CFG["input_dirs"] or data.find_class_roots(CFG["data_root"])
assert roots, "Không tìm thấy thư mục dạng <tên_món>/*.jpg trong input"
print("Thư mục gốc dùng:", roots)
class_images = data.list_class_images(roots)
print(f"{len(class_images)} món, {sum(map(len, class_images.values()))} ảnh")
items = data.sample_images(class_images, CFG["classes"], CFG["n_classes"], CFG["per_class"], CFG["seed"])
seen = {}
for it in items:  # trùng tên file trong cùng món (vd. từ train/ và valid/) -> thêm hậu tố
    n = seen.get(it["key"], 0)
    seen[it["key"]] = n + 1
    if n:
        it["stem"] += f"_{n}"
        it["key"] = f"{it['class']}/{it['stem']}"
with open(f"{WORK}/items.json", "w") as f:
    json.dump(items, f)
print(f"Tập con: {len(items)} ảnh, món: {sorted({it['class'] for it in items})}")

# %% [markdown]
# ## 2. Chạy model trên 2 GPU (mỗi GPU một nửa số ảnh)

# %%
# Chia ảnh cho các GPU (shard), chạy song song, in log cuối
n_gpu = max(torch.cuda.device_count(), 1)
jobs = []
for g in range(n_gpu):
    args = ["--items", f"{WORK}/items.json", "--models", CFG["model"], "--prompts", "|".join(CFG["prompts"]),
            "--out", f"{WORK}/preds", "--shard", str(g), "--nshards", str(n_gpu)] + (["--sam"] if CFG["sam"] else [])
    jobs.append((g, args, f"{WORK}/logs/detect_gpu{g}.log"))
codes = launch(jobs, timeout_min=CFG["timeout_min"], cwd="/tmp/foodbox_models")  # trọng số model không vào output
for _, _, log in jobs:
    print(f"----- {log}")
    print("\n".join(open(log, errors="replace").read().splitlines()[-8:]))
variant = CFG["model"] + ("+sam" if CFG["sam"] else "")
preds = load_preds(f"{WORK}/preds")
assert variant in preds, f"Không có kết quả {variant}: xem log ở trên"
results = preds[variant]["images"]
print(f"{len(results)}/{len(items)} ảnh có kết quả; thời gian detect {(time.time() - T0) / 60:.1f} phút")

# %% [markdown]
# ## 3. Hậu xử lý và xuất bộ nhãn
# NMS (box trùng) → lọc diện tích (quá nhỏ/quá to) → bỏ box ôm cả khay → gán class → ghi ảnh, nhãn, hình vẽ, báo cáo.

# %%
# Ghi bộ nhãn YOLO hoàn chỉnh
report, failed = build_dataset(items, results, OUT, conf=CFG["conf"], nms_iou=CFG["nms_iou"],
                               min_area=CFG["min_area"], max_area=CFG["max_area"],
                               drop_container=CFG["drop_container"], main_rule=CFG["main_rule"],
                               review_conf=CFG["review_conf"], max_boxes=CFG["max_boxes"],
                               val_frac=CFG["val_frac"], seed=CFG["seed"], other_conf=CFG["other_conf"])
rep = pd.DataFrame(report)
print(f"Ảnh có ít nhất 1 box: {(rep.n_boxes > 0).mean():.1%} | box/ảnh trung bình: {rep.n_boxes.mean():.2f} | "
      f"cần xem lại: {rep.needs_review.sum()} ảnh | failed: {len(failed)}")
print("\nTheo món:")
print(rep.groupby("class").agg(n_img=("image", "size"), box_per_img=("n_boxes", "mean"),
                               mean_conf=("max_conf", lambda s: pd.to_numeric(s, errors="coerce").mean()),
                               review=("needs_review", "sum")).round(2).to_string())
print("\nLý do cần xem lại:", rep.reasons[rep.reasons != ""].str.split(";").explode().value_counts().to_dict())

# %%
# Lưới hình mẫu (2 ảnh mỗi món) để xem nhanh, và tóm tắt cuối
tiles = []
for c in sorted({it["class"] for it in items}):
    for it in [x for x in items if x["class"] == c][:2]:
        p = f"{OUT}/visualized/{it['class']}/{it['stem']}.jpg"
        if os.path.exists(p):
            tiles.append(viz.titled(viz.fit(data.load_image(p), 360), it["key"]))
viz.grid(tiles, ncols=4).save(f"{OUT}/../preview_grid.jpg", quality=80)
with open(f"{OUT}/../run_summary.json", "w") as f:
    json.dump({"config": CFG, "roots": roots, "n_images": len(items), "minutes": (time.time() - T0) / 60,
               "exit_codes": codes, "detector_meta": preds[variant]["meta"]}, f, indent=1, default=str)
print(f"Xong sau {(time.time() - T0) / 60:.1f} phút")
!du -sh {OUT} {OUT}/*
