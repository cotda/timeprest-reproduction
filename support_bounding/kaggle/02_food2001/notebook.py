# %% [markdown]
# # Gán bounding box cho dataset món ăn của nhóm (2001_food_dataset, 300 món × 5 ảnh)
# - Input: dataset Kaggle chứa thư mục `images/<tên_món>/*.jpg` (tự dò, file `food_dataset.csv` bỏ qua).
# - Model + tham số giống module `foodbox`: Grounding DINO base, prompt "food", box ≥ 0.35, box phụ ≥ 0.45, và GỘP các
#   miếng cùng đĩa/thớt thành 1 khung (chạy thêm 1 lượt tìm đĩa/bát/khay/thớt; NOTES v5). Tắt: `group=False`.
# - Kết quả: 1 file `/kaggle/working/ketqua_2001.zip` gồm
#   - `theo_anh/<món>/001.jpg, 001_bbox.jpg (khung đánh số), 001_1.jpg, 001_2.jpg ... (cắt đúng từng khung)`
#   - `yolo/`: bộ nhãn YOLO để train (`images/`, `labels/`, `classes.txt`, `data.yaml`, `train.txt`, `val.txt`),
#     `visualized/`, `crops/`, `report.csv` (mọi ảnh, cờ cần xem lại), `failed.csv` (ảnh không có box).
# - Chạy thử nhanh trước: đặt `per_class=1` (300 ảnh) hoặc `n_classes=5`.
# - File nguồn: `support_bounding/kaggle/02_food2001/notebook.py` (build bằng `tools/build_notebook.py`).

# %%
# Môi trường: Grounding DINO chạy bằng torch + transformers có sẵn của Kaggle (không cần pip install)
import sys, torch, transformers, PIL
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| pillow", PIL.__version__)
print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %% [foodbox]

# %% [markdown]
# ## Tham số

# %%
# Toàn bộ tham số của lần chạy
CFG = dict(
    data_root="/kaggle/input",   # nơi Kaggle gắn dataset (chỉ đọc)
    input_dirs=None,             # None = tự dò thư mục images/; hoặc list đường dẫn cụ thể
    n_classes=None,              # None = tất cả 300 món (thử nhanh: 5)
    per_class=None,              # None = tất cả ảnh mỗi món (thử nhanh: 1)
    seed=0,
    prompts=["food"],
    conf=0.35,                   # ngưỡng giữ box (box lớn nhất = món theo tên thư mục)
    other_conf=0.45,             # ngưỡng cho box phụ (food_other)
    nms_iou=0.5, min_area=0.002, max_area=1.0,
    max_boxes=4,                 # nhiều box hơn thế thì đánh dấu xem lại trong report.csv
    val_frac=0.2,                # chia train/val 80/20 theo từng món
    crop_pad=0.0,                # 0 = cắt đúng box
    group=True,                  # gộp các miếng cùng đĩa/thớt thành 1 khung (chậm gấp ~2 vì thêm lượt tìm vật chứa)
    timeout_min=90,              # quá thời gian thì dừng job detect (kết quả đã lưu theo đợt vẫn dùng được)
    zip_name="ketqua_2001.zip",
)

# %%
# Nạp thư viện, tạo thư mục
import csv, json, os, random, shutil, time, zipfile
import pandas as pd
from foodbox import data, viz
from foodbox.autolabel import build_dataset
from foodbox.crop import write_image_folder
from foodbox.run_detect import launch, load_preds
W = "/kaggle/working"
OUT, FOLDERS, WORK = f"{W}/yolo", f"{W}/theo_anh", f"{W}/tmp_preds"
for d in [OUT, FOLDERS, f"{WORK}/logs"]:
    os.makedirs(d, exist_ok=True)
T0 = time.time()

# %% [markdown]
# ## 1. Dò dataset

# %%
# In cấu trúc input, tìm thư mục <món>/*.jpg, chọn ảnh (mặc định: tất cả)
data.print_tree(CFG["data_root"], max_depth=4)
roots = CFG["input_dirs"] or data.find_class_roots(CFG["data_root"])
assert roots, "Không tìm thấy thư mục dạng <tên_món>/*.jpg trong input: kiểm tra đã Add Input dataset chưa"
print("Thư mục gốc dùng:", roots)
class_images = data.list_class_images(roots)
print(f"{len(class_images)} món, {sum(map(len, class_images.values()))} ảnh; vài món: {list(class_images)[:5]}")
items = data.dedupe_keys(data.sample_images(class_images, None, CFG["n_classes"], CFG["per_class"], CFG["seed"]))
with open(f"{WORK}/items.json", "w") as f:
    json.dump(items, f)
print(f"Chạy {len(items)} ảnh")

# %% [markdown]
# ## 2. Chạy Grounding DINO trên 2 GPU (mỗi GPU một nửa số ảnh, lưu theo đợt mỗi 50 ảnh; + lượt tìm vật chứa nếu group)

# %%
# Mở 1 tiến trình mỗi GPU, in log cuối
n_gpu = max(torch.cuda.device_count(), 1)
jobs = [(g, ["--items", f"{WORK}/items.json", "--models", "gdino", "--prompts", "|".join(CFG["prompts"]),
             "--out", f"{WORK}/preds", "--shard", str(g), "--nshards", str(n_gpu)]
         + (["--containers"] if CFG["group"] else []), f"{WORK}/logs/detect_gpu{g}.log")
        for g in range(n_gpu)]
codes = launch(jobs, timeout_min=CFG["timeout_min"], cwd="/tmp/foodbox_models")  # trọng số model không vào output
for _, _, log in jobs:
    print(f"----- {log}")
    print("\n".join(open(log, encoding="utf-8", errors="replace").read().splitlines()[-6:]))
preds = load_preds(f"{WORK}/preds")
assert "gdino" in preds, "Không có kết quả: xem log ở trên"
results = preds["gdino"]["images"]
print(f"{len(results)}/{len(items)} ảnh có kết quả; detect {(time.time() - T0) / 60:.1f} phút")

# %% [markdown]
# ## 3. Bộ nhãn YOLO + ảnh cắt (thư mục yolo/)

# %%
# Hậu xử lý → nhãn YOLO, ảnh vẽ box, ảnh cắt, report.csv, failed.csv
report, failed = build_dataset(items, results, OUT, conf=CFG["conf"], nms_iou=CFG["nms_iou"],
                               min_area=CFG["min_area"], max_area=CFG["max_area"], max_boxes=CFG["max_boxes"],
                               val_frac=CFG["val_frac"], seed=CFG["seed"], other_conf=CFG["other_conf"],
                               crops=True, crop_pad=CFG["crop_pad"], group=CFG["group"])
rep = pd.DataFrame(report)
print(f"Ảnh có box: {(rep.n_boxes > 0).sum()}/{len(items)} ({(rep.n_boxes > 0).mean():.1%}) | "
      f"box/ảnh {rep.n_boxes.mean():.2f} | cần xem lại {rep.needs_review.sum()} | failed {len(failed)}")
print("Lý do cần xem lại:", rep.reasons[rep.reasons != ""].str.split(";").explode().value_counts().to_dict())
print("Số box mỗi ảnh:", rep.n_boxes.value_counts().sort_index().to_dict())

# %% [markdown]
# ## 4. Thư mục theo từng ảnh: 001.jpg, 001_bbox.jpg, 001_1.jpg, 001_2.jpg ... (theo_anh/<món>/)

# %%
# Mỗi ảnh có box: ảnh gốc + ảnh có khung đánh số + ảnh cắt đánh số từ 1 (giống đầu ra của foodbox-detect)
rows = list(csv.DictReader(open(f"{OUT}/crops/crops.csv", encoding="utf-8")))
by_img = {}
for r in rows:
    by_img.setdefault(r["image"], []).append(r)
index = []
for img_rel, crops in sorted(by_img.items()):  # img_rel = images/<món>/<ảnh>.jpg
    cls, fname = img_rel.split("/")[-2:]
    index += write_image_folder(f"{OUT}/{img_rel}", crops, f"{OUT}/crops", f"{FOLDERS}/{cls}",
                                os.path.splitext(fname)[0])
pd.DataFrame(index).to_csv(f"{FOLDERS}/index.csv", index=False)
print(f"{len(by_img)} ảnh, {len(index)} ảnh cắt -> {FOLDERS}/<món>/")

# %%
# Hình xem nhanh: 40 ảnh có khung chọn ngẫu nhiên (seed 0)
picks = sorted(random.Random(0).sample(sorted(by_img), min(40, len(by_img))))
tiles = []
for img_rel in picks:
    cls, fname = img_rel.split("/")[-2:]
    stem = os.path.splitext(fname)[0]
    tiles.append(viz.titled(viz.fit(data.load_image(f"{FOLDERS}/{cls}/{stem}_bbox.jpg"), 320), f"{cls}/{stem}", 14))
viz.grid(tiles, ncols=5).save(f"{W}/preview_grid.jpg", quality=80)

# %% [markdown]
# ## 5. Nén kết quả thành 1 file zip (tải về 1 file thay vì hàng nghìn file)

# %%
# Nén yolo/ + theo_anh/ (không nén lại ảnh JPG: ZIP_STORED cho nhanh), rồi xoá thư mục đã nén để output gọn
with open(f"{W}/run_summary.json", "w") as f:
    json.dump({"config": CFG, "roots": roots, "n_images": len(items), "n_with_box": int((rep.n_boxes > 0).sum()),
               "n_crops": len(index), "minutes": (time.time() - T0) / 60, "exit_codes": codes,
               "detector_meta": preds["gdino"]["meta"]}, f, indent=1, default=str)
zpath = f"{W}/{CFG['zip_name']}"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
    for base in [OUT, FOLDERS]:
        for root, _, files in os.walk(base):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, W))
    z.write(f"{W}/run_summary.json", "run_summary.json")
for d in [OUT, FOLDERS]:
    shutil.rmtree(d)
print(f"Xong sau {(time.time() - T0) / 60:.1f} phút: {zpath} ({os.path.getsize(zpath) / 2**20:.0f} MB)")
!ls -la /kaggle/working
