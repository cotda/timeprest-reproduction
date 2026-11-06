# %% [markdown]
# # 01 — So sánh model auto bounding box trên UECFOOD-256 (tập con)
# - Dữ liệu: `kaggleolap/uecfood256-mini` (10 món × 20 ảnh, có box thật trong `bb_info.txt`).
# - Model: Grounding DINO, OWLv2, YOLO-World; mỗi model thêm biến thể `+sam` (box lấy từ mask SAM 2.1, ôm sát thức ăn hơn).
# - Đo: AP@0.5, Precision/Recall/F1, IoU trung bình, tốc độ/ảnh, VRAM. Hình so sánh: box thật xanh lá, box model đỏ.
# - Dùng 2 GPU: mỗi GPU chạy một nhóm model cùng lúc.
# - File nguồn: `support_bounding/kaggle/01_compare/notebook.py` (build bằng `tools/build_notebook.py`).

# %%
# Cài Ultralytics (YOLO-World, SAM 2.1) với phiên bản cố định; torch/transformers dùng bản có sẵn của Kaggle
!pip install -q ultralytics==8.4.170
import sys, torch, transformers, ultralytics, PIL
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| ultralytics", ultralytics.__version__, "| pillow", PIL.__version__)
print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
!nvidia-smi --query-gpu=index,name,memory.total --format=csv

# %% [foodbox]

# %% [markdown]
# ## Tham số
# Sửa ở đây (không sửa trong thư viện). `gpu_jobs`: GPU nào chạy model nào.

# %%
# Toàn bộ tham số của lần chạy
CFG = dict(
    data_root="/kaggle/input",                           # nơi Kaggle gắn dataset (chỉ đọc)
    uec_class_ids=[1, 7, 11, 18, 23, 36, 87, 98, 206, 209],  # id món trong category.txt
    per_class=20,                                        # số ảnh mỗi món
    seed=0,
    prompts=["food"],                                    # câu chữ mô tả thứ cần tìm
    gpu_jobs={0: ["gdino"], 1: ["owlv2", "yoloworld"]},  # chia model cho 2 GPU
    sam=True,                                            # thêm biến thể <model>+sam
    iou_thr=0.5,                                         # IoU để tính là phát hiện đúng
    nms_iou=0.5, min_area=0.002, max_area=1.0,           # hậu xử lý: NMS, lọc box quá nhỏ (tỉ lệ so với ảnh)
    drop_container=False,                                # True = bỏ box chứa >= 2 box khác (v1: làm giảm AP, xem NOTES.md)
    timeout_min=45,                                      # quá thời gian thì dừng job (tránh treo mất quota)
    out="/kaggle/working/output",
)

# %%
# Nạp thư viện, tạo thư mục output
import json, os, time
import numpy as np, pandas as pd
from foodbox import data, labels, metrics, viz
from foodbox import postprocess as pp
from foodbox.run_detect import as_arrays, launch, load_preds
OUT = CFG["out"]
for sub in ["logs", "preds", "gt_labels", "compare"]:
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
T0 = time.time()

# %% [markdown]
# ## 1. Dò dataset và lấy box thật
# Mỗi bản upload có cấu trúc khác nhau nên in cây thư mục rồi tự tìm thư mục có `category.txt` + `<id>/bb_info.txt`.

# %%
# In cấu trúc thư mục input và tìm gốc UECFOOD-256
data.print_tree(CFG["data_root"], max_depth=4)
uec_root = data.find_uec_root(CFG["data_root"])
assert uec_root, "Không tìm thấy UECFOOD-256 (category.txt + 1/bb_info.txt) trong /kaggle/input"
cats = data.read_uec_categories(uec_root)
uec_boxes = data.read_uec_boxes(uec_root)
print(f"UEC root: {uec_root} | {len(cats)} món | {len(uec_boxes)} ảnh có box trong bb_info.txt")

# %%
# Chọn tập con (10 món x 20 ảnh) và chuyển box thật sang YOLO (gt_labels/<món>/<ảnh>.txt)
class_images = data.uec_class_images(uec_root, CFG["uec_class_ids"], cats)
items = data.sample_images(class_images, n_classes=None, per_class=CFG["per_class"], seed=CFG["seed"])
cat_of_class = {data.safe_name(cats[i]): i for i in CFG["uec_class_ids"]}
gts = data.convert_uec_to_yolo(items, uec_boxes, cat_of_class, f"{OUT}/gt_labels")
with open(f"{OUT}/items.json", "w") as f:
    json.dump(items, f)
n_box = [len(g) for g in gts.values()]
print(f"{len(items)} ảnh, {sum(n_box)} box thật, ảnh nhiều box: {sum(n > 1 for n in n_box)}, ảnh không có box: {n_box.count(0)}")
print("ảnh có tag EXIF xoay (box thật có thể lệch):", [it["key"] for it in items if data.exif_rotated(it["path"])])
print(pd.Series({c: len(v) for c, v in class_images.items()}, name="ảnh/món trong dataset").to_string())

# %% [markdown]
# ## 2. Chạy các model trên 2 GPU
# Mỗi GPU là một tiến trình riêng (`python -m foodbox.run_detect`); log từng tiến trình ở `logs/`.

# %%
# Mở các tiến trình detect song song (nếu chỉ có 1 GPU thì dồn hết vào GPU 0)
n_gpu = torch.cuda.device_count()
gpu_jobs = CFG["gpu_jobs"] if n_gpu >= len(CFG["gpu_jobs"]) else {0: sum(CFG["gpu_jobs"].values(), [])}
jobs = []
for gpu, models in gpu_jobs.items():
    args = ["--items", f"{OUT}/items.json", "--models", ",".join(models), "--prompts", "|".join(CFG["prompts"]),
            "--out", f"{OUT}/preds"] + (["--sam"] if CFG["sam"] else [])
    jobs.append((gpu, args, f"{OUT}/logs/detect_gpu{gpu}.log"))
codes = launch(jobs, timeout_min=CFG["timeout_min"], cwd="/tmp/foodbox_models")  # trọng số model không vào output
for _, _, log in jobs:  # in 15 dòng cuối log của mỗi GPU
    print(f"----- {log}")
    print("\n".join(open(log, encoding="utf-8", errors="replace").read().splitlines()[-15:]))
print(f"Thời gian detect: {(time.time() - T0) / 60:.1f} phút")

# %% [markdown]
# ## 3. Đánh giá
# - `ap50`: AP@0.5 không phân biệt món (box nào là thức ăn), không phụ thuộc ngưỡng.
# - `ap30`: AP ở IoU 0.3. Box thật của UEC thường ôm cả bát/đĩa, nên box ôm sát thức ăn (đúng ý mình) bị phạt ở IoU 0.5.
# - `conf`: ngưỡng confidence cho F1 cao nhất của từng model (thang điểm mỗi model khác nhau); P/R/F1/IoU tính ở ngưỡng này.
# - `mean_iou`: mỗi box thật lấy IoU cao nhất với box dự đoán (0 nếu bỏ sót), lấy trung bình.
# - `area_ratio`: trung vị diện tích box dự đoán / box thật (của các cặp đúng); > 1 là box rộng hơn box thật (vd. ôm cả đĩa).
# - `main_acc_*`: tỉ lệ ảnh chọn đúng box món chính khi lấy box lớn nhất / điểm cao nhất.

# %%
# Hậu xử lý (NMS + lọc diện tích + bỏ box ôm cả khay) rồi tính chỉ số cho từng biến thể
preds = load_preds(f"{OUT}/preds")
assert preds, "Không có file kết quả nào trong preds/: xem log detect ở trên"
rows, pp_preds, per_class = [], {}, []
for variant, r in sorted(preds.items()):
    P = {}
    for key, x in r["images"].items():
        b, s = as_arrays(x)
        P[key] = pp.postprocess(b, s, max(x["w"], 1), max(x["h"], 1), conf=0.0, nms_iou=CFG["nms_iou"],
                                min_area=CFG["min_area"], max_area=CFG["max_area"],
                                drop_container=CFG["drop_container"], max_det=100)
    pp_preds[variant] = P
    m = metrics.evaluate(P, gts, CFG["iou_thr"])
    ap30 = metrics.evaluate(P, gts, 0.3)["ap50"]  # AP ở IoU 0.3: box thật UEC ôm cả bát/đĩa, box ôm sát thức ăn bị phạt ở IoU 0.5
    ok = [x for x in r["images"].values() if "error" not in x]
    rows.append({"model": variant, **m, "ap30": ap30,
                 "sec_per_img": float(np.mean([x["time_s"] for x in ok])) if ok else float("nan"),
                 "peak_vram_mb": max(mm["peak_vram_mb"] for mm in r["meta"]),
                 "load_s": max(mm["load_s"] for mm in r["meta"]),
                 "n_errors": len(r["images"]) - len(ok)})
    for c in sorted(class_images):  # AP theo từng món: món nào khó
        g = {k: v for k, v in gts.items() if k.startswith(c + "/")}
        mc = metrics.evaluate(P, g, CFG["iou_thr"], conf=m["conf"])
        per_class.append({"model": variant, "class": c, "ap50": mc["ap50"], "recall": mc["recall"],
                          "precision": mc["precision"], "mean_iou": mc["mean_iou"]})
df = pd.DataFrame(rows).sort_values("ap50", ascending=False)
df.to_csv(f"{OUT}/results.csv", index=False)
pc = pd.DataFrame(per_class)
pc.to_csv(f"{OUT}/results_per_class.csv", index=False)
cols = ["model", "ap50", "ap30", "precision", "recall", "f1", "mean_iou", "area_ratio", "main_acc_largest", "main_acc_score",
        "conf", "sec_per_img", "peak_vram_mb", "images_no_box", "n_errors"]
print(df[cols].round(3).to_string(index=False))
print("\nAP@0.5 theo món:")
print(pc.pivot(index="class", columns="model", values="ap50").round(2).to_string())

# %% [markdown]
# ## 4. Hình so sánh
# Mỗi ảnh một hình: mỗi ô là một biến thể, box thật xanh lá, box model đỏ (ở ngưỡng `conf` của biến thể đó).

# %%
# Vẽ hình so sánh cho từng ảnh (compare/<món>/<ảnh>.jpg) và một lưới tóm tắt (compare_grid.jpg)
variants = list(df["model"])
best_conf = dict(zip(df["model"], df["conf"]))
summary_tiles = []
for k, it in enumerate(items):
    img = data.load_image(it["path"])
    gt_b = [g[2] for g in gts[it["key"]]]
    tiles = []
    for v in variants:
        b, s = pp_preds[v].get(it["key"], (np.zeros((0, 4)), np.zeros(0)))
        keep = s >= best_conf[v]
        im = viz.draw_boxes(img, gt_b, colors=[viz.GREEN] * len(gt_b))
        im = viz.draw_boxes(im, b[keep], labels=[f"{x:.2f}" for x in s[keep]], colors=[viz.RED] * int(keep.sum()))
        tiles.append(viz.titled(viz.fit(im, 360), v))
    panel = viz.grid(tiles, ncols=3)
    os.makedirs(f"{OUT}/compare/{it['class']}", exist_ok=True)
    panel.save(f"{OUT}/compare/{it['class']}/{it['stem']}.jpg", quality=85)
    if k % CFG["per_class"] < 2:  # 2 ảnh đầu mỗi món vào lưới tóm tắt
        summary_tiles.append(viz.titled(viz.fit(viz.grid(tiles, ncols=len(tiles)), 1800), it["key"], size=22))
viz.grid(summary_tiles, ncols=1).save(f"{OUT}/compare_grid.jpg", quality=80)
print(f"Đã lưu {len(items)} hình so sánh vào {OUT}/compare/ và compare_grid.jpg")

# %%
# Tóm tắt cuối: ghi summary.json, in dung lượng output
summary = {"config": CFG, "uec_root": uec_root, "n_images": len(items), "minutes": (time.time() - T0) / 60,
           "results": df[cols].round(4).to_dict(orient="records"), "exit_codes": codes}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=str)
print(f"Xong sau {summary['minutes']:.1f} phút")
!du -sh {OUT} {OUT}/*
