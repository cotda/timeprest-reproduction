# %% [markdown]
# # 03 — Thử text prompt khác cho Grounding DINO (xử lý box ôm cả đĩa)
# Bốn biến thể, cùng model Grounding DINO base:
# - `_food`: câu chữ "food." (như bước 02)
# - `_food_pb`: "food. plate. bowl. cup. tray." rồi bỏ box mà model gán cho đĩa/bát/cốc/khay
# - `_class`: tên món theo thư mục, vd. "french toast."
# - `_class_pb`: tên món + "plate. bowl. cup. tray."
#
# Chạy trên 2 tập: UECFOOD-256 mini (có box thật → tính AP) trên GPU 0, và đúng 200 ảnh Food-101 tiny của bước 02 trên GPU 1.
# File nguồn: `support_bounding/kaggle/03_prompts/notebook.py`.

# %%
# Môi trường (Grounding DINO chạy qua transformers có sẵn của Kaggle, không cần cài thêm)
import sys, torch, transformers, PIL
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| pillow", PIL.__version__)
print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %% [foodbox]

# %% [markdown]
# ## Tham số

# %%
# Toàn bộ tham số của lần chạy
EXTRAS = ["plate", "bowl", "cup", "tray"]  # vật chứa: model gán đĩa/bát cho các từ này thay vì cho "food"
CFG = dict(
    data_root="/kaggle/input",
    uec_class_ids=[1, 7, 11, 18, 23, 36, 87, 98, 206, 209],  # giống bước 01
    uec_per_class=20,
    food_n_classes=10, food_per_class=20,                    # giống bước 02 (cùng 200 ảnh)
    seed=0,
    prompt_sets=[
        {"tag": "_food", "mode": "fixed", "prompts": ["food"], "extras": []},
        {"tag": "_food_pb", "mode": "fixed", "prompts": ["food"], "extras": EXTRAS},
        {"tag": "_class", "mode": "class", "prompts": [], "extras": []},
        {"tag": "_class_pb", "mode": "class", "prompts": [], "extras": EXTRAS},
    ],
    food_conf=0.35,          # ngưỡng của bước 02 v2.1 (để so trực tiếp)
    timeout_min=45,
    out="/kaggle/working/output",
)

# %%
# Nạp thư viện, tạo thư mục
import json, os, time, random
import numpy as np, pandas as pd
from foodbox import data, metrics, viz
from foodbox import postprocess as pp
from foodbox.run_detect import as_arrays, launch, load_preds
OUT = CFG["out"]
for sub in ["logs", "preds_uec", "preds_food", "gt_labels", "compare_uec", "compare_food"]:
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
with open(f"{OUT}/prompt_sets.json", "w") as f:
    json.dump(CFG["prompt_sets"], f, indent=1)
T0 = time.time()

# %% [markdown]
# ## 1. Dữ liệu: UEC mini (có box thật) và Food-101 tiny (cùng 200 ảnh như bước 02)

# %%
# UEC: tìm gốc, chọn 10 món x 20 ảnh, lấy box thật
data.print_tree(CFG["data_root"], max_depth=4)
uec_root = data.find_uec_root(CFG["data_root"])
assert uec_root, "Không tìm thấy UECFOOD-256"
cats = data.read_uec_categories(uec_root)
uec_images = data.uec_class_images(uec_root, CFG["uec_class_ids"], cats)
uec_items = data.sample_images(uec_images, n_classes=None, per_class=CFG["uec_per_class"], seed=CFG["seed"])
cat_of_class = {data.safe_name(cats[i]): i for i in CFG["uec_class_ids"]}
gts = data.convert_uec_to_yolo(uec_items, data.read_uec_boxes(uec_root), cat_of_class, f"{OUT}/gt_labels")
print(f"UEC: {len(uec_items)} ảnh, {sum(map(len, gts.values()))} box thật")

# %%
# Food-101: các thư mục <món>/*.jpg không nằm trong UEC, chọn giống hệt bước 02
food_roots = [r for r in data.find_class_roots(CFG["data_root"]) if not r.startswith(uec_root)]
print("Food-101:", food_roots)
food_images = data.list_class_images(food_roots)
food_items = data.dedupe_keys(data.sample_images(food_images, None, CFG["food_n_classes"], CFG["food_per_class"],
                                                 CFG["seed"]))
print(f"Food-101: {len(food_items)} ảnh, món {sorted({it['class'] for it in food_items})}")
for name, its in [("uec", uec_items), ("food", food_items)]:
    with open(f"{OUT}/items_{name}.json", "w") as f:
        json.dump(its, f)

# %% [markdown]
# ## 2. Chạy 4 biến thể prompt: GPU 0 = UEC, GPU 1 = Food-101

# %%
# Mỗi GPU một tiến trình, nạp model 1 lần rồi chạy lần lượt 4 bộ prompt; lưu theo đợt mỗi 50 ảnh
gpus = [0, 1] if torch.cuda.device_count() >= 2 else [0, 0]
jobs = [(g, ["--items", f"{OUT}/items_{name}.json", "--models", "gdino", "--prompt-sets", f"{OUT}/prompt_sets.json",
             "--out", f"{OUT}/preds_{name}", "--ckpt-every", "50"], f"{OUT}/logs/detect_{name}.log")
        for g, name in zip(gpus, ["uec", "food"])]
if gpus[0] == gpus[1]:  # 1 GPU: chạy lần lượt
    codes = [launch([j], timeout_min=CFG["timeout_min"], cwd="/tmp/foodbox_models")[0] for j in jobs]
else:
    codes = launch(jobs, timeout_min=CFG["timeout_min"], cwd="/tmp/foodbox_models")
for _, _, log in jobs:
    print(f"----- {log}")
    print("\n".join(open(log, encoding="utf-8", errors="replace").read().splitlines()[-6:]))
print(f"Thời gian detect: {(time.time() - T0) / 60:.1f} phút")

# %% [markdown]
# ## 3. Đánh giá trên UEC (có box thật)
# Mỗi biến thể có thang điểm khác (tên món thường cho điểm thấp hơn "food"), nên P/R/F1 tính ở ngưỡng F1 tốt nhất của từng biến thể.
# `area_ratio` < 1: box nhỏ hơn box thật (box thật UEC ôm cả bát) → ôm sát thức ăn hơn.

# %%
# Bảng UEC
P_uec = load_preds(f"{OUT}/preds_uec")
assert P_uec, "Không có kết quả UEC"
rows, pp_uec = [], {}
for v, r in sorted(P_uec.items()):
    P = {k: pp.postprocess(*as_arrays(x), max(x["w"], 1), max(x["h"], 1), conf=0.0, max_det=100)
         for k, x in r["images"].items()}
    pp_uec[v] = P
    m = metrics.evaluate(P, gts)
    rows.append({"variant": v, **m, "ap30": metrics.evaluate(P, gts, 0.3)["ap50"],
                 "sec_per_img": float(np.mean([x["time_s"] for x in r["images"].values()])),
                 "partial": any(mm.get("partial") for mm in r["meta"])})
uec_df = pd.DataFrame(rows).sort_values("ap50", ascending=False)
uec_df.to_csv(f"{OUT}/prompts_uec.csv", index=False)
best_conf = dict(zip(uec_df["variant"], uec_df["conf"]))
print(uec_df[["variant", "ap50", "ap30", "precision", "recall", "f1", "mean_iou", "area_ratio", "conf",
              "images_no_box", "sec_per_img", "partial"]].round(3).to_string(index=False))

# %% [markdown]
# ## 4. Thống kê trên Food-101 (không có box thật)
# Ở 2 ngưỡng: 0.35 (bước 02) và ngưỡng F1 tốt nhất của biến thể đó trên UEC.
# `main_area` = diện tích box món chính / ảnh (nhỏ hơn = ôm sát hơn); `near_full` = số ảnh có box món chính > 90% ảnh.

# %%
# Bảng Food-101
P_food = load_preds(f"{OUT}/preds_food")
rows, pp_food = [], {}
for v, r in sorted(P_food.items()):
    pp_food[v] = {k: pp.postprocess(*as_arrays(x), max(x["w"], 1), max(x["h"], 1), conf=0.0, max_det=100)
                  for k, x in r["images"].items()}
    for tag, c in [("0.35", CFG["food_conf"]), ("best_uec", best_conf.get(v, CFG["food_conf"]))]:
        n_box, areas, n_full, n_img = [], [], 0, 0
        for k, (b, s) in pp_food[v].items():
            x = r["images"][k]
            keep = s >= c
            n_box.append(int(keep.sum()))
            if keep.any():
                n_img += 1
                main = pp.pick_main(b[keep], s[keep], "largest")
                a = float(pp.area(b[keep][main:main + 1])[0]) / (x["w"] * x["h"])
                areas.append(a)
                n_full += a > 0.9
        rows.append({"variant": v, "threshold": tag, "conf": round(float(c), 3), "images_with_box": n_img,
                     "box_per_img": float(np.mean(n_box)), "main_area": float(np.mean(areas)) if areas else float("nan"),
                     "near_full": n_full})
food_df = pd.DataFrame(rows)
food_df.to_csv(f"{OUT}/prompts_food.csv", index=False)
print(food_df.round(3).to_string(index=False))

# %% [markdown]
# ## 5. Hình so sánh
# UEC: box thật xanh lá, box model đỏ. Food-101: box model đỏ. Mỗi ô một biến thể, ở ngưỡng F1 tốt nhất trên UEC.

# %%
# Vẽ hình so sánh: 2 ảnh/món của UEC và 3 ảnh/món của Food-101 (chọn ngẫu nhiên, seed 0)
variants = sorted(P_uec)
rng = random.Random(0)
def pick(items, k):
    out = []
    for c in sorted({it["class"] for it in items}):
        cand = [it for it in items if it["class"] == c]
        out += rng.sample(cand, min(k, len(cand)))
    return out
for name, its, Ppp, k in [("uec", uec_items, pp_uec, 2), ("food", food_items, pp_food, 3)]:
    for it in pick(its, k):
        img = data.load_image(it["path"])
        tiles = []
        for v in variants:
            b, s = Ppp[v].get(it["key"], (np.zeros((0, 4)), np.zeros(0)))
            keep = s >= best_conf.get(v, CFG["food_conf"])
            im = img
            if name == "uec":
                g = [x[2] for x in gts[it["key"]]]
                im = viz.draw_boxes(im, g, colors=[viz.GREEN] * len(g))
            im = viz.draw_boxes(im, b[keep], labels=[f"{x:.2f}" for x in s[keep]], colors=[viz.RED] * int(keep.sum()))
            tiles.append(viz.titled(viz.fit(im, 360), v))
        viz.grid(tiles, ncols=4).save(f"{OUT}/compare_{name}/{it['class']}__{it['stem']}.jpg", quality=85)
print("Đã lưu hình so sánh:", len(os.listdir(f"{OUT}/compare_uec")), "UEC,", len(os.listdir(f"{OUT}/compare_food")), "Food-101")

# %%
# Tóm tắt
with open(f"{OUT}/summary.json", "w") as f:
    json.dump({"config": CFG, "minutes": (time.time() - T0) / 60, "exit_codes": codes,
               "uec": uec_df.round(4).to_dict(orient="records"), "food": food_df.round(4).to_dict(orient="records")},
              f, indent=1, default=str)
print(f"Xong sau {(time.time() - T0) / 60:.1f} phút")
!du -sh {OUT}
