# %% [markdown]
# # 04 — Gộp các miếng cùng đĩa/thớt thành 1 món: chạy Grounding DINO + vật chứa
# Mỗi ảnh chạy 2 lượt: "food" (như cũ) và "plate. bowl. tray. cutting board." (vật chứa). Lưu cả hai để đánh giá trên máy
# (`tools/eval_group.py`): gộp / không gộp, trên UEC mini (có box thật, 58 ảnh khay nhiều món) và Food-101 tiny (200 ảnh bước 02).
# File nguồn: `support_bounding/kaggle/04_group/notebook.py`.

# %%
# Môi trường (chỉ cần torch + transformers có sẵn của Kaggle)
import sys, torch, transformers, PIL
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| pillow", PIL.__version__)
print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %% [foodbox]

# %%
# Tham số (giống bước 01 / 02 để dùng đúng các ảnh cũ)
CFG = dict(
    data_root="/kaggle/input",
    uec_class_ids=[1, 7, 11, 18, 23, 36, 87, 98, 206, 209], uec_per_class=20,
    food_n_classes=10, food_per_class=20,
    seed=0, timeout_min=45, out="/kaggle/working/output",
)

# %%
# Dữ liệu: UEC (box thật) + Food-101 tiny, chọn giống hệt bước 01 / 02
import json, os, time
from foodbox import data
from foodbox.run_detect import launch, load_preds
OUT = CFG["out"]
for sub in ["logs", "preds_uec", "preds_food", "gt_labels"]:
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
T0 = time.time()
uec_root = data.find_uec_root(CFG["data_root"])
assert uec_root, "Không tìm thấy UECFOOD-256"
cats = data.read_uec_categories(uec_root)
uec_items = data.sample_images(data.uec_class_images(uec_root, CFG["uec_class_ids"], cats), n_classes=None,
                               per_class=CFG["uec_per_class"], seed=CFG["seed"])
gts = data.convert_uec_to_yolo(uec_items, data.read_uec_boxes(uec_root),
                               {data.safe_name(cats[i]): i for i in CFG["uec_class_ids"]}, f"{OUT}/gt_labels")
food_roots = [r for r in data.find_class_roots(CFG["data_root"]) if not r.startswith(uec_root)]
food_items = data.dedupe_keys(data.sample_images(data.list_class_images(food_roots), None, CFG["food_n_classes"],
                                                 CFG["food_per_class"], CFG["seed"]))
for name, its in [("uec", uec_items), ("food", food_items)]:
    with open(f"{OUT}/items_{name}.json", "w", encoding="utf-8") as f:
        json.dump(its, f)
print(f"UEC {len(uec_items)} ảnh ({sum(len(g) >= 2 for g in gts.values())} ảnh nhiều món), Food-101 {len(food_items)} ảnh")

# %%
# GPU 0 = UEC, GPU 1 = Food-101; mỗi ảnh 2 lượt (food + vật chứa), lưu theo đợt mỗi 50 ảnh
gpus = [0, 1] if torch.cuda.device_count() >= 2 else [0, 0]
jobs = [(g, ["--items", f"{OUT}/items_{name}.json", "--models", "gdino", "--containers", "--out", f"{OUT}/preds_{name}"],
         f"{OUT}/logs/detect_{name}.log") for g, name in zip(gpus, ["uec", "food"])]
if gpus[0] == gpus[1]:
    codes = [launch([j], timeout_min=CFG["timeout_min"], cwd="/tmp/foodbox_models")[0] for j in jobs]
else:
    codes = launch(jobs, timeout_min=CFG["timeout_min"], cwd="/tmp/foodbox_models")
for _, _, log in jobs:
    print(f"----- {log}")
    print("\n".join(open(log, encoding="utf-8", errors="replace").read().splitlines()[-5:]))
for name in ["uec", "food"]:
    P = load_preds(f"{OUT}/preds_{name}")["gdino"]["images"]
    n_c = sum(len(x.get("c_boxes", [])) > 0 for x in P.values())
    print(f"{name}: {len(P)} ảnh có kết quả, {n_c} ảnh có box vật chứa")
with open(f"{OUT}/summary.json", "w", encoding="utf-8") as f:
    json.dump({"config": CFG, "minutes": (time.time() - T0) / 60, "exit_codes": codes}, f, indent=1)
print(f"Xong sau {(time.time() - T0) / 60:.1f} phút")
