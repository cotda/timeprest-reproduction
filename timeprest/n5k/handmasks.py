"""B7b (n5k/PLAN.md mục 10, 2026-10-17 21:32 / 21:40): location of the maps against real ingredient masks.

    python -m timeprest.n5k.handmasks --config n5k/configs/n5k_handmasks.yaml --set name=n5k_timeprest_w3_s0 \
        --ckpt <dir with best.pt or best_stage*.pt>
    python -m timeprest.n5k.handmasks --config n5k/configs/n5k_handmasks.yaml --check-groups <dir of
        dish_metadata_cafe{1,2}.csv>          # mass check of handmasks.groups_file (no model, no images)

HandMasks = the ingredient-level masks of Hand et al. 2026 (Frontiers in Nutrition 13:1927530, CC BY 4.0) for
1056 overhead images of Nutrition5k: one indexed PNG per dish (640 x 480, pixel = region label) and one row per region
(group = `annotation_name`, one of 40; true kcal / mass). They are a separate, read-only source: nothing here writes to
the Nutrition5k cache or splits, and the 40 groups never enter the training labels. A dish is used only if it is in
our val or test split (checked); HandMasks dishes of our train split are skipped and counted.

Masks are resized like the images (whole 640 x 480 frame -> image_size^2, NEAREST), then reduced to the occupancy of
every region in each cell of the head grid (7 x 7 at 224 px). Maps (explain.py): ingredient calorie map C and ReLU of
the direct kcal map. Per dish:
  M1  share of the map inside the food (union of the masks) vs the food's area share (= a uniform map)
  M2  per-region kcal share of the map vs the true kcal share; MAE of the shares, against area shares (dishes with
      >= 2 regions, all with true kcal)
  M3  pointing game per group present in the dish: peak cell of sum_{k in group} p_k ReLU(CAM_k) (groups_file maps
      our 195 ingredients to the 40 groups) is a hit if a region of the group touches the cell; chance = share of the
      food cells touched by the group
Rules (handmasks.rules): M1 mean(in - area) > m1_margin and one-sided Wilcoxon p < alpha; M2 relative MAE gain over
the area shares > m2_rel_gain and Wilcoxon p < alpha; M3 hit rate - chance > m3_margin. They say how well the maps
locate the real food at the head's resolution (32 px cells of the 224 px image, ~91 x 69 px of the photo).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image

from .. import utils
from ..config import load_config
from .common import prepare
from .explain import _wilcoxon_greater, collect, load_model

MAPS = ("cmap", "tmap")                     # ingredient calorie map, direct kcal map (ReLU)


# ---------------------------------------------------------------------------- HandMasks data
def find_root(hc: dict) -> str:
    for pattern in hc["root_search"]:
        hits = sorted(glob.glob(pattern))
        if hits and os.path.exists(os.path.join(hits[0], "ingredient_nutrition.csv")):
            return hits[0]
    raise FileNotFoundError(f"no HandMasks folder (ingredient_nutrition.csv + masks/) in {hc['root_search']}")


def read_regions(root: str) -> dict:
    """dish_id -> list of regions {label, group, kcal, mass, fat, carb, protein} (nutrients None if missing)."""
    out = defaultdict(list)
    with open(os.path.join(root, "ingredient_nutrition.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ok = r["has_nutrition"] == "1"
            out[r["dish_id"]].append({"label": int(r["label_value"]), "group": r["annotation_name"],
                                      **{k: (float(r[c]) if ok else None) for k, c in
                                         (("kcal", "calories"), ("mass", "mass"), ("fat", "fat"), ("carb", "carb"),
                                          ("protein", "protein"))}})
    return dict(out)


def read_mask(root: str, dish_id: str, hw: tuple[int, int]) -> np.ndarray:
    m = Image.open(os.path.join(root, "masks", f"{dish_id}.png"))
    if (m.height, m.width) != tuple(hw):
        raise ValueError(f"HandMasks {dish_id}: mask is {m.height} x {m.width}, expected {hw[0]} x {hw[1]}")
    return np.asarray(m)


def cell_occupancy(mask: np.ndarray, labels: list[int], size: int, grid: tuple[int, int]) -> np.ndarray:
    """[len(labels) + 1, h, w]: share of each cell covered by each region label, last row = any food (mask > 0).
    The mask is resized like the images: whole frame -> size x size, NEAREST."""
    m = np.asarray(Image.fromarray(mask).resize((size, size), Image.NEAREST))
    h, w = grid
    if size % h or size % w:
        raise ValueError(f"image size {size} is not a multiple of the grid {grid}")
    layers = [m == lab for lab in labels] + [m > 0]
    return np.stack([a.reshape(h, size // h, w, size // w).mean((1, 3)) for a in layers]).astype(np.float64)


def load_groups(path: str, ingredients: list[dict], groups_known: set[str]) -> list[str | None]:
    """Group of each of our K ingredients (meta order); every ingredient must be listed, groups must exist.
    path "identity": the ingredients are the groups themselves (added HM2, labels = HandMasks groups)."""
    if path == "identity":
        if any(d["id"] not in groups_known for d in ingredients):
            raise ValueError("groups_file identity: some ingredient ids are not HandMasks groups")
        return [d["id"] for d in ingredients]
    with open(path, encoding="utf-8") as f:
        table = yaml.safe_load(f)["groups"]
    out = []
    for d in ingredients:
        if d["id"] not in table:
            raise KeyError(f"{path}: ingredient {d['id']} ({d.get('name')}) has no entry")
        g = table[d["id"]].get("group")
        if g is not None and g not in groups_known:
            raise ValueError(f"{path}: {d['id']} -> unknown HandMasks group {g!r}")
        out.append(g)
    return out


# ---------------------------------------------------------------------------- metrics
def dish_metrics(maps: dict, occ: np.ndarray, regions: list[dict], cam: np.ndarray | None, prob: np.ndarray | None,
                 groups: list[str | None] | None) -> dict:
    """maps: name -> non-negative [h, w]; occ from cell_occupancy (rows = regions in order, last = food).
    cam [K, h, w], prob [K], groups [K] for M3 (None to skip)."""
    food = occ[-1]
    area = float(food.mean())
    out = {"area_food": area, "n_regions": len(regions)}
    reg_area = occ[:-1].sum((1, 2))
    kc = [r["kcal"] for r in regions]
    m2_ok = len(regions) >= 2 and all(k is not None for k in kc) and sum(kc) > 0 and reg_area.sum() > 0
    for name, m in maps.items():
        tot = float(m.sum())
        out[f"{name}_m1_in"] = float((m * food).sum()) / tot if tot > 0 else float("nan")
        if m2_ok:
            per = (occ[:-1] * m[None]).sum((1, 2))
            share = per / per.sum() if per.sum() > 0 else np.full(len(regions), np.nan)
            true = np.array(kc) / sum(kc)
            out[f"{name}_m2_mae"] = float(np.abs(share - true).mean())
            out["area_m2_mae"] = float(np.abs(reg_area / reg_area.sum() - true).mean())
    if cam is not None:
        by_group = defaultdict(list)
        for j, r in enumerate(regions):
            by_group[r["group"]].append(j)
        hits = []
        food_cells = food > 0
        for g, js in sorted(by_group.items()):
            ks = [k for k, gk in enumerate(groups) if gk == g]
            touched = occ[js].sum(0) > 0
            if not ks or not food_cells.any():
                continue
            pooled = (np.clip(cam[ks], 0, None) * prob[ks, None, None]).sum(0)
            peak = np.unravel_index(int(np.argmax(pooled)), pooled.shape)
            hits.append({"group": g, "hit": bool(touched[peak]),
                         "chance": float((touched & food_cells).sum() / food_cells.sum())})
        out["m3"] = hits
    return out


def summarize(rows: list[dict], rules: dict) -> dict:
    """Means, Wilcoxon tests and the verdict of each rule over the dishes of one split."""
    res = {"n_dishes": len(rows)}
    area = np.array([r["area_food"] for r in rows])
    for name in MAPS:
        if not rows or f"{name}_m1_in" not in rows[0]:
            continue
        m1 = np.array([r[f"{name}_m1_in"] for r in rows])
        d1 = m1 - area
        p1 = _wilcoxon_greater(d1)
        res[f"{name}_m1"] = {"in": float(np.nanmean(m1)), "area": float(area.mean()), "gain": float(np.nanmean(d1)),
                             "p": p1, "pass": bool(np.nanmean(d1) > rules["m1_margin"] and p1 < rules["alpha"])}
        r2 = [r for r in rows if f"{name}_m2_mae" in r and np.isfinite(r[f"{name}_m2_mae"])]
        if r2:
            a = np.array([r[f"{name}_m2_mae"] for r in r2])
            b = np.array([r["area_m2_mae"] for r in r2])
            gain = 1.0 - a.mean() / b.mean()
            p2 = _wilcoxon_greater(b - a)
            res[f"{name}_m2"] = {"n": len(r2), "mae_map": float(a.mean()), "mae_area": float(b.mean()),
                                 "rel_gain": float(gain), "p": p2,
                                 "pass": bool(gain > rules["m2_rel_gain"] and p2 < rules["alpha"])}
    m3 = [h for r in rows for h in r.get("m3", [])]
    if m3:
        hit = float(np.mean([h["hit"] for h in m3]))
        chance = float(np.mean([h["chance"] for h in m3]))
        per = defaultdict(list)
        for h in m3:
            per[h["group"]].append(h)
        res["m3"] = {"n": len(m3), "hit": hit, "chance": chance, "pass": bool(hit - chance > rules["m3_margin"]),
                     "per_group": {g: {"n": len(v), "hit": float(np.mean([x["hit"] for x in v])),
                                       "chance": float(np.mean([x["chance"] for x in v]))}
                                   for g, v in sorted(per.items(), key=lambda kv: -len(kv[1]))}}
    return res


# ---------------------------------------------------------------------------- group mapping check
def check_groups(cfg: dict, meta_dir: str, log=print) -> dict:
    """Mass check of groups_file against the regions: for a dish with one region of group g, the region mass should
    equal the grams of the dish's ingredients mapped to g (Nutrition5k dish_metadata_cafe*.csv)."""
    hc = cfg["handmasks"]
    regions = read_regions(find_root(hc))
    with open(hc["groups_file"], encoding="utf-8") as f:
        table = yaml.safe_load(f)["groups"]
    dish = {}
    for p in sorted(glob.glob(os.path.join(meta_dir, "dish_metadata_cafe*.csv"))):
        with open(p, encoding="utf-8") as f:
            for row in csv.reader(f):
                dish[row[0]] = [(row[i], float(row[i + 2])) for i in range(6, len(row) - 6, 7)]
    n = ok = 0
    bad = []
    for d, regs in regions.items():
        count = defaultdict(int)
        for r in regs:
            count[r["group"]] += 1
        for r in regs:
            if r["mass"] is None or count[r["group"]] > 1 or d not in dish:
                continue
            pred = sum(w for i, w in dish[d] if (table.get(i) or {}).get("group") == r["group"])
            n += 1
            if abs(pred - r["mass"]) <= max(1.0, 0.02 * r["mass"]):
                ok += 1
            else:
                bad.append({"dish_id": d, "group": r["group"], "mass": r["mass"], "mapped_grams": round(pred, 2)})
    res = {"regions_checked": n, "match": ok, "share": ok / n if n else float("nan"), "mismatches": bad[:50]}
    log(f"[handmasks] group mass check: {ok}/{n} single-group regions match ({res['share']:.1%})")
    return res


# ---------------------------------------------------------------------------- main
def run(cfg: dict, ckpt_dir: str, device, data=None, log=print) -> dict:
    hc, ec = cfg["handmasks"], cfg["explain"]
    root = find_root(hc)
    out_dir = hc["dir"].format(name=cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    built, (_, val, test), meta, task = prepare(cfg, data, log=log)
    regions = read_regions(root)
    groups = load_groups(hc["groups_file"], meta["ingredients"], {r["group"] for v in regions.values() for r in v})
    model, epoch = load_model(built, ckpt_dir, test[0][0].unsqueeze(0))
    model.to(device)
    size = int(cfg["data"]["image_size"])
    ours = {s: set(meta["ids"][s]) for s in ("train", "val", "test")}
    skipped = {s: sum(d in ours[s] for d in regions) for s in ours}
    skipped["not_in_our_splits"] = sum(all(d not in v for v in ours.values()) for d in regions)
    log(f"[handmasks] {root}: {len(regions)} dishes; in our train {skipped['train']} (skipped), "
        f"val {skipped['val']}, test {skipped['test']}, elsewhere {skipped['not_in_our_splits']} (skipped)")
    bs = int(cfg["runtime"]["eval_batch_size"])
    res = {"name": cfg["name"], "checkpoint": ckpt_dir, "best_epoch": epoch, "handmasks_root": root,
           "dish_counts": skipped, "handmasks": hc, "code_hash": utils.code_hash()}
    for split, ds in (("val", val), ("test", test)):
        ids = meta["ids"][split]
        idx = [i for i, d in enumerate(ids) if d in regions]
        if any(ids[i] in ours["train"] for i in idx):
            raise RuntimeError("a HandMasks dish of our train split reached the evaluation")
        sub = torch.utils.data.Subset(ds, idx)
        base = collect(model, task, sub, ec, device, bs)
        grid = tuple(base["cam"].shape[2:])
        rows = []
        for j, i in enumerate(idx):
            regs = regions[ids[i]]
            occ = cell_occupancy(read_mask(root, ids[i], hc["mask_hw"]), [r["label"] for r in regs], size, grid)
            maps = {"cmap": base["cmap"][j].numpy().astype(np.float64)}
            if base["tmap"] is not None:
                maps["tmap"] = np.clip(base["tmap"][j].numpy().astype(np.float64), 0, None)
            r = dish_metrics(maps, occ, regs, base["cam"][j].numpy(), base["prob"][j].numpy(), groups)
            rows.append({"dish_id": ids[i], **r})
        utils.save_json(rows, os.path.join(out_dir, f"handmasks_{split}_dishes.json"))
        res[split] = summarize(rows, hc["rules"])
        # kcal MAE on exactly these dishes (direct branch if any, as test_mae_kcal), to compare models on the same dishes
        kcal = base["kcal_direct"] if base["kcal_direct"] is not None else base["totals_ingr"][:, 0]
        res[split]["mae_kcal"] = float((kcal - ds.targets[idx, task.K]).abs().mean()) if idx else float("nan")
        log(f"[handmasks] {split}: {len(rows)} dishes " + json.dumps({k: v for k, v in res[split].items()
                                                                     if k != "m3"}, default=float)[:600])
    utils.save_json(res, os.path.join(out_dir, "handmasks.json"))
    short = {s: {k: ({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "per_group"}
                     if isinstance(v, dict) else v) for k, v in res[s].items()} for s in ("val", "test")}
    print("N5K_HANDMASKS_JSON", json.dumps({"name": cfg["name"], "best_epoch": epoch, **short}), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", default=None, help="run directory with best.pt or best_stage*.pt (default output.dir)")
    ap.add_argument("--check-groups", default=None, metavar="META_DIR",
                    help="only run the mass check of handmasks.groups_file (dir of dish_metadata_cafe*.csv)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.check_groups:
        res = check_groups(cfg, args.check_groups)
        print("N5K_HANDMASKS_GROUPS_JSON", json.dumps({k: v for k, v in res.items() if k != "mismatches"}), flush=True)
        return
    utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    device = utils.resolve_device(cfg["runtime"]["device"])
    run(cfg, args.ckpt or cfg["output"]["dir"], device)


if __name__ == "__main__":
    main()
