"""Nutrition5k data (n5k/PLAN.md mục 4–5): fixed split, RGB-D overhead images decoded once into a
.npy cache, labels = grams per ingredient + dish totals (layout of timeprest.n5k.task).

Source (Kaggle `gillesokhin/nutrition5k-dataset`):
  dish_nutrition_values.csv  dish_id, calories, mass, fat, carb, protein
  dish_ingredients.csv       dish_id, ingr_id, ingr_name, grams, calories, fat, carb, protein
  ingredients_metadata.csv   ingr_name, ingr_id, cal/g, fat(g), carb(g), protein(g)
  imagery/realsense_overhead/<dish_id>/{rgb.png, depth_raw.png}
Split: n5k/splits/{train,val,test}.txt (dish ids of Exp 1–3). Dishes with a missing/unreadable
image, no nutrition row or no ingredient row are dropped and listed in meta.json with the reason.

    python -m timeprest.n5k.data --config n5k/configs/n5k_quick.yaml     # build/verify cache, print report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import _find_dir_with
from .task import TOTALS

SPLITS = ("train", "val", "test")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CACHE_VERSION = 1
NUTRITION_COLS = ("calories", "mass", "fat", "carb", "protein")   # dish_nutrition_values.csv, order of TOTALS
PER_GRAM_COLS = ("cal/g", None, "fat(g)", "carb(g)", "protein(g)")  # ingredients_metadata.csv (mass/g = 1)


class BadSample(Exception):
    pass


# ---------------------------------------------------------------------------- source tables
def _rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def load_tables(base: str) -> dict:
    nutrition = {r["dish_id"]: [float(r[c]) for c in NUTRITION_COLS]
                 for r in _rows(os.path.join(base, "dish_nutrition_values.csv"))}
    dish_ingr: dict = {}
    names = {}
    for r in _rows(os.path.join(base, "dish_ingredients.csv")):
        d = dish_ingr.setdefault(r["dish_id"], {})
        vals = np.array([float(r["grams"]), float(r["calories"]), float(r["fat"]), float(r["carb"]),
                         float(r["protein"])])
        d[r["ingr_id"]] = d.get(r["ingr_id"], 0) + vals          # same ingredient twice in a dish: summed
        names[r["ingr_id"]] = r["ingr_name"]
    per_gram = {}
    for r in _rows(os.path.join(base, "ingredients_metadata.csv")):
        per_gram[r["ingr_id"]] = [1.0 if c is None else float(r[c]) for c in PER_GRAM_COLS]
        names.setdefault(r["ingr_id"], r["ingr_name"])
    return {"nutrition": nutrition, "dish_ingr": dish_ingr, "per_gram": per_gram, "names": names}


def read_split(split_dir: str) -> dict:
    out = {}
    for s in SPLITS:
        with open(os.path.join(split_dir, f"{s}.txt"), encoding="utf-8") as f:
            out[s] = [l.strip() for l in f if l.strip()]
    seen = [d for s in SPLITS for d in out[s]]
    if len(seen) != len(set(seen)):
        raise ValueError("a dish id appears in more than one split")
    return out


# ---------------------------------------------------------------------------- images
def decode_dish(folder: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """rgb uint8 [size, size, 3] (bilinear) and raw depth uint16 [size, size] (nearest: 0 = no
    measurement must not blend with valid depths). Raises BadSample with the reason."""
    from PIL import Image
    paths = {k: os.path.join(folder, f) for k, f in (("rgb", "rgb.png"), ("depth", "depth_raw.png"))}
    for k, p in paths.items():
        if not os.path.isfile(p):
            raise BadSample(f"missing {os.path.basename(p)}")
        if os.path.getsize(p) == 0:
            raise BadSample(f"empty {os.path.basename(p)}")
    try:
        with Image.open(paths["rgb"]) as im:
            rgb = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
        with Image.open(paths["depth"]) as im:
            d = np.asarray(im, dtype=np.int64)
    except Exception as e:  # PIL raises several types for corrupt files
        raise BadSample(f"unreadable image ({type(e).__name__})") from None
    if d.ndim != 2 or d.min() < 0 or d.max() > 65535:
        raise BadSample(f"depth not single-channel 16-bit (shape {d.shape})")
    if not (d > 0).any():
        raise BadSample("depth has no valid pixel")
    depth = np.asarray(Image.fromarray(d.astype(np.int32), mode="I").resize((size, size), Image.NEAREST),
                       dtype=np.int64).astype(np.uint16)
    return rgb, depth


# ---------------------------------------------------------------------------- cache
FILES = [f"{s}_{k}.npy" for s in SPLITS for k in ("rgb", "depth", "y")]


def _complete(d: str) -> bool:
    return all(os.path.exists(os.path.join(d, f)) for f in FILES + ["meta.json"])


def _cache_size(d: str) -> int | None:
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        return json.load(f).get("image_size")


def _find_cache(root: str, size: int | None = None) -> str | None:
    """First complete cache (meta.json + arrays) under `root`, e.g. a cache uploaded as a Kaggle dataset;
    with `size`, only a cache of that image size."""
    top = root.split("*")[0].rstrip("/\\") or "/"
    if not os.path.isdir(top):
        return None
    base_depth = top.rstrip("/\\").count(os.sep)
    for d, subdirs, files in os.walk(top, followlinks=True):
        # never descend into image folders (Nutrition5k: thousands of dish folders, minutes to walk)
        subdirs[:] = sorted(s for s in subdirs if s != "imagery") if d.count(os.sep) - base_depth < 6 else []
        if "meta.json" in files and _complete(d) and (size is None or _cache_size(d) == size):
            return d
    return None


def _consistency(dishes: list, ingr_rows: list, from_grams: np.ndarray, nutrition: dict) -> dict:
    """How well the ingredient table explains the dish totals: (a) Σ ingredient calories of
    dish_ingredients.csv, (b) Σ grams × per-gram values (`from_grams`, [n, T]) vs the dish totals."""
    out = {}
    if not dishes:
        return out
    tot = np.array([nutrition[d] for d in dishes])                    # [n, T]
    sum_ingr_kcal = np.array([sum(v[1] for v in rows.values()) for rows in ingr_rows])
    out["kcal_sum_ingredients_mae"] = float(np.abs(sum_ingr_kcal - tot[:, 0]).mean())
    for t, name in enumerate(TOTALS):
        err = np.abs(from_grams[:, t] - tot[:, t])
        out[f"{name}_from_grams_mae"] = float(err.mean())
        out[f"{name}_from_grams_median_rel"] = float(np.median(err / np.maximum(tot[:, t], 1e-6)))
    return out


def build_cache(data_cfg: dict, log=print) -> str:
    """Decode and label every dish of the split once; return the cache directory. A complete cache
    under data.cache_dir or data.cache_search is reused as is."""
    cache = data_cfg.get("cache_dir") or os.path.join(os.environ.get("TMPDIR", "/tmp"), "timeprest_n5k")
    size = int(data_cfg.get("image_size", 224))
    if _complete(cache) and _cache_size(cache) != size:
        raise RuntimeError(f"{cache} holds {_cache_size(cache)}px images, data.image_size is {size}: "
                           "set another data.cache_dir")
    if not _complete(cache) and data_cfg.get("cache_search"):
        found = _find_cache(data_cfg["cache_search"], size)
        if found:
            return found
    if _complete(cache):
        return cache
    os.makedirs(cache, exist_ok=True)
    lock = os.path.join(cache, ".building")
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:                       # another rank builds it: wait for meta.json
        t0 = time.time()
        while not _complete(cache):
            if time.time() - t0 > 3600:
                raise TimeoutError(f"waited 1 h for {cache}; delete {lock} if a previous run crashed")
            time.sleep(2)
        return cache
    try:
        _build(cache, data_cfg, log)
    finally:
        os.remove(lock)
    return cache


def _build(cache: str, data_cfg: dict, log):
    t0 = time.time()
    base = _find_dir_with(data_cfg["root"], "dish_nutrition_values.csv")
    tables = load_tables(base)
    split = read_split(data_cfg["split_dir"])
    size = int(data_cfg.get("image_size", 224))
    img_root = os.path.join(base, "imagery", "realsense_overhead")
    threads = int(data_cfg.get("decode_threads", 16))   # file reads on a lazily mounted dataset are latency bound
    every = int(data_cfg.get("progress_every", 250))

    def one(d):
        try:
            if d not in tables["nutrition"]:
                raise BadSample("no row in dish_nutrition_values.csv")
            if not tables["dish_ingr"].get(d):
                raise BadSample("no row in dish_ingredients.csv")
            return decode_dish(os.path.join(img_root, d), size), None
        except BadSample as e:
            return None, str(e)

    kept, imgs, dropped = {}, {}, []
    total, done = sum(len(split[s]) for s in SPLITS), 0
    log(f"[n5k data] decoding {total} dishes from {img_root} with {threads} threads")
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for s in SPLITS:
            kept[s], imgs[s] = [], []
            for d, (img, reason) in zip(split[s], pool.map(one, split[s])):   # map keeps the split order
                if reason is None:
                    imgs[s].append(img)
                    kept[s].append(d)
                else:
                    dropped.append({"dish_id": d, "split": s, "reason": reason})
                done += 1
                if every and done % every == 0:
                    el = time.time() - t0
                    log(f"[n5k data] {done}/{total} dishes  {el:.0f}s  eta {el / done * (total - done):.0f}s")
            log(f"[n5k data] {s}: kept {len(kept[s])}/{len(split[s])}  ({time.time() - t0:.0f}s)")

    # ingredient vocabulary = ingredients present (grams > 0) in the kept train dishes
    vocab = sorted({i for d in kept["train"] for i, v in tables["dish_ingr"][d].items() if v[0] > 0})
    index = {i: k for k, i in enumerate(vocab)}
    K = len(vocab)
    # per-gram values: ingredients_metadata.csv; fallback = train ratios of dish_ingredients.csv
    pg, fallback = np.zeros((K, len(TOTALS))), []
    for i, k in index.items():
        if i in tables["per_gram"]:
            pg[k] = tables["per_gram"][i]
        else:
            g = sum(tables["dish_ingr"][d][i] for d in kept["train"] if i in tables["dish_ingr"][d])
            pg[k] = [g[1] / g[0], 1.0, g[2] / g[0], g[3] / g[0], g[4] / g[0]]
            fallback.append(i)

    labels, report = {}, {"dropped": len(dropped), "vocab_size": K, "per_gram_fallback": fallback}
    for s in SPLITS:
        y = np.zeros((len(kept[s]), K + len(TOTALS)), dtype=np.float32)
        oov_dishes, oov_kcal, all_kcal = 0, 0.0, 0.0
        for n, d in enumerate(kept[s]):
            for i, v in tables["dish_ingr"][d].items():
                all_kcal += v[1]
                if i in index:
                    y[n, index[i]] = v[0]
                elif v[0] > 0:
                    oov_kcal += v[1]
            oov_dishes += any(i not in index and v[0] > 0 for i, v in tables["dish_ingr"][d].items())
            y[n, K:] = tables["nutrition"][d]
        labels[s] = y
        rows = kept[s]
        report[s] = {"n": len(rows), "dishes_with_unknown_ingredient": oov_dishes,
                     "unknown_ingredient_kcal_share": oov_kcal / all_kcal if all_kcal else 0.0,
                     **_consistency(rows, [tables["dish_ingr"][d] for d in rows],
                                    y[:, :K].astype(np.float64) @ pg, tables["nutrition"])}

    ytr = labels["train"].astype(np.float64)
    present = ytr[:, :K] > 0
    gram_scale = np.maximum((ytr[:, :K] * present).sum(0) / np.maximum(present.sum(0), 1), 1.0)
    totals = ytr[:, K:]
    rng = np.random.default_rng(0)
    px = np.concatenate([dep[rng.integers(0, size, 64), rng.integers(0, size, 64)]
                         for _, dep in imgs["train"]]).astype(np.float64)
    lo, hi = (float(v) for v in np.percentile(px[px > 0], [1, 99]))
    d01 = np.where(px > 0, np.clip((px - lo) / max(hi - lo, 1.0), 0, 1), 0.0)   # as N5kImages
    meta = {"version": CACHE_VERSION, "source": base, "image_size": size,
            "ingredients": [{"id": i, "name": tables["names"].get(i, "")} for i in vocab],
            "totals": list(TOTALS), "per_gram": pg.tolist(), "gram_scale": gram_scale.tolist(),
            "total_mean": totals.mean(0).tolist(), "total_std": totals.std(0).tolist(),
            "depth_lo": lo, "depth_hi": hi, "depth_mean": float(d01.mean()), "depth_std": float(d01.std()),
            "rgb_mean": list(IMAGENET_MEAN), "rgb_std": list(IMAGENET_STD),
            "ids": kept, "dropped": dropped, "report": report, "build_time_s": round(time.time() - t0, 1)}
    for s in SPLITS:                           # write-then-rename; meta.json last marks completion
        arrays = {"rgb": np.stack([r for r, _ in imgs[s]]) if imgs[s] else np.zeros((0, size, size, 3), np.uint8),
                  "depth": np.stack([d for _, d in imgs[s]]) if imgs[s] else np.zeros((0, size, size), np.uint16),
                  "y": labels[s]}
        for k, a in arrays.items():
            path = os.path.join(cache, f"{s}_{k}.npy")
            tmp = f"{path}.{os.getpid()}.tmp.npy"
            np.save(tmp, a)
            os.replace(tmp, path)
    tmp = os.path.join(cache, f"meta.json.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, os.path.join(cache, "meta.json"))
    log(f"[n5k data] cache written to {cache} in {meta['build_time_s']}s")


# ---------------------------------------------------------------------------- dataset
class N5kImages(Dataset):
    """4-channel input: ImageNet-normalised RGB + depth clipped to [depth_lo, depth_hi], scaled to
    [0, 1] (0 = no measurement) and standardised. Augmentation: the same random horizontal /
    vertical flip for RGB and depth. Labels: float [K + T] (timeprest.n5k.task layout)."""

    def __init__(self, rgb: np.ndarray, depth: np.ndarray, y: np.ndarray, meta: dict, augment: bool):
        self.rgb = torch.from_numpy(np.ascontiguousarray(rgb))                       # [N, H, W, 3] uint8
        lo, hi = meta["depth_lo"], meta["depth_hi"]
        d01 = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
        d01[depth == 0] = 0.0
        self.depth = torch.from_numpy(d01.astype(np.float16))                        # [N, H, W]
        self.targets = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))   # [N, K + T]
        self.mean = torch.tensor(meta["rgb_mean"]).view(3, 1, 1)
        self.std = torch.tensor(meta["rgb_std"]).view(3, 1, 1)
        self.dmean, self.dstd = meta["depth_mean"], max(meta["depth_std"], 1e-6)
        self.augment = augment

    def __len__(self):
        return self.rgb.shape[0]

    def subset(self, idx) -> "N5kImages":
        """Same dataset restricted to `idx` (keeps `targets`, unlike torch Subset)."""
        idx = torch.as_tensor(list(idx), dtype=torch.long)
        out = object.__new__(N5kImages)
        out.__dict__.update(self.__dict__)
        out.rgb, out.depth, out.targets = self.rgb[idx], self.depth[idx], self.targets[idx]
        return out

    def __getitem__(self, i):
        rgb = (self.rgb[i].permute(2, 0, 1).float().div_(255.0) - self.mean) / self.std
        dep = ((self.depth[i].float() - self.dmean) / self.dstd).unsqueeze(0)
        x = torch.cat([rgb, dep], 0)
        if self.augment:
            if torch.rand(()) < 0.5:
                x = x.flip(2)
            if torch.rand(()) < 0.5:
                x = x.flip(1)
        return x, self.targets[i]


def load_datasets(data_cfg: dict, log=print):
    """(train, val, test, meta). Builds the cache first if needed."""
    cache = build_cache(data_cfg, log)
    with open(os.path.join(cache, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("version") != CACHE_VERSION:
        raise RuntimeError(f"{cache}: cache version {meta.get('version')} != {CACHE_VERSION}; rebuild it")
    if meta.get("image_size") != int(data_cfg.get("image_size", 224)):
        raise RuntimeError(f"{cache}: {meta.get('image_size')}px cache, data.image_size is {data_cfg.get('image_size')}")
    out = []
    for s in SPLITS:
        arr = {k: np.load(os.path.join(cache, f"{s}_{k}.npy")) for k in ("rgb", "depth", "y")}
        out.append(N5kImages(arr["rgb"], arr["depth"], arr["y"], meta, augment=data_cfg.get("augment", True) and s == "train"))
    meta["cache_dir"] = cache
    return (*out, meta)


def format_report(meta: dict) -> str:
    lines = [f"cache: {meta.get('cache_dir')}  source: {meta['source']}  image {meta['image_size']}px  "
             f"vocab K={len(meta['ingredients'])}  build {meta['build_time_s']}s",
             f"depth clip [{meta['depth_lo']:.0f}, {meta['depth_hi']:.0f}]  "
             f"per-gram fallback for {len(meta['report']['per_gram_fallback'])} ingredient(s)",
             f"dropped {len(meta['dropped'])}: " + "; ".join(f"{d['dish_id']} ({d['split']}: {d['reason']})"
                                                           for d in meta["dropped"])]
    for s in SPLITS:
        r = meta["report"][s]
        lines.append(f"{s:>5}: n={r['n']}  unknown-ingredient dishes={r['dishes_with_unknown_ingredient']} "
                     f"(kcal share {r['unknown_ingredient_kcal_share'] * 100:.2f}%)  "
                     f"sum(ingredient kcal) vs dish MAE={r.get('kcal_sum_ingredients_mae', float('nan')):.2f}  "
                     + "  ".join(f"{t} from grams MAE={r.get(f'{t}_from_grams_mae', float('nan')):.2f}"
                                 f" (median rel {r.get(f'{t}_from_grams_median_rel', float('nan')) * 100:.1f}%)"
                                 for t in TOTALS))
    return "\n".join(lines)


def main(argv=None):
    from ..config import load_config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    train, val, test, meta = load_datasets(cfg["data"])
    print(format_report(meta))
    x, y = train[0]
    print(f"sample: x {tuple(x.shape)} rgb mean {x[:3].mean():.3f} depth mean {x[3].mean():.3f}  y {tuple(y.shape)}")
    print("N5K_DATA_JSON", json.dumps({"n": [len(train), len(val), len(test)], "K": len(meta["ingredients"]),
                                       "dropped": meta["dropped"], "report": meta["report"]}))


if __name__ == "__main__":
    main()
