"""HM (n5k/added/README.md, 2026-10-17 22:25 / 22:31; exploratory): train only on the Nutrition5k dishes that have
HandMasks (Hand et al. 2026), with our splits (741 train / 157 val / 154 test), then locate with timeprest.n5k.handmasks.

Same images, cache and trainers as the main task; nothing of data.py / the cache / n5k/splits changes. The datasets are
loaded as usual, restricted with N5kImages.subset and handed to the trainer through `data=`.
  added.hm.labels: ingredients   HM1: the 195 ingredient labels of the main task (only less data)
  added.hm.labels: groups        HM2: the 40 HandMasks groups as the ingredient labels:
      present = the dish has a region of the group; grams = summed region masses (regions without a mass: grams of
      our ingredients mapped to the group by handmasks.groups_file, else the group's train median); dish totals
      unchanged; per gram (kcal, 1, fat, carb, protein) = median over the train regions; gram_scale as data.py
      (mean grams of the train dishes holding it, >= 1). Oils / sugar have no group, so the ingredient-derived kcal
      is low; the direct-totals branch (the kcal metric) is unaffected.

    python -m timeprest.n5k.added.hm train --config n5k/added/configs/hm_single.yaml [--set added.hm.labels=groups]
    torchrun --standalone --nproc_per_node=2 -m timeprest.n5k.added.hm train --config n5k/added/configs/hm_timeprest.yaml
    python -m timeprest.n5k.added.hm locate --config <same> --ckpt <run dir>     # handmasks M1-M3 on the same data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from ... import utils
from ...config import load_config
from ..data import SPLITS, load_datasets
from ..handmasks import find_root, load_groups, read_regions
from ..handmasks import run as locate_run
from ..task import TOTALS
from ..train import EngineTrainer, checks_ok, write_summary


def group_labels(datasets, meta: dict, regions: dict, groups_file: str, log=print):
    """HM2: replace the K ingredient labels by the HandMasks groups. -> (datasets, meta)."""
    groups = sorted({r["group"] for v in regions.values() for r in v})
    gi = {g: j for j, g in enumerate(groups)}
    K, G = len(meta["ingredients"]), len(groups)
    of_k = load_groups(groups_file, meta["ingredients"], set(groups))
    members = defaultdict(list)
    for k, g in enumerate(of_k):
        if g is not None:
            members[g].append(k)
    fallback = defaultdict(int)
    grams = []
    for s, ds in zip(SPLITS, datasets):
        y = ds.targets.double()
        out = torch.zeros(len(ds), G, dtype=torch.float64)
        for i, d in enumerate(meta["ids"][s]):
            by = defaultdict(list)
            for r in regions[d]:
                by[r["group"]].append(r["mass"])
            for g, ms in by.items():
                if all(m is not None for m in ms):
                    out[i, gi[g]] = sum(ms)
                else:
                    out[i, gi[g]] = float(y[i, members[g]].sum()) if members[g] else 0.0
                    fallback["mapped" if out[i, gi[g]] > 0 else "median"] += 1
        grams.append(out)
    tr = grams[0]
    med = [float(tr[tr[:, j] > 0, j].median()) if (tr[:, j] > 0).any() else 1.0 for j in range(G)]
    for s, out in zip(SPLITS, grams):                       # regions with no mass at all: the group's train median
        for i, d in enumerate(meta["ids"][s]):
            for r in regions[d]:
                j = gi[r["group"]]
                if out[i, j] <= 0:
                    out[i, j] = med[j]
    regs = defaultdict(list)
    for d in meta["ids"]["train"]:
        for r in regions[d]:
            if r["mass"]:
                regs[r["group"]].append(r)
    pg_k = np.asarray(meta["per_gram"], dtype=np.float64)          # [K, T] of our ingredients (fallback)
    per_gram = []
    for g in groups:
        row = []
        for t in TOTALS:
            if t == "mass":
                row.append(1.0)
            elif regs.get(g):
                row.append(float(np.median([r[t] / r["mass"] for r in regs[g]])))
            else:                                                 # no train region with a mass: mapped ingredients
                row.append(float(pg_k[members[g], TOTALS.index(t)].mean()) if members[g] else 0.0)
        per_gram.append(row)
    present = tr > 0
    scale = np.maximum((tr * present).sum(0).numpy() / np.maximum(present.sum(0).numpy(), 1), 1.0)
    out_ds = []
    for ds, gr in zip(datasets, grams):
        new = ds.subset(range(len(ds)))
        new.targets = torch.cat([gr.float(), ds.targets[:, K:]], 1).contiguous()
        out_ds.append(new)
    meta = dict(meta, ingredients=[{"id": g, "name": g} for g in groups], per_gram=per_gram,
                gram_scale=scale.tolist())
    meta["hm"]["group_fallback"] = dict(fallback)
    log(f"[hm] labels = {G} HandMasks groups (was {K} ingredients); regions without a mass: {dict(fallback)}")
    return out_ds, meta


def hm_data(cfg: dict, log=print):
    """(train, val, test, meta) restricted to the HandMasks dishes of our splits."""
    hc, labels = cfg["handmasks"], cfg["added"]["hm"]["labels"]
    if labels not in ("ingredients", "groups"):
        raise ValueError(f"added.hm.labels must be ingredients or groups, got {labels!r}")
    train, val, test, meta = load_datasets(cfg["data"], log)
    regions = read_regions(find_root(hc))
    out, ids = [], {}
    for s, ds in zip(SPLITS, (train, val, test)):
        idx = [i for i, d in enumerate(meta["ids"][s]) if d in regions]
        out.append(ds.subset(idx))
        ids[s] = [meta["ids"][s][i] for i in idx]
    meta = dict(meta, ids=ids)
    meta["hm"] = {"labels": labels, "dishes": {s: len(v) for s, v in ids.items()}}
    log(f"[hm] HandMasks dishes kept: {meta['hm']['dishes']} (of {len(regions)}); labels: {labels}")
    if labels == "groups":
        out, meta = group_labels(out, meta, regions, hc["groups_file"], log)
    return (*out, meta)


def locate_cfg(cfg: dict) -> dict:
    """HM2 predicts the groups themselves: M3 pools each group's own CAM (identity mapping)."""
    if cfg["added"]["hm"]["labels"] == "groups":
        cfg = dict(cfg, handmasks=dict(cfg["handmasks"], groups_file="identity"))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "locate"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ckpt", default=None, help="locate: run directory (default output.dir)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    out_dir = cfg["output"]["dir"]
    utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    if args.cmd == "locate":
        device = utils.resolve_device(cfg["runtime"]["device"])
        locate_run(locate_cfg(cfg), args.ckpt or out_dir, device, data=hm_data(cfg))
        return
    one_stage = int(cfg["pipeline"]["num_stages"]) == 1
    rank0 = int(os.environ.get("RANK", "0")) == 0
    if cfg["output"].get("require_checks") and not args.force:
        ok, msg = checks_ok(cfg)
        if not ok:
            if rank0:
                print(f"[refuse] {msg} (use --force to override)")
            sys.exit(2)
    if os.path.exists(os.path.join(out_dir, "meta.json" if not one_stage else "checkpoint.pt")):
        if rank0:
            print(f"[refuse] {out_dir} already has a run; change output.dir")
        sys.exit(2)
    data = hm_data(cfg, log=print if rank0 else (lambda *_: None))
    if one_stage:
        device = utils.resolve_device(cfg["runtime"]["device"])
        print("env:", json.dumps(utils.env_info()), "| code", utils.code_hash(), flush=True)
        tr = EngineTrainer(cfg, device, out_dir, data=data)
        tr.fit()
        test = tr.test_best()
        write_summary(cfg, out_dir, test, {"hm": data[3]["hm"]})
        print("N5K_HM_JSON " + json.dumps({"name": cfg["name"], "labels": data[3]["hm"]["labels"],
                                           "test_mae_kcal": test["test_mae_kcal"]}), flush=True)
        return
    import torch.distributed as dist
    from ..dist_train import N5kDistTrainer
    tr = N5kDistTrainer(cfg, out_dir, data=data)
    if tr.rank == 0:
        print("env:", json.dumps(utils.env_info()), "| code", utils.code_hash(), "| gpus", torch.cuda.device_count(),
              flush=True)
    rows = tr.fit()
    if rows and rows[-1]["epoch"] >= cfg["training"]["epochs"]:
        test = tr.test_best()
        if tr.rank == 0:
            write_summary(cfg, out_dir, test, {"hm": data[3]["hm"]})
            print("N5K_HM_JSON " + json.dumps({"name": cfg["name"], "labels": data[3]["hm"]["labels"],
                                               "test_mae_kcal": test["test_mae_kcal"]}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
