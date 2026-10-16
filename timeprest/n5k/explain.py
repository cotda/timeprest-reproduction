"""B7 (n5k/PLAN.md mục 7, 9, 10 - 2026-10-16): calorie maps, figures and the deletion test of a trained model.

    python -m timeprest.n5k.explain --config n5k/configs/n5k_timeprest.yaml --set seed=0 name=n5k_timeprest_s0
    (checkpoint: --ckpt DIR holding best.pt (engine, 1 GPU) or best_stage<r>.pt (pipeline); default output.dir)

Maps at the head's resolution h x w (7 x 7 for 224 px images):
  CAM_k                    presence map of ingredient k (N5kHead.maps)
  ingredient calorie map   C = sum over k predicted present of kcal_k * A_k, A_k = ReLU(CAM_k) / sum ReLU(CAM_k)
                           (uniform where ReLU(CAM_k) is 0 everywhere), kcal_k = predicted grams_k * kcal per gram;
                           sum of C = kcal of the ingredients predicted present
  direct kcal map          kcal channel of the direct-totals map (ranking only; its sum has no meaning)
Deletion test: an image cell (image_size / h px) is masked with 0 (= the data mean after normalisation). For each
(dish, k) with k predicted and truly present: mask the c cells with the highest CAM_k versus c random cells (mean of
`random_draws` draws, shared by every k of the dish); per dish the same for the two kcal maps. Rule (PLAN mục 10):
a map is faithful to the model if at c = rule_cells the drop with the map is larger than with random cells, one-sided
paired Wilcoxon p < rule_alpha. It says nothing about the true location of the food.
Outputs in explain.dir: deletion_pairs.csv, deletion_dishes.csv, deletion.json, figures/dish_<id>.png, figures.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils
from ..config import load_config
from ..models import build_blocks, build_stages
from .common import prepare
from .task import TOTALS

PAIR_MEASURES = ("drop_p", "drop_g", "off_p")          # presence drop of k, relative gram drop of k, |dp| of others
DISH_MEASURES = ("drop_kcal_ingr", "drop_kcal_direct")  # relative kcal drop (ingredient map / direct map)
RULES = {"cam_presence": ("pairs", "drop_p"), "cam_gram": ("pairs", "drop_g"),
         "calorie_map_ingr": ("dishes", "drop_kcal_ingr"), "kcal_map_direct": ("dishes", "drop_kcal_direct")}


# ---------------------------------------------------------------------------- model
def merge_stage_states(states: list[dict], bounds: list[int]) -> dict:
    """State dicts of the stages (nn.Sequential of blocks bounds[s]..bounds[s+1]) -> one state dict of the
    whole nn.Sequential of blocks."""
    out = {}
    for start, sd in zip(bounds[:-1], states):
        for k, v in sd.items():
            i, rest = k.split(".", 1)
            out[f"{int(i) + start}.{rest}"] = v
    return out


def load_model(cfg: dict, ckpt_dir: str, sample: torch.Tensor | None = None) -> tuple[nn.Sequential, int]:
    """Whole model (nn.Sequential of blocks) with the best checkpoint of a run; returns (model, best epoch).
    cfg: config with the model section completed by `prepare` (num_ingredients, priors)."""
    mcfg = dict(cfg["model"], pretrained=None)               # the checkpoint holds every weight
    one = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(one):
        st = torch.load(one, map_location="cpu", weights_only=False)
        states, epoch = st["engine"]["stages"], st["epoch"]
    else:
        files = sorted(glob.glob(os.path.join(ckpt_dir, "best_stage*.pt")),
                       key=lambda p: int(re.search(r"best_stage(\d+)\.pt$", p).group(1)))
        if not files:
            raise FileNotFoundError(f"no best.pt or best_stage*.pt in {ckpt_dir}")
        sts = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
        states, epoch = [s["module"] for s in sts], sts[0]["epoch"]
    blocks = build_blocks(mcfg)
    if len(states) == 1:
        bounds = [0, len(blocks)]
    else:
        _, bounds = build_stages(mcfg, len(states), cfg["pipeline"]["partition"], sample)
    model = nn.Sequential(*blocks)
    model.load_state_dict(merge_stage_states(states, bounds))
    return model.eval(), int(epoch)


@torch.no_grad()
def forward_maps(model: nn.Sequential, task, x: torch.Tensor):
    """(predictions, CAM [B, K, h, w], direct kcal map [B, h, w] or None)."""
    feats = model[:-1](x)
    head = model[-1]
    cam, _ = head.maps(feats)
    pred = task.predict(head(feats))
    tmap = head.total_map(feats)[:, TOTALS.index("kcal")] if head.total is not None else None
    return pred, cam, tmap


def ingredient_calorie_map(cam: torch.Tensor, grams: torch.Tensor, present: torch.Tensor,
                           kcal_per_gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cam [B, K, h, w], grams [B, K], present bool [B, K], kcal_per_gram [K] -> (C [B, h, w], A [B, K, h, w])."""
    a = cam.clamp_min(0)
    s = a.sum((2, 3), keepdim=True)
    A = torch.where(s > 0, a / s.clamp_min(1e-12), torch.full_like(a, 1.0 / (a.shape[2] * a.shape[3])))
    kcal = grams * kcal_per_gram.to(grams) * present.to(grams)
    return (kcal[:, :, None, None] * A).sum(1), A


def mask_cells(x: torch.Tensor, cells, grid: tuple[int, int]) -> torch.Tensor:
    """Copy of x [C, H, W] with the given flat cells of an h x w grid set to 0."""
    h, w = grid
    ch, cw = x.shape[1] // h, x.shape[2] // w
    y = x.clone()
    for c in cells:
        r, q = divmod(int(c), w)
        y[:, r * ch:(r + 1) * ch, q * cw:(q + 1) * cw] = 0
    return y


def top_cells(m: torch.Tensor, c: int) -> list[int]:
    return torch.argsort(m.flatten(), descending=True)[:c].tolist()


# ---------------------------------------------------------------------------- deletion test
def _wilcoxon_greater(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if len(d) == 0 or not np.any(d != 0):
        return 1.0
    from scipy.stats import wilcoxon
    return float(wilcoxon(d, alternative="greater").pvalue)


def stats(rows: list[dict], measures, cells) -> dict:
    out = {}
    for c in cells:
        rc = [r for r in rows if r["c"] == c]
        for m in measures:
            a = np.array([r[m + "_map"] for r in rc], dtype=float)
            b = np.array([r[m + "_rand"] for r in rc], dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            out[f"{m}@{c}"] = {"n": int(len(a)), "map": float(a.mean()) if len(a) else None,
                               "rand": float(b.mean()) if len(b) else None,
                               "frac_map_gt_rand": float((a > b).mean()) if len(a) else None,
                               "p_wilcoxon": _wilcoxon_greater(a - b)}
    return out


@torch.no_grad()
def deletion_test(model, task, ds, ids, ec: dict, device, base: dict, log=print) -> tuple[list, list]:
    """Per-pair and per-dish rows (see module docstring). base: outputs of `collect` on ds."""
    thr, cells, R = ec["presence_threshold"], list(ec["cells"]), int(ec["random_draws"])
    rng = np.random.default_rng(ec["seed"])
    kpg = task.per_gram[:, TOTALS.index("kcal")].float()
    bs = base["eval_batch_size"]
    pairs, dishes = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        p0, g0 = base["prob"][i], base["grams"][i]
        present = p0 >= thr
        ks = torch.nonzero(present & (y[:task.K] > 0)).flatten().tolist()
        grid = tuple(base["cam"].shape[2:])
        kin0 = float((g0 * kpg * present).sum())
        kdir0 = float(base["kcal_direct"][i]) if base["kcal_direct"] is not None else float("nan")
        variants = []                                   # (kind, c, k or None, masked image)
        for c in cells:
            for _ in range(R):
                variants.append(("rand", c, None, mask_cells(x, rng.choice(grid[0] * grid[1], c, replace=False), grid)))
            for k in ks:
                variants.append(("cam", c, k, mask_cells(x, top_cells(base["cam"][i, k], c), grid)))
            variants.append(("cmap", c, None, mask_cells(x, top_cells(base["cmap"][i], c), grid)))
            if base["tmap"] is not None:
                variants.append(("tmap", c, None, mask_cells(x, top_cells(base["tmap"][i], c), grid)))
        outs = []
        for s in range(0, len(variants), bs):
            xb = torch.stack([v[3] for v in variants[s:s + bs]]).to(device)
            pr = task.predict(model(xb))
            outs += [(torch.sigmoid(pr["logits"]).float().cpu(), pr["grams"].float().cpu(),
                      pr["totals"][:, TOTALS.index("kcal")].float().cpu())]
        prob = torch.cat([o[0] for o in outs]); grams = torch.cat([o[1] for o in outs])
        kdir = torch.cat([o[2] for o in outs])

        def pair_vals(j, k):
            others = [o for o in torch.nonzero(present).flatten().tolist() if o != k]
            return (float(p0[k] - prob[j, k]), float((g0[k] - grams[j, k]) / g0[k].clamp_min(1e-6)),
                    float((p0[others] - prob[j, others]).abs().mean()) if others else float("nan"))

        def dish_vals(j):
            kin = float((grams[j] * kpg * present).sum())
            return ((kin0 - kin) / max(kin0, 1e-6), (kdir0 - float(kdir[j])) / max(kdir0, 1e-6))

        for c in cells:
            rj = [j for j, v in enumerate(variants) if v[0] == "rand" and v[1] == c]
            for k in ks:
                j = next(j for j, v in enumerate(variants) if v[0] == "cam" and v[1] == c and v[2] == k)
                m = pair_vals(j, k)
                r = np.nanmean(np.array([pair_vals(jj, k) for jj in rj]), axis=0) if rj else [np.nan] * 3
                pairs.append({"dish_id": ids[i], "k": k, "c": c, "p0": round(float(p0[k]), 5), "grams0": round(float(g0[k]), 3),
                              **{f"{n}_map": round(a, 6) for n, a in zip(PAIR_MEASURES, m)},
                              **{f"{n}_rand": round(float(a), 6) for n, a in zip(PAIR_MEASURES, r)}})
            jc = next(j for j, v in enumerate(variants) if v[0] == "cmap" and v[1] == c)
            jt = next((j for j, v in enumerate(variants) if v[0] == "tmap" and v[1] == c), None)
            rnd = np.array([dish_vals(jj) for jj in rj]).mean(0) if rj else [np.nan] * 2
            dishes.append({"dish_id": ids[i], "c": c, "kcal_ingr0": round(kin0, 3), "kcal_direct0": round(kdir0, 3),
                           "drop_kcal_ingr_map": round(dish_vals(jc)[0], 6), "drop_kcal_ingr_rand": round(float(rnd[0]), 6),
                           "drop_kcal_direct_map": round(dish_vals(jt)[1], 6) if jt is not None else float("nan"),
                           "drop_kcal_direct_rand": round(float(rnd[1]), 6)})
        if (i + 1) % 100 == 0:
            log(f"[explain] deletion {i + 1}/{len(ds)} dishes, {len(pairs)} pair rows")
    return pairs, dishes


# ---------------------------------------------------------------------------- figures
@torch.no_grad()
def collect(model, task, ds, ec: dict, device, bs: int) -> dict:
    """Predictions and maps of every image of ds (no augmentation), on the CPU."""
    kpg = task.per_gram[:, TOTALS.index("kcal")].float()
    keep = {"prob": [], "grams": [], "totals": [], "totals_ingr": [], "cam": [], "tmap": [], "cmap": []}
    for s in range(0, len(ds), bs):
        x = torch.stack([ds[i][0] for i in range(s, min(len(ds), s + bs))]).to(device)
        pred, cam, tmap = forward_maps(model, task, x)
        prob, grams = torch.sigmoid(pred["logits"]).float().cpu(), pred["grams"].float().cpu()
        cam = cam.float().cpu()
        cmap, _ = ingredient_calorie_map(cam, grams, prob >= ec["presence_threshold"], kpg)
        for k, v in (("prob", prob), ("grams", grams), ("totals", pred["totals"].float().cpu()),
                     ("totals_ingr", pred["totals_ingr"].float().cpu()), ("cam", cam), ("cmap", cmap),
                     ("tmap", tmap.float().cpu() if tmap is not None else None)):
            keep[k].append(v)
    out = {k: (torch.cat(v) if v and v[0] is not None else None) for k, v in keep.items()}
    out["kcal_direct"] = out["totals"][:, TOTALS.index("kcal")] if task.total_scale is not None else None
    out["eval_batch_size"] = bs
    return out


def pick_dishes(err: np.ndarray, n: int, seed: int) -> list[int]:
    """n // 2 dishes at the (2j+1)/(2 n//2) quantiles of the error, the rest drawn at random from the others."""
    order = np.argsort(err, kind="stable")
    nq = n // 2
    picks = [int(order[int(round((2 * j + 1) / (2 * nq) * (len(err) - 1)))]) for j in range(nq)] if nq else []
    rest = [i for i in range(len(err)) if i not in picks]
    rng = np.random.default_rng(seed)
    picks += [int(i) for i in rng.choice(rest, min(n - nq, len(rest)), replace=False)]
    return picks


def _up(m: torch.Tensor, size) -> np.ndarray:
    return F.interpolate(m[None, None].float(), size=size, mode="bilinear", align_corners=False)[0, 0].numpy()


def dish_figure(path: str, i: int, dish_id: str, ds, y: torch.Tensor, base: dict, names: list[str], task, ec: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = task.K
    kpg = task.per_gram[:, TOTALS.index("kcal")].float()
    rgb = ds.rgb[i].numpy()
    dep = ds.depth[i].float().numpy()
    size = rgb.shape[:2]
    p, g = base["prob"][i], base["grams"][i]
    gt = y[:K]
    present = p >= ec["presence_threshold"]
    top = [k for k in torch.argsort(p, descending=True)[:ec["top_cams"]].tolist()]
    fig = plt.figure(figsize=(16, 12.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8])
    axes = np.array([[fig.add_subplot(gs[r, q]) for q in range(4)] for r in range(2)])
    panels = [("RGB", None), ("depth", None), ("calories from ingredients (kcal / cell)", base["cmap"][i]),
              ("direct kcal map (relative)", base["tmap"][i] if base["tmap"] is not None else None)]
    A = base["cam"][i].clamp_min(0)
    for k in top:
        panels.append((f"CAM {names[k][:24]} (p={float(p[k]):.2f})", A[k]))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, m) in zip(axes.flat, panels):
        if title == "depth":
            ax.imshow(dep, cmap="gray")
        else:
            ax.imshow(rgb)
            if m is not None:
                im = ax.imshow(_up(m, size), cmap="jet", alpha=0.45)
                if title.startswith("calories"):
                    fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=9)
    rows = sorted(set(torch.nonzero(present).flatten().tolist()) | set(torch.nonzero(gt > 0).flatten().tolist()),
                  key=lambda k: -max(float(g[k] * kpg[k]) * bool(present[k]), float(gt[k] * kpg[k])))[:12]
    tot_t = y[K:].tolist()
    lines = [f"{'ingredient':<26}{'p':>6}{'g pred':>9}{'g true':>9}{'kcal pred':>11}{'kcal true':>11}"]
    for k in rows:
        lines.append(f"{names[k][:25]:<26}{float(p[k]):>6.2f}{float(g[k]):>9.1f}{float(gt[k]):>9.1f}"
                     f"{float(g[k] * kpg[k]) * bool(present[k]):>11.1f}{float(gt[k] * kpg[k]):>11.1f}")
    lines.append("")
    lines.append(f"{'dish total':<26}" + "".join(f"{t:>10}" for t in TOTALS))
    lines.append(f"{'predicted (direct)':<26}" + "".join(f"{float(v):>10.1f}" for v in base["totals"][i]))
    lines.append(f"{'predicted (ingredients)':<26}" + "".join(f"{float(v):>10.1f}" for v in base["totals_ingr"][i]))
    lines.append(f"{'true':<26}" + "".join(f"{v:>10.1f}" for v in tot_t))
    lines.append(f"(kcal pred is shown for ingredients with p >= {ec['presence_threshold']}; "
                 f"sum of the calorie map = {float(base['cmap'][i].sum()):.1f} kcal)")
    tab = fig.add_subplot(gs[2, :])
    tab.axis("off")
    tab.text(0, 1, "\n".join(lines), family="monospace", fontsize=9, va="top", transform=tab.transAxes)
    fig.suptitle(f"{dish_id}: kcal true {tot_t[0]:.0f}, predicted {float(base['totals'][i][0]):.0f}")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.01, wspace=0.12, hspace=0.18)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return {"dish_id": dish_id, "index": i, "kcal_true": tot_t[0], "kcal_pred": float(base["totals"][i][0]),
            "top_ingredients": [{"name": names[k], "p": round(float(p[k]), 4), "grams_pred": round(float(g[k]), 2),
                                 "grams_true": float(gt[k])} for k in top]}


# ---------------------------------------------------------------------------- main
def run(cfg: dict, ckpt_dir: str, device, data=None, log=print) -> dict:
    ec = cfg["explain"]
    out_dir = ec["dir"].format(name=cfg["name"])
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    built, (_, _, test), meta, task = prepare(cfg, data, log=log)
    model, epoch = load_model(built, ckpt_dir, test[0][0].unsqueeze(0))
    model.to(device)
    ids = meta["ids"]["test"] if "ids" in meta else [str(i) for i in range(len(test))]
    names = [d.get("name") or str(d.get("id")) for d in meta["ingredients"]]
    bs = int(cfg["runtime"]["eval_batch_size"])
    base = collect(model, task, test, ec, device, bs)
    y = test.targets
    kcal_pred = (base["kcal_direct"] if base["kcal_direct"] is not None else base["totals_ingr"][:, 0]).numpy()
    err = np.abs(kcal_pred - y[:, task.K].numpy())
    log(f"[explain] {ckpt_dir} (best epoch {epoch}): test MAE kcal {err.mean():.2f} on {len(test)} dishes")
    figs = [dish_figure(os.path.join(out_dir, "figures", f"dish_{ids[i]}.png"), i, ids[i], test, y[i], base, names,
                        task, ec) for i in pick_dishes(err, int(ec["figure_dishes"]), int(ec["seed"]))]
    utils.save_json(figs, os.path.join(out_dir, "figures.json"))
    pairs, dishes = deletion_test(model, task, test, ids, ec, device, base, log)
    for name, rows in (("deletion_pairs.csv", pairs), ("deletion_dishes.csv", dishes)):
        if os.path.exists(os.path.join(out_dir, name)):
            os.remove(os.path.join(out_dir, name))
        for r in rows:
            utils.append_csv(r, os.path.join(out_dir, name))
    st = {"pairs": stats(pairs, PAIR_MEASURES, ec["cells"]), "dishes": stats(dishes, DISH_MEASURES, ec["cells"])}
    c = ec["rule_cells"]
    verdict = {}
    for rule, (kind, m) in RULES.items():
        s = st[kind].get(f"{m}@{c}", {})
        verdict[rule] = bool(s.get("n") and s["map"] > s["rand"] and s["p_wilcoxon"] < ec["rule_alpha"])
    res = {"name": cfg["name"], "checkpoint": ckpt_dir, "best_epoch": epoch, "test_mae_kcal": float(err.mean()),
           "n_dishes": len(test), "n_pairs": len({(r["dish_id"], r["k"]) for r in pairs}),
           "rule": {"cells": c, "alpha": ec["rule_alpha"]}, "faithful": verdict, "stats": st,
           "explain": ec, "code_hash": utils.code_hash()}
    utils.save_json(res, os.path.join(out_dir, "deletion.json"))
    short = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in st[RULES[k][0]][f"{RULES[k][1]}@{c}"].items()}
             for k in RULES}
    print("N5K_EXPLAIN_JSON", json.dumps({"name": cfg["name"], "test_mae_kcal": round(float(err.mean()), 3),
                                          "n_pairs": res["n_pairs"], "faithful": verdict, f"at_c{c}": short}), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", default=None, help="run directory with best.pt or best_stage*.pt (default output.dir)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    device = utils.resolve_device(cfg["runtime"]["device"])
    run(cfg, args.ckpt or cfg["output"]["dir"], device)


if __name__ == "__main__":
    main()
