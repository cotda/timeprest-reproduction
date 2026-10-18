"""Figures of the B8 slides (n5k/PLAN.md mục 9-11) from the downloaded Kaggle outputs in n5k/results/kaggle/.

    python n5k/report/make_figures.py        # -> n5k/report/figures/*.png (matplotlib, CPU only)

Every system keeps one colour and one line style in every figure (TiMePReSt blue solid, PipeDream orange dashed,
as in report/README.md); series are also labelled directly. Runs that are not downloaded yet are skipped.
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K = os.path.join(ROOT, "n5k", "results", "kaggle")
OUT = os.path.join(ROOT, "n5k", "report", "figures")
SEEDS = (0, 1, 2)
# categorical slots of the validated palette (dataviz reference), fixed per system
STYLE = {"timeprest": ("TiMePReSt N=2", "#2a78d6", "-", 2.2), "pipedream": ("PipeDream", "#eb6834", "--", 2.0),
         "single": ("1 GPU", "#1baf7a", ":", 2.0), "ddp": ("DDP", "#eda100", "-.", 2.0),
         "timeprest_n4": ("TiMePReSt N=4", "#e87ba4", "-", 1.2)}
GRAY, INK, INK2 = "#9b9a96", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "legend.frameon": False})


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main_run(system, s):
    """Run dir of the main configuration: B6 with w_total = 3 (run18, n5k/PLAN.md mục 10, 2026-10-17 20:27) when
    downloaded, else B6 with w_total = 1 (run6)."""
    p = os.path.join(K, f"run18_s{s}", "n5k_results", "runs", f"n5k_{system}_w3_s{s}")
    return p if os.path.exists(p) else os.path.join(K, f"run6_s{s}", "n5k_results", "runs", f"n5k_{system}_s{s}")


def main_explain(system):
    """explain dir of the B7 run of `system` (run18_s0 when downloaded, else run11_b7 / run13_pd)."""
    p = os.path.join(K, "run18_s0", "n5k_results", "explain", f"n5k_{system}_w3_s0")
    run = "run13_pd" if system == "pipedream" else "run11_b7"
    return p if os.path.exists(p) else os.path.join(K, run, "n5k_results", "explain", f"b7_{system}_s0")


def b6_runs(system):
    """[(cum train time [E], val MAE kcal [E], val mAP [E])] of the main B6 runs (main_run)."""
    out = []
    for s in SEEDS:
        p = os.path.join(main_run(system, s), "metrics.csv")
        if os.path.exists(p):
            h = read_csv(p)
            t = np.cumsum([float(r.get("epoch_time_s") or r["wall_time_s"]) for r in h])
            out.append((t, np.array([float(r["val_mae_kcal"]) for r in h]), np.array([float(r["val_ingr_map"]) for r in h])))
    return out


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_mae_curves():
    """(a) val MAE kcal by epoch (mean of 3 seeds); (b) best-so-far val MAE kcal by train time."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9))
    grid = np.linspace(0, 800, 401)
    for system in STYLE:
        runs = b6_runs(system)
        if not runs:
            continue
        name, color, ls, lw = STYLE[system]
        m = np.mean([r[1] for r in runs], 0)
        ep = np.arange(1, len(m) + 1)
        a.plot(ep, m, color=color, ls=ls, lw=lw, label=name)
        best = []
        for t, v, _ in runs:
            bs = np.minimum.accumulate(v)
            idx = np.searchsorted(t, grid, side="right") - 1
            best.append(np.where(idx >= 0, bs[np.clip(idx, 0, None)], np.nan))
        bm = np.nanmean(best, 0) if all(np.isfinite(x).any() for x in best) else None
        stop = grid <= np.mean([t[-1] for t, _, _ in runs])
        b.plot(grid[stop], bm[stop], color=color, ls=ls, lw=lw, label=name)
    a.set(xlabel="epoch", ylabel="val MAE kcal (mean of 3 seeds)", ylim=(40, 150), xlim=(1, 30))
    a.set_title("(a) by epoch", loc="left", fontsize=10, color=INK)
    b.set(xlabel="training time (s, same machine per seed)", ylabel="best val MAE kcal so far", ylim=(40, 150),
          xlim=(0, 800))
    b.set_title("(b) at equal training time", loc="left", fontsize=10, color=INK)
    a.legend(loc="upper right", fontsize=8.5, ncol=2)
    b.legend(loc="upper right", fontsize=8.5, ncol=2)
    save(fig, "fig_mae_epoch_time.png")


def fig_oscillation():
    """Per-seed val MAE kcal of TiMePReSt and PipeDream: TiMePReSt oscillates in epochs 6-18."""
    tp, pd = b6_runs("timeprest"), b6_runs("pipedream")
    if not tp or not pd:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    top = max(170, 10 * np.ceil((max(max(r[1]) for r in tp + pd) + 15) / 10))
    for s, ax in enumerate(axes):
        ax.axvspan(6, 18, color="#f0efec", lw=0)
        for system, runs in (("pipedream", pd), ("timeprest", tp)):
            name, color, ls, lw = STYLE[system]
            v = runs[s][1]
            ax.plot(np.arange(1, len(v) + 1), v, color=color, ls=ls, lw=lw, marker="o", ms=3, label=name)
        ax.set_title(f"seed {s}", loc="left", fontsize=10, color=INK)
        ax.set(xlabel="epoch", ylim=(40, top))
    axes[0].set_ylabel("val MAE kcal")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[1].text(12, top - 7, "epoch 6-18", ha="center", fontsize=8.5, color=INK2)
    save(fig, "fig_oscillation.png")


def fig_time():
    """(a) epoch time of the 5 systems (main B6, mean of 3 seeds); (b) the systems at emulated bandwidth (run17)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    names, vals = [], []
    for system in ("single", "ddp", "pipedream", "timeprest", "timeprest_n4"):
        ss = [json.load(open(os.path.join(main_run(system, s), "summary.json"))) for s in SEEDS]
        names.append(system)
        vals.append(np.mean([x["mean_epoch_time_s"] for x in ss]))
    x = np.arange(len(names))
    a.bar(x, vals, width=0.62, color=[STYLE[n][1] for n in names], edgecolor="#fcfcfb", linewidth=2)
    for i, v in enumerate(vals):
        a.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    a.set_xticks(x, [STYLE[n][0].replace(" N=", "\nN=") for n in names], fontsize=8.5)
    a.set(ylabel="s / epoch (mean of 3 seeds)", ylim=(0, max(vals) * 1.18))
    a.set_title("(a) 2 x T4, one machine (B6)", loc="left", fontsize=10, color=INK)
    a.grid(axis="x", visible=False)
    # run17 measured DDP (twice, around the pipeline bench) and the pipeline on one machine; run6_bench had no DDP
    r17 = os.path.join(K, "run17_ddp_slow", "n5k_results")
    systems = ("timeprest", "pipedream")
    if os.path.exists(os.path.join(r17, "bench_comm_ddp", "rep2", "bench_comm_ddp.json")):
        rows = json.load(open(os.path.join(r17, "bench_comm", "bench_comm.json")))["rows"]
        reps = [json.load(open(os.path.join(r17, "bench_comm_ddp", f"rep{i}", "bench_comm_ddp.json")))["rows"] for i in (1, 2)]
        for bw in (None, 2, 1):
            rows.append({"system": "ddp", "bandwidth_gbps": bw, "epoch_time_s": float(np.mean(
                [next(r["epoch_time_s"] for r in rep if r["system"] == "ddp" and r["bandwidth_gbps"] == bw) for rep in reps]))})
        systems = ("timeprest", "pipedream", "ddp")
    else:
        rows = json.load(open(os.path.join(K, "run6_bench", "n5k_results", "bench_comm", "bench_comm.json")))["rows"]
    bws = [None, 2, 1]
    w = 0.72 / len(systems)
    for j, system in enumerate(systems):
        v = [next(r["epoch_time_s"] for r in rows if r["system"] == system and r["bandwidth_gbps"] == bw) for bw in bws]
        xs = np.arange(3) + (j - (len(systems) - 1) / 2) * w
        b.bar(xs, v, width=w, color=STYLE[system][1], edgecolor="#fcfcfb", linewidth=2, label=STYLE[system][0])
        for xx, vv in zip(xs, v):
            b.text(xx, vv + 0.6, f"{vv:.1f}", ha="center", fontsize=8.5, color=INK)
    for i, bw in enumerate(bws):
        tp = next(r["epoch_time_s"] for r in rows if r["system"] == "timeprest" and r["bandwidth_gbps"] == bw)
        pd = next(r["epoch_time_s"] for r in rows if r["system"] == "pipedream" and r["bandwidth_gbps"] == bw)
        b.text(i, max(tp, pd) + 4.5, f"TP/PD x{tp / pd:.2f}", ha="center", fontsize=8, color=INK2)
    b.set_xticks(range(3), ["same machine", "2 Gbit/s", "1 Gbit/s"])
    b.set(ylabel="s / epoch", ylim=(0, 60))
    b.set_title("(b) emulated link between the 2 GPUs" + (" (run17, one machine)" if "ddp" in systems else ""), loc="left",
                fontsize=10, color=INK)
    b.legend(loc="upper left", fontsize=8.5)
    b.grid(axis="x", visible=False)
    save(fig, "fig_time.png")


def fig_deletion():
    """Deletion test at the decision c (map vs random cells), one panel per system with an explain run."""
    found = []
    for system in ("timeprest", "pipedream", "single"):
        p = os.path.join(main_explain(system), "deletion.json")
        if os.path.exists(p):
            found.append((system, json.load(open(p))))
    if not found:
        return
    measures = [("pairs", "drop_p", "prob.\nof k"), ("pairs", "drop_g", "grams\nof k"),
                ("dishes", "drop_kcal_ingr", "kcal\n(ingr. map)"), ("dishes", "drop_kcal_direct", "kcal\n(direct map)")]
    fig, axes = plt.subplots(1, len(found), figsize=(4.4 * len(found), 3.3), sharey=True, squeeze=False)
    for ax, (system, d) in zip(axes[0], found):
        c = d["rule"]["cells"]
        x = np.arange(len(measures))
        mv = [d["stats"][k][f"{m}@{c}"]["map"] for k, m, _ in measures]
        rv = [d["stats"][k][f"{m}@{c}"]["rand"] for k, m, _ in measures]
        ax.bar(x - 0.19, mv, width=0.38, color=STYLE[system][1], edgecolor="#fcfcfb", linewidth=2, label="mask map cells")
        ax.bar(x + 0.19, rv, width=0.38, color=GRAY, edgecolor="#fcfcfb", linewidth=2, label="mask random cells")
        for xx, v in zip(x - 0.19, mv):
            ax.text(xx, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set_xticks(x, [m[2] for m in measures], fontsize=8.5)
        ax.set_title(f"{STYLE[system][0]}, c = {c} of 49 cells", loc="left", fontsize=10, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0][0].set(ylabel="drop after masking", ylim=(-0.15, 0.8))
    axes[0][0].legend(loc="upper right", fontsize=8)
    save(fig, "fig_deletion.png")


def crop_examples():
    """The image + map rows of two B7 figures (TiMePReSt seed 0) for the slides."""
    src = os.path.join(main_explain("timeprest"), "figures")
    for dish in ("1562170356", "1563295400"):
        p = os.path.join(src, f"dish_dish_{dish}.png")
        if os.path.exists(p):
            im = Image.open(p)
            w, h = im.size
            im.crop((0, int(0.045 * h), w, int(0.70 * h))).save(os.path.join(OUT, f"fig_map_{dish}.png"))
            print("wrote", f"fig_map_{dish}.png")


def fig_lr_sweep():
    """30-epoch lr sweep of TiMePReSt (run 12) and PipeDream (run 13) with the B6 lr: mean best val MAE kcal."""
    rows = []
    for system, b6_lr, run, lrs in (("timeprest", 0.01, "run12_tp_lr", (0.02, 0.005)),
                                    ("pipedream", 0.02, "run13_pd", (0.01, 0.005))):
        for lr in sorted((b6_lr,) + lrs, reverse=True):
            vals = []
            for s in SEEDS:
                if lr == b6_lr:
                    p = os.path.join(K, f"run6_s{s}", "n5k_results", "runs", f"n5k_{system}_s{s}", "metrics.csv")
                else:
                    p = os.path.join(K, run, "n5k_results", "runs", f"n5k_{system}_lr{lr:g}_s{s}", "metrics.csv")
                if os.path.exists(p):
                    vals.append(min(float(r["val_mae_kcal"]) for r in read_csv(p)))
            if len(vals) == len(SEEDS):
                rows.append((system, lr, np.mean(vals), np.std(vals, ddof=1)))
    if len({r[0] for r in rows}) < 1 or not any(r[1] not in (0.01, 0.02) for r in rows):
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    for system in ("timeprest", "pipedream"):
        rr = [r for r in rows if r[0] == system]
        if rr:
            name, color, ls, lw = STYLE[system]
            ax.errorbar([r[1] for r in rr], [r[2] for r in rr], yerr=[r[3] for r in rr], color=color, ls=ls, lw=lw,
                        marker="o", ms=6, capsize=3, label=name)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([0.005, 0.01, 0.02], ["0.005", "0.01", "0.02"])
    ax.set(xlabel="lr", ylabel="best val MAE kcal (3 seeds)")
    ax.legend(fontsize=8.5)
    save(fig, "fig_lr_sweep.png")


if __name__ == "__main__":
    fig_mae_curves()
    fig_oscillation()
    fig_time()
    fig_deletion()
    crop_examples()
    fig_lr_sweep()
