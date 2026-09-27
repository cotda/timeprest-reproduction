"""Figures of our runs laid out like the paper's figures, with the paper's values (read from its
figures, paper_notes §11) drawn as reference marks, so the two can be compared side by side.

    python report/make_figures.py          # writes report/figures/*.png
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures")
R = lambda *p: os.path.join(ROOT, "results", *p)

# validated categorical palette (dataviz reference, light): fixed order, identity follows the system
COL = {"TiMePReSt": "#2a78d6", "PipeDream": "#eb6834", "PipeDream-vsync": "#1baf7a", "DeepSpeed": "#eda100",
       "PipeDream M=64": "#e87ba4", "Variant 1": "#1baf7a", "Variant 2": "#eda100",
       "N=2": "#1baf7a", "N=2, M=384": "#eda100"}
DASH = {"TiMePReSt": "-", "PipeDream": "--", "PipeDream-vsync": "-.", "DeepSpeed": ":", "PipeDream M=64": (0, (5, 1, 1, 1)),
        "Variant 1": "-.", "Variant 2": ":", "N=2": "-.", "N=2, M=384": ":"}
INK, INK2, GRID, SURF, PAPER = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def load(path: str) -> dict:
    rows = list(csv.DictReader(open(os.path.join(path, "metrics.csv"), encoding="utf-8")))
    f = lambda k: [float(r[k]) for r in rows]
    return {"acc1": f("test_acc1"), "acc5": f("test_acc5"), "loss": f("test_loss"), "cum": f("cum_time_s"),
            "t": f("epoch_time_s"), "mem": rows[-1]["peak_mem_mb"], "epoch": list(range(1, len(rows) + 1))}


def line(ax, x, y, name, label_end=True, dy=0.0):
    ax.plot(x, y, color=COL[name], ls=DASH[name], label=name)
    if label_end:   # end value; dy staggers the labels of series that end close together
        ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(4, dy), textcoords="offset points", fontsize=8,
                    color=INK2, va="center")


def paper_range(ax, x, lo, hi, text, color):
    """A value read from a paper figure: a vertical range bar with a hollow marker."""
    ax.errorbar([x], [(lo + hi) / 2], yerr=[[(hi - lo) / 2], [(hi - lo) / 2]], fmt="o", mfc=SURF, mec=color,
                ecolor=color, ms=8, mew=2, capsize=4, zorder=5)
    ax.annotate(text, (x, hi), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)


def grouped_bars(ax, data: dict, value, label, hatch_parts: bool = False):
    """data: workload -> {system: raw}; value(raw) -> list of stacked parts. Workload names on the x axis,
    systems identified by color (legend), value printed on top of each bar."""
    from matplotlib.patches import Patch
    x, centers, seen = 0.0, [], {}
    for wl, systems in data.items():
        start = x
        for name, raw in systems.items():
            parts, bottom = value(raw), 0.0
            for k, p in enumerate(parts):
                ax.bar(x, p, bottom=bottom, width=0.8, color=COL[name], edgecolor=SURF, linewidth=2,
                       hatch="///" if (hatch_parts and k > 0) else None, alpha=1.0 if k == 0 else 0.55)
                bottom += p
            ax.annotate(label(bottom), (x, bottom), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK2)
            seen[name] = Patch(color=COL[name], label=name)
            x += 1
        centers.append(((start + x - 1) / 2, wl))
        x += 0.8
    ax.set_xticks([c for c, _ in centers], [w.replace(" / ", "\n") for _, w in centers], fontsize=8.5)
    ax.grid(axis="x", visible=False)
    return list(seen.values())


def curves_figure(runs: dict, title: str, fname: str, time_ok: bool, paper: dict | None = None, note: str = ""):
    """Paper Fig.4/5/8 layout: rows = top-1, top-5, test loss; left = time points, right = epochs."""
    ref = st.mean(next(iter(runs.values()))["t"])        # 1 time point = mean epoch time of the first system
    cols = 2 if time_ok else 1
    if time_ok:
        fig, axes = plt.subplots(3, cols, figsize=(5.2 * cols, 9.6), squeeze=False)
    else:                                  # epoch axis only: three panels side by side
        fig, row_axes = plt.subplots(1, 3, figsize=(15, 4.4))
        axes = [[a] for a in row_axes]
    for row, (key, lab) in enumerate((("acc1", "Top-1 accuracy (%)"), ("acc5", "Top-5 accuracy (%)"), ("loss", "Test loss"))):
        order = sorted(runs, key=lambda n: runs[n][key][-1], reverse=True)   # stagger end labels top to bottom
        for name, d in runs.items():
            dy = (len(order) - 1) / 2 * 9 - order.index(name) * 9
            if time_ok:
                line(axes[row][0], [c / ref for c in d["cum"]], d[key], name, dy=dy)
            line(axes[row][cols - 1], d["epoch"], d[key], name, dy=dy)
        if time_ok:
            axes[row][0].set_xlabel(f"Time points (1 = mean epoch time of {next(iter(runs))})")
        axes[row][cols - 1].set_xlabel("Epochs")
        for ax in axes[row]:
            ax.set_ylabel(lab)
        if paper and key in paper:
            for (panel, x, lo, hi, text, name) in paper[key]:
                ax = axes[row][0 if panel == "time" and time_ok else cols - 1]
                if panel == "epochs":            # right of the data, so the end labels stay readable
                    n = len(next(iter(runs.values()))["epoch"])
                    x = n * (1.12 if name == "TiMePReSt" else 1.22)
                    ax.set_xlim(right=n * 1.3)
                paper_range(ax, x, lo, hi, text, COL[name])
    axes[0][0].legend(loc="lower right")
    fig.suptitle(title, fontweight="bold", color=INK)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, (0.035 if time_ok else 0.09) if note else 0, 1, 0.97 if time_ok else 0.9))
    fig.savefig(os.path.join(OUT, fname), dpi=140)
    plt.close(fig)


def mean_band(ax, runs: list[dict], key: str, name: str):
    n = min(len(r[key]) for r in runs)
    ys = [[r[key][i] for r in runs] for i in range(n)]
    m = [st.mean(v) for v in ys]
    s = [st.stdev(v) for v in ys]
    x = list(range(1, n + 1))
    ax.fill_between(x, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], color=COL[name], alpha=0.18, lw=0)
    line(ax, x, m, f"{name}", label_end=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    c2 = R("phase2", "run2_lr0.02_timeprest-graph")
    tin = R("tinyimagenet")
    PAPER_NOTE = "Hollow markers with range bars = values read from the paper's figure (approximate, paper_notes §11)."

    # Fig.4: VGG-16 / CIFAR-100 (both runs in one Kaggle session -> time axis valid)
    curves_figure({"TiMePReSt": load(os.path.join(c2, "kaggle_cifar100_vgg16_timeprest_graph")),
                   "PipeDream": load(os.path.join(c2, "kaggle_cifar100_vgg16_pipedream_lr002"))},
                  "Like paper Fig.4 - VGG-16 / CIFAR-100, W=2, 160 epochs (ours: Kaggle 2xT4, same host)",
                  "fig04_vgg16_cifar100.png", True,
                  {"acc1": [("time", 27, 72, 75, "paper\nTiMePReSt", "TiMePReSt"), ("time", 27, 44, 48, "paper\nPipeDream", "PipeDream"),
                            ("epochs", 160, 72, 75, "paper T.", "TiMePReSt"), ("epochs", 155, 73, 77, "paper P.", "PipeDream")],
                   "acc5": [("time", 27, 88, 91, "paper T.", "TiMePReSt"), ("time", 27, 74, 78, "paper P.", "PipeDream")]},
                  PAPER_NOTE + " Paper: Cluster A (2 machines); PipeDream 5-6x slower per epoch, hence its low curve on the time axis.")

    # Fig.8 / S4: VGG-16 / Tiny-ImageNet: seed 1 pair ran on one account (time valid); epochs = 3-seed mean +- sd
    s1 = {"TiMePReSt": load(os.path.join(tin, "run2_seeds_bn", "runs", "tin_vgg16_timeprest_s1")),
          "PipeDream": load(os.path.join(tin, "run2_seeds_bn", "runs", "tin_vgg16_pipedream_s1"))}
    curves_figure(s1, "Like paper Fig.8 / S4 - VGG-16 / Tiny-ImageNet, W=2, 80 epochs (seed 1, same host)",
                  "fig08_vgg16_tinyimagenet.png", True,
                  {"acc1": [("time", 25, 69, 73, "paper T.\n(W=3)", "TiMePReSt"), ("time", 25, 31, 35, "paper P.\n(W=3)", "PipeDream"),
                            ("epochs", 80, 68, 72, "paper T.", "TiMePReSt"), ("epochs", 74, 73, 77, "paper P.", "PipeDream")]},
                  PAPER_NOTE + " Paper values are Cluster B (3 machines, Fig.8) at ~140 epochs; ours stop at 80.")
    seeds = {"TiMePReSt": [], "PipeDream": []}
    for s, base in ((0, ("run1_80ep_lr0.02", "runs", "tin_vgg16_")), (1, ("run2_seeds_bn", "runs", "tin_vgg16_")), (2, ("run2_seeds_bn", "runs", "tin_vgg16_"))):
        for name, sysn in (("TiMePReSt", "timeprest"), ("PipeDream", "pipedream")):
            seeds[name].append(load(os.path.join(tin, *base[:-1], base[-1] + sysn + ("" if s == 0 else f"_s{s}"))))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for name in seeds:
        mean_band(axes[0], seeds[name], "acc1", name)
        mean_band(axes[1], seeds[name], "loss", name)
    for ax, lab in zip(axes, ("Top-1 accuracy (%)", "Test loss")):
        ax.set_xlabel("Epochs"); ax.set_ylabel(lab)
    axes[0].axhline(50, color=GRID, lw=1)
    axes[0].legend(loc="lower right")
    fig.suptitle("VGG-16 / Tiny-ImageNet: mean +- sd over 3 seeds (paper: TiMePReSt needs MORE epochs; ours: fewer mid-training, equal at the end)",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(OUT, "fig08b_vgg16_tinyimagenet_3seeds.png"), dpi=140); plt.close(fig)

    # Fig.5: ResNet-50 / Tiny-ImageNet
    r3 = os.path.join(tin, "run3_resnet50", "runs")
    curves_figure({"TiMePReSt": load(os.path.join(r3, "tin_r50_timeprest")), "PipeDream": load(os.path.join(r3, "tin_r50_pipedream"))},
                  "Like paper Fig.5 - ResNet-50 / Tiny-ImageNet, W=2, 80 epochs (ours, 1 seed)", "fig05_resnet50_tinyimagenet.png", True,
                  note="Paper Fig.5: same comparison on Cluster A; TiMePReSt ahead over time (no numbers read from the figure).")

    # Fig.9: all systems on VGG-16 / Tiny-ImageNet (epochs only: runs come from different machines / runtimes)
    r1 = os.path.join(tin, "run1_80ep_lr0.02", "runs")
    curves_figure({"TiMePReSt": load(os.path.join(r1, "tin_vgg16_timeprest")), "PipeDream": load(os.path.join(r1, "tin_vgg16_pipedream")),
                   "PipeDream-vsync": load(os.path.join(r1, "tin_vgg16_pipedream_vsync")),
                   "DeepSpeed": load(os.path.join(tin, "run4_deepspeed", "runs", "tin_vgg16_deepspeed")),
                   "PipeDream M=64": load(os.path.join(tin, "run5_pipedream_m64", "runs", "tin_vgg16_pipedream_m64"))},
                  "Like paper Fig.9 - five systems, VGG-16 / Tiny-ImageNet, 80 epochs, seed 0 (epoch axis)", "fig09_vgg16_tinyimagenet_systems.png", False,
                  note="Paper Fig.9 (Cluster C, 4 machines, time axis): TiMePReSt 69-73 % vs other systems 24-35 % at ~21 time points. "
                       "Ours: time axis omitted (different machines/runtimes); per-epoch times in fig16.")

    # Fig.13: ablation (TiMePReSt / Variant 1 on account A, Variant 2 on account B -> epochs)
    curves_figure({"TiMePReSt": load(os.path.join(r1, "tin_vgg16_timeprest")), "Variant 1": load(os.path.join(r1, "tin_vgg16_variant1")),
                   "Variant 2": load(os.path.join(r1, "tin_vgg16_variant2"))},
                  "Like paper Fig.13 - ablation, VGG-16 / Tiny-ImageNet, 80 epochs (Variant 1 = nF1B + stashing, Variant 2 = 1F1B, no stashing)",
                  "fig13_ablation_vgg16_tinyimagenet.png", False,
                  note="Paper Fig.13 (Cluster C, time axis): Variant 1 78-82 % > TiMePReSt 63-67 % > Variant 2 33-37 %. Ours: all three ~59 % at the end.")

    # Fig.11: number of micro-batches (VGG-16 / CIFAR-100, phase 3)
    p3 = R("phase3", "run1_gd3", "runs")
    curves_figure({"TiMePReSt": load(os.path.join(c2, "kaggle_cifar100_vgg16_timeprest_graph")),
                   "N=2": load(os.path.join(p3, "kaggle_cifar100_vgg16_timeprest_n2")),
                   "N=2, M=384": load(os.path.join(p3, "kaggle_cifar100_vgg16_timeprest_n2_m384"))},
                  "Like paper Fig.11 - TiMePReSt with N=3 (M=192), N=2 (M=192), N=2 (M=384), VGG-16 / CIFAR-100", "fig11_microbatches_vgg16_cifar100.png", False,
                  note="Paper Fig.11 (W=4, VGG-16 / Tiny-ImageNet): N=3 best, N=2 with larger mini-batch worst. With W=2 our N=2 keeps v=1.")

    # Fig.15: memory per stage (stacked)
    mem = {"VGG-16 / CIFAR-100": {"TiMePReSt": load(os.path.join(c2, "kaggle_cifar100_vgg16_timeprest_graph"))["mem"],
                                  "PipeDream": load(os.path.join(c2, "kaggle_cifar100_vgg16_pipedream_lr002"))["mem"]},
           "VGG-16 / Tiny-ImageNet": {"TiMePReSt": load(os.path.join(r1, "tin_vgg16_timeprest"))["mem"],
                                      "PipeDream": load(os.path.join(r1, "tin_vgg16_pipedream"))["mem"],
                                      "DeepSpeed": load(os.path.join(tin, "run4_deepspeed", "runs", "tin_vgg16_deepspeed"))["mem"],
                                      "PipeDream M=64": load(os.path.join(tin, "run5_pipedream_m64", "runs", "tin_vgg16_pipedream_m64"))["mem"]},
           "ResNet-50 / Tiny-ImageNet": {"TiMePReSt": load(os.path.join(r3, "tin_r50_timeprest"))["mem"],
                                         "PipeDream": load(os.path.join(r3, "tin_r50_pipedream"))["mem"]}}
    paper_mem = {"VGG-16 / CIFAR-100": {"TiMePReSt": 7, "PipeDream": 11, "DeepSpeed": 6},
                 "VGG-16 / Tiny-ImageNet": {"TiMePReSt": 10, "PipeDream": 15, "DeepSpeed": 8},
                 "ResNet-50 / Tiny-ImageNet": {"TiMePReSt": 12, "PipeDream": 17, "DeepSpeed": 10}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    h = grouped_bars(axes[0], mem, lambda m: [float(v) / 1024 for v in m.split("|")], lambda v: f"{v:.2f}", hatch_parts=True)
    from matplotlib.patches import Patch
    axes[0].legend(handles=h + [Patch(facecolor=SURF, edgecolor=INK2, hatch="///", label="stage 2 (GPU1)")], fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("Peak allocated per GPU, stacked (GB)")
    axes[0].set_title("Ours: W=2 (solid = stage 1 / GPU0, hatched = stage 2 / GPU1)")
    h = grouped_bars(axes[1], paper_mem, lambda v: [v], lambda v: f"~{v:.0f}")
    axes[1].legend(handles=h, fontsize=7.5, loc="upper left")
    axes[1].set_ylabel("Sum over 4 stages (GB, read from figure)")
    axes[1].set_title("Paper Fig.15: Cluster C (W=4)")
    fig.suptitle("Like paper Fig.15 - memory: paper ~30 % saving for TiMePReSt vs PipeDream; ours 0-6 % (and PipeDream M=64 / DeepSpeed lowest)",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(OUT, "fig15_memory.png"), dpi=140); plt.close(fig)

    # Fig.16/17: epoch time and throughput, same machine (bench, no network emulation)
    def bench(path):
        rows = json.load(open(path, encoding="utf-8"))["rows"]
        return {r["system"]: r["epoch_time_s"] for r in rows if r["bandwidth_gbps"] is None}
    bc = bench(R("phase3", "run1_gd3", "bench_comm", "bench_comm.json"))
    bt = bench(os.path.join(tin, "run5_pipedream_m64", "bench_comm", "bench_comm.json"))
    br = bench(os.path.join(tin, "run3_resnet50", "bench_comm", "bench_comm.json"))
    ours = {"VGG-16 / CIFAR-100": {"TiMePReSt": bc["timeprest"], "PipeDream": bc["pipedream"]},
            "VGG-16 / Tiny-ImageNet": {"TiMePReSt": bt["timeprest"], "PipeDream": bt["pipedream"], "PipeDream M=64": bt["pipedream_m64"]},
            "ResNet-50 / Tiny-ImageNet": {"TiMePReSt": br["timeprest"], "PipeDream": br["pipedream"]}}
    paper_t = {"VGG-16 / CIFAR-100": {"TiMePReSt": 11, "PipeDream": 62.5}, "VGG-16 / Tiny-ImageNet": {"TiMePReSt": 75.5, "PipeDream": 130.5},
               "ResNet-50 / Tiny-ImageNet": {"TiMePReSt": 62.5, "PipeDream": 122.5}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, data, title in ((axes[0], ours, "Ours: same machine (Kaggle 2xT4), bench"),
                            (axes[1], paper_t, "Paper Fig.16a: Cluster B (W=3), minutes/epoch read")):
        rel = {wl: {n: v / sy["TiMePReSt"] for n, v in sy.items()} for wl, sy in data.items()}
        h = grouped_bars(ax, rel, lambda v: [v], lambda v: f"{v:.2f}x")
        ax.axhline(1, color=INK2, lw=1)
        ax.legend(handles=h, fontsize=7.5, loc="upper left")
        ax.set_ylabel("Epoch time relative to TiMePReSt"); ax.set_title(title)
    axes[0].set_ylim(0, 2.1); axes[1].set_ylim(0, 6.5)
    fig.suptitle("Like paper Fig.16a/17a - epoch time: paper PipeDream 1.7-5.7x slower; ours 0.89-1.05x (different y-scales!)",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(OUT, "fig16_epoch_time.png"), dpi=140); plt.close(fig)

    # Not in the paper: epoch time vs emulated bandwidth (tests the paper's communication argument, §3.8)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
    for ax, (wl, path) in zip(axes, (("VGG-16 / CIFAR-100", R("phase3", "run1_gd3", "bench_comm", "bench_comm.json")),
                                     ("VGG-16 / Tiny-ImageNet", os.path.join(tin, "run5_pipedream_m64", "bench_comm", "bench_comm.json")),
                                     ("ResNet-50 / Tiny-ImageNet", os.path.join(tin, "run3_resnet50", "bench_comm", "bench_comm.json")))):
        rows = json.load(open(path, encoding="utf-8"))["rows"]
        finite = sorted({r["bandwidth_gbps"] for r in rows if r["bandwidth_gbps"]}, reverse=True)
        xs_map = {None: 0, **{bw: i + 1 for i, bw in enumerate(finite)}}     # fastest (same host) -> slowest
        for sysn, name in (("timeprest", "TiMePReSt"), ("pipedream", "PipeDream"), ("pipedream_m64", "PipeDream M=64")):
            pts = sorted(((xs_map[r["bandwidth_gbps"]], r["epoch_time_s"]) for r in rows if r["system"] == sysn))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COL[name], ls=DASH[name], marker="o", ms=5, label=name)
        ax.set_xticks(range(len(finite) + 1), ["same\nhost"] + [f"{b:g}" for b in finite])
        ax.set_xlabel("Bandwidth per direction (Gbit/s)  → slower")
        ax.set_ylabel("Epoch time (s)"); ax.set_title(wl)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Not in the paper - TiMePReSt gains only when communication is expensive (1-2 Gbit/s: 9-16 % faster); same host: none",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(OUT, "figX_bandwidth.png"), dpi=140); plt.close(fig)
    print("saved", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
