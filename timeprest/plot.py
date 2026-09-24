"""Compare runs like paper Fig.4 (top-1 / top-5 / loss vs time and vs epoch) + a summary table.

    python -m timeprest.plot --runs <run_dir_timeprest> <run_dir_pipedream> --out results/cifar100
"""
from __future__ import annotations

import argparse
import json
import os

from .utils import read_csv, save_json


def load_run(path: str) -> dict:
    rows = read_csv(os.path.join(path, "metrics.csv"))
    cfg_path = os.path.join(path, "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
    f = lambda k: [float(r[k]) if r.get(k) not in (None, "", "None") else float("nan") for r in rows]
    return {"name": cfg.get("system", os.path.basename(path.rstrip("/\\"))), "path": path, "cfg": cfg,
            "epoch": f("epoch"), "test_acc1": f("test_acc1"), "test_acc5": f("test_acc5"),
            "test_loss": f("test_loss"), "train_loss": f("train_loss"), "train_acc1": f("train_acc1"),
            "wall": f("wall_time_s"), "est": f("est_pipe_time_s"), "cum_wall": f("cum_wall_s"),
            "cum_est": f("cum_est_pipe_s"), "peak_mem_mb": f("peak_mem_mb"),
            "stage_mem": [r.get("stage_mem_mb") for r in rows], "versions": [r.get("max_weight_versions") for r in rows]}


def first_reach(xs, accs, target):
    for x, a in zip(xs, accs):
        if a >= target:
            return x
    return None


def summarize(runs: list[dict], targets=(50.0, 60.0, 70.0)) -> list[dict]:
    out = []
    for r in runs:
        n = len(r["epoch"])
        best = max(range(n), key=lambda i: r["test_acc1"][i])
        s = {
            "system": r["name"], "epochs": n,
            "final_test_acc1": r["test_acc1"][-1], "final_test_acc5": r["test_acc5"][-1],
            "best_test_acc1": r["test_acc1"][best], "best_epoch": int(r["epoch"][best]),
            "final_test_loss": r["test_loss"][-1], "final_train_loss": r["train_loss"][-1],
            "mean_epoch_1gpu_s": sum(r["wall"]) / n,
            "mean_epoch_est_pipeline_s": sum(r["est"]) / n,
            "peak_mem_mb_max": max(r["peak_mem_mb"]) if r["peak_mem_mb"] else None,
            "stage_mem_mb_last": r["stage_mem"][-1], "max_weight_versions": r["versions"][-1],
        }
        for t in targets:
            s[f"epochs_to_{int(t)}"] = first_reach(r["epoch"], r["test_acc1"], t)
            h = first_reach(r["cum_est"], r["test_acc1"], t)
            s[f"est_pipeline_hours_to_{int(t)}"] = round(h / 3600, 3) if h is not None else None
        out.append(s)
    return out


def plot(runs: list[dict], out_prefix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref_epoch = runs[0]["est"] and sum(runs[0]["est"]) / len(runs[0]["est"])
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    panels = [("test_acc1", "Top-1 accuracy (%)"), ("test_acc5", "Top-5 accuracy (%)"), ("test_loss", "Test loss")]
    colors = ["tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple"]
    for row, (key, label) in enumerate(panels):
        for c, r in zip(colors, runs):
            # time axis in "time points": cumulative estimated pipeline time / mean epoch time of runs[0]
            tp = [t / ref_epoch for t in r["cum_est"]] if ref_epoch else r["cum_wall"]
            axes[row][0].plot(tp, r[key], color=c, label=r["name"])
            axes[row][1].plot(r["epoch"], r[key], color=c, label=r["name"])
        axes[row][0].set_xlabel(f"Time points (1 = mean est. epoch time of {runs[0]['name']})")
        axes[row][1].set_xlabel("Epochs")
        for ax in axes[row]:
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("VGG-16 / CIFAR-100, W=2 (single-GPU simulation; time = estimated 2-GPU pipeline time)")
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=120)
    print("saved", out_prefix + ".png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/compare")
    args = ap.parse_args(argv)
    runs = [load_run(p) for p in args.runs]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summ = summarize(runs)
    save_json(summ, args.out + "_summary.json")
    keys = list(summ[0].keys())
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for s in summ:
        print("| " + " | ".join(f"{s[k]:.3f}" if isinstance(s[k], float) else str(s[k]) for k in keys) + " |")
    plot(runs, args.out)


if __name__ == "__main__":
    main()
