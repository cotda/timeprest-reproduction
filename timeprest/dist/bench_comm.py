"""Phase 3, E3: epoch time vs (emulated) network bandwidth between the two stages.

The paper runs on clusters of single-GPU machines connected by a network (§4.1); on Kaggle both
GPUs share one host, so stage-to-stage transfers are almost free. Every stage-to-stage message is
put through an `EmulatedLink` (bandwidth + per-message latency, one link per direction), and a
short run of each system is timed at each bandwidth. Training math is unchanged.

    torchrun --standalone --nproc_per_node=2 -m timeprest.dist.bench_comm --config configs/kaggle_bench_comm.yaml

Writes <output.dir>/bench_comm.csv, bench_comm.json and bench_comm.png on rank 0.
"""
from __future__ import annotations

import argparse
import copy
import os
import shutil

import torch.distributed as dist

from .. import utils
from ..config import _load_yaml, apply_overrides, resolve
from ..data import build_datasets
from .train import DistTrainer, init_dist


def _stat(row: dict, key: str) -> list[float]:
    return [float(x) for x in str(row[key]).split("|")]


def plot(rows: list[dict], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    finite = [r["bandwidth_gbps"] for r in rows if r["bandwidth_gbps"]]
    inf_x = max(finite) * 2 if finite else 1.0           # "no emulation" drawn right of the others
    for c, system in zip(["tab:green", "tab:blue", "tab:orange", "tab:red"], dict.fromkeys(r["system"] for r in rows)):
        rs = sorted((r for r in rows if r["system"] == system), key=lambda r: r["bandwidth_gbps"] or inf_x)
        xs = [r["bandwidth_gbps"] or inf_x for r in rs]
        ax[0].plot(xs, [r["epoch_time_s"] for r in rs], "o-", color=c, label=system)
        ax[1].plot(xs, [max(r["busy_frac"]) for r in rs], "o-", color=c, label=system)
    for a, lab in zip(ax, ["Epoch time (s)", "Busy fraction of the busiest GPU"]):
        a.set_xscale("log")
        a.set_xlabel("Emulated bandwidth per direction (Gbit/s); rightmost = no emulation")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Epoch time vs stage-to-stage bandwidth (Kaggle T4x2, emulated link)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/kaggle_bench_comm.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    raw = apply_overrides(_load_yaml(args.config), args.set)
    bench = raw.pop("bench")
    base_cfg = resolve(copy.deepcopy(raw))
    rank, W, _ = init_dist(base_cfg)
    base_dir = base_cfg["output"]["dir"]
    if rank == 0:
        os.makedirs(base_dir, exist_ok=True)
        print("env:", utils.env_info(), "| code", utils.code_hash(), flush=True)
    datasets = build_datasets(base_cfg["data"], base_cfg["model"]["num_classes"], base_cfg["seed"])
    rows = []
    for bw in bench["bandwidths_gbps"]:
        for system in bench["systems"]:
            r = copy.deepcopy(raw)
            r["system"] = system
            r.setdefault("dist", {})["emulate_bandwidth_gbps"] = bw
            r["dist"]["emulate_latency_ms"] = bench.get("latency_ms", 0.0)
            r.setdefault("training", {})["epochs"] = bench["epochs"]
            r["name"] = f"bench_{system}_bw{bw if bw else 'inf'}"
            cfg = resolve(r)
            out = os.path.join(base_dir, cfg["name"])
            if rank == 0:
                shutil.rmtree(out, ignore_errors=True)
            dist.barrier()
            utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
            hist = DistTrainer(cfg, out, datasets, verbose=False).fit()
            last = hist[-1]                      # later epochs: no first-epoch warm-up effects
            if rank == 0:
                row = {"system": system, "bandwidth_gbps": bw, "latency_ms": bench.get("latency_ms", 0.0),
                       "epoch_time_s": last["epoch_time_s"], "busy_frac": _stat(last, "busy_frac"),
                       "wait_time_s": _stat(last, "wait_time_s"), "mb_sent": _stat(last, "mb_sent"),
                       "msgs_sent": _stat(last, "msgs_sent"), "peak_mem_mb": _stat(last, "peak_mem_mb"),
                       "epochs_timed": len(hist)}
                rows.append(row)
                print(f"[bw {str(bw or 'inf'):>5} Gbit/s] {system:10s} epoch {row['epoch_time_s']:7.2f}s "
                      f"busy {row['busy_frac']} wait {row['wait_time_s']}", flush=True)
                utils.append_csv({k: ("|".join(map(str, v)) if isinstance(v, list) else v) for k, v in row.items()},
                                 os.path.join(base_dir, "bench_comm.csv"))
    if rank == 0:
        utils.save_json({"code_hash": utils.code_hash(), "config": args.config, "overrides": args.set,
                         "bench": bench, "env": utils.env_info(), "rows": rows}, os.path.join(base_dir, "bench_comm.json"))
        systems = list(dict.fromkeys(r["system"] for r in rows))
        print("\n| bandwidth (Gbit/s) | " + " | ".join(f"{s} epoch (s)" for s in systems) + " | ratio "
              + "/".join(systems) + " |")
        print("|---|" + "---|" * (len(systems) + 1))
        for bw in bench["bandwidths_gbps"]:
            t = {r["system"]: r["epoch_time_s"] for r in rows if r["bandwidth_gbps"] == bw}
            ratio = t[systems[0]] / t[systems[1]] if len(systems) > 1 and t.get(systems[1]) else float("nan")
            print(f"| {bw or 'inf'} | " + " | ".join(f"{t.get(s, float('nan')):.2f}" for s in systems)
                  + f" | {ratio:.3f} |")
        plot(rows, os.path.join(base_dir, "bench_comm.png"))
        print("saved", os.path.join(base_dir, "bench_comm.png"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
