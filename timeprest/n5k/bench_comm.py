"""Epoch time vs (emulated) stage-to-stage bandwidth for the nutrition task (n5k/PLAN.md mục 7, B6).
Same procedure as timeprest.dist.bench_comm (every system, every bandwidth, on the SAME machine; the
last of `bench.epochs` epochs is timed), with the nutrition pipeline trainer; the data cache is loaded once.

    torchrun --standalone --nproc_per_node=2 -m timeprest.n5k.bench_comm --config n5k/configs/n5k_bench_comm.yaml

Writes <output.dir>/bench_comm.csv, bench_comm.json and bench_comm.png on rank 0.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil

import torch.distributed as dist

from .. import utils
from ..config import _load_yaml, apply_overrides, resolve
from ..dist.bench_comm import _stat, bench_variants, plot
from ..dist.train import init_dist
from .data import load_datasets
from .dist_train import N5kDistTrainer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="n5k/configs/n5k_bench_comm.yaml")
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
    data = load_datasets(base_cfg["data"], print if rank == 0 else (lambda *_: None))
    rows = []
    for bw in bench["bandwidths_gbps"]:
        for name, system, over in bench_variants(bench["systems"]):
            r = apply_overrides(copy.deepcopy(raw), over)
            r["system"] = system
            r.setdefault("dist", {})["emulate_bandwidth_gbps"] = bw
            r["dist"]["emulate_latency_ms"] = bench.get("latency_ms", 0.0)
            r.setdefault("training", {})["epochs"] = bench["epochs"]
            r["name"] = f"bench_{name}_bw{bw if bw else 'inf'}"
            cfg = resolve(r)
            out = os.path.join(base_dir, cfg["name"])
            if rank == 0:
                shutil.rmtree(out, ignore_errors=True)
            dist.barrier()
            utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
            hist = N5kDistTrainer(cfg, out, data, verbose=False).fit()
            last = hist[-1]                      # later epochs: no first-epoch warm-up effects
            if rank == 0:
                row = {"system": name, "bandwidth_gbps": bw, "latency_ms": bench.get("latency_ms", 0.0),
                       "epoch_time_s": last["epoch_time_s"], "busy_frac": _stat(last, "busy_frac"),
                       "wait_time_s": _stat(last, "wait_time_s"), "mb_sent": _stat(last, "mb_sent"),
                       "msgs_sent": _stat(last, "msgs_sent"), "peak_mem_mb": _stat(last, "peak_mem_mb"),
                       "val_mae_kcal": last["val_mae_kcal"], "epochs_timed": len(hist)}
                rows.append(row)
                print(f"[bw {str(bw or 'inf'):>5} Gbit/s] {name:14s} epoch {row['epoch_time_s']:7.2f}s "
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
        print("N5K_BENCH_JSON " + json.dumps(rows), flush=True)
        plot(rows, os.path.join(base_dir, "bench_comm.png"))
        print("saved", os.path.join(base_dir, "bench_comm.png"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
