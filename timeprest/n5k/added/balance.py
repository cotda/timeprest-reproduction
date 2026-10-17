"""C (n5k/added/README.md): load balance of the 2-GPU pipeline from the op traces already logged.

Each `op_trace_epochE.json` holds, per stage, the ops of one epoch as [kind, mini-batch, micro-batch | None,
start_s, end_s] (CUDA events relative to the stage's epoch start). Per stage this gives the busy time of F and B,
the idle time before the first op (fill), after the last op (drain) and between ops, the work per mini-batch and
the steady-state cycle (time between successive backward starts). A kernel folder that also ran single / DDP on
the same machine gives the reference times. Busy time includes the GPU->CPU copy of the sent tensor (`t.cpu()` in
the gloo channel is synchronous), so "busy" is compute + copy, not compute alone.

    python -m timeprest.n5k.added.balance --config n5k/added/configs/added_base.yaml n5k/results/kaggle/run6_s0 ...
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as stt

from ...config import load_config


def stage_stats(ops: list, epoch_s: float, trim: float) -> dict:
    """Busy / idle split and per-mini-batch work of one stage in one epoch."""
    f = [o for o in ops if o[0] == "F"]
    b = [o for o in ops if o[0] == "B"]
    busy_f = sum(o[4] - o[3] for o in f)
    busy_b = sum(o[4] - o[3] for o in b)
    first, last = min(o[3] for o in ops), max(o[4] for o in ops)
    work = {}
    for o in ops:
        work[o[1]] = work.get(o[1], 0.0) + o[4] - o[3]
    starts = sorted(o[3] for o in b)
    cyc = [y - x for x, y in zip(starts, starts[1:])]
    k = int(len(cyc) * trim)
    mid = cyc[k:len(cyc) - k] or cyc
    keys = sorted(work)
    kw = int(len(keys) * trim)
    wmid = [work[i] for i in keys[kw:len(keys) - kw]] or list(work.values())
    busy = busy_f + busy_b
    return {"busy_s": busy, "busy_f_s": busy_f, "busy_b_s": busy_b, "busy_frac": busy / epoch_s,
            "fill_s": first, "drain_s": max(0.0, epoch_s - last), "gaps_s": max(0.0, (last - first) - busy),
            "f_ms": 1e3 * busy_f / max(1, len(f)), "b_ms": 1e3 * busy_b / max(1, len(b)),
            "work_ms": 1e3 * stt.median(wmid), "cycle_ms": 1e3 * stt.median(mid) if mid else float("nan"),
            "minibatches": len(b)}


def _mean_dicts(ds: list[dict]) -> dict:
    return {k: sum(d[k] for d in ds) / len(ds) for k in ds[0]}


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_stats(run_dir: str, min_epoch: int, trim: float) -> dict | None:
    """Mean over the traced epochs >= min_epoch of a pipeline run; None for runs without traces."""
    traces = {int(re.search(r"epoch(\d+)", p).group(1)): p for p in glob.glob(os.path.join(run_dir, "op_trace_epoch*.json"))}
    traces = {e: p for e, p in traces.items() if e >= min_epoch}
    if not traces:
        return None
    rows = {int(r["epoch"]): r for r in _read_csv(os.path.join(run_dir, "metrics.csv"))}
    per_epoch = []
    for e, p in sorted(traces.items()):
        with open(p, encoding="utf-8") as fh:
            ops = json.load(fh)["ops_per_stage"]
        if not all(ops):
            continue
        ep = float(rows[e]["epoch_time_s"])
        per_epoch.append((ep, [stage_stats(o, ep, trim) for o in ops]))
    if not per_epoch:
        return None
    W = len(per_epoch[0][1])
    stages = [_mean_dicts([s[i] for _, s in per_epoch]) for i in range(W)]
    ep = sum(e for e, _ in per_epoch) / len(per_epoch)
    total_busy = sum(s["busy_s"] for s in stages)
    bott = max(s["work_ms"] for s in stages)
    return {"epoch_s": ep, "stages": stages, "traced_epochs": sorted(traces), "sum_busy_s": total_busy,
            "efficiency": total_busy / (W * ep),        # 1 = every GPU busy the whole epoch
            "bottleneck_stage": max(range(W), key=lambda i: stages[i]["work_ms"]),
            "bottleneck_epoch_s": bott * stages[0]["minibatches"] / 1e3,   # epoch time if only the slowest stage limited it
            "imbalance": bott / (sum(s["work_ms"] for s in stages) / W)}  # 1 = equal work per stage


def ref_epoch_s(run_dir: str, min_epoch: int) -> float | None:
    """Mean epoch time (epochs >= 2) of a single / DDP run in a kernel folder."""
    p = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(p):
        return None
    rows = [r for r in _read_csv(p) if int(r["epoch"]) >= 2]
    t = [float(r.get("wall_time_s") or r["epoch_time_s"]) for r in rows]
    return sum(t) / len(t) if t else None


def analyze_kernel(kdir: str, min_epoch: int, trim: float) -> dict:
    """Every run of one kernel folder (same machine)."""
    out = {"kernel": os.path.basename(os.path.normpath(kdir)), "runs": {}, "ref": {}}
    for rd in sorted(glob.glob(os.path.join(kdir, "**", "metrics.csv"), recursive=True)):
        rd = os.path.dirname(rd)
        name = os.path.basename(rd)
        cfg_p = os.path.join(rd, "config.json")
        system = json.load(open(cfg_p, encoding="utf-8")).get("system") if os.path.exists(cfg_p) else None
        st = run_stats(rd, min_epoch, trim)
        if st is not None:
            out["runs"][name] = {"system": system, **st}
        elif system in ("single", "ddp") or name.startswith(("n5k_single", "n5k_ddp")):
            t = ref_epoch_s(rd, min_epoch)
            if t is not None:
                out["ref"][name] = {"system": system, "epoch_s": t}
    single = [v["epoch_s"] for k, v in out["ref"].items() if "single" in k or v["system"] == "single"]
    if single:
        s = single[0]
        for v in out["runs"].values():
            v["speedup_vs_single"] = s / v["epoch_s"]
            v["busy_over_single"] = v["sum_busy_s"] / s     # > 1: pipeline does more GPU work than one GPU
    return out


def table(kernels: list[dict]) -> str:
    head = ("| kernel | run | s/epoch | so với single | GPU bận | F ms (s0/s1) | B ms (s0/s1) | việc/mini-batch ms "
            "| chu kỳ ms | lấp/rút/khe s | tổng bận / single |")
    lines = [head, "|" + "---|" * 11]
    for k in kernels:
        for name, v in k["runs"].items():
            s = v["stages"]
            j = lambda key, fmt="{:.0f}": " / ".join(fmt.format(x[key]) for x in s)
            sp = f"{v['speedup_vs_single']:.2f}×" if "speedup_vs_single" in v else "—"
            ob = f"{v['busy_over_single']:.2f}" if "busy_over_single" in v else "—"
            lines.append(f"| {k['kernel']} | {name} | {v['epoch_s']:.1f} | {sp} "
                         f"| {j('busy_frac', '{:.2f}')} | {j('f_ms')} | {j('b_ms')} | {j('work_ms')} | {j('cycle_ms')} "
                         f"| " + " ; ".join(f"{x['fill_s']:.1f}/{x['drain_s']:.1f}/{x['gaps_s']:.1f}" for x in s)
                         + f" | {ob} |")
        for name, v in k["ref"].items():
            lines.append(f"| {k['kernel']} | {name} | {v['epoch_s']:.1f} | | | | | | | | |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--json", default=None, help="also write the full numbers to this file")
    ap.add_argument("kernels", nargs="+", help="kernel result folders (runs of one folder ran on the same machine)")
    args = ap.parse_args(argv)
    bc = load_config(args.config, args.set)["added"]["balance"]
    ks = [analyze_kernel(k, bc["min_trace_epoch"], bc["steady_trim"]) for k in args.kernels]
    print(table(ks))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(ks, f, indent=1)
    print("N5K_BALANCE_JSON " + json.dumps([{k["kernel"]: {n: {"epoch_s": round(v["epoch_s"], 2),
                                            "efficiency": round(v["efficiency"], 3), "imbalance": round(v["imbalance"], 3),
                                            "speedup_vs_single": round(v.get("speedup_vs_single", float("nan")), 3)}
                                            for n, v in k["runs"].items()}} for k in ks]))


if __name__ == "__main__":
    main()
