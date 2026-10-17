"""run17_ddp_slow: DDP vs TiMePReSt / PipeDream on an emulated slow link, one machine (n5k/PLAN.md mục 10, 18:21).

    python n5k/analysis/analyze_run17.py n5k/results/kaggle/run17_ddp_slow

DDP = mean of the two DDP benches (before / after the pipeline bench). Rule (set before the run): at each bandwidth
"pipeline wins" if TiMePReSt s/epoch < DDP by more than 5%, "DDP wins" if the reverse by more than 5%, else "tie".
"""
import json
import os
import sys

d = os.path.join(sys.argv[1], "n5k_results")
reps = [json.load(open(os.path.join(d, "bench_comm_ddp", f"rep{i}", "bench_comm_ddp.json")))["rows"] for i in (1, 2)]
pipe = json.load(open(os.path.join(d, "bench_comm", "bench_comm.json")))["rows"]


def get(rows, system, bw, key="epoch_time_s"):
    return next(r[key] for r in rows if r["system"] == system and r["bandwidth_gbps"] == bw)


print("native DDP (no hook), inf:", [get(r, "native", None) for r in reps])
print("| bw | DDP rep1 / rep2 | DDP mean | link busy | DDP - DDP(inf) | TiMePReSt | PipeDream | TP/DDP | PD/DDP | TP/PD | rule |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
base = None
for bw in (None, 2, 1):
    dd = [get(r, "ddp", bw) for r in reps]
    m = sum(dd) / 2
    base = m if bw is None else base
    tp, pd = get(pipe, "timeprest", bw), get(pipe, "pipedream", bw)
    rule = "pipeline wins" if tp < 0.95 * m else "DDP wins" if m < 0.95 * tp else "tie"
    print(f"| {bw or 'inf'} | {dd[0]:.2f} / {dd[1]:.2f} | {m:.2f} | {get(reps[0], 'ddp', bw, 'link_busy_s'):.1f} | "
          f"{m - base:+.2f} | {tp:.2f} | {pd:.2f} | {tp / m:.3f} | {pd / m:.3f} | {tp / pd:.3f} | {rule} |")
print("all-reduce MB/epoch:", get(reps[0], "ddp", None, "allreduce_mb"), "| pipeline MB/epoch per direction:",
      get(pipe, "timeprest", None, "mb_sent"))
print("peak mem MB  DDP:", get(reps[0], "ddp", 1, "peak_mem_mb"), "| TiMePReSt:", get(pipe, "timeprest", 1, "peak_mem_mb"))
print("val MAE kcal after 2 epochs (GPU non-determinism only; the link does not change the math):",
      {f"{s}@{bw or 'inf'}": round(get(r, s, bw, "val_mae_kcal"), 1) for r in reps[:1] for s, bw in
       (("native", None), ("ddp", None), ("ddp", 2), ("ddp", 1))},
      {f"{s}@{bw or 'inf'}": round(get(pipe, s, bw, "val_mae_kcal"), 1) for s in ("timeprest", "pipedream")
       for bw in (None, 2, 1)})
