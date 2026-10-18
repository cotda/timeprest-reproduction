"""run18_s{0,1,2}: B6 again with loss.w_total = 3 (TiMePReSt lr 0.02), against B6 (run6_s*, w_total = 1) and the
predictions of n5k/PLAN.md mục 10 (2026-10-17 20:27).

    python n5k/analysis/analyze_run18.py n5k/results/kaggle
"""
import csv
import glob
import json
import os
import statistics as st
import sys

K = sys.argv[1]
SYSTEMS = ("single", "ddp", "pipedream", "timeprest", "timeprest_n4")


def summary(kernel, name):
    p = glob.glob(os.path.join(K, kernel, "**", name, "summary.json"), recursive=True)
    return json.load(open(p[0])) if p else None


def osc(name, kernel):
    p = glob.glob(os.path.join(K, kernel, "**", name, "metrics.csv"), recursive=True)
    v = [float(r["val_mae_kcal"]) for r in csv.DictReader(open(p[0]))]
    return max(v[i] - v[i - 1] for i in range(6, 18))


def ms(xs):
    return f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


print("| hệ | test kcal w3 | test kcal B6 (w1) | chênh | mass w3 (max) | mAP w3 | dao động w3 | s/epoch w3 |")
print("|---|---|---|---|---|---|---|---|")
res = {}
for s in SYSTEMS:
    new = [summary(f"run18_s{i}", f"n5k_{s}_w3_s{i}") for i in range(3)]
    old = [summary(f"run6_s{i}", f"n5k_{s}_s{i}") for i in range(3)]
    t = [x["test_mae_kcal"] for x in new]
    o = [x["test_mae_kcal"] for x in old if x]
    res[s] = {"test": t, "old": o, "mass": [x["test_mae_mass"] for x in new], "map": [x["test_ingr_map"] for x in new],
              "time": [x["mean_epoch_time_s"] for x in new], "osc": [osc(f"n5k_{s}_w3_s{i}", f"run18_s{i}") for i in range(3)]}
    r = res[s]
    print(f"| {s} | {ms(t)} | {ms(o)} | {st.mean(t) - st.mean(o):+.2f} | {max(r['mass']):.1f} | {st.mean(r['map']):.1f} | "
          f"{st.mean(r['osc']):.1f} | {st.mean(r['time']):.1f} |")
print("\nTiMePReSt - PipeDream (w3):", round(st.mean(res["timeprest"]["test"]) - st.mean(res["pipedream"]["test"]), 2),
      "| paired per seed:", [round(a - b, 2) for a, b in zip(res["timeprest"]["test"], res["pipedream"]["test"])])
print("time ratios per kernel (w3) TP/PD:", [round(a / b, 3) for a, b in zip(res["timeprest"]["time"], res["pipedream"]["time"])],
      "DDP/PD:", [round(a / b, 3) for a, b in zip(res["ddp"]["time"], res["pipedream"]["time"])])
old_t = {s: [summary(f"run6_s{i}", f"n5k_{s}_s{i}")["mean_epoch_time_s"] for i in range(3)] for s in ("timeprest", "pipedream", "ddp")}
print("time ratios per kernel (B6) TP/PD:", [round(a / b, 3) for a, b in zip(old_t["timeprest"], old_t["pipedream"])],
      "DDP/PD:", [round(a / b, 3) for a, b in zip(old_t["ddp"], old_t["pipedream"])])
ex = {}
for s in ("single", "pipedream", "timeprest"):
    p = glob.glob(os.path.join(K, "run18_s0", "**", f"n5k_{s}_w3_s0", "deletion.json"), recursive=True)
    ex[s] = json.load(open(p[0]))["faithful"] if p else None
print("B7 faithful (run18_s0):", json.dumps(ex))
