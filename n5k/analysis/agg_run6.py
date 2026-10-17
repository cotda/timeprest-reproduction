"""B6 table: mean +- std over the seeds of every system (run6_s{0,1,2}). Usage: python agg_run6.py <repo>"""
import csv
import json
import os
import statistics as st
import sys

K = os.path.join(sys.argv[1], "n5k/results/kaggle")
SYSTEMS = ["single", "ddp", "pipedream", "timeprest", "timeprest_n4"]
KEYS = [("test_mae_kcal", "kcal"), ("test_pmae_kcal", "kcal%"), ("test_mae_mass", "mass"), ("test_mae_fat", "fat"),
        ("test_mae_carb", "carb"), ("test_mae_protein", "prot"), ("test_ingr_f1", "F1"), ("test_ingr_map", "mAP"),
        ("test_mae_gram_present", "gram"), ("test_mae_kcal_ingr", "kcal_ingr")]


def ms(v):
    return f"{st.mean(v):6.2f} +- {st.stdev(v):4.2f}" if len(v) > 1 else f"{v[0]:6.2f}" if v else "   -"


out = {}
for s in SYSTEMS:
    rows = {k: [] for k, _ in KEYS}
    rows.update(epoch_s=[], peak=[], best_epoch=[], val_kcal_ep10=[], val_kcal_ep20=[], seeds=[])
    for seed in (0, 1, 2):
        d = os.path.join(K, f"run6_s{seed}", "n5k_results", "runs", f"n5k_{s}_s{seed}")
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        sm = json.load(open(os.path.join(d, "summary.json")))
        m = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
        rows["seeds"].append(seed)
        for k, _ in KEYS:
            if k in sm:
                rows[k].append(float(sm[k]))
        t = [float(r.get("epoch_time_s") or r.get("wall_time_s")) for r in m][1:]
        rows["epoch_s"].append(st.mean(t))
        rows["peak"].append(max(float(x) for x in m[-1]["peak_mem_mb"].split("|")))
        rows["best_epoch"].append(sm.get("best_epoch"))
        rows["val_kcal_ep10"].append(float(m[9]["val_mae_kcal"]))
        rows["val_kcal_ep20"].append(float(m[19]["val_mae_kcal"]))
    out[s] = rows

print("seeds:", {s: out[s]["seeds"] for s in SYSTEMS})
print(f"{'system':14s} " + " ".join(f"{lab:>15s}" for _, lab in KEYS[:2] + KEYS[6:8]))
for s in SYSTEMS:
    print(f"{s:14s} " + " ".join(f"{ms(out[s][k]):>15s}" for k, _ in KEYS[:2] + KEYS[6:8]))
print()
print(f"{'system':14s} " + " ".join(f"{lab:>15s}" for _, lab in KEYS[2:6] + KEYS[8:]))
for s in SYSTEMS:
    print(f"{s:14s} " + " ".join(f"{ms(out[s][k]):>15s}" for k, _ in KEYS[2:6] + KEYS[8:]))
print()
print(f"{'system':14s} {'s/epoch':>15s} {'peak MB (max GPU)':>18s} {'val kcal ep10':>15s} {'val kcal ep20':>15s} best epochs")
for s in SYSTEMS:
    r = out[s]
    print(f"{s:14s} {ms(r['epoch_s']):>15s} {ms(r['peak']):>18s} {ms(r['val_kcal_ep10']):>15s} "
          f"{ms(r['val_kcal_ep20']):>15s} {r['best_epoch']}")
