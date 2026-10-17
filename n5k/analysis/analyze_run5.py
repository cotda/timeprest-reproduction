"""run5a + run5b decision (PLAN mục 10): partition at 320 and image size. Usage: python analyze_run5.py <repo>"""
import csv
import glob
import json
import os
import sys

R = sys.argv[1]
K5 = os.path.join(R, "n5k/results/kaggle")

print("== partition at 320 (D4, 4 epochs) ==")
rs = json.load(open(os.path.join(K5, "run5b/RUN_SUMMARY.json")))
for st in rs["steps"]:
    print(f"  step {st['step']:16s} {st['status']:8s} {st['minutes']} min")
ok = []
for k in (5, 4, 6):
    st = next(s for s in rs["steps"] if s["step"] == f"p{k}_320")
    js = [l for l in st["json"] if l.startswith("N5K_CHECKS_JSON")]
    if not js:
        print(f"  p{k}: no result ({st['status']}); tail: {st['tail'][-3:]}")
        continue
    res = json.loads(js[-1].split(" ", 1)[1])["results"]
    d4 = next((x for x in res if x["id"] == 4), None)
    if d4 is None or d4["status"] != "PASS":
        print(f"  p{k}: D4 {d4 and d4['status']}: {d4 and d4.get('message')}")
        continue
    m = d4["metrics"]
    row = {}
    for sysname in ("pipedream", "timeprest"):
        peak = [float(v) for v in m[sysname]["peak_mem_mb_per_gpu"].split("|")]
        row[sysname] = (m[sysname]["epoch_time_s"], peak, m[sysname]["busy_frac_per_gpu"])
        print(f"  p{k} [0,{k},18] {sysname:10s} {m[sysname]['epoch_time_s']:7.2f} s/epoch  peak {peak}  "
              f"busy {m[sysname]['busy_frac_per_gpu']}  sent {m[sysname]['mb_sent_per_gpu']}")
    if all(max(row[s][1]) < 14000 for s in row):
        ok.append((row["pipedream"][0], k))
print("  eligible (peak < 14 GB on both GPUs):", sorted(ok), "-> choose", min(ok)[1] if ok else None)


def best(path):
    rows = list(csv.DictReader(open(path)))
    b = min(rows, key=lambda r: float(r["val_mae_kcal"]))
    return float(b["val_mae_kcal"]), float(b["val_ingr_map"]), int(b["epoch"]), float(rows[-1].get("epoch_time_s") or rows[-1].get("epoch_s") or rows[-1].get("time_s") or "nan")


print("== image size (single, lr 0.02, 8 epochs; best val kcal and mAP at that epoch) ==")
avg = {}
for r, run in ((224, "run5a"), (320, "run5b")):
    vals = []
    for s in (0, 1):
        p = glob.glob(os.path.join(K5, run, f"n5k_results/sweep/sweep_single_r{r}s{s}_lr0.02/metrics.csv"))
        if not p:
            print(f"  {r} seed {s}: missing")
            continue
        v = best(p[0])
        vals.append(v)
        print(f"  {r} seed {s}: kcal {v[0]:.2f} mAP {v[1]:.2f} (epoch {v[2]}) epoch {v[3]:.1f}s")
    if len(vals) == 2:
        avg[r] = (sum(v[0] for v in vals) / 2, sum(v[1] for v in vals) / 2)
        print(f"  {r} mean: kcal {avg[r][0]:.2f} mAP {avg[r][1]:.2f}")
if len(avg) == 2:
    score = (avg[320][0] / avg[224][0] - 1) + (1 - avg[320][1] / avg[224][1])
    print(f"  score = (kcal320/kcal224 - 1) + (1 - mAP320/mAP224) = {score:+.4f} -> "
          + ("use 320" if score < -0.05 else "keep 224"))
