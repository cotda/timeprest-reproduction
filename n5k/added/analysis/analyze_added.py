"""A/B/B2 (n5k/added/README.md): live vs EMA test / best val / oscillation of every added run, means per system.
Usage: python n5k/added/analysis/analyze_added.py n5k/results/added/kaggle n5k/results/added/ema_wtotal_summary.json"""
import csv, glob, json, os, sys, statistics as st
root = sys.argv[1]
TAGS = ["ema0.98", "ema0.99", "ema0.995"]
def osc(v, a=6, b=18):
    return max(v[i] - v[i - 1] for i in range(a, b))   # epochs a+1..b (1-based a..b), same as earlier scripts
res = {}
for d in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
    rd = os.path.dirname(d); name = os.path.basename(rd)
    s = json.load(open(d)); rows = list(csv.DictReader(open(os.path.join(rd, "metrics.csv"))))
    live = [float(r["val_mae_kcal"]) for r in rows]
    r = {"live": {"test": s["test_mae_kcal"], "best_val": min(live), "best_ep": s["best_epoch"], "osc": osc(live),
                  "ep10": live[9], "ep15": live[14], "ep20": live[19]}}
    for t in TAGS:
        v = [float(x[f"{t}_val_mae_kcal"]) for x in rows]
        r[t] = {"test": s["ema"][t]["test_mae_kcal"], "best_val": min(v), "best_ep": s["ema"][t]["best_epoch"], "osc": osc(v),
                "ep10": v[9], "ep15": v[14], "ep20": v[19]}
    r["gshare_total_ep15"] = float(rows[14]["val_gshare_total"]); r["train_mae_ep15"] = float(rows[14]["train_mae_kcal"])
    r["s_per_ep"] = st.mean(float(x["epoch_time_s"]) for x in rows[1:])
    res[name] = r
for n, r in res.items():
    print(n, f"s/ep {r['s_per_ep']:.1f} gshare_tot {r['gshare_total_ep15']:.2f} trainMAE15 {r['train_mae_ep15']:.0f}")
    for k in ["live"] + TAGS:
        x = r[k]; print(f"   {k:8s} test {x['test']:.2f} bestval {x['best_val']:.1f}@{x['best_ep']} osc {x['osc']:.1f} val10/15/20 {x['ep10']:.0f}/{x['ep15']:.0f}/{x['ep20']:.0f}")
print("== A means over seeds")
for sysn in ["timeprest", "pipedream"]:
    runs = [r for n, r in res.items() if n.startswith(f"added_{sysn}_ema_s")]
    for k in ["live"] + TAGS:
        print(sysn, k, "test %.2f±%.2f" % (st.mean(r[k]["test"] for r in runs), st.stdev(r[k]["test"] for r in runs)),
              "bestval %.2f" % st.mean(r[k]["best_val"] for r in runs), "osc %.1f" % st.mean(r[k]["osc"] for r in runs),
              "diff vs live %.2f" % st.mean(r[k]["test"] - r["live"]["test"] for r in runs))
json.dump(res, open(sys.argv[2], "w"), indent=1)
print("== B/B2: w_total = 3 (3 seeds) vs w_total = 1 (A, same seeds)")
for sysn in ["timeprest", "pipedream"]:
    w = [res.get(f"added_{sysn}_wtot3_s{s}") for s in range(3)]
    a = [res.get(f"added_{sysn}_ema_s{s}") for s in range(3)]
    if not all(w) or not all(a):
        continue
    for k in ["live"] + TAGS:
        t = [x[k]["test"] for x in w]
        print(sysn, k, "test %.2f±%.2f" % (st.mean(t), st.stdev(t)), "bestval %.2f" % st.mean(x[k]["best_val"] for x in w),
              "osc %.1f" % st.mean(x[k]["osc"] for x in w))
    print("  diff vs A live %.2f" % st.mean(x["live"]["test"] - y["live"]["test"] for x, y in zip(w, a)),
          "paired", [round(x["live"]["test"] - y["live"]["test"], 2) for x, y in zip(w, a)])
