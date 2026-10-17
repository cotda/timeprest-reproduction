"""run12/run13 + B6: 30-epoch lr sweep of TiMePReSt and PipeDream (3 seeds)."""
import csv, json, os, sys, numpy as np
K = sys.argv[1]
def run_dir(system, lr, seed):
    b6 = {"timeprest": 0.01, "pipedream": 0.02}[system]
    if lr == b6:
        return f"{K}/run6_s{seed}/n5k_results/runs/n5k_{system}_s{seed}"
    r = {"timeprest": "run12_tp_lr", "pipedream": "run13_pd"}[system]
    return f"{K}/{r}/n5k_results/runs/n5k_{system}_lr{lr:g}_s{seed}"
res = {}
for system in ("timeprest", "pipedream"):
    for lr in (0.02, 0.01, 0.005):
        rows = []
        for s in (0, 1, 2):
            d = run_dir(system, lr, s)
            h = list(csv.DictReader(open(f"{d}/metrics.csv")))
            sm = json.load(open(f"{d}/summary.json"))
            v = np.array([float(r["val_mae_kcal"]) for r in h])
            jumps = np.diff(v[5:18])                       # epochs 6..18
            rows.append(dict(best_val=v.min(), test=sm["test_mae_kcal"], test_map=sm["test_ingr_map"], best_ep=sm["best_epoch"],
                             osc=max(0.0, jumps.max()), train_last=float(h[-1]["train_mae_kcal"]), s_ep=sm["mean_epoch_time_s"]))
        m = {k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows], ddof=1)) for k in rows[0]}
        res[(system, lr)] = (m, rows)
        print(f"{system:<10} lr {lr:<6} best val {m['best_val'][0]:6.2f}±{m['best_val'][1]:.2f}  test {m['test'][0]:6.2f}±{m['test'][1]:.2f}"
              f"  mAP {m['test_map'][0]:5.1f}  osc {m['osc'][0]:5.1f}  train_last {m['train_last'][0]:5.1f}  best_ep {[r['best_ep'] for r in rows]}  s/ep {m['s_ep'][0]:.1f}")
        print("    per seed best val", [round(r["best_val"], 1) for r in rows], "test", [round(r["test"], 1) for r in rows])
for system in ("timeprest", "pipedream"):
    c = {lr: res[(system, lr)][0]["best_val"][0] for lr in (0.02, 0.01, 0.005)}
    best = min(c, key=c.get); srt = sorted(c.values())
    print(system, "best lr by val:", best, "| gap to runner-up %.2f%%" % (100 * (srt[1] / srt[0] - 1)))
