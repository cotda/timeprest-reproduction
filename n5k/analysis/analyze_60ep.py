"""run14 (TiMePReSt lr .01 + PipeDream lr .02) and run15 (TiMePReSt lr .02), 60 epochs, seed 0, vs 30-epoch seed 0."""
import csv, json, os, sys, numpy as np
K = sys.argv[1]
runs = {"TP lr0.01 60ep": f"{K}/run14_60ep/n5k_results/runs/n5k_timeprest_ep60_s0",
        "PD lr0.02 60ep": f"{K}/run14_60ep/n5k_results/runs/n5k_pipedream_ep60_s0",
        "TP lr0.02 60ep": f"{K}/run15_tp60_lr02/n5k_results/runs/n5k_timeprest_lr0.02_ep60_s0",
        "TP lr0.01 30ep": f"{K}/run6_s0/n5k_results/runs/n5k_timeprest_s0",
        "PD lr0.02 30ep": f"{K}/run6_s0/n5k_results/runs/n5k_pipedream_s0",
        "TP lr0.02 30ep": f"{K}/run12_tp_lr/n5k_results/runs/n5k_timeprest_lr0.02_s0"}
out = {}
for name, d in runs.items():
    if not os.path.exists(f"{d}/summary.json"):
        print(f"{name:<16} missing"); continue
    h = list(csv.DictReader(open(f"{d}/metrics.csv"))); sm = json.load(open(f"{d}/summary.json"))
    v = np.array([float(r["val_mae_kcal"]) for r in h]); E = len(v)
    lo, hi = int(round(E * 6 / 30)) - 1, int(round(E * 18 / 30))      # the same share of the schedule as epochs 6-18 of 30
    osc = max(0.0, np.diff(v[lo:hi]).max())
    out[name] = sm["test_mae_kcal"]
    print(f"{name:<16} ep {E:>2} best_ep {sm['best_epoch']:>2} best val {v.min():6.2f} test {sm['test_mae_kcal']:6.2f} mAP {sm['test_ingr_map']:5.1f}"
          f" osc {osc:5.1f} train_last {float(h[-1]['train_mae_kcal']):5.1f} s/ep {sm['mean_epoch_time_s']:5.1f}")
for a, b in (("TP lr0.01 60ep", "TP lr0.01 30ep"), ("PD lr0.02 60ep", "PD lr0.02 30ep"), ("TP lr0.02 60ep", "TP lr0.02 30ep")):
    if a in out and b in out:
        dlt = out[a] - out[b]
        print(f"{a} vs {b}: test {dlt:+.2f} kcal -> {'60 epoch tot hon (> 2 kcal)' if dlt < -2 else 'khong tot hon ro (<= 2 kcal)'}")
for tp in ("TP lr0.01 60ep", "TP lr0.02 60ep"):
    if tp in out and "PD lr0.02 60ep" in out:
        print(f"gap {tp} - PD 60ep: {out[tp] - out['PD lr0.02 60ep']:+.2f} kcal (30ep seed0: lr.01 {out['TP lr0.01 30ep'] - out['PD lr0.02 30ep']:+.2f}, lr.02 {out['TP lr0.02 30ep'] - out['PD lr0.02 30ep']:+.2f})")
