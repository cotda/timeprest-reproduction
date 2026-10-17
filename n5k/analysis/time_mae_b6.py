"""B6: val MAE kcal / mAP by epoch and at equal wall-clock time (cum train time), mean of 3 seeds."""
import csv, sys, numpy as np
root = sys.argv[1]
SYS = ["single", "ddp", "pipedream", "timeprest", "timeprest_n4"]
H = {}
for s in SYS:
    runs = []
    for seed in (0, 1, 2):
        p = f"{root}/run6_s{seed}/n5k_results/runs/n5k_{s}_s{seed}/metrics.csv"
        h = list(csv.DictReader(open(p)))
        t = np.cumsum([float(r.get("epoch_time_s") or r["wall_time_s"]) for r in h])
        runs.append((t, np.array([float(r["val_mae_kcal"]) for r in h]), np.array([float(r["val_ingr_map"]) for r in h])))
    H[s] = runs
print("val MAE kcal by epoch (mean of 3 seeds) / best-so-far")
for ep in (3, 5, 10, 15, 20, 30):
    print(f"ep{ep:>3} " + " ".join(f"{s}={np.mean([r[1][ep-1] for r in H[s]]):6.1f}/{np.mean([r[1][:ep].min() for r in H[s]]):5.1f}" for s in SYS))
print("mAP by epoch")
for ep in (3, 5, 10, 20, 30):
    print(f"ep{ep:>3} " + " ".join(f"{s}={np.mean([r[2][ep-1] for r in H[s]]):5.1f}" for s in SYS))
print("best-so-far val MAE kcal at equal train time (s), mean of 3 seeds")
for T in (60, 120, 180, 270, 390, 560):
    row = []
    for s in SYS:
        v = [r[1][r[0] <= T].min() if (r[0] <= T).any() else np.nan for r in H[s]]
        row.append(f"{s}={np.mean(v):6.1f}")
    print(f"t={T:>4}s " + " ".join(row))
print("total train time s:", {s: round(np.mean([r[0][-1] for r in H[s]])) for s in SYS})
print("time to reach best-so-far <= X kcal (s, mean over seeds that reach it)")
for X in (70, 65, 60, 57):
    row = []
    for s in SYS:
        ts = [r[0][np.argmax(np.minimum.accumulate(r[1]) <= X)] for r in H[s] if (r[1] <= X).any()]
        row.append(f"{s}={np.mean(ts):5.0f}({len(ts)})" if ts else f"{s}=  -  ")
    print(f"<= {X}: " + " ".join(row))
