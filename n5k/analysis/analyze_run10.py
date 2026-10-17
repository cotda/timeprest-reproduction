"""run10_2s_sweep: best val kcal (+ mAP at that epoch), last epoch, test, s/epoch (skip epoch 1), peak, busy."""
import csv, glob, json, os, sys
root = sys.argv[1]
rows = {}
for p in sorted(glob.glob(os.path.join(root, "**", "sweep_*_2s_lr*", "metrics.csv"), recursive=True)):
    name = os.path.basename(os.path.dirname(p))
    h = list(csv.DictReader(open(p, encoding="utf-8")))
    b = min(h, key=lambda r: float(r["val_mae_kcal"]))
    t = [float(r.get("epoch_time_s") or r.get("wall_time_s")) for r in h]
    sm = os.path.join(os.path.dirname(p), "summary.json")
    s = json.load(open(sm)) if os.path.exists(sm) else {}
    rows[name] = dict(ep=len(h), best_ep=int(b["epoch"]), kcal=float(b["val_mae_kcal"]), mAP=float(b["val_ingr_map"]),
                      last_kcal=float(h[-1]["val_mae_kcal"]), last_mAP=float(h[-1]["val_ingr_map"]),
                      test_kcal=s.get("test_mae_kcal"), test_mAP=s.get("test_ingr_map"),
                      s_ep=sum(t[1:]) / max(1, len(t[1:])), peak=h[-1].get("peak_mem_mb"), busy=h[-1].get("busy_frac"))
for k, v in rows.items():
    print(f"{k:<28}" + " ".join(f"{a}={b:.2f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
g = lambda n: rows.get(n, {}).get("s_ep")
base = g("sweep_single_2s_lr0.02")
for n in ("ddp", "pipedream", "timeprest"):
    x = g(f"sweep_{n}_2s_lr0.02")
    if base and x: print(f"time {n}/single lr0.02 = {x / base:.3f}")
for lr in ("0.02", "0.01"):
    a, b = g(f"sweep_timeprest_2s_lr{lr}"), g(f"sweep_pipedream_2s_lr{lr}")
    if a and b: print(f"time timeprest/pipedream lr{lr} = {a / b:.3f}")
