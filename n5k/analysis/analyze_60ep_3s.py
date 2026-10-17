"""60 epochs x 3 seeds (seed 0: run14/run15; seeds 1/2: run16_s1/s2) vs 30 epochs x 3 seeds (B6 / run12)."""
import csv, json, os, sys, numpy as np
K = sys.argv[1]
def d60(cfg, s):
    if s == 0:
        return {"tp01": f"{K}/run14_60ep/n5k_results/runs/n5k_timeprest_ep60_s0", "pd02": f"{K}/run14_60ep/n5k_results/runs/n5k_pipedream_ep60_s0",
                "tp02": f"{K}/run15_tp60_lr02/n5k_results/runs/n5k_timeprest_lr0.02_ep60_s0"}[cfg]
    sys_, lr = {"tp01": ("timeprest", "0.01"), "tp02": ("timeprest", "0.02"), "pd02": ("pipedream", "0.02")}[cfg]
    return f"{K}/run16_s{s}/n5k_results/runs/n5k_{sys_}_lr{lr}_ep60_s{s}"
def d30(cfg, s):
    return {"tp01": f"{K}/run6_s{s}/n5k_results/runs/n5k_timeprest_s{s}", "pd02": f"{K}/run6_s{s}/n5k_results/runs/n5k_pipedream_s{s}",
            "tp02": f"{K}/run12_tp_lr/n5k_results/runs/n5k_timeprest_lr0.02_s{s}"}[cfg]
def load(d):
    if not os.path.exists(f"{d}/summary.json"):
        return None
    h = list(csv.DictReader(open(f"{d}/metrics.csv"))); sm = json.load(open(f"{d}/summary.json"))
    v = np.array([float(r["val_mae_kcal"]) for r in h]); E = len(v)
    lo, hi = int(round(E * 6 / 30)) - 1, int(round(E * 18 / 30))
    return dict(val=v.min(), test=sm["test_mae_kcal"], map=sm["test_ingr_map"], osc=max(0.0, np.diff(v[lo:hi]).max()),
                train=float(h[-1]["train_mae_kcal"]), sep=sm["mean_epoch_time_s"], tot=sum(float(r.get("epoch_time_s") or r["wall_time_s"]) for r in h))
M = {}
for cfg in ("tp01", "tp02", "pd02"):
    for ep, f in ((30, d30), (60, d60)):
        rows = [load(f(cfg, s)) for s in (0, 1, 2)]
        ok = [r for r in rows if r]
        if not ok: continue
        m = {k: (np.mean([r[k] for r in ok]), np.std([r[k] for r in ok], ddof=1) if len(ok) > 1 else 0) for k in ok[0]}
        M[(cfg, ep)] = m
        print(f"{cfg} {ep}ep n={len(ok)}  val {m['val'][0]:.2f}±{m['val'][1]:.2f}  test {m['test'][0]:.2f}±{m['test'][1]:.2f}  mAP {m['map'][0]:.1f}"
              f"  osc {m['osc'][0]:.1f}  train {m['train'][0]:.1f}  s/ep {m['sep'][0]:.1f}  total {m['tot'][0]:.0f}s  per-seed test {[round(r['test'],1) if r else None for r in rows]}")
for cfg in ("tp01", "tp02", "pd02"):
    if (cfg, 30) in M and (cfg, 60) in M:
        dl = M[(cfg, 60)]["test"][0] - M[(cfg, 30)]["test"][0]
        print(f"(a) {cfg}: 60 vs 30 test {dl:+.2f} -> {'60 tot hon' if dl < -2 else 'khong tot hon ro'}")
if all((c, 60) in M for c in ("tp01", "tp02", "pd02")):
    best = min(("tp01", "tp02"), key=lambda c: M[(c, 60)]["val"][0])
    g = M[(best, 60)]["test"][0] - M[("pd02", 60)]["test"][0]
    print(f"(b) TP best lr at 60ep by val: {best}; test gap vs PD60 {g:+.2f} -> {'bat kip' if g <= 2 else 'chua bat kip'}")
    print(f"(c) TP {best} 60ep test {M[(best, 60)]['test'][0]:.2f} vs PD 30ep {M[('pd02', 30)]['test'][0]:.2f}; total time {M[(best, 60)]['tot'][0]:.0f}s vs {M[('pd02', 30)]['tot'][0]:.0f}s")
