"""run9_2s_part: partition of the two-stream model (PLAN mục 10). Usage: python analyze_run9.py <repo>"""
import json
import os
import sys

K = os.path.join(sys.argv[1], "n5k/results/kaggle/run9_2s_part")
rs = json.load(open(os.path.join(K, "RUN_SUMMARY.json")))
for st in rs["steps"]:
    print(f"step {st['step']:10s} {st['status']:8s} {st['minutes']} min")
ok = []
for k in (5, 4, 6):
    p = os.path.join(K, "n5k_results/partition2s", f"p{k}.json")
    if not os.path.exists(p):
        st = next(s for s in rs["steps"] if s["step"] == f"p{k}_2s")
        print(f"p{k}: no checks file; tail: {st['tail'][-4:]}")
        continue
    res = {r["id"]: r for r in json.load(open(p))["results"]}
    print(f"p{k}: " + " ".join(f"D{i}={res[i]['status']}" for i in sorted(res)))
    d4 = res.get(4)
    if not d4 or d4["status"] != "PASS":
        print("   D4:", d4 and d4["message"][:200])
        continue
    rows = {}
    for s in ("pipedream", "timeprest"):
        m = d4["metrics"][s]
        peak = [float(v) for v in str(m["peak_mem_mb_per_gpu"]).split("|")]
        rows[s] = (m["epoch_time_s"], peak)
        print(f"   {s:10s} {m['epoch_time_s']:7.2f} s/epoch peak {peak} busy {m['busy_frac_per_gpu']} "
              f"sent {m['mb_sent_per_gpu']}")
    if all(res[i]["status"] == "PASS" for i in (0, 1, 2, 3)) and all(max(rows[s][1]) < 14000 for s in rows):
        ok.append((rows["pipedream"][0], k))
print("eligible:", sorted(ok), "-> choose", f"[0,{min(ok)[1]},18]" if ok else None)
