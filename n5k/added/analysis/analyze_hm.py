"""HM (n5k/added/README.md, 2026-10-17 22:25 / 22:31): HM1 / HM2 (trained on the HandMasks dishes only) against the
main models (run18 seed 0, located by run19) on the same 154 test dishes, and the rule set before the run:
(a) test MAE kcal not worse than the main model by more than 2 kcal; (b) location clearly better (M2 rel_gain + 0.05
or M3 hit + 0.05) in at least 2 of the 3 systems. The rule is a proposal; HM becomes main only if the user says so.

    python n5k/added/analysis/analyze_hm.py n5k/results/added/kaggle/run20_hm n5k/results/kaggle/run19_handmasks
"""
import glob
import json
import os
import sys

hm_dir, main_dir = sys.argv[1], sys.argv[2]


def load(root, name):
    p = glob.glob(os.path.join(root, "**", name, "handmasks.json"), recursive=True)
    return json.load(open(p[0])) if p else None


def summ(root, name):
    p = glob.glob(os.path.join(root, "**", name, "summary.json"), recursive=True)
    return json.load(open(p[0])) if p else None


print("| hệ | mô hình | MAE kcal (154 test) | mass | mAP | M1 kcal trên thức ăn | M2 MAE tỉ lệ (gain) | M3 trúng (mốc) |")
print("|---|---|---|---|---|---|---|---|")
verdict = {}
for system in ("single", "pipedream", "timeprest"):
    main = load(main_dir, f"n5k_{system}_w3_s0")
    rows = {"chính": (main, None)}
    for tag in ("hm1", "hm2"):
        rows[tag.upper()] = (load(hm_dir, f"{tag}_{system}_s0"), summ(hm_dir, f"{tag}_{system}_s0"))
    for label, (h, s) in rows.items():
        t = h["test"]
        mass = f"{s['test_mae_mass']:.1f}" if s else "—"
        mp = f"{s['test_ingr_map']:.1f}" if s else "—"
        print(f"| {system} | {label} | {t['mae_kcal']:.1f} | {mass} | {mp} | {t['cmap_m1']['in']:.3f} | "
              f"{t['cmap_m2']['mae_map']:.3f} ({t['cmap_m2']['rel_gain']:+.2f}) | {t['m3']['hit']:.3f} ({t['m3']['chance']:.3f}) |")
    for tag in ("HM1", "HM2"):
        h, m = rows[tag][0]["test"], main["test"]
        a = h["mae_kcal"] - m["mae_kcal"] <= 2.0
        b = (h["cmap_m2"]["rel_gain"] - m["cmap_m2"]["rel_gain"] > 0.05) or (h["m3"]["hit"] - m["m3"]["hit"] > 0.05)
        verdict.setdefault(tag, []).append((system, round(h["mae_kcal"] - m["mae_kcal"], 2), a,
                                            round(h["cmap_m2"]["rel_gain"] - m["cmap_m2"]["rel_gain"], 3),
                                            round(h["m3"]["hit"] - m["m3"]["hit"], 3), b))
print()
for tag, v in verdict.items():
    a_all = all(x[2] for x in v)
    b_count = sum(x[5] for x in v)
    print(tag, "per system (Δkcal, a, ΔM2 gain, ΔM3 hit, b):", v)
    print(f"  (a) every system within 2 kcal: {a_all}; (b) systems with better location: {b_count}/3 -> "
          f"rule {'met' if a_all and b_count >= 2 else 'not met'}")
