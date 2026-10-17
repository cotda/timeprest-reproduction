"""C (n5k/added/README.md): op-trace load-balance analysis on a hand-made 2-stage trace."""
import csv
import json
import os

import pytest

from timeprest.n5k.added import balance


def _trace(K=8, f0=0.10, b0=0.20, f1=0.05, b1=0.30):
    """Stage 0 limits the cycle: per mini-batch 2 F of f0 + B of b0; stage 1 starts 0.2 s later."""
    s0, s1 = [], []
    t = 0.0
    for i in range(K):
        for j in range(2):
            s0.append(["F", i, j, t, t + f0]); t += f0
        s0.append(["B", i, None, t, t + b0]); t += b0
    c = 2 * f0 + b0
    for i in range(K):
        u = 0.2 + i * c
        s1.append(["F", i, 0, u, u + f1]); s1.append(["F", i, 1, u + f1, u + 2 * f1])
        s1.append(["B", i, None, u + 2 * f1, u + 2 * f1 + b1])
    return [s0, s1], t


def test_stage_stats_splits_busy_idle_and_cycle():
    (s0, s1), end = _trace()
    ep = end + 0.5
    a = balance.stage_stats(s0, ep, 0.25)
    assert a["busy_s"] == pytest.approx(8 * 0.4)
    assert a["fill_s"] == 0.0 and a["gaps_s"] == pytest.approx(0.0, abs=1e-9)
    assert a["drain_s"] == pytest.approx(0.5)
    assert a["f_ms"] == pytest.approx(100) and a["b_ms"] == pytest.approx(200)
    assert a["work_ms"] == pytest.approx(400) and a["cycle_ms"] == pytest.approx(400)
    b = balance.stage_stats(s1, ep, 0.25)
    assert b["fill_s"] == pytest.approx(0.2)
    assert b["work_ms"] == pytest.approx(400)          # 2 x 50 + 300
    assert b["gaps_s"] == pytest.approx(0.0, abs=1e-9)  # stage 1 never waits inside: equal work per stage


def test_analyze_kernel_reads_runs_and_single_reference(tmp_path):
    ops, end = _trace(f1=0.025, b1=0.15)                # stage 1 does half the work of stage 0
    ep = end + 0.1
    run = tmp_path / "runs" / "n5k_timeprest_s0"
    os.makedirs(run)
    json.dump({"ops_per_stage": ops}, open(run / "op_trace_epoch10.json", "w"))
    json.dump({"system": "timeprest"}, open(run / "config.json", "w"))
    with open(run / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["epoch", "epoch_time_s"]); w.writeheader()
        for e in range(1, 11):
            w.writerow({"epoch": e, "epoch_time_s": ep})
    single = tmp_path / "runs" / "n5k_single_s0"
    os.makedirs(single)
    json.dump({"system": "single"}, open(single / "config.json", "w"))
    with open(single / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["epoch", "wall_time_s"]); w.writeheader()
        for e in range(1, 4):
            w.writerow({"epoch": e, "wall_time_s": 9.0 if e == 1 else 4.0})   # epoch 1 ignored
    k = balance.analyze_kernel(str(tmp_path), 10, 0.25)
    r = k["runs"]["n5k_timeprest_s0"]
    assert k["ref"]["n5k_single_s0"]["epoch_s"] == pytest.approx(4.0)
    assert r["bottleneck_stage"] == 0
    assert r["imbalance"] == pytest.approx(400 / 300)
    assert r["speedup_vs_single"] == pytest.approx(4.0 / ep)
    assert r["sum_busy_s"] == pytest.approx(8 * 0.4 + 8 * 0.2)
    assert "n5k_timeprest_s0" in balance.table([k])
