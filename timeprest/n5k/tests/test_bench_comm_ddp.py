"""DDP on a slow network (n5k/PLAN.md mục 10, 2026-10-17 18:21): link timing of the emulated ring all-reduce, and on
2 processes (gloo, CPU, fake dataset) the hook leaves training identical to plain DDP while the slow link makes the
epoch slower: at least the link busy time, at most that on top of the full-speed epoch (overlap with backward)."""
import json
import os
import subprocess
import sys

import pytest
import torch

from timeprest import utils
from timeprest.n5k.bench_comm_ddp import EmulatedAllReduce

from .test_train import overrides, write_fake

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_ring_allreduce_hold_time():
    link = EmulatedAllReduce(1.0, 0.1, 2, torch.device("cpu"))     # 1 Gbit/s = 125e6 B/s
    tx, lat = link.hold_s(125_000_000)
    assert tx == pytest.approx(1.0) and lat == pytest.approx(2e-4)  # W=2: each rank sends the whole buffer once
    tx4, lat4 = EmulatedAllReduce(1.0, 0.1, 4, torch.device("cpu")).hold_s(125_000_000)
    assert tx4 == pytest.approx(1.5) and lat4 == pytest.approx(6e-4)
    assert EmulatedAllReduce(None, 0.1, 2, torch.device("cpu")).hold_s(10**9) == (0.0, pytest.approx(2e-4))


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("n5k_bench_ddp")
    write_fake(tmp)
    return tmp


def test_bench_ddp_identical_training_and_slow_link(fake, tmp_path):
    out = tmp_path / "bench"
    cmd = [sys.executable, "-m", "timeprest.dist.launch_local", "--nproc", "2", "-m", "timeprest.n5k.bench_comm_ddp",
           "--config", os.path.join("n5k", "configs", "n5k_bench_comm_ddp.yaml"),
           "--set", *overrides(fake, (f"output.dir={out}", "bench.epochs=1", "bench.bandwidths_gbps=[null,0.5]",
                                      "training.lr=0.001"))]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, (p.stdout + p.stderr)[-4000:]
    rows = utils.read_csv(out / "bench_comm_ddp.csv")
    assert [(r["system"], r["bandwidth_gbps"]) for r in rows] == [("native", ""), ("ddp", ""), ("ddp", "0.5")]
    native, fast, slow = rows
    assert float(fast["val_mae_kcal"]) == float(native["val_mae_kcal"]) == float(slow["val_mae_kcal"])
    steps = 48 // 8                                       # fake train split / batch size
    assert int(slow["allreduce_buckets"]) >= steps and float(slow["allreduce_mb"]) > 0
    busy = float(slow["link_busy_s"])
    assert busy == pytest.approx(float(slow["allreduce_mb"]) * 2**20 / (0.5e9 / 8), rel=1e-3)
    t_fast, t_slow = float(fast["epoch_time_s"]), float(slow["epoch_time_s"])
    # the link is one serial resource (epoch >= its busy time); transfers may overlap backward (epoch <= fast + busy)
    assert max(t_fast, busy) < t_slow < t_fast + busy + 1.0
    assert len(json.load(open(out / "bench_comm_ddp.json"))["rows"]) == 3
