"""B6 (n5k/PLAN.md mục 7): bench_comm for the nutrition pipeline, 2 processes (gloo, CPU) on the fake dataset:
one row per (bandwidth, system), and the emulated slow link makes the epoch slower."""
import json
import os
import subprocess
import sys

import pytest

from timeprest import utils

from .test_train import overrides, write_fake

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("n5k_bench")
    write_fake(tmp)
    return tmp


def test_bench_comm_rows_and_slow_link(fake, tmp_path):
    out = tmp_path / "bench"
    cmd = [sys.executable, "-m", "timeprest.dist.launch_local", "--nproc", "2", "-m", "timeprest.n5k.bench_comm",
           "--config", os.path.join("n5k", "configs", "n5k_bench_comm.yaml"),
           "--set", *overrides(fake, ("pipeline.partition=auto", f"output.dir={out}", "bench.epochs=1",
                                      "bench.bandwidths_gbps=[null,0.002]",
                                      "bench.systems=[timeprest,pipedream]", "training.lr=0.001"))]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, (p.stdout + p.stderr)[-4000:]
    rows = utils.read_csv(out / "bench_comm.csv")
    assert [(r["system"], r["bandwidth_gbps"]) for r in rows] == [
        ("timeprest", ""), ("pipedream", ""), ("timeprest", "0.002"), ("pipedream", "0.002")]
    t = {(r["system"], r["bandwidth_gbps"]): float(r["epoch_time_s"]) for r in rows}
    for s in ("timeprest", "pipedream"):
        assert t[(s, "0.002")] > t[(s, "")]
    assert len(json.load(open(out / "bench_comm.json"))["rows"]) == 4
    assert (out / "bench_comm.png").exists()
