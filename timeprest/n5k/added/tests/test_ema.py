"""A (n5k/added/README.md): weight EMA (update rule, swap-in for evaluation) and the EMA pipeline trainer on two
CPU/gloo processes (training identical to timeprest.n5k.dist_train, EMA columns and test metrics written)."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from timeprest.n5k.added.ema import WeightEMA, tag
from timeprest.n5k.tests.test_train import ROOT, overrides, write_fake


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 2))


def _step(m, opt):
    m.train()
    loss = m(torch.randn(8, 4)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()


def test_update_rule_and_integer_buffers():
    m = _net()
    ema = WeightEMA(m, [0.0, 0.5], warmup=False)
    s0 = {k: v.clone() for k, v in m.state_dict().items()}
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    _step(m, opt)
    ema.update(m)
    live = m.state_dict()
    for k, v in live.items():
        torch.testing.assert_close(ema.shadows[0.0][k], v, rtol=0, atol=0)          # decay 0 = live weights
        if v.dtype.is_floating_point:
            torch.testing.assert_close(ema.shadows[0.5][k], 0.5 * s0[k] + 0.5 * v)
        else:
            assert torch.equal(ema.shadows[0.5][k], v)                              # num_batches_tracked copied
    assert ema.updates == 1


def test_warmup_decay_ramps_to_the_target():
    ema = WeightEMA(_net(), [0.99], warmup=True)
    assert ema.effective(0.99) == pytest.approx(1 / 10)
    ema.updates = 10_000
    assert ema.effective(0.99) == 0.99


def test_applied_swaps_in_place_and_restores_live_weights():
    m = _net()
    ema = WeightEMA(m, [0.9], warmup=False)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    opt.register_step_post_hook(lambda *_: ema.update(m))
    for _ in range(3):
        _step(m, opt)
    assert ema.updates == 3                                  # the post-hook ran once per optimizer step
    live = {k: v.clone() for k, v in m.state_dict().items()}
    ptrs = [p.data_ptr() for p in m.parameters()]
    x = torch.randn(5, 4)
    m.eval()
    with ema.applied(m, 0.9):
        y_ema = m(x)
        assert [p.data_ptr() for p in m.parameters()] == ptrs
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, ema.shadows[0.9][k], rtol=0, atol=0)
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, live[k], rtol=0, atol=0)
    assert not torch.allclose(y_ema, m(x))


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("n5k_ema")
    write_fake(tmp)
    return tmp


def _run(module, fake, out, extra=()):
    cmd = [sys.executable, "-m", "timeprest.dist.launch_local", "--nproc", "2", "-m", module,
           "--config", os.path.join("n5k", "added", "configs", "added_timeprest.yaml"), "--force",
           "--set", *overrides(fake, ("training.lr=0.001", "added.ema.decays=[0.5,0.9]", *extra)),
           f"output.dir={fake / out}"]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200)


def test_ema_trainer_on_two_processes_matches_dist_train(fake):
    r = _run("timeprest.n5k.added.ema_train", fake, "ema")
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    ref = _run("timeprest.n5k.dist_train", fake, "plain")
    assert ref.returncode == 0, (ref.stdout + ref.stderr)[-4000:]
    rows = list(csv.DictReader(open(fake / "ema" / "metrics.csv")))
    rref = list(csv.DictReader(open(fake / "plain" / "metrics.csv")))
    assert len(rows) == len(rref) == 2
    for a, b in zip(rows, rref):                             # the EMA never feeds back into training
        for k in ("train_loss", "val_loss", "val_mae_kcal"):
            assert float(a[k]) == pytest.approx(float(b[k]), rel=1e-6), k
        for d in (0.5, 0.9):
            assert np.isfinite(float(a[f"{tag(d)}_val_mae_kcal"]))
    s = json.load(open(fake / "ema" / "summary.json"))
    assert set(s["ema"]) == {"ema0.5", "ema0.9"}
    for v in s["ema"].values():
        assert np.isfinite(v["test_mae_kcal"]) and 1 <= v["best_epoch"] <= 2
    assert [l for l in r.stdout.splitlines() if l.startswith("N5K_EMA_JSON")]
    for st in (0, 1):
        assert (fake / "ema" / f"best_ema0.9_stage{st}.pt").exists()
