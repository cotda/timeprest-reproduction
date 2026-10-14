"""B5 (n5k/PLAN.md mục 9): DDP baseline. The shards of the ranks are exactly the single-GPU mini-batches;
2 processes (gloo, CPU) train, write the usual metrics.csv / summary.json and resume seamlessly."""
import json
import os
import subprocess
import sys

import pytest
import torch
from torch.utils.data import Dataset

from timeprest import utils
from timeprest.data import train_loader
from timeprest.n5k.ddp_train import shard_batches

from .test_train import overrides, write_fake

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Index(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


@pytest.mark.parametrize("n,M,world,epoch", [(50, 8, 2, 0), (50, 8, 2, 3), (37, 12, 4, 1)])
def test_shards_are_the_single_gpu_minibatches(n, M, world, epoch):
    single = [b.tolist() for b in train_loader(Index(n), M, seed=7, epoch=epoch, num_workers=0)]
    shards = [shard_batches(n, M, 7, epoch, r, world) for r in range(world)]
    assert len(single) == n // M
    for b, batch in enumerate(single):
        assert sum((shards[r][b] for r in range(world)), []) == batch


def test_batch_must_split_evenly():
    with pytest.raises(ValueError):
        shard_batches(20, 6, 0, 0, 0, 4)


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("n5k_ddp")
    write_fake(tmp)
    return tmp


def run_ddp(fake, out, extra=(), args=()):
    cmd = [sys.executable, "-m", "timeprest.dist.launch_local", "--nproc", "2", "-m", "timeprest.n5k.ddp_train",
           "--config", os.path.join("n5k", "configs", "n5k_ddp.yaml"), "--force", *args,
           "--set", *overrides(fake, ("training.lr=0.001", f"output.dir={out}", *extra))]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200)


def test_ddp_trains_writes_summary_and_resumes(fake, tmp_path):
    a = run_ddp(fake, tmp_path / "a")
    assert a.returncode == 0, (a.stdout + a.stderr)[-4000:]
    s = json.load(open(tmp_path / "a" / "summary.json"))
    assert s["trainer"] == "ddp" and s["ranks"] == 2 and s["epochs_done"] == 2
    assert all(k in s for k in ("test_mae_kcal", "test_ingr_map", "best_epoch"))
    rows_a = utils.read_csv(tmp_path / "a" / "metrics.csv")
    assert len(rows_a) == 2 and len(rows_a[0]["peak_mem_mb"].split("|")) == 2
    assert int(rows_a[0]["samples"]) == (48 // 8) * 8

    b1 = run_ddp(fake, tmp_path / "b", args=("--stop-after", "1"))
    assert b1.returncode == 0, (b1.stdout + b1.stderr)[-4000:]
    b2 = run_ddp(fake, tmp_path / "b", args=("--resume",))
    assert b2.returncode == 0, (b2.stdout + b2.stderr)[-4000:]
    rows_b = utils.read_csv(tmp_path / "b" / "metrics.csv")
    assert [r["train_loss"] for r in rows_b] == [r["train_loss"] for r in rows_a]
    assert [r["val_mae_kcal"] for r in rows_b] == [r["val_mae_kcal"] for r in rows_a]
    pa = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)["model"]
    pb = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)["model"]
    for k in pa:
        torch.testing.assert_close(pa[k], pb[k], rtol=0, atol=0)
