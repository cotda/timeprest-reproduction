"""Parts of the DeepSpeed baseline that do not need DeepSpeed (not installable on the test machine)."""
import csv
import glob

import torch
import torch.nn.functional as F

from timeprest.config import load_config
from timeprest.data import epoch_order
from timeprest.dist.deepspeed_train import LossMeter, epoch_row, eval_micro_batches, micro_batch_indices


def test_micro_batches_follow_the_phase2_epoch_order():
    n, M, N = 1000, 192, 3
    mbs = micro_batch_indices(n, seed=0, epoch=4, M=M, N=N)
    order = epoch_order(n, 0, 4)
    assert len(mbs) == (n // M) * N and all(len(m) == M // N for m in mbs)
    flat = [i for m in mbs for i in m]
    assert flat == order[:(n // M) * M].tolist()           # same samples, same order as timeprest.dist.train
    assert len(eval_micro_batches(10000, 192, 3)) == 52 * 3


def test_loss_meter_matches_the_mini_batch_loss_of_the_other_runs():
    torch.manual_seed(0)
    out, y = torch.randn(192, 7), torch.randint(0, 7, (192,))
    m = LossMeter()
    losses = [m(o, t) for o, t in zip(torch.tensor_split(out, 3), torch.tensor_split(y, 3))]
    # DeepSpeed averages the micro-batch losses; equal micro-batches -> sum CE / M
    torch.testing.assert_close(sum(losses) / 3, F.cross_entropy(out, y, reduction="sum") / 192)
    s = m.stats()
    assert s["n"] == 192 and abs(s["loss"] - F.cross_entropy(out, y).item()) < 1e-5
    assert abs(s["acc1"] - (out.argmax(1) == y).sum().item() / 192 * 100) < 1e-9


def test_epoch_row_has_the_phase2_columns():
    row = epoch_row(0, 0.02, 10.0, {"loss": 1.0, "acc1": 50.0, "n": 99}, {"loss": 1.2, "acc1": 40.0, "acc5": 70.0},
                    [1000.0, 500.0], 10.0)
    ref = glob.glob("results/tinyimagenet/run1_80ep_lr0.02/runs/tin_vgg16_timeprest/metrics.csv")[0]
    header = next(csv.reader(open(ref)))
    assert list(row) == header


def test_deepspeed_config_matches_the_timeprest_run():
    ds, tp = load_config("configs/tin_deepspeed.yaml"), load_config("configs/tin_timeprest.yaml")
    for k in ("model", "data"):
        assert ds[k] == tp[k]
    for k in ("batch_size", "lr", "epochs", "warmup_epochs", "momentum", "weight_decay", "lr_schedule"):
        assert ds["training"][k] == tp["training"][k]
    assert ds["seed"] == tp["seed"] and ds["pipeline"]["num_microbatches"] == 3 and ds["pipeline"]["partition"] == tp["pipeline"]["partition"]
