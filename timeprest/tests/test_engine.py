import copy
import os

import pytest
import torch
import torch.nn as nn

from timeprest.config import load_config
from timeprest.engine import PipelineEngine
from timeprest.models import mlp_blocks, vgg16_bn_blocks
from timeprest.reference import full_grad_at, mixed_rule_grads, plain_training
from timeprest.runner import Trainer

TRAIN = {"epochs": 1, "batch_size": 12, "optimizer": "sgd", "lr": 0.05, "momentum": 0.9,
         "nesterov": False, "weight_decay": 1e-3, "lr_schedule": "cosine", "warmup_epochs": 0}
TOL = dict(rtol=1e-9, atol=1e-12)


def pipe_cfg(schedule, N, backward_version, vertical_sync=True, max_inflight="pipedream", W=2):
    return {"num_stages": W, "num_microbatches": N, "schedule": schedule,
            "backward_version": backward_version, "vertical_sync": vertical_sync,
            "max_inflight": max_inflight}


def make_model(kind, W=2):
    torch.manual_seed(0)
    blocks = mlp_blocks(12, 16, 5, 4) if kind == "mlp" else vgg16_bn_blocks(5, width=1 / 16)
    cut = [round(len(blocks) * s / W) for s in range(W + 1)]
    return [nn.Sequential(*blocks[a:b]).double() for a, b in zip(cut[:-1], cut[1:])]


def make_batches(kind, K, M, seed=1):
    g = torch.Generator().manual_seed(seed)
    shape = (M, 12) if kind == "mlp" else (M, 3, 32, 32)
    return [(torch.randn(*shape, generator=g, dtype=torch.float64),
             torch.randint(0, 5, (M,), generator=g)) for _ in range(K)]


def flat_params(mods):
    return torch.cat([p.detach().flatten() for m in mods for p in m.parameters()])


def run_recorded(stages, batches, pc):
    eng = PipelineEngine(stages, pc, TRAIN, "cpu", len(batches))
    eng.record_versions = True
    got = {}
    eng.on_backward = lambda s, i, g, used: got.__setitem__((s, i), (g, used))
    eng.run_epoch(batches)
    return eng, got


@pytest.mark.parametrize("kind,N", [("mlp", 1), ("mlp", 3), ("vgg", 3), ("vgg", 1)])
def test_sequential_pipeline_equals_plain_training(kind, N):
    """No staleness (one mini-batch in flight) -> identical to ordinary training (check 3)."""
    stages = make_model(kind)
    batches = make_batches(kind, K=4, M=12)
    ref = plain_training(stages, batches, N, TRAIN, len(batches))
    eng = PipelineEngine([copy.deepcopy(s) for s in stages],
                         pipe_cfg("nF1B" if N > 1 else "1F1B", N, "stashed", max_inflight=1),
                         TRAIN, "cpu", len(batches))
    eng.run_epoch(batches)
    torch.testing.assert_close(flat_params(eng.stages), flat_params(ref), **TOL)
    # BN running stats are updated exactly once per micro-batch forward (not in the recompute)
    for a, b in zip([b for m in eng.stages for b in m.buffers()], ref.buffers()):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("kind", ["mlp", "vgg"])
def test_pipedream_stashed_gradients_are_consistent(kind):
    """1F1B + stashing + vertical sync: each gradient equals the full-model gradient at the
    single version used by that mini-batch's forward, on every stage."""
    batches = make_batches(kind, K=6, M=12)
    eng, got = run_recorded(make_model(kind), batches, pipe_cfg("1F1B", 1, "stashed"))
    stale = 0
    for i, (x, y) in enumerate(batches):
        k = got[(eng.W - 1, i)][1][0]
        assert all(got[(s, i)][1] == [k] for s in range(eng.W))
        stale += k < i
        ref = full_grad_at(eng.stages, eng.version_params, [k] * eng.W, x, y, 1)
        for s in range(eng.W):
            for n, g in got[(s, i)][0].items():
                torch.testing.assert_close(g, ref[s][n], **TOL)
    assert stale > 0  # really ran with stale (stashed) versions


@pytest.mark.parametrize("kind,W,N", [("mlp", 2, 3), ("vgg", 2, 3), ("mlp", 3, 2)])
def test_timeprest_mixed_version_rule(kind, W, N):
    """TiMePReSt: gradients follow the declared rule (local VJP at the committed version on the
    stage input produced with the micro-batch's forward version)."""
    batches = make_batches(kind, K=6, M=12)
    eng, got = run_recorded(make_model(kind, W), batches, pipe_cfg("nF1B", N, "committed", W=W))
    tr = eng.last_trace
    mixed = 0
    for i, (x, y) in enumerate(batches):
        kb = got[(W - 1, i)][1][0]
        assert all(got[(s, i)][1] == [kb] * N for s in range(W))  # vertical sync in backward
        kf = [tr.fwd_version[(0, i, j)] for j in range(N)]
        mixed += sum(k != kb for k in kf)
        ref = mixed_rule_grads(eng.stages, eng.version_params, kf, kb, x, y, N)
        for s in range(W):
            for n, g in got[(s, i)][0].items():
                torch.testing.assert_close(g, ref[s][n], **TOL)
    assert mixed > 0  # some micro-batches really were forwarded with an older version


def test_versions_continue_across_epochs():
    stages = make_model("mlp", W=3)
    batches = make_batches("mlp", K=7, M=12)
    for bv in ("committed", "stashed"):
        eng = PipelineEngine(copy.deepcopy(stages), pipe_cfg("nF1B", 2, bv, W=3), TRAIN, "cpu", 14)
        eng.run_epoch(batches)
        eng.run_epoch(batches)
        assert eng.version == [14, 14, 14]


@pytest.mark.parametrize("system", ["timeprest", "pipedream"])
def test_checkpoint_resume_is_seamless(tmp_path, system):
    over = [f"system={system}", "data.dataset=synthetic", "data.synthetic_train=240",
            "data.synthetic_test=64", "data.num_workers=0", "model.width=0.0625",
            "model.num_classes=10", "training.batch_size=48", "training.epochs=3",
            "runtime.device=cpu", "runtime.profile_ops=false"]
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "quick.yaml"), over)
    a = Trainer(cfg, torch.device("cpu"), str(tmp_path / "a"), verbose=False)
    a.fit()
    b = Trainer(cfg, torch.device("cpu"), str(tmp_path / "b"), verbose=False)
    b.fit(stop_after=1)
    b2 = Trainer(cfg, torch.device("cpu"), str(tmp_path / "b"), verbose=False)
    assert b2.try_resume() and b2.start_epoch == 1
    b2.fit()
    torch.testing.assert_close(flat_params(b2.engine.stages), flat_params(a.engine.stages), rtol=0, atol=0)
    ha, hb = a.history(), b2.history()
    assert [r["train_loss"] for r in ha] == [r["train_loss"] for r in hb]
