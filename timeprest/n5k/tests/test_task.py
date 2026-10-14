"""B1 (n5k/PLAN.md mục 9): nutrition task plugged into the phase-1 engine and the phase-2 runtime.
Same checks as the classification suite: no staleness == plain training (check 3), TiMePReSt /
PipeDream gradients follow their rule (check 4), runtime on 2 processes == engine."""
import copy
import math
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from timeprest.engine import PipelineEngine
from timeprest.n5k.synthetic import K, make_batches, make_model, make_task
from timeprest.n5k.task import TOTALS, NutritionTask, mean_average_precision, micro_f1
from timeprest.reference import full_grad_at, pipedream_swap_grads, plain_training

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN = {"epochs": 1, "batch_size": 12, "optimizer": "sgd", "lr": 0.05, "momentum": 0.9,
         "nesterov": False, "weight_decay": 1e-3, "lr_schedule": "cosine", "warmup_epochs": 0}
TOL = dict(rtol=1e-9, atol=1e-12)


def pipe_cfg(schedule, N, backward_version, max_inflight="pipedream", vertical_sync=True):
    return {"num_stages": 2, "num_microbatches": N, "schedule": schedule, "backward_version": backward_version,
            "vertical_sync": vertical_sync, "max_inflight": max_inflight, "backward_rule": "graph"}


def flat(mods):
    return torch.cat([p.detach().flatten() for m in mods for p in m.parameters()])


# ---------------------------------------------------------------- task math
def test_loss_matches_manual_formula():
    task = make_task(w_presence=0.5, w_gram=2.0, w_total=3.0)
    g = torch.Generator().manual_seed(3)
    out = torch.randn(5, 2 * K, generator=g, dtype=torch.float64)
    y = make_batches("mlp", task, 1, 5)[0][1]
    grams = F.softplus(out[:, K:]) * task.gram_scale
    totals = grams @ task.per_gram
    present = (y[:, :K] > 0).double()
    bce = F.binary_cross_entropy_with_logits(out[:, :K], present, reduction="none").mean(1)
    gl = F.smooth_l1_loss(grams / task.gram_scale, y[:, :K] / task.gram_scale, reduction="none").mean(1)
    tl = F.smooth_l1_loss((totals - y[:, K:]) / task.total_std, torch.zeros_like(totals), reduction="none").mean(1)
    torch.testing.assert_close(task.loss_sum(out, y), (0.5 * bce + 2.0 * gl + 3.0 * tl).sum(), **TOL)


def test_sum_reduction_and_loss_terms():
    """loss.ingredient_reduce=sum multiplies the presence / gram terms by K; eval reports the unweighted
    terms (their weighted sum = loss) and gradient shares that sum to 1."""
    mean = make_task(w_presence=0.5, w_gram=2.0, w_total=3.0)
    s = NutritionTask(mean.per_gram, mean.gram_scale, mean.total_std, 0.5, 2.0, 3.0, ingredient_reduce="sum")
    g = torch.Generator().manual_seed(4)
    out = torch.randn(6, 2 * K, generator=g, dtype=torch.float64)
    y = make_batches("mlp", mean, 1, 6)[0][1]
    cm, cs = mean.components(out, y), s.components(out, y)
    torch.testing.assert_close(cs[0], K * cm[0], **TOL)
    torch.testing.assert_close(cs[1], K * cm[1], **TOL)
    torch.testing.assert_close(cs[2], cm[2], **TOL)
    for task, c in ((mean, cm), (s, cs)):
        st = task.eval_start()
        task.eval_update(st, out, y)
        res = task.eval_result(st)
        assert math.isclose(res["loss"], 0.5 * res["loss_presence"] + 2.0 * res["loss_gram"] + 3.0 * res["loss_total"],
                            rel_tol=1e-9)
        assert math.isclose(res["loss_presence"], c[0].mean().item(), rel_tol=1e-9)
        assert math.isclose(sum(res[f"gshare_{n}"] for n in ("presence", "gram", "total")), 1.0, rel_tol=1e-9)
    with pytest.raises(ValueError):
        NutritionTask(mean.per_gram, mean.gram_scale, mean.total_std, ingredient_reduce="max")


def test_perfect_prediction_has_zero_regression_error():
    task = make_task()
    y = make_batches("mlp", task, 1, 6)[0][1]
    grams = y[:, :K]
    raw = torch.where(grams > 0, torch.log(torch.expm1(grams / task.gram_scale)), torch.full_like(grams, -40.0))
    logits = torch.where(grams > 0, 40.0, -40.0).double()
    y = torch.cat([grams, grams @ task.per_gram], 1)              # totals consistent with the grams
    st = task.eval_start()
    task.eval_update(st, torch.cat([logits, raw], 1), y)
    res = task.eval_result(st)
    for name in TOTALS:
        assert res[f"mae_{name}"] < 1e-6
    assert res["mae_gram_present"] < 1e-6 and res["ingr_f1"] == 100.0 and res["ingr_map"] == 100.0


def test_eval_metrics_on_known_values():
    true = torch.tensor([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=torch.bool)
    scores = torch.tensor([[0.9, 0.2], [0.8, 0.6], [0.3, 0.7], [0.1, 0.4]])
    # col 0: ranking 0(+),1(-),2(+),3 -> AP = (1/1 + 2/3) / 2; col 1: ranking 2(+),1(-),3(+),0 -> same
    assert math.isclose(mean_average_precision(scores, true), (1 + 2 / 3) / 2 * 100)
    # threshold 0.5: pred col0 = [1,1,0,0], col1 = [0,1,1,0] -> tp 2, fp 2, fn 2
    assert math.isclose(micro_f1(scores >= 0.5, true), 50.0)


def test_rejects_wrong_output_width():
    with pytest.raises(ValueError):
        make_task().loss_sum(torch.zeros(2, 2 * K + 1, dtype=torch.float64), torch.zeros(2, K + 5, dtype=torch.float64))


def test_from_meta_reads_loss_weights():
    t = make_task()
    meta = {"per_gram": t.per_gram.tolist(), "gram_scale": t.gram_scale.tolist(), "total_std": t.total_std.tolist()}
    t2 = NutritionTask.from_meta(meta, {"w_gram": 0.25})
    assert t2.w == (1.0, 0.25, 1.0) and t2.K == K


# ---------------------------------------------------------------- engine (phase 1)
@pytest.mark.parametrize("kind,N", [("mlp", 1), ("mlp", 2), ("resnet", 2)])
def test_sequential_pipeline_equals_plain_training(kind, N):
    """Check 3 with the nutrition loss: one mini-batch in flight == ordinary training."""
    task = make_task()
    stages = make_model(kind, 2)
    batches = make_batches(kind, task, 4, 12)
    ref = plain_training(stages, batches, N, TRAIN, len(batches), loss_sum=task.loss_sum)
    eng = PipelineEngine([copy.deepcopy(s) for s in stages],
                         pipe_cfg("nF1B" if N > 1 else "1F1B", N, "stashed", max_inflight=1), TRAIN, "cpu",
                         len(batches), task=task)
    stats = eng.run_epoch(batches)
    torch.testing.assert_close(flat(eng.stages), flat(ref), **TOL)
    assert "train_mae_kcal" in stats and math.isfinite(stats["train_loss"])


@pytest.mark.parametrize("kind", ["mlp", "resnet"])
def test_timeprest_gradients_follow_graph_rule(kind):
    """Check 4 with the nutrition loss: TiMePReSt (N=2, committed backward) == PipeDream's stored graph
    with the backward version copied in; some micro-batches really are forwarded with an older version."""
    task = make_task()
    batches = make_batches(kind, task, 6, 12)
    eng = PipelineEngine(make_model(kind, 2), pipe_cfg("nF1B", 2, "committed"), TRAIN, "cpu", len(batches), task=task)
    eng.record_versions = True
    got = {}
    eng.on_backward = lambda s, i, g, used: got.__setitem__((s, i), (g, used))
    eng.run_epoch(batches)
    mixed = 0
    for i, (x, y) in enumerate(batches):
        kb = got[(1, i)][1][0]
        kf = [eng.last_trace.fwd_version[(0, i, j)] for j in range(2)]
        mixed += sum(k != kb for k in kf)
        ref = pipedream_swap_grads(eng.stages, eng.version_params, kf, kb, x, y, 2, loss_sum=task.loss_sum)
        for s in range(2):
            for n, g in got[(s, i)][0].items():
                torch.testing.assert_close(g, ref[s][n], **TOL)
    assert mixed > 0


def test_pipedream_stashed_gradients_with_nutrition_loss():
    task = make_task()
    batches = make_batches("mlp", task, 6, 12)
    eng = PipelineEngine(make_model("mlp", 2), pipe_cfg("1F1B", 1, "stashed", vertical_sync=False), TRAIN, "cpu",
                         len(batches), task=task)
    eng.record_versions = True
    got = {}
    eng.on_backward = lambda s, i, g, used: got.__setitem__((s, i), (g, used))
    eng.run_epoch(batches)
    for i, (x, y) in enumerate(batches):
        ks = [got[(s, i)][1][0] for s in range(2)]
        ref = full_grad_at(eng.stages, eng.version_params, ks, x, y, 1, loss_sum=task.loss_sum)
        for s in range(2):
            for n, g in got[(s, i)][0].items():
                torch.testing.assert_close(g, ref[s][n], **TOL)


def test_engine_evaluate_returns_nutrition_metrics():
    task = make_task()
    batches = make_batches("mlp", task, 2, 12)
    eng = PipelineEngine(make_model("mlp", 2), pipe_cfg("nF1B", 2, "committed"), TRAIN, "cpu", 2, task=task)
    res = eng.evaluate(batches)
    for key in (["loss", "pmae_kcal", "mae_gram_present", "ingr_f1", "ingr_map"] + [f"mae_{t}" for t in TOTALS]
                + [f"{k}_{c}" for k in ("loss", "gshare") for c in ("presence", "gram", "total")]):
        assert key in res and math.isfinite(res[key]), key


# ---------------------------------------------------------------- runtime (phase 2, 2 processes)
def run_dist(tmp_path, **kw):
    args = [sys.executable, "-m", "timeprest.dist.launch_local", "--nproc", "2", "-m", "timeprest.n5k.dist_selftest",
            "--out", str(tmp_path)] + [a for k, v in kw.items() for a in (f"--{k}", str(v))]
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return [torch.load(os.path.join(tmp_path, f"rank{s}.pt"), weights_only=False) for s in range(2)]


@pytest.mark.parametrize("system,model", [("timeprest", "mlp"), ("timeprest", "resnet"), ("pipedream", "mlp")])
def test_runtime_static_matches_engine(tmp_path, system, model):
    from timeprest.dist.selftest import pipeline_cfg
    res = run_dist(tmp_path, system=system, model=model, order="static", K=6, N=2, epochs=2)
    task = make_task()
    eng = PipelineEngine(make_model(model, 2), pipeline_cfg(system, 2, "static", "auto", 2, "graph"),
                         dict(TRAIN, epochs=2), "cpu", 6, task=task)
    batches = make_batches(model, task, 6, 12)
    for _ in range(2):
        eng.run_epoch(batches)
    for s in range(2):
        for n, p in eng.stages[s].named_parameters():
            torch.testing.assert_close(res[s]["final"][n], p.detach(), **TOL)
    ev = eng.evaluate(batches)
    assert res[0]["eval"] is None
    for k, v in ev.items():
        assert math.isclose(res[1]["eval"][k], v, rel_tol=1e-9, abs_tol=1e-9), k
    assert "abs_err_kcal" in res[1]["stats"][0]


def test_runtime_dynamic_timeprest_gradients_follow_rule(tmp_path):
    Kb, N = 8, 2
    res = run_dist(tmp_path, system="timeprest", model="mlp", order="dynamic", K=Kb, N=N)
    task = make_task()
    stages = make_model("mlp", 2)
    vp = {(s, k): dict(res[s]["versions"][k]) for s in range(2) for k in res[s]["versions"]}
    fwd = {s: {(i, j): k for (kind, i, j, k, _) in res[s]["op_log"] if kind == "F"} for s in range(2)}
    for i, (x, y) in enumerate(make_batches("mlp", task, Kb, 12)):
        assert fwd[1] == {key: fwd[0][key] for key in fwd[1]}          # vertical sync
        for s in range(2):
            assert res[s]["grads"][(0, i)][1] == [i] * N               # backward at the live version
        ref = pipedream_swap_grads(stages, vp, [fwd[0][(i, j)] for j in range(N)], i, x, y, N, loss_sum=task.loss_sum)
        for s in range(2):
            for n, g in res[s]["grads"][(0, i)][0].items():
                torch.testing.assert_close(g, ref[s][n], **TOL)
