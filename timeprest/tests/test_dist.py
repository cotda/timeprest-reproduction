"""Phase-2 runtime on 2 real processes (gloo, CPU, float64)."""
import copy
import os
import subprocess
import sys

import pytest
import torch

from timeprest.dist.selftest import TRAIN, make_batches, make_model, pipeline_cfg
from timeprest.engine import PipelineEngine
from timeprest.reference import full_grad_at, mixed_rule_grads, pipedream_swap_grads

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOL = dict(rtol=1e-9, atol=1e-12)


def run_dist(tmp_path, **kw):
    args = [sys.executable, "-m", "timeprest.dist.launch_local", "--nproc", "2", "-m", "timeprest.dist.selftest",
            "--out", str(tmp_path)]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return [torch.load(os.path.join(tmp_path, f"rank{s}.pt"), weights_only=False) for s in range(2)]


def engine_reference(system, model, K, N, epochs=1, rule="graph"):
    pc = pipeline_cfg(system, N, "static", "recompute", 2, rule)
    eng = PipelineEngine(make_model(model, 2), pc, dict(TRAIN, epochs=epochs), "cpu", K)
    batches = make_batches(model, K, TRAIN["batch_size"])
    for _ in range(epochs):
        eng.run_epoch(batches)
    return eng


@pytest.mark.parametrize("system,model,rule,mode", [
    ("timeprest", "mlp", "graph", "auto"), ("timeprest", "vgg", "graph", "auto"), ("pipedream", "vgg", "graph", "auto"),
    ("timeprest", "mlp", "recompute", "recompute"), ("timeprest", "mlp", "recompute", "auto"),
    ("pipedream", "mlp", "recompute", "auto"), ("timeprest", "vgg", "recompute", "auto")])
def test_static_order_matches_single_process_engine(tmp_path, system, model, rule, mode):
    """Static (Fig.2) order on 2 processes == phase-1 engine: same versions, same parameters."""
    res = run_dist(tmp_path, system=system, model=model, order="static", backward_mode=mode, backward_rule=rule,
                   K=6, N=3, epochs=2)
    eng = engine_reference(system, model, 6, 3, epochs=2, rule=rule)
    for s in range(2):
        for n, p in eng.stages[s].named_parameters():
            torch.testing.assert_close(res[s]["final"][n], p.detach(), **TOL)
        for a, b in zip(res[s]["buffers"], eng.stages[s].buffers()):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("system,rule", [("timeprest", "graph"), ("pipedream", "graph"), ("timeprest", "recompute"),
                                         ("pipedream", "recompute")])
def test_dynamic_order_gradients_follow_rule(tmp_path, system, rule):
    """Dynamic order: versions depend on timing, but every backward must match the reference rule
    for the versions actually used (logged by the runtime)."""
    K, N = 8, 3
    res = run_dist(tmp_path, system=system, model="mlp", order="dynamic", backward_mode="auto", backward_rule=rule,
                   K=K, N=N)
    stages = make_model("mlp", 2)
    vp = {(s, k): {n: t for n, t in res[s]["versions"][k].items()} for s in range(2) for k in res[s]["versions"]}
    fwd = {s: {(i, j): k for (kind, i, j, k, _) in res[s]["op_log"] if kind == "F"} for s in range(2)}
    batches = make_batches("mlp", K, TRAIN["batch_size"])
    for i, (x, y) in enumerate(batches):
        if system == "timeprest":
            assert fwd[1] == {key: fwd[0][key] for key in fwd[1]}  # vertical sync
            kb = i
            for s in range(2):
                assert res[s]["grads"][(0, i)][1] == [kb] * N  # backward at the live version
            ref = (pipedream_swap_grads if rule == "graph" else mixed_rule_grads)(
                stages, vp, [fwd[0][(i, j)] for j in range(N)], kb, x, y, N)
        else:
            ks = [fwd[s][(i, 0)] for s in range(2)]
            for s in range(2):
                assert res[s]["grads"][(0, i)][1] == [ks[s]]  # stashed
            ref = full_grad_at(stages, vp, ks, x, y, 1)
        for s in range(2):
            for n, g in res[s]["grads"][(0, i)][0].items():
                torch.testing.assert_close(g, ref[s][n], **TOL)


def test_handle_staged_receive_copies_once_after_completion():
    """gloo channel with a GPU tensor: the receive lands in a CPU buffer, copied into the target on
    wait() (exercised here with CPU tensors and a fake work that completes on a helper thread)."""
    import threading
    from timeprest.dist.runtime import Handle

    go = threading.Event()

    class Work:
        def wait(self):
            go.wait()
            staged.fill_(7.0)

    staged, dst = torch.zeros(3), torch.zeros(3)
    h = Handle(Work(), threaded=True, staged=staged, dst=dst)
    assert not h.is_completed()
    go.set()
    h.wait()
    assert h.is_completed() and torch.equal(dst, torch.full((3,), 7.0))
    dst.zero_()
    h.wait()                                   # second wait must not copy again
    assert torch.equal(dst, torch.zeros(3))
