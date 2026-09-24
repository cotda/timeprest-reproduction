"""The seven pre-flight checks from CLAUDE.md. Each returns (status, metrics, message)."""
from __future__ import annotations

import contextlib
import copy
import math
import os
import shutil
import time

import torch
import torch.nn.functional as F

from .. import utils
from ..config import resolve
from ..data import build_datasets, fixed_batch_loader
from ..engine import PipelineEngine, frozen_bn_stats
from ..reference import fig2_grid, FIG2, full_grad_at, max_rel_err, mixed_rule_grads, plain_training
from ..runner import Trainer, build, make_train_loader
from ..schedule import (build_schedule, render_grid, steady_state_v, trace_versions,
                        version_difference_formula)

PASS, FAIL = "PASS", "FAIL"


class Ctx:
    """Shared state: base config, device, datasets (loaded once)."""

    def __init__(self, cfg: dict, device: torch.device, work_dir: str):
        self.cfg, self.device, self.work_dir = cfg, device, work_dir
        self.cc = cfg["checks"]
        self.systems = self.cc.get("systems", [cfg["system"]])
        # N used by the nF1B systems (the config may have been resolved for a 1F1B system)
        self.nf1b_N = self.cc.get("num_microbatches", cfg["pipeline"]["num_microbatches"]
                                  if cfg["pipeline"]["schedule"] == "nF1B" else 3)
        self._data = None

    def data(self):
        if self._data is None:
            self._data = build_datasets(self.cfg["data"], self.cfg["model"]["num_classes"], self.cfg["seed"])
        return self._data

    def sys_cfg(self, system: str, **over) -> dict:
        raw = copy.deepcopy(self.cfg)
        raw["system"] = system
        for k in ("schedule", "vertical_sync", "backward_version"):
            raw["pipeline"][k] = None  # re-apply the preset of the new system
        if system in ("timeprest", "variant1"):
            raw["pipeline"]["num_microbatches"] = self.nf1b_N
        for path, v in over.items():
            node = raw
            parts = path.split(".")
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = v
        return resolve(raw)

    def first_batches(self, K: int, M: int, dtype=None):
        train, _ = self.data()
        n = K * M
        if len(train) < n:
            raise RuntimeError(f"need {n} training samples, have {len(train)}")
        xs, ys = zip(*[train[i] for i in range(n)])
        x, y = torch.stack(xs), torch.tensor(ys)
        if dtype is not None:
            x = x.to(dtype)
        return [(x[k * M:(k + 1) * M], y[k * M:(k + 1) * M]) for k in range(K)]


def _params(stages):
    return torch.cat([p.detach().flatten().float().cpu() for m in stages for p in m.parameters()])


# ---------------------------------------------------------------------------------------- 1
def check_seed(ctx: Ctx):
    steps = ctx.cc.get("seed_steps", 12)
    metrics, ok = {}, True
    for system in ctx.systems:
        cfg = ctx.sys_cfg(system)
        runs = []
        for _ in range(2):
            utils.set_seed(cfg["seed"], deterministic=True)
            eng, train, _, _ = build(cfg, ctx.device, ctx.data())
            st = eng.run_epoch(make_train_loader(cfg, train, 0, ctx.device), max_minibatches=steps)
            runs.append((st["train_loss"], _params(eng.stages)))
        dl = abs(runs[0][0] - runs[1][0])
        dp = (runs[0][1] - runs[1][1]).abs().max().item()
        metrics[system] = {"loss": round(runs[0][0], 6), "loss_diff": dl, "param_max_diff": dp}
        ok &= dl <= 1e-6 and dp <= 1e-5
    return (PASS if ok else FAIL), metrics, f"{steps} mini-batches twice with the same seed (tol loss 1e-6, params 1e-5)"


# ---------------------------------------------------------------------------------------- 2
def check_init_loss(ctx: Ctx):
    cfg = ctx.sys_cfg(ctx.systems[0])
    utils.set_seed(cfg["seed"])
    eng, train, _, _ = build(cfg, ctx.device, ctx.data())
    x, y = ctx.first_batches(1, min(256, len(train)))[0]
    x, y = x.to(ctx.device), y.to(ctx.device)
    C = cfg["model"]["num_classes"]
    with torch.no_grad(), contextlib.ExitStack() as stack:
        for m in eng.stages:
            stack.enter_context(frozen_bn_stats(m))
        out = eng.forward_full(x)
    loss = F.cross_entropy(out.float(), y).item()
    expected = math.log(C)
    tol = ctx.cc.get("init_loss_tol", 0.3)
    shape_ok = tuple(out.shape) == (x.shape[0], C)
    finite = bool(torch.isfinite(out).all())
    ok = shape_ok and finite and abs(loss - expected) <= tol
    return (PASS if ok else FAIL), {"loss": round(loss, 4), "ln_C": round(expected, 4), "shape": list(out.shape),
                                    "finite": finite}, f"|loss - ln({C})| <= {tol}, shape (B,{C}), no NaN/Inf"


# ---------------------------------------------------------------------------------------- 3
def check_grad_equivalence(ctx: Ctx):
    """(a) one mini-batch in flight == plain training; (b) PipeDream stashed gradients equal the
    full-model gradient at the forward version. float64 on the real model."""
    K, M = ctx.cc.get("equiv_steps", 4), ctx.cc.get("equiv_batch", 12)
    rtol, atol = ctx.cc.get("equiv_rtol", 1e-5), ctx.cc.get("equiv_atol", 1e-8)
    batches = [(x.to(ctx.device), y.to(ctx.device)) for x, y in ctx.first_batches(K + 2, M, torch.float64)]
    metrics, ok = {}, True
    for n_micro, sched in ((ctx.nf1b_N, "nF1B"), (1, "1F1B")):
        cfg = ctx.sys_cfg("timeprest" if sched == "nF1B" else "pipedream",
                          **{"training.batch_size": M, "training.epochs": 1})
        utils.set_seed(cfg["seed"], deterministic=True)
        eng, *_ = build(cfg, ctx.device, ctx.data())
        stages0 = [copy.deepcopy(m).double() for m in eng.stages]
        pc = dict(cfg["pipeline"], max_inflight=1)
        seq = PipelineEngine([copy.deepcopy(m) for m in stages0], pc, cfg["training"], ctx.device, K)
        seq.run_epoch(batches[:K])
        # same LR schedule as the engine (epochs=1, steps_per_epoch=K, warmup in epochs)
        ref = plain_training(stages0, batches[:K], n_micro, cfg["training"], K,
                             int(cfg["training"].get("warmup_epochs", 0) * K))
        a, b = _params(seq.stages).double(), _params(ref).double()
        err = max_rel_err(a, b)
        good = torch.allclose(a, b, rtol=rtol, atol=atol)
        metrics[f"sequential_{sched}_N{n_micro}_vs_plain_rel_err"] = err
        ok &= good
    # (b) stale pipeline with stashing
    cfg = ctx.sys_cfg("pipedream", **{"training.batch_size": M, "training.epochs": 1})
    utils.set_seed(cfg["seed"], deterministic=True)
    eng0, *_ = build(cfg, ctx.device, ctx.data())
    eng = PipelineEngine([copy.deepcopy(m).double() for m in eng0.stages], cfg["pipeline"], cfg["training"],
                         ctx.device, K + 2)
    eng.record_versions = True
    got = {}
    eng.on_backward = lambda s, i, g, used: got.__setitem__((s, i), (g, used))
    eng.run_epoch(batches)
    worst, stale = 0.0, 0
    for i, (x, y) in enumerate(batches):
        k = got[(eng.W - 1, i)][1][0]
        stale += k < i
        refg = full_grad_at(eng.stages, eng.version_params, [k] * eng.W, x, y, 1)
        for s in range(eng.W):
            for n, g in got[(s, i)][0].items():
                worst = max(worst, max_rel_err(g, refg[s][n]))
                ok &= torch.allclose(g, refg[s][n], rtol=rtol, atol=atol)
    metrics["pipedream_stashed_grad_max_rel_err"] = worst
    metrics["pipedream_stale_minibatches"] = stale
    ok &= stale > 0
    return (PASS if ok else FAIL), metrics, f"float64, torch.allclose rtol={rtol} atol={atol}"


# ---------------------------------------------------------------------------------------- 4
def check_mechanism(ctx: Ctx):
    metrics, ok, notes = {}, True, []
    # (a) schedules vs paper Fig.2
    for (W, N) in sorted(FIG2):
        same = render_grid(build_schedule(W, N, 12), W, 18) == fig2_grid(W, N)
        metrics[f"fig2_W{W}_N{N}"] = same
        ok &= same
    # (b) Eq.(2) and Eq.(3)
    eq2_bad, eq3_bad, eq3_bad_outside = [], [], []
    for W in range(2, 9):
        for N in range(2, 8):
            v = steady_state_v(trace_versions(build_schedule(W, N, 30), W, N))
            if (v == 1) != (W <= N + 1):
                eq2_bad.append((W, N, v))
            if v != version_difference_formula(W, N):
                (eq3_bad if W <= 5 else eq3_bad_outside).append((W, N, v, version_difference_formula(W, N)))
    metrics["eq2_violations"] = eq2_bad
    metrics["eq3_violations_W<=5"] = eq3_bad
    metrics["eq3_differences_W>5(info)"] = eq3_bad_outside
    ok &= not eq2_bad and not eq3_bad
    # (c) engine trace on the configured pipeline, both systems
    for system in ctx.systems:
        cfg = ctx.sys_cfg(system)
        utils.set_seed(cfg["seed"])
        eng, train, _, _ = build(cfg, ctx.device, ctx.data())
        eng.log_ops = True
        batches = ctx.first_batches(8, cfg["training"]["batch_size"])
        eng.run_epoch(batches)
        log = eng.op_log
        tr = eng.last_trace
        W, N = eng.W, eng.N
        vertical = all(tr.fwd_version[(s, i, j)] == tr.fwd_version[(0, i, j)] for (s, i, j) in tr.fwd_version)
        b_ops = [r for r in log if r["kind"] == "B"]
        if cfg["pipeline"]["backward_version"] == "stashed":
            horiz = all(tr.bwd_version[(s, i, j)] == tr.fwd_version[(s, i, j)] for (s, i, j) in tr.bwd_version)
            no_horizontal_stash = None
        else:
            horiz = None
            no_horizontal_stash = all(all(v == r["live"] for v in r["versions"]) for r in b_ops)
        v_meas = steady_state_v(tr)
        m = {"schedule": cfg["pipeline"]["schedule"], "W": W, "N": N, "vertical_sync_forward": vertical,
             "max_weight_versions_per_stage": tr.max_versions, "v_measured": v_meas}
        if horiz is not None:
            m["backward_uses_forward_version"] = horiz
            ok &= horiz
        if no_horizontal_stash is not None:
            m["backward_uses_live_weights"] = no_horizontal_stash
            m["micro_batches_forwarded_with_older_version"] = sum(
                1 for (s, i, j), k in tr.fwd_version.items() if s == 0 and k < tr.bwd_version[(0, i, j)])
            ok &= no_horizontal_stash
            if W >= 2 and N >= 2:
                m["v_formula"] = version_difference_formula(W, N)
                ok &= v_meas == m["v_formula"] and ((v_meas == 1) == (W <= N + 1))
        ok &= vertical
        m["trace_first_minibatches"] = [
            f"mb{r['mb'] + 1}{'' if r['kind'] == 'B' else chr(65 + r['micro'])} {r['kind']}@stage{r['stage'] + 1} "
            f"slot{r['slot'] + 1} ver={r.get('version', r.get('versions'))} live={r['live']}"
            for r in log if r["mb"] < 3]
        metrics[system] = m
    # (d) mixed-version gradient rule on the real model (float64)
    cfg = ctx.sys_cfg("timeprest", **{"training.batch_size": ctx.cc.get("equiv_batch", 12),
                                      "training.epochs": 1})
    if cfg["pipeline"]["num_microbatches"] > 1:
        utils.set_seed(cfg["seed"], deterministic=True)
        eng0, *_ = build(cfg, ctx.device, ctx.data())
        eng = PipelineEngine([copy.deepcopy(m).double() for m in eng0.stages], cfg["pipeline"], cfg["training"],
                             ctx.device, 6)
        eng.record_versions = True
        got = {}
        eng.on_backward = lambda s, i, g, used: got.__setitem__((s, i), (g, used))
        batches = [(x.to(ctx.device), y.to(ctx.device))
                   for x, y in ctx.first_batches(6, cfg["training"]["batch_size"], torch.float64)]
        eng.run_epoch(batches)
        worst, mixed = 0.0, 0
        for i, (x, y) in enumerate(batches):
            kb = got[(eng.W - 1, i)][1][0]
            kf = [eng.last_trace.fwd_version[(0, i, j)] for j in range(eng.N)]
            mixed += sum(k != kb for k in kf)
            refg = mixed_rule_grads(eng.stages, eng.version_params, kf, kb, x, y, eng.N)
            for s in range(eng.W):
                for n, g in got[(s, i)][0].items():
                    worst = max(worst, max_rel_err(g, refg[s][n]))
        metrics["mixed_version_rule_max_rel_err"] = worst
        metrics["mixed_version_micro_batches"] = mixed
        ok &= worst < 1e-6 and mixed > 0
    if eq3_bad_outside:
        notes.append("Eq.(3) differs from the simulated schedule for some W>5 (outside the paper's experiments)")
    return (PASS if ok else FAIL), metrics, "; ".join(notes) or "Fig.2 schedules, Eq.(2)/(3), version trace"


# ---------------------------------------------------------------------------------------- 5
def check_overfit(ctx: Ctx):
    steps = ctx.cc.get("overfit_steps", 300)
    M = ctx.cc.get("overfit_batch", 96)
    target = ctx.cc.get("overfit_target_loss", 0.05)
    raw_data = copy.deepcopy(ctx.cfg["data"])
    raw_data["augment"] = False
    train, _ = build_datasets(raw_data, ctx.cfg["model"]["num_classes"], ctx.cfg["seed"])
    xs, ys = zip(*[train[i] for i in range(M)])
    x, y = torch.stack(xs), torch.tensor(ys)
    metrics, ok = {}, True
    for system in ctx.systems:
        cfg = ctx.sys_cfg(system, **{"training.batch_size": M, "training.epochs": 1,
                                     "training.lr": ctx.cc.get("overfit_lr", 0.02),
                                     "training.weight_decay": 0.0, "training.lr_schedule": "constant",
                                     "training.warmup_epochs": 0})
        utils.set_seed(cfg["seed"])
        eng, *_ = build(cfg, ctx.device, ctx.data())
        chunk = 50
        curve = []
        for _ in range(steps // chunk):
            st = eng.run_epoch(fixed_batch_loader(x, y, chunk))
            curve.append(round(st["train_loss"], 4))
        final = curve[-1]
        acc = st["train_acc1"]
        metrics[system] = {"loss_curve_per_50_steps": curve, "final_loss": final, "final_acc1": round(acc, 2)}
        ok &= final < target and math.isfinite(final)
    return (PASS if ok else FAIL), metrics, f"one fixed batch of {M}, {steps} updates, loss < {target}"


# ---------------------------------------------------------------------------------------- 6
def check_short_run(ctx: Ctx):
    metrics, ok = {}, True
    full_epochs = ctx.cc.get("full_epochs", 160)
    full_train = ctx.cc.get("full_train_samples", 50000)
    min_acc = ctx.cc.get("short_min_acc1", 2.0)
    for system in ctx.systems:
        cfg = ctx.sys_cfg(system)
        utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
        out = os.path.join(ctx.work_dir, f"short_{system}")
        shutil.rmtree(out, ignore_errors=True)
        tr = Trainer(cfg, ctx.device, out, ctx.data(), verbose=False)
        rows = tr.fit()
        r1, r2 = rows[0], rows[-1]
        n = r2["samples"]
        sec_per_sample = r2["wall_time_s"] / max(1, n)
        est_per_sample = (r2["est_pipe_time_s"] or 0) / max(1, n)
        m = {
            "train_loss_by_epoch": [r["train_loss"] for r in rows],
            "test_acc1_by_epoch": [r["test_acc1"] for r in rows],
            "epoch_time_1gpu_s": r2["wall_time_s"],
            "epoch_time_est_2gpu_s": r2["est_pipe_time_s"],
            "peak_mem_mb": r2["peak_mem_mb"],
            "stage_mem_mb": r2["stage_mem_mb"],
            "max_weight_versions": r2["max_weight_versions"],
            "est_full_run_hours_1gpu": round(sec_per_sample * full_train * full_epochs / 3600, 2),
            "est_full_epoch_min_2gpu": round(est_per_sample * full_train / 60, 2),
        }
        metrics[system] = m
        ok &= r2["train_loss"] < r1["train_loss"] and r2["test_acc1"] > min_acc and math.isfinite(r2["train_loss"])
    return (PASS if ok else FAIL), metrics, f"train loss decreases, test top-1 > {min_acc}%"


# ---------------------------------------------------------------------------------------- 7
def check_resume(ctx: Ctx):
    metrics, ok = {}, True
    n = ctx.cc.get("resume_subset", 960)
    train, test = ctx.data()
    from torch.utils.data import Subset
    small = (Subset(train, range(min(n, len(train)))), Subset(test, range(min(256, len(test)))))
    for system in ctx.systems:
        cfg = ctx.sys_cfg(system, **{"training.epochs": 2, "data.num_workers": 0})
        base = os.path.join(ctx.work_dir, f"resume_{system}")
        shutil.rmtree(base, ignore_errors=True)
        utils.set_seed(cfg["seed"], deterministic=True)
        a = Trainer(cfg, ctx.device, os.path.join(base, "continuous"), small, verbose=False)
        a.fit()
        utils.set_seed(cfg["seed"], deterministic=True)
        b = Trainer(cfg, ctx.device, os.path.join(base, "resumed"), small, verbose=False)
        b.fit(stop_after=1)
        del b
        b2 = Trainer(cfg, ctx.device, os.path.join(base, "resumed"), small, verbose=False)
        resumed = b2.try_resume()
        b2.fit()
        dp = (_params(a.engine.stages) - _params(b2.engine.stages)).abs().max().item()
        la = [float(r["train_loss"]) for r in a.history()]
        lb = [float(r["train_loss"]) for r in b2.history()]
        dl = max(abs(p - q) for p, q in zip(la, lb)) if len(la) == len(lb) else float("inf")
        metrics[system] = {"resumed": resumed, "param_max_diff": dp, "loss_max_diff": dl,
                           "loss_continuous": la, "loss_resumed": lb}
        ok &= resumed and dp <= 1e-5 and dl <= 1e-5
    return (PASS if ok else FAIL), metrics, "2 epochs continuous vs 1 epoch + resume (tol 1e-5)"


CHECKS = [
    (1, "seed_reproducibility", check_seed),
    (2, "initial_loss", check_init_loss),
    (3, "gradient_equivalence", check_grad_equivalence),
    (4, "timeprest_mechanism", check_mechanism),
    (5, "overfit_one_batch", check_overfit),
    (6, "short_run", check_short_run),
    (7, "checkpoint_resume", check_resume),
]


def run_checks(ctx: Ctx, ids: list[int]) -> list[dict]:
    results = []
    for cid, name, fn in CHECKS:
        if cid not in ids:
            continue
        t0 = time.perf_counter()
        print(f"\n=== check {cid}: {name} ===", flush=True)
        try:
            status, metrics, msg = fn(ctx)
        except Exception as e:  # a crash is a FAIL with the error message
            import traceback
            traceback.print_exc()
            status, metrics, msg = FAIL, {}, f"{type(e).__name__}: {e}"
        dt = time.perf_counter() - t0
        res = {"id": cid, "name": name, "status": status, "seconds": round(dt, 1), "message": msg,
               "metrics": metrics}
        print(f"[{status}] {name} ({dt:.1f}s) - {msg}")
        _print_metrics(metrics)
        results.append(res)
    return results


def _print_metrics(metrics: dict, indent: str = "    "):
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{indent}{k}:")
            _print_metrics(v, indent + "    ")
        elif isinstance(v, list) and v and isinstance(v[0], str):
            print(f"{indent}{k}:")
            for line in v:
                print(f"{indent}    {line}")
        else:
            print(f"{indent}{k}: {v}")
