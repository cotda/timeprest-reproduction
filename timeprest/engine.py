"""Single-process pipeline simulator: executes a 1F1B / nF1B schedule stage by stage with
explicit weight versions, so the version semantics are exactly those of a real W-stage pipeline.

Backward rule (`pipeline.backward_rule`; the paper does not spell out how a backward with newer
weights than the forward is computed, see paper_notes.md §14.4, §22.5):
  graph (default, PipeDream's mechanism): every stage keeps the autograd graph of its in-flight
      forwards and runs the backward on it with the weight version chosen by the policy: saved
      activations and BN batch statistics are those of the forward, every saved weight is read at
      the backward version (`swap.GraphForward`). For PipeDream (stashed) both versions coincide.
  recompute (ablation): each stage keeps only its input; at backward time it re-runs its own
      forward on that input with the chosen version and back-propagates the gradient received from
      the next stage (local VJP at the chosen version); BN running statistics frozen during it.
BN running statistics are updated once, in the original forward.
"""
from __future__ import annotations

import contextlib
import math
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .schedule import Op, build_schedule, trace_versions
from .swap import GraphForward, release_storage


@contextlib.contextmanager
def frozen_bn_stats(module: nn.Module):
    bns = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [(m.running_mean.clone() if m.running_mean is not None else None,
              m.running_var.clone() if m.running_var is not None else None,
              m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None) for m in bns]
    try:
        yield
    finally:
        with torch.no_grad():
            for m, (rm, rv, nb) in zip(bns, saved):
                if rm is not None:
                    m.running_mean.copy_(rm)
                if rv is not None:
                    m.running_var.copy_(rv)
                if nb is not None:
                    m.num_batches_tracked.copy_(nb)


def lr_lambda_factory(total_steps: int, warmup_steps: int, schedule: str) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        if schedule == "cosine":
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
        raise ValueError(f"unknown lr_schedule {schedule!r}")
    return fn


def _upcast(t: torch.Tensor) -> torch.Tensor:
    """Compute losses in at least fp32 (keeps fp64 in the float64 tests)."""
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class PipelineEngine:
    def __init__(self, stages: list[nn.Module], pipeline_cfg: dict, training_cfg: dict,
                 device: torch.device | str, steps_per_epoch: int, profile_ops: bool = False):
        self.device = torch.device(device)
        self.stages = [m.to(self.device) for m in stages]
        self.W = len(stages)
        self.N = int(pipeline_cfg["num_microbatches"])
        self.schedule_name = pipeline_cfg["schedule"]
        self.vertical_sync = bool(pipeline_cfg["vertical_sync"])
        self.backward_version = pipeline_cfg["backward_version"]
        self.max_inflight = pipeline_cfg.get("max_inflight", "pipedream")
        self.backward_rule = pipeline_cfg.get("backward_rule") or "graph"
        if self.backward_rule not in ("graph", "recompute"):
            raise ValueError("pipeline.backward_rule must be graph|recompute")
        self.profile_ops = profile_ops
        self.named = [list(m.named_parameters()) for m in self.stages]
        tc = training_cfg
        total_steps = tc["epochs"] * steps_per_epoch
        warmup = int(tc.get("warmup_epochs", 0) * steps_per_epoch)
        self.optimizers, self.schedulers = [], []
        for m in self.stages:
            if tc["optimizer"] != "sgd":
                raise ValueError("only sgd is implemented")
            opt = torch.optim.SGD(m.parameters(), lr=tc["lr"], momentum=tc["momentum"],
                                  weight_decay=tc["weight_decay"], nesterov=tc.get("nesterov", False))
            self.optimizers.append(opt)
            self.schedulers.append(torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda_factory(total_steps, warmup, tc["lr_schedule"])))
        self.version = [0] * self.W
        # --- debug / check hooks ---
        self.on_backward = None        # fn(stage, mb, grads: dict, versions: list[int])
        self.record_versions = False   # keep a copy of every version (tests only)
        self.version_params: dict = {}
        self.log_ops = False           # fill op_log with the version used by every op
        self.op_log: list[dict] = []
        self.last_ops: list[Op] = []
        self.last_trace = None

    # ------------------------------------------------------------------ helpers
    def _live(self, s: int) -> dict:
        return {n: p for n, p in self.named[s]}

    def _clone_live(self, s: int) -> dict:
        return {n: p.detach().clone().requires_grad_(True) for n, p in self.named[s]}

    def _params(self, s: int, k: int, stored: list[dict]) -> dict:
        if k == self.version[s]:
            return self._live(s)
        if k not in stored[s]:
            raise RuntimeError(f"stage {s}: weight version {k} requested but not stored "
                               f"(live={self.version[s]}, stored={sorted(stored[s])})")
        return stored[s][k]

    def _run(self, s: int, params: dict, x: torch.Tensor) -> torch.Tensor:
        return functional_call(self.stages[s], params, (x,))

    def _run_keep_graph(self, s: int, params: dict, x: torch.Tensor) -> torch.Tensor:
        """Forward whose graph is kept until the backward. BN running-stat buffers are passed as
        private copies and written back (same running-stat trajectory), so later forwards never
        modify a tensor this graph saved."""
        bufs = {n: b.clone() for n, b in self.stages[s].named_buffers()}
        out = functional_call(self.stages[s], {**params, **bufs}, (x,))
        with torch.no_grad():
            for n, b in self.stages[s].named_buffers():
                b.copy_(bufs[n])
        return out

    def _drop_version(self, s: int, stored: list[dict], k: int):
        params = stored[s].pop(k)
        if self.backward_rule == "graph":   # graphs keep these only as gradient leaves
            release_storage(params)

    def _snapshot_version(self, s: int):
        if self.record_versions:
            self.version_params[(s, self.version[s])] = {n: p.detach().clone() for n, p in self.named[s]}

    def lrs(self) -> list[float]:
        return [o.param_groups[0]["lr"] for o in self.optimizers]

    def train(self, mode: bool = True):
        for m in self.stages:
            m.train(mode)

    # ------------------------------------------------------------------ epoch
    def run_epoch(self, loader, max_minibatches: int | None = None) -> dict:
        K = len(loader) if max_minibatches is None else min(len(loader), max_minibatches)
        W, N, last = self.W, self.N, self.W - 1
        ops = build_schedule(W, N, K, self.max_inflight)
        trace = trace_versions(ops, W, N, self.vertical_sync, self.backward_version)
        self.last_ops, self.last_trace = ops, trace
        self.train(True)
        if self.record_versions:
            for s in range(W):
                self._snapshot_version(s)

        base = self.version[0]  # trace versions are relative to the epoch start (pipeline drained)
        if any(v != base for v in self.version):
            raise AssertionError("stages have different versions at epoch start")
        it = iter(loader)
        stored = [dict() for _ in range(W)]   # stage -> absolute version -> params
        pending_x, labels, sizes = {}, {}, {}
        inputs = {}      # (s, i, j) -> activation waiting for forward at stage s
        saved = {}       # (s, i, j) -> stage input (recompute) or forward graph record (graph)
        grads_in = {}    # (s, i, j) -> gradient w.r.t. stage-s output
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        correct = torch.zeros((), device=self.device, dtype=torch.long)
        n_seen = 0
        comm_bytes = [0] * len(ops)
        timers = []
        param_bytes = [sum(_nbytes(p) for _, p in self.named[s]) for s in range(W)]
        mem_max = [0] * W
        on_cuda = self.device.type == "cuda"
        t_epoch = time.perf_counter()
        data_time = 0.0

        for idx, op in enumerate(ops):
            s, i = op.stage, op.mb
            if self.version[s] - base != trace.live_at_op[idx]:
                raise AssertionError("engine/trace version mismatch")
            if op.kind == "F" and s == 0 and op.micro == 0:
                td = time.perf_counter()
                x, y = next(it)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                pending_x[i] = list(torch.tensor_split(x, N))
                labels[i] = list(torch.tensor_split(y, N))
                sizes[i] = x.shape[0]
                data_time += time.perf_counter() - td
            if self.profile_ops:
                if on_cuda:
                    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    ev0.record()
                else:
                    t0 = time.perf_counter()

            if op.kind == "F":
                j = op.micro
                k = base + trace.fwd_version[(s, i, j)]
                x = pending_x[i][j] if s == 0 else inputs.pop((s, i, j))
                params = self._params(s, k, stored)
                if self.backward_rule == "graph":
                    xin = x.detach().requires_grad_(s > 0)
                    gf = GraphForward(params)
                    with torch.enable_grad():
                        out = gf.run(lambda: self._run_keep_graph(s, params, xin))
                        rec = {"x": xin, "gf": gf, "leaves": [params[n] for n, _ in self.named[s]],
                               "target": out if s < last else
                               F.cross_entropy(_upcast(out), labels[i][j], reduction="sum") / sizes[i]}
                    saved[(s, i, j)] = rec
                    out = out.detach()
                else:
                    with torch.no_grad():
                        out = self._run(s, params, x)
                    saved[(s, i, j)] = x
                if s < last:
                    inputs[(s + 1, i, j)] = out
                    comm_bytes[idx] = _nbytes(out)
                else:
                    y = labels[i][j]
                    loss_sum += F.cross_entropy(_upcast(out), y, reduction="sum").double()
                    correct += (out.argmax(1) == y).sum()
                    n_seen += y.shape[0]
                if s == 0:
                    pending_x[i][j] = None
                if self.log_ops:
                    self.op_log.append({"kind": "F", "stage": s, "mb": i, "micro": j, "version": k,
                                        "live": self.version[s], "slot": op.slot})
            else:
                grads = [None] * len(self.named[s])
                used = []
                M = sizes[i]
                for j in range(N):
                    k = base + trace.bwd_version[(s, i, j)]
                    used.append(k)
                    params = self._params(s, k, stored)
                    if self.backward_rule == "graph":
                        rec = saved.pop((s, i, j))
                        rec["gf"].use(params)
                        gout = None if s == last else grads_in.pop((s, i, j))
                        g = torch.autograd.grad(rec["target"], rec["leaves"] + ([rec["x"]] if s > 0 else []),
                                                grad_outputs=gout)
                        for n in range(len(rec["leaves"])):
                            grads[n] = g[n] if grads[n] is None else grads[n] + g[n]
                        if s > 0:
                            grads_in[(s - 1, i, j)] = g[-1]
                            comm_bytes[idx] += _nbytes(g[-1])
                        continue
                    plist = [params[n] for n, _ in self.named[s]]
                    x = saved.pop((s, i, j))
                    if s > 0:
                        x = x.detach().requires_grad_(True)
                    with torch.enable_grad(), frozen_bn_stats(self.stages[s]):
                        out = self._run(s, params, x)
                        if s == last:
                            target = F.cross_entropy(_upcast(out), labels[i][j], reduction="sum") / M
                            gout = None
                        else:
                            target, gout = out, grads_in.pop((s, i, j))
                        wrt = plist + ([x] if s > 0 else [])
                        g = torch.autograd.grad(target, wrt, grad_outputs=gout)
                    for n in range(len(plist)):
                        grads[n] = g[n] if grads[n] is None else grads[n] + g[n]
                    if s > 0:
                        grads_in[(s - 1, i, j)] = g[-1]
                        comm_bytes[idx] += _nbytes(g[-1])
                if s == last:
                    del labels[i]
                if s == 0:
                    del pending_x[i], sizes[i]
                if self.on_backward is not None:
                    self.on_backward(s, i, {n: g for (n, _), g in zip(self.named[s], grads)}, used)
                if self.log_ops:
                    self.op_log.append({"kind": "B", "stage": s, "mb": i, "versions": used,
                                        "live": self.version[s], "slot": op.slot})
                for kk in [kk for kk in stored[s] if trace.last_use.get((s, kk - base), -1) <= idx]:
                    self._drop_version(s, stored, kk)   # release before cloning so copies never pile up
                cur = self.version[s]
                if trace.last_use.get((s, cur - base), -1) > idx:
                    stored[s][cur] = self._clone_live(s)
                for (_, p), g in zip(self.named[s], grads):
                    p.grad = g
                self.optimizers[s].step()
                self.optimizers[s].zero_grad(set_to_none=True)
                self.schedulers[s].step()
                self.version[s] += 1
                self._snapshot_version(s)

            if self.profile_ops:
                if on_cuda:
                    ev1.record()
                    timers.append((ev0, ev1))
                else:
                    timers.append(time.perf_counter() - t0)
            # release weight versions that no later op needs
            for kk in [kk for kk in stored[s] if trace.last_use.get((s, kk - base), -1) <= idx]:
                self._drop_version(s, stored, kk)
            # memory accounting for this stage (weights incl. stored versions + held activations)
            # (graph rule: only the stage input of each kept graph is counted; the real total is
            #  the measured CUDA peak)
            held = sum(_nbytes(t["x"] if isinstance(t, dict) else t) for (ss, _, _), t in saved.items() if ss == s)
            held += sum(_nbytes(t) for (ss, _, _), t in inputs.items() if ss == s)
            held += sum(_nbytes(t) for (ss, _, _), t in grads_in.items() if ss == s)
            mem = param_bytes[s] * (1 + len(stored[s])) + held
            mem_max[s] = max(mem_max[s], mem)

        if any(stored[s] for s in range(W)) or saved or inputs or grads_in or labels:
            raise AssertionError("pipeline not drained at end of epoch")
        if on_cuda:
            torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - t_epoch
        if self.profile_ops:
            durations = [a.elapsed_time(b) / 1000.0 for a, b in timers] if on_cuda else timers
        else:
            durations = None
        return {
            "minibatches": K,
            "samples": n_seen,
            "train_loss": (loss_sum / max(1, n_seen)).item(),
            "train_acc1": correct.item() / max(1, n_seen) * 100.0,
            "wall_time_s": wall,
            "data_time_s": data_time,
            "op_durations_s": durations,
            "comm_bytes": comm_bytes,
            "max_weight_versions": list(trace.max_versions),
            "v_max": max(trace.v.values()),
            "stage_mem_accounted_bytes": mem_max,
            "stage_param_bytes": param_bytes,
        }

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.train(False)
        loss, c1, c5, n = 0.0, 0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = x
            for m in self.stages:
                out = m(out)
            out = _upcast(out)
            loss += F.cross_entropy(out, y, reduction="sum").item()
            top = out.topk(min(5, out.shape[1]), 1).indices
            c1 += (top[:, 0] == y).sum().item()
            c5 += (top == y[:, None]).any(1).sum().item()
            n += y.shape[0]
        self.train(True)
        return {"loss": loss / n, "acc1": c1 / n * 100.0, "acc5": c5 / n * 100.0}

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.stages:
            x = m(x)
        return x

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        return {
            "stages": [m.state_dict() for m in self.stages],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "schedulers": [s.state_dict() for s in self.schedulers],
            "version": list(self.version),
        }

    def load_state_dict(self, sd: dict):
        for m, d in zip(self.stages, sd["stages"]):
            m.load_state_dict(d)
        for o, d in zip(self.optimizers, sd["optimizers"]):
            o.load_state_dict(d)
        for s, d in zip(self.schedulers, sd["schedulers"]):
            s.load_state_dict(d)
        self.version = list(sd["version"])
