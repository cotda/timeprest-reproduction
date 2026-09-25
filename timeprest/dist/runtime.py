"""Real multi-process pipeline runtime (one process = one stage = one GPU), phase 2.

Scheduling (paper §3.2), `order: dynamic`: when the stage is free it runs the next backward if
its input gradient has arrived (last stage: all N forwards of that mini-batch done), otherwise the
next forward if its input has arrived and at most W-s mini-batches are in flight, otherwise waits.
`order: static` replays the idealised slot order of `schedule.build_schedule` (= phase-1 engine).

Weight versions are decided at run time:
  forward : stage 0 uses its live version and tags every micro-batch with it; with vertical sync
            the later stages reuse the tagged version (kept until no forward can still need it);
  backward: "stashed" -> the micro-batch's forward version (PipeDream);
            "latest"  -> the stage's live version, which is always `epoch_base + i` for mini-batch i
                         because each stage updates in mini-batch order (TiMePReSt; equals
                         phase-1 "committed" whenever v = 1).
Backward math is the phase-1 rule (`pipeline.backward_rule`, see engine.py):
  graph (default, PipeDream's mechanism): the forward graph is kept and its backward reads every
      saved weight at the backward version (`swap.GraphForward`); no recompute.
  recompute (ablation): local VJP at the backward version on the stored stage input; when forward
      and backward versions coincide the forward graph is kept instead (`backward_mode: auto`,
      identical math).
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from ..engine import _upcast, frozen_bn_stats, lr_lambda_factory
from ..schedule import build_schedule
from ..swap import GraphForward, release_storage


class Handle:
    """Pollable wrapper of an async p2p work. NCCL works report completion from the CUDA event;
    gloo works only complete inside wait(), so a helper thread waits for them (as PipeDream's
    receive helper threads do). With a gloo channel a GPU tensor travels through a CPU copy:
    `staged` is that copy, `dst` the GPU tensor filled from it on the first wait()."""

    def __init__(self, work, threaded: bool, staged=None, dst=None):
        self.work, self.staged, self.dst = work, staged, dst
        self.event = None
        if threaded:
            self.event = threading.Event()

            def _wait():
                work.wait()
                self.event.set()
            threading.Thread(target=_wait, daemon=True).start()

    def is_completed(self) -> bool:
        return self.event.is_set() if self.event is not None else self.work.is_completed()

    def wait(self):
        if self.event is not None:
            self.event.wait()
        else:
            self.work.wait()
        if self.dst is not None:
            self.dst.copy_(self.staged)
            self.dst = None


class EmulatedLink:
    """A slower network in front of one direction of a channel (phase 3, paper_notes §23): each
    message is transmitted after the previous one (a link carries one message at a time), takes
    `bytes / bandwidth`, and arrives `latency` later; only then is it handed to the real send.
    Transfers are serialised per link, and the fwd and bwd links are independent (full duplex).
    Contents and order are unchanged, so training is identical; only timing changes."""

    def __init__(self, bandwidth_gbps: float | None, latency_ms: float, send_fn):
        self.bytes_per_s = None if not bandwidth_gbps else bandwidth_gbps * 1e9 / 8
        self.latency_s = latency_ms / 1000.0
        self.send_fn = send_fn           # (tensor, peer, group) -> work with wait()
        self.q: queue.Queue = queue.Queue()
        self.free_at = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    class _Work:
        def __init__(self):
            self.event = threading.Event()
            self.error = None

        def wait(self):
            self.event.wait()
            if self.error is not None:
                raise self.error

        def is_completed(self):
            return self.event.is_set()

    def send(self, t, peer, group):
        w = self._Work()
        self.q.put((t, peer, group, w, time.perf_counter()))
        return w

    def transfer_s(self, nbytes: int) -> float:
        return nbytes / self.bytes_per_s if self.bytes_per_s else 0.0

    def _loop(self):
        while True:
            t, peer, group, w, t_post = self.q.get()
            try:
                start = max(t_post, self.free_at)
                self.free_at = start + self.transfer_s(t.numel() * t.element_size())
                delay = self.free_at + self.latency_s - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                self.send_fn(t, peer, group).wait()
            except Exception as e:  # surfaced to whoever waits on the send
                w.error = e
            w.event.set()


class Channels:
    """Point-to-point channels between neighbouring stages. Forward activations and backward
    gradients use separate process groups, so their message orders never interfere.

    backend "gloo" (default, as PipeDream's runtime for pipelined configs): tensors are staged
    through CPU memory and every pending receive waits on a CPU thread. Nothing ever sits on the
    GPU waiting for the peer, so no GPU-side deadlock is possible. With NCCL a posted receive is a
    GPU kernel that only ends when the peer sends; any context-wide sync in the meantime (e.g. a
    kernel's first launch loading its module) then deadlocks both stages (seen on Kaggle T4 x2).
    backend None: same backend as the default process group.
    emulate_bandwidth_gbps / emulate_latency_ms: put an `EmulatedLink` in front of every send
    (gloo only), to reproduce slower inter-machine networks (phase 3, E3)."""

    def __init__(self, rank: int, world: int, backend: str | None = "gloo",
                 emulate_bandwidth_gbps: float | None = None, emulate_latency_ms: float = 0.0):
        self.rank, self.world = rank, world
        self.backend = backend or dist.get_backend()
        kw = {} if backend is None else {"backend": backend}
        self.fwd = [dist.new_group([s, s + 1], **kw) for s in range(world - 1)]
        self.bwd = [dist.new_group([s, s + 1], **kw) for s in range(world - 1)]
        self.threaded = self.backend != "nccl"
        self.stage = self.backend != "nccl"      # gloo p2p needs CPU tensors
        self._warm_up()
        self.links = {}
        if emulate_bandwidth_gbps or emulate_latency_ms:
            if self.backend == "nccl":
                raise ValueError("network emulation needs the gloo p2p backend")
            real = lambda t, peer, g: dist.isend(t, peer, group=g)
            for g in self.fwd + self.bwd:   # one link per direction actually used by this rank
                self.links[id(g)] = EmulatedLink(emulate_bandwidth_gbps, emulate_latency_ms, real)

    def _warm_up(self):
        """Create every communicator and connect both directions of each group up front, in the same
        order on all ranks. NCCL creates a group's communicator, and connects each direction a -> b,
        lazily at the first send/recv, blocking the host until the peer joins: a gradient receive
        (s+1 -> s) posted first on one rank while the other waits for an activation deadlocks."""
        nccl = self.backend == "nccl"
        dev = torch.device("cuda", torch.cuda.current_device()) if nccl else torch.device("cpu")
        t = torch.zeros(1, device=dev)
        for groups in (self.fwd, self.bwd):
            for s, g in enumerate(groups):
                for src, dst in ((s, s + 1), (s + 1, s)):
                    if self.rank == src:
                        dist.send(t, dst, group=g)
                    elif self.rank == dst:
                        dist.recv(t, src, group=g)
                    # finish this transfer completely before any rank starts the next handshake
                    if nccl:
                        torch.cuda.synchronize(dev)
                    dist.barrier()

    def _isend(self, t, peer, g):
        if self.stage and t.is_cuda:
            t = t.cpu()
        link = self.links.get(id(g))
        work = link.send(t, peer, g) if link is not None else dist.isend(t, peer, group=g)
        return Handle(work, self.threaded, staged=t)

    def _irecv(self, t, peer, g):
        if self.stage and t.is_cuda:
            cpu = torch.empty(t.shape, dtype=t.dtype)
            return Handle(dist.irecv(cpu, peer, group=g), self.threaded, staged=cpu, dst=t)
        return Handle(dist.irecv(t, peer, group=g), self.threaded)

    def isend_fwd(self, t):
        return self._isend(t, self.rank + 1, self.fwd[self.rank])

    def irecv_fwd(self, t):
        return self._irecv(t, self.rank - 1, self.fwd[self.rank - 1])

    def isend_bwd(self, t):
        return self._isend(t, self.rank - 1, self.bwd[self.rank - 1])

    def irecv_bwd(self, t):
        return self._irecv(t, self.rank + 1, self.bwd[self.rank])

    def send_fwd(self, t):
        self.isend_fwd(t).wait()

    def recv_fwd(self, t):
        self.irecv_fwd(t).wait()


def _done(work) -> bool:
    return work.is_completed()


class StageRuntime:
    def __init__(self, rank: int, world: int, module: nn.Module, pipeline_cfg: dict, training_cfg: dict,
                 device, steps_per_epoch: int, in_feat: tuple | None, out_feat: tuple | None,
                 channels: Channels, batch_size: int, dtype=torch.float32):
        self.s, self.W, self.last = rank, world, world - 1
        self.device = torch.device(device)
        self.module = module.to(self.device)
        self.N = int(pipeline_cfg["num_microbatches"])
        self.vertical_sync = bool(pipeline_cfg["vertical_sync"])
        bv = pipeline_cfg["backward_version"]
        if bv == "committed":
            bv = "latest"   # identical when v = 1; the dynamic runtime has no global commit
        self.backward_version = bv
        self.order = pipeline_cfg.get("order", "dynamic")
        self.backward_rule = pipeline_cfg.get("backward_rule") or "graph"
        self.backward_mode = pipeline_cfg.get("backward_mode", "auto")  # recompute rule only: auto | recompute
        mi = pipeline_cfg.get("max_inflight", "pipedream")
        self.cap = (world - rank) if mi == "pipedream" else (float("inf") if mi in (None, "none") else int(mi))
        self.max_inflight_cfg = mi
        self.sync_each_op = pipeline_cfg.get("sync_each_op", True)
        self.prefetch = int(pipeline_cfg.get("recv_prefetch") or 2 * self.N + 2)
        self.timeout_s = float(pipeline_cfg.get("timeout_s", 600))
        self.in_feat, self.out_feat = in_feat, out_feat
        self.ch = channels
        self.dtype = dtype
        self.named = list(self.module.named_parameters())
        self.M = int(batch_size)
        self.micro = [c.shape[0] for c in torch.tensor_split(torch.empty(self.M), self.N)]
        tc = training_cfg
        self.opt = torch.optim.SGD(self.module.parameters(), lr=tc["lr"], momentum=tc["momentum"],
                                   weight_decay=tc["weight_decay"], nesterov=tc.get("nesterov", False))
        total = tc["epochs"] * steps_per_epoch
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda_factory(total, int(tc.get("warmup_epochs", 0) * steps_per_epoch), tc["lr_schedule"]))
        self.version = 0
        self.snaps: dict = {}          # version -> detached params (never modified in place)
        self.pins: dict = {}           # version -> number of pending backwards that read it
        self.profile_ops = True
        self.on_backward = None        # debug hook fn(mb, grads, versions)
        self.record_versions = False
        self.version_params: dict = {}
        self.op_log: list = []
        self._kernels_warm = False

    # ------------------------------------------------------------------ versions
    def _live(self) -> dict:
        return {n: p for n, p in self.named}

    def _snapshot(self, k: int) -> dict:
        if k in self.snaps:
            return self.snaps[k]
        if k != self.version:
            raise RuntimeError(f"stage {self.s}: version {k} needed but not kept (live {self.version})")
        snap = {n: p.detach().clone().requires_grad_(True) for n, p in self.named}
        self.snaps[k] = snap
        return snap

    def _prune(self, keep_from: int):
        for k in [k for k in self.snaps if k < keep_from and k != self.version and not self.pins.get(k)]:
            snap = self.snaps.pop(k)
            if self.backward_rule == "graph":   # kept graphs hold these only as gradient leaves
                release_storage(snap)

    def _run(self, params: dict, x):
        return functional_call(self.module, params, (x,))

    def _run_keep_graph(self, params: dict, x):
        """Forward whose graph is kept until a later backward. BatchNorm saves its running-stat
        buffers for backward and every later forward updates them in place, so the forward runs
        on private copies that are then written back (same running-stat trajectory)."""
        bufs = {n: b.clone() for n, b in self.module.named_buffers()}
        out = functional_call(self.module, {**params, **bufs}, (x,))
        with torch.no_grad():
            for n, b in self.module.named_buffers():
                b.copy_(bufs[n])
        return out

    def warm_up_kernels(self):
        """Run every kind of compute of this stage once (train forward/backward for each micro-batch
        size, an optimizer step, eval forward) before any NCCL receive is posted. First launches load
        CUDA modules / cuDNN sub-libraries, which needs a context-wide sync: done later while a
        posted receive is still waiting for the peer, it would deadlock. No side effects: buffers,
        parameters, optimizer and RNG state are left untouched."""
        if self._kernels_warm or self.device.type != "cuda" or self.in_feat is None:
            return
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        was = self.module.training
        params = {n: p.detach().clone().requires_grad_(True) for n, p in self.named}
        bufs = {n: b.clone() for n, b in self.module.named_buffers()}
        opt = torch.optim.SGD(list(params.values()), **{k: self.opt.defaults[k] for k in
                                                        ("lr", "momentum", "weight_decay", "nesterov")})
        for train in (True, False):
            self.module.train(train)
            for m in sorted(set(self.micro) | {self.M}):
                x = torch.randn((m, *self.in_feat), dtype=self.dtype, device=self.device,
                                requires_grad=train and self.s > 0)
                with torch.enable_grad() if train else torch.no_grad():
                    out = functional_call(self.module, {**params, **bufs}, (x,))
                    if not train:
                        continue
                    if self.s == self.last:   # same bookkeeping ops as do_forward
                        y = torch.randint(0, out.shape[1], (m,), device=self.device)
                        l = F.cross_entropy(_upcast(out), y, reduction="sum")
                        acc_l = torch.zeros((), device=self.device, dtype=torch.float64)
                        acc_c = torch.zeros((), device=self.device, dtype=torch.long)
                        acc_l += l.detach().double()
                        acc_c += (out.detach().argmax(1) == y).sum()
                        (l / self.M).backward()
                        acc_l.item(), acc_c.item()
                    else:
                        out.backward(torch.randn_like(out))
                    if self.s > 0:            # do_backward: summed grads, concatenated input grads
                        g = [p.grad + p.grad for p in params.values()]
                        torch.cat([x.grad, x.grad]).contiguous()
                        del g
            if train:
                opt.step()
                opt.zero_grad(set_to_none=True)
        self.module.train(was)
        torch.cuda.synchronize(self.device)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, self.device)
        self._kernels_warm = True

    # ------------------------------------------------------------------ epoch
    def run_epoch(self, K: int, batches=None, labels=None) -> dict:
        """batches: iterable of input mini-batches (stage 0); labels: list of label mini-batches
        (last stage). Returns local statistics."""
        s, N, last = self.s, self.N, self.last
        self.warm_up_kernels()
        base = self.version
        self.module.train()
        if self.record_versions:
            self.version_params[self.version] = {n: p.detach().cpu().clone() for n, p in self.named}
        fwd_list = [(i, j) for i in range(K) for j in range(N)]
        it = iter(batches) if s == 0 else None
        chunks = {}                   # stage 0: i -> list of micro inputs
        lab = {}
        records = {}                  # (i, j) -> dict
        act_q: deque = deque()        # posted forward receives (s > 0)
        grad_q: deque = deque()       # posted gradient receives (s < last)
        n_act_posted = n_grad_posted = 0
        sends: list = []
        f_idx = b_idx = 0
        inflight: set = set()
        last_pin = base
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        correct = torch.zeros((), device=self.device, dtype=torch.long)
        n_seen = 0
        stats = {"recompute_micro": 0, "graph_micro": 0, "max_snaps": 0, "bytes_sent": 0,
                 "msgs_sent": 0, "waits": 0, "wait_time_s": 0.0}
        ev_log = []
        on_cuda = self.device.type == "cuda"
        static_ops = [o for o in build_schedule(self.W, N, K, self.max_inflight_cfg) if o.stage == s] \
            if self.order == "static" else None
        op_pos = 0
        if on_cuda:
            ev_epoch = torch.cuda.Event(enable_timing=True)
            ev_epoch.record()
        t0 = time.perf_counter()

        def post_recvs():
            nonlocal n_act_posted, n_grad_posted
            if s > 0:
                while n_act_posted < K * N and len(act_q) < self.prefetch:
                    i, j = fwd_list[n_act_posted]
                    m = self.micro[j]
                    tag = torch.empty(1, dtype=torch.int64, device=self.device)
                    buf = torch.empty((m, *self.in_feat), dtype=self.dtype, device=self.device)
                    act_q.append((i, j, self.ch.irecv_fwd(tag), self.ch.irecv_fwd(buf), tag, buf))
                    n_act_posted += 1
            if s < last:
                while n_grad_posted < K and len(grad_q) < 2:
                    buf = torch.empty((self.M, *self.out_feat), dtype=self.dtype, device=self.device)
                    grad_q.append((n_grad_posted, self.ch.irecv_bwd(buf), buf))
                    n_grad_posted += 1

        def b_ready() -> bool:
            if b_idx >= K:
                return False
            if s == last:
                return all((b_idx, j) in records for j in range(N))
            return bool(grad_q) and grad_q[0][0] == b_idx and _done(grad_q[0][1])

        def f_ready() -> bool:
            if f_idx >= K * N:
                return False
            i, _ = fwd_list[f_idx]
            if not (i in inflight or len(inflight) < self.cap):
                return False
            if s == 0:
                return True
            h = act_q[0]
            return _done(h[2]) and _done(h[3])

        def do_forward():
            nonlocal f_idx, last_pin, loss_sum, correct, n_seen
            i, j = fwd_list[f_idx]
            if s == 0:
                if j == 0:
                    x_all = next(it).to(self.device, non_blocking=True)
                    chunks[i] = list(torch.tensor_split(x_all, N))
                x = chunks[i][j]
                chunks[i][j] = None
                k = self.version
            else:
                ii, jj, wt, wa, tag, buf = act_q.popleft()
                assert (ii, jj) == (i, j)
                wt.wait()
                wa.wait()
                x = buf
                k = int(tag.item()) if self.vertical_sync else self.version
                last_pin = max(last_pin, int(tag.item()))
            vi = base + i                       # live version when B(i) runs on this stage
            kb = k if self.backward_version == "stashed" else vi
            graph = self.backward_mode == "auto" and kb == k
            if self.backward_rule == "graph":
                fparams = self._live() if k == self.version else self._snapshot(k)
                # backward version: live at B time when it will be kb (= base + i), else kept
                bparams = None if kb == vi else self._snapshot(kb)
                if bparams is not None:
                    self.pins[kb] = self.pins.get(kb, 0) + 1
                xin = x.detach().requires_grad_(s > 0)
                gf = GraphForward(fparams)
                with torch.enable_grad():
                    out = gf.run(lambda: self._run_keep_graph(fparams, xin))
                rec = {"mode": "graph", "x": xin, "gf": gf, "bparams": bparams,
                       "plist": [fparams[n] for n, _ in self.named], "k": k, "kb": kb}
                stats["graph_micro"] += 1
            elif graph:
                params = self._live() if (k == self.version == vi) else self._snapshot(k)
                xin = x.detach().requires_grad_(s > 0)
                out = self._run_keep_graph(params, xin)
                rec = {"mode": "graph", "x": xin, "plist": [params[n] for n, _ in self.named], "k": k, "kb": kb}
                stats["graph_micro"] += 1
            else:
                fparams = self._live() if k == self.version else self._snapshot(k)
                with torch.no_grad():
                    out = self._run(fparams, x)
                # backward params: live at B time ("latest"), or the stashed version
                bparams = None if self.backward_version == "latest" else \
                    (self._live() if k == self.version == vi else self._snapshot(k))
                rec = {"mode": "recompute", "x": x.detach(), "bparams": bparams, "k": k, "kb": kb}
                stats["recompute_micro"] += 1
            if s < last:
                o = out.detach().contiguous()
                tag = torch.tensor([k], dtype=torch.int64, device=self.device)
                sends.append((self.ch.isend_fwd(tag), tag))
                sends.append((self.ch.isend_fwd(o), o))
                stats["bytes_sent"] += o.numel() * o.element_size()
                stats["msgs_sent"] += 1
                if rec["mode"] == "graph":
                    rec["out"] = out
            else:
                y = lab[i][j]
                l = F.cross_entropy(_upcast(out), y, reduction="sum")
                loss_sum += l.detach().double()
                correct += (out.detach().argmax(1) == y).sum()
                n_seen += y.shape[0]
                if rec["mode"] == "graph":
                    rec["loss"] = l / self.M
            records[(i, j)] = rec
            inflight.add(i)
            f_idx += 1
            if self.record_versions:
                self.op_log.append(("F", i, j, k, self.version))
            return ("F", i, j)

        def do_backward():
            nonlocal b_idx
            i = b_idx
            if s < last:
                ii, wg, gbuf = grad_q.popleft()
                assert ii == i
                wg.wait()
                gchunks = list(torch.tensor_split(gbuf, N))
            grads = [None] * len(self.named)
            gins, used = [], []
            for j in range(N):
                rec = records.pop((i, j))
                used.append(rec["kb"])
                if rec["mode"] == "graph":
                    if "gf" in rec:
                        rec["gf"].use(self._live() if rec["bparams"] is None else rec["bparams"])
                        if rec["bparams"] is not None:
                            self.pins[rec["kb"]] -= 1
                    target = rec["loss"] if s == last else rec["out"]
                    wrt = rec["plist"] + ([rec["x"]] if s > 0 else [])
                    g = torch.autograd.grad(target, wrt, grad_outputs=None if s == last else gchunks[j])
                else:
                    params = self._live() if rec["bparams"] is None else rec["bparams"]
                    x = rec["x"].requires_grad_(s > 0)
                    with torch.enable_grad(), frozen_bn_stats(self.module):
                        out = self._run(params, x)
                        if s == last:
                            target = F.cross_entropy(_upcast(out), lab[i][j], reduction="sum") / self.M
                        else:
                            target = out
                        wrt = [params[n] for n, _ in self.named] + ([x] if s > 0 else [])
                        g = torch.autograd.grad(target, wrt, grad_outputs=None if s == last else gchunks[j])
                for n in range(len(self.named)):
                    grads[n] = g[n] if grads[n] is None else grads[n] + g[n]
                if s > 0:
                    gins.append(g[-1])
            if s > 0:
                gall = torch.cat(gins).contiguous()        # one backward message per mini-batch
                sends.append((self.ch.isend_bwd(gall), gall))
                stats["bytes_sent"] += gall.numel() * gall.element_size()
                stats["msgs_sent"] += 1
            if s == last:
                del lab[i]
            if s == 0:
                del chunks[i]
            if self.on_backward is not None:
                self.on_backward(i, {n: g for (n, _), g in zip(self.named, grads)}, used)
            if self.vertical_sync and s > 0:
                self._snapshot(self.version)   # later forwards may still be tagged with it
            for (_, p), g in zip(self.named, grads):
                p.grad = g
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()
            self.version += 1
            if self.record_versions:
                self.version_params[self.version] = {n: p.detach().cpu().clone() for n, p in self.named}
                self.op_log.append(("B", i, None, used, self.version - 1))
            inflight.discard(i)
            b_idx += 1
            return ("B", i, None)

        if s == last:
            for i in range(K):
                lab[i] = list(torch.tensor_split(labels[i].to(self.device), N))

        last_progress = time.perf_counter()
        while b_idx < K:
            post_recvs()
            if static_ops is not None:
                o = static_ops[op_pos]
                kind = o.kind
                if kind == "F" and s > 0:
                    act_q[0][2].wait()
                    act_q[0][3].wait()
                if kind == "B" and s < last:
                    grad_q[0][1].wait()
                ready_kind = kind
                op_pos += 1
            else:
                ready_kind = "B" if b_ready() else ("F" if f_ready() else None)
            if ready_kind is None:
                tw = time.perf_counter()
                while not (b_ready() or f_ready()):
                    post_recvs()
                    time.sleep(0.0001)
                    if time.perf_counter() - last_progress > self.timeout_s:
                        raise RuntimeError(f"stage {s}: no progress for {self.timeout_s}s "
                                           f"(f_idx={f_idx}, b_idx={b_idx}, inflight={sorted(inflight)})")
                stats["waits"] += 1
                stats["wait_time_s"] += time.perf_counter() - tw
                continue
            if on_cuda and self.profile_ops:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
            done = do_backward() if ready_kind == "B" else do_forward()
            if on_cuda and self.profile_ops:
                e1.record()
                ev_log.append((done, e0, e1))
            elif self.profile_ops:
                ev_log.append((done, time.perf_counter(), None))
            if self.sync_each_op and on_cuda:
                torch.cuda.current_stream(self.device).synchronize()
            sends[:] = [(w, t) for w, t in sends if not _done(w)]
            stats["max_snaps"] = max(stats["max_snaps"], len(self.snaps))
            keep = last_pin if (self.vertical_sync and s > 0) else self.version
            self._prune(keep)
            last_progress = time.perf_counter()

        for w, _ in sends:
            w.wait()
        if on_cuda:
            torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - t0
        if records or chunks or lab or act_q or grad_q:
            raise AssertionError(f"stage {s}: pipeline not drained")
        self.snaps.clear()
        self.pins.clear()
        ops = []
        if on_cuda and self.profile_ops:
            for (kind, i, j), a, b in ev_log:
                ops.append((kind, i, j, ev_epoch.elapsed_time(a) / 1000.0, ev_epoch.elapsed_time(b) / 1000.0))
        busy = sum(b - a for _, _, _, a, b in ops) if ops else None
        return {
            "stage": s, "wall_s": wall, "busy_s": busy, "ops": ops,
            "loss_sum": loss_sum.item(), "correct": int(correct.item()), "samples": n_seen,
            "peak_mem_mb": torch.cuda.max_memory_allocated(self.device) / 2**20 if on_cuda else None,
            **stats,
        }

    # ------------------------------------------------------------------ eval (synchronous)
    @torch.no_grad()
    def evaluate(self, batches=None, labels=None, n_batches: int = 0) -> dict | None:
        self.module.eval()
        s, last = self.s, self.last
        loss = 0.0
        c1 = c5 = n = 0
        for b in range(n_batches):
            if s == 0:
                x = next(batches).to(self.device)
            else:
                shape = torch.empty(1 + len(self.in_feat), dtype=torch.int64, device=self.device)
                self.ch.recv_fwd(shape)
                x = torch.empty(tuple(shape.tolist()), dtype=self.dtype, device=self.device)
                self.ch.recv_fwd(x)
            out = self.module(x)
            if s < last:
                o = out.contiguous()
                self.ch.send_fwd(torch.tensor(list(o.shape), dtype=torch.int64, device=self.device))
                self.ch.send_fwd(o)
            else:
                y = labels[b].to(self.device)
                o = _upcast(out)
                loss += F.cross_entropy(o, y, reduction="sum").item()
                top = o.topk(min(5, o.shape[1]), 1).indices
                c1 += (top[:, 0] == y).sum().item()
                c5 += (top == y[:, None]).any(1).sum().item()
                n += y.shape[0]
        self.module.train()
        if s == last:
            return {"loss": loss / n, "acc1": c1 / n * 100.0, "acc5": c5 / n * 100.0}
        return None

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict(), "version": self.version}

    def load_state_dict(self, sd: dict):
        self.module.load_state_dict(sd["module"])
        self.opt.load_state_dict(sd["optimizer"])
        self.sched.load_state_dict(sd["scheduler"])
        self.version = sd["version"]
