"""Pipeline schedules (1F1B / nF1B) on idealised time slots, and symbolic weight-version traces.

Schedule rule (implementation choice, verified against paper Fig.2a–e, p.4):
  * every op (forward of one micro-batch, or backward of one mini-batch) takes one slot;
  * an op may only use results finished in an *earlier* slot;
  * backward has priority over forward (paper §3.2);
  * stage s (0-based) keeps at most `W - s` mini-batches in flight (PipeDream's NOAM rule),
    which is what turns N=1 into PipeDream's 1F1B; it never binds for the Fig.2 nF1B cases.

Version rule (see `trace_versions`): version k of a stage = its weights after k updates.
Updates happen once per mini-batch per stage, in mini-batch order.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Op:
    kind: str          # "F" (forward of one micro-batch) or "B" (backward of one mini-batch)
    stage: int
    mb: int            # 0-based mini-batch index within the epoch
    micro: int | None  # micro-batch index for F, None for B
    slot: int          # 0-based idealised time slot

    def label(self) -> str:
        if self.kind == "F":
            return f"{self.mb + 1}{chr(ord('A') + self.micro)}"
        return f"{self.mb + 1}"


def _caps(num_stages: int, max_inflight) -> list[float]:
    if max_inflight in (None, "none"):
        return [math.inf] * num_stages
    if max_inflight == "pipedream":
        return [num_stages - s for s in range(num_stages)]
    return [int(max_inflight)] * num_stages


def build_schedule(num_stages: int, num_microbatches: int, num_minibatches: int,
                   max_inflight="pipedream") -> list[Op]:
    """Return all ops of one epoch (drained at the end), sorted by (slot, stage)."""
    W, N, K = num_stages, num_microbatches, num_minibatches
    if W < 1 or N < 1 or K < 1:
        raise ValueError("num_stages, num_microbatches, num_minibatches must be >= 1")
    caps = _caps(W, max_inflight)
    fwd_order = [(i, j) for i in range(K) for j in range(N)]
    next_f = [0] * W
    next_b = [0] * W
    f_done = [dict() for _ in range(W)]   # (i, j) -> slot
    b_done = [dict() for _ in range(W)]   # i -> slot
    inflight = [set() for _ in range(W)]
    ops: list[Op] = []
    t = 0
    while next_b[0] < K:
        for s in range(W):
            i = next_b[s]
            if i < K:
                if s == W - 1:
                    ready = all(f_done[s].get((i, j), math.inf) < t for j in range(N))
                else:
                    ready = b_done[s + 1].get(i, math.inf) < t
                if ready:
                    ops.append(Op("B", s, i, None, t))
                    b_done[s][i] = t
                    inflight[s].discard(i)
                    next_b[s] += 1
                    continue
            if next_f[s] < len(fwd_order):
                i, j = fwd_order[next_f[s]]
                dep_ok = s == 0 or f_done[s - 1].get((i, j), math.inf) < t
                cap_ok = i in inflight[s] or len(inflight[s]) < caps[s]
                if dep_ok and cap_ok:
                    ops.append(Op("F", s, i, j, t))
                    f_done[s][(i, j)] = t
                    inflight[s].add(i)
                    next_f[s] += 1
        t += 1
        if t > 10 * (K * (N + 1) + W) + 100:
            raise RuntimeError("schedule did not terminate (deadlock?)")
    ops.sort(key=lambda o: (o.slot, o.stage))
    return ops


def render_grid(ops: list[Op], num_stages: int, max_slots: int | None = None) -> list[list[str]]:
    """Grid[stage][slot] of labels like Fig.2 ('.' = idle, '1A' = forward, '1' = backward)."""
    n_slots = max(o.slot for o in ops) + 1
    if max_slots is not None:
        n_slots = min(n_slots, max_slots)
    grid = [["." for _ in range(n_slots)] for _ in range(num_stages)]
    for o in ops:
        if o.slot < n_slots:
            grid[o.stage][o.slot] = o.label()
    return grid


def format_grid(grid: list[list[str]]) -> str:
    return "\n".join(f"M{s + 1}: " + " ".join(f"{c:>3}" for c in row) for s, row in enumerate(grid))


def version_difference_formula(W: int, N: int) -> int:
    """Paper Eq.(3)/(16): v = floor((W + N - 2) / N), stated for W >= 2, N >= 2."""
    return (W + N - 2) // N


@dataclass
class VersionTrace:
    fwd_version: dict        # (stage, mb, micro) -> version used in forward
    bwd_version: dict        # (stage, mb, micro) -> version used in backward for that micro
    committed_at_bwd: dict   # mb -> stage-0 version when the backward starts at the last stage
    v: dict                  # mb -> version difference (paper §3.4 definition)
    last_use: dict           # (stage, version) -> index (in ops) of last op that uses it
    superseded_at: dict      # (stage, version) -> index of the op that creates version+1
    max_versions: list       # per stage: max number of simultaneously stored weight versions
    live_at_op: list         # per op index: live version of that stage just before the op


def trace_versions(ops: list[Op], num_stages: int, num_microbatches: int,
                   vertical_sync: bool = True, backward_version: str = "committed") -> VersionTrace:
    """Symbolically assign weight versions to every op.

    forward : stage 0 uses its live version when the micro-batch enters the pipeline; with
              vertical_sync every later stage reuses that same version (paper §3, p.3; p.5:
              "the previous version is stored until a forward pass, that uses it, is completed").
    backward: "stashed"   -> the micro-batch's own forward version (PipeDream horizontal stashing)
              "committed" -> the version already applied on *all* stages (= stage 0's version)
                             when the backward starts at the last stage; same for all stages
                             (TiMePReSt: latest weights + vertical sync of the backward pass)
              "latest"    -> each stage's live version at its backward (no vertical sync).
    """
    W, N = num_stages, num_microbatches
    last = W - 1
    ver = [0] * W
    upd_slots = [[] for _ in range(W)]
    pinned = {}
    fwd, bwd, committed, v = {}, {}, {}, {}
    last_use: dict = {}
    superseded: dict = {}
    live_at_op = []

    def use(s, k, idx):
        if k > ver[s]:
            raise AssertionError(f"stage {s} asked for future version {k} (live {ver[s]})")
        last_use[(s, k)] = idx

    for idx, o in enumerate(ops):
        s = o.stage
        live_at_op.append(ver[s])
        if o.kind == "F":
            key = (o.mb, o.micro)
            if s == 0:
                pinned[key] = ver[0]
            k = pinned[key] if vertical_sync else ver[s]
            fwd[(s, o.mb, o.micro)] = k
            use(s, k, idx)
        else:
            if s == last:
                committed[o.mb] = sum(1 for u in upd_slots[0] if u < o.slot)
                v[o.mb] = o.mb - committed[o.mb] + 1
            for j in range(N):
                if backward_version == "stashed":
                    k = fwd[(s, o.mb, j)]
                elif backward_version == "committed":
                    k = committed[o.mb]
                elif backward_version == "latest":
                    k = ver[s]
                else:
                    raise ValueError(backward_version)
                bwd[(s, o.mb, j)] = k
                use(s, k, idx)
            superseded[(s, ver[s])] = idx
            ver[s] += 1
            upd_slots[s].append(o.slot)

    # count simultaneously stored versions per stage: live + old versions kept after the update
    # that superseded them (op sup) until their last use (op lu), i.e. during ops sup < idx <= lu
    max_versions = [1] * W
    events = defaultdict(list)  # stage -> list of (start_idx, end_idx) retention intervals
    for (s, k), sup_idx in superseded.items():
        lu = last_use.get((s, k), -1)
        if lu > sup_idx:
            events[s].append((sup_idx, lu))
    for s, intervals in events.items():
        for idx in range(len(ops)):
            n = 1 + sum(1 for a, b in intervals if a < idx <= b)
            max_versions[s] = max(max_versions[s], n)
    return VersionTrace(fwd, bwd, committed, v, last_use, superseded, max_versions, live_at_op)


def steady_state_v(trace: VersionTrace) -> int:
    return max(trace.v.values()) if trace.v else 1
