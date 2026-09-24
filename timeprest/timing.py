"""Estimate the wall-clock time of an epoch on a real W-GPU pipeline from per-op durations
measured in the single-GPU simulation (estimate only; real numbers come from phase 2)."""
from __future__ import annotations

from .schedule import Op


def estimate_pipeline_time(ops: list[Op], durations: list[float], comm_bytes: list[int],
                           num_stages: int, num_microbatches: int,
                           bandwidth_Bps: float = float("inf"), latency_s: float = 0.0) -> float:
    """Event simulation: each stage runs its ops in schedule order; an op starts when its
    stage is free and its inputs have arrived (sender finish + latency + bytes / bandwidth).
    Communication is asynchronous (overlaps with the sender's next op); no link contention."""
    W, N, last = num_stages, num_microbatches, num_stages - 1
    stage_free = [0.0] * W
    f_ready: dict = {}   # (s, i, j) -> time the forward output is available at stage s+1
    f_fin: dict = {}     # (s, i, j) -> finish time of forward at stage s
    b_ready: dict = {}   # (s, i) -> time the backward gradient is available at stage s-1
    for idx, o in enumerate(ops):
        s = o.stage
        if o.kind == "F":
            dep = 0.0 if s == 0 else f_ready[(s - 1, o.mb, o.micro)]
        else:
            if s == last:
                dep = max(f_fin[(s, o.mb, j)] for j in range(N))
            else:
                dep = b_ready[(s + 1, o.mb)]
        start = max(stage_free[s], dep)
        fin = start + durations[idx]
        stage_free[s] = fin
        comm = latency_s + (comm_bytes[idx] / bandwidth_Bps if bandwidth_Bps != float("inf") else 0.0)
        if o.kind == "F":
            f_fin[(s, o.mb, o.micro)] = fin
            if s < last:
                f_ready[(s, o.mb, o.micro)] = fin + comm
        elif s > 0:
            b_ready[(s, o.mb)] = fin + comm
    return max(stage_free)


def summarize_durations(ops: list[Op], durations: list[float]) -> dict:
    """Mean duration per (kind, stage)."""
    acc: dict = {}
    for o, d in zip(ops, durations):
        key = f"{o.kind}{o.stage}"
        a = acc.setdefault(key, [0.0, 0])
        a[0] += d
        a[1] += 1
    return {k: v[0] / v[1] for k, v in sorted(acc.items())}
