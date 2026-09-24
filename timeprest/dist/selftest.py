"""Small multi-process scenario used by the tests/checks: runs the phase-2 runtime on a tiny
float64 model and dumps per-rank results (final params, every weight version, gradients, op log).

    python -m timeprest.dist.launch_local --nproc 2 -m timeprest.dist.selftest --out DIR --system timeprest --order static
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from ..config import SYSTEM_PRESETS
from ..models import mlp_blocks, vgg16_bn_blocks
from .runtime import Channels, StageRuntime

TRAIN = {"epochs": 1, "batch_size": 12, "optimizer": "sgd", "lr": 0.05, "momentum": 0.9,
         "nesterov": False, "weight_decay": 1e-3, "lr_schedule": "cosine", "warmup_epochs": 0}


def make_model(kind: str, W: int):
    torch.manual_seed(0)
    blocks = mlp_blocks(12, 16, 5, 4) if kind == "mlp" else vgg16_bn_blocks(5, width=1 / 16)
    cut = [round(len(blocks) * s / W) for s in range(W + 1)]
    return [nn.Sequential(*blocks[a:b]).double() for a, b in zip(cut[:-1], cut[1:])]


def make_batches(kind: str, K: int, M: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    shape = (M, 12) if kind == "mlp" else (M, 3, 32, 32)
    return [(torch.randn(*shape, generator=g, dtype=torch.float64),
             torch.randint(0, 5, (M,), generator=g)) for _ in range(K)]


def pipeline_cfg(system: str, N: int, order: str, backward_mode: str, W: int) -> dict:
    p = dict(SYSTEM_PRESETS[system])
    if p["schedule"] == "nF1B":
        p["num_microbatches"] = N
    p.update(num_stages=W, order=order, backward_mode=backward_mode, max_inflight="pipedream",
             sync_each_op=True, timeout_s=120)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", default="timeprest")
    ap.add_argument("--order", default="static")
    ap.add_argument("--backward-mode", default="recompute")
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    if os.environ.get("TIMEPREST_ANOMALY"):
        torch.autograd.set_detect_anomaly(True)
    from .train import init_dist
    rank, W, device = init_dist({"runtime": {"device": "cpu"}, "pipeline": {"num_stages": int(os.environ["WORLD_SIZE"])}})
    pc = pipeline_cfg(args.system, args.N, args.order, args.backward_mode, W)
    stages = make_model(args.model, W)
    batches = make_batches(args.model, args.K, TRAIN["batch_size"])
    feats, x = [], batches[0][0][:1]
    with torch.no_grad():
        for m in stages:
            x = m.eval()(x)
            m.train()
            feats.append(tuple(x.shape[1:]))
    ch = Channels(rank, W)
    tc = dict(TRAIN, epochs=args.epochs)
    rt = StageRuntime(rank, W, stages[rank], pc, tc, device, args.K, feats[rank - 1] if rank else None,
                      feats[rank] if rank < W - 1 else None, ch, TRAIN["batch_size"], dtype=torch.float64)
    rt.profile_ops = False
    rt.record_versions = True
    grads = {}
    base = [0]
    rt.on_backward = lambda i, g, used: grads.__setitem__((base[0], i), ({n: t.detach().clone() for n, t in g.items()}, used))
    for ep in range(args.epochs):
        base[0] = ep
        rt.run_epoch(args.K, (b[0] for b in batches) if rank == 0 else None,
                     [b[1] for b in batches] if rank == W - 1 else None)
    torch.save({"final": {n: p.detach().clone() for n, p in rt.named}, "versions": rt.version_params,
                "grads": grads, "op_log": rt.op_log, "buffers": [b.clone() for b in rt.module.buffers()]},
               os.path.join(args.out, f"rank{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
