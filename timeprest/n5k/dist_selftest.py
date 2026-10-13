"""2-process self-test of the stage runtime with the nutrition task (synthetic, float64, CPU/gloo):
dumps final params, every weight version, gradients and the op log per rank (as timeprest.dist.selftest).

    python -m timeprest.dist.launch_local --nproc 2 -m timeprest.n5k.dist_selftest --out DIR --system timeprest
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from ..dist.runtime import Channels, StageRuntime
from ..dist.selftest import TRAIN, pipeline_cfg
from .synthetic import make_batches, make_model, make_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", default="timeprest")
    ap.add_argument("--order", default="static")
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    from ..dist.train import init_dist
    rank, W, device = init_dist({"runtime": {"device": "cpu"}, "pipeline": {"num_stages": int(os.environ["WORLD_SIZE"])}})
    pc = pipeline_cfg(args.system, args.N, args.order, "auto", W, "graph")
    task = make_task()
    stages = make_model(args.model, W)
    batches = make_batches(args.model, task, args.K, TRAIN["batch_size"])
    feats, x = [], batches[0][0][:1]
    with torch.no_grad():
        for m in stages:
            x = m.eval()(x)
            m.train()
            feats.append(tuple(x.shape[1:]))
    ch = Channels(rank, W)
    rt = StageRuntime(rank, W, stages[rank], pc, dict(TRAIN, epochs=args.epochs), device, args.K,
                      feats[rank - 1] if rank else tuple(batches[0][0].shape[1:]),
                      feats[rank] if rank < W - 1 else None, ch, TRAIN["batch_size"], dtype=torch.float64, task=task)
    rt.profile_ops = False
    rt.record_versions = True
    grads, stats = {}, []
    base = [0]
    rt.on_backward = lambda i, g, used: grads.__setitem__((base[0], i), ({n: t.detach().clone() for n, t in g.items()}, used))
    for ep in range(args.epochs):
        base[0] = ep
        stats.append(rt.run_epoch(args.K, (b[0] for b in batches) if rank == 0 else None,
                                  [b[1] for b in batches] if rank == W - 1 else None))
    ev = rt.evaluate((b[0] for b in batches) if rank == 0 else None,
                     [b[1] for b in batches] if rank == W - 1 else None, args.K)
    torch.save({"final": {n: p.detach().clone() for n, p in rt.named}, "versions": rt.version_params,
                "grads": grads, "op_log": rt.op_log, "buffers": [b.clone() for b in rt.module.buffers()],
                "stats": [{k: v for k, v in s.items() if k != "ops"} for s in stats], "eval": ev},
               os.path.join(args.out, f"rank{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
