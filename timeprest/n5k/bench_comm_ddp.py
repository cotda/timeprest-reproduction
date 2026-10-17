"""Epoch time of the DDP baseline vs (emulated) network bandwidth (n5k/PLAN.md mục 10, 2026-10-17 18:21): the DDP
counterpart of `timeprest.n5k.bench_comm` (TiMePReSt / PipeDream), run on the same machine in the same kernel.

The slower network sits in a DDP communication hook. When a gradient bucket is ready during backward it is queued on
one emulated link (one bucket at a time, full duplex like the pipeline's EmulatedLink) that holds it for
    2 (W-1)/W * bytes / bandwidth  (+ 2 (W-1) * latency)          (ring all-reduce: each rank sends that much)
from the moment the link is free; only then is the bucket really all-reduced (mean over the ranks, as DDP's default).
Values are unchanged, so training is identical to `timeprest.n5k.ddp_train`; DDP still overlaps the transfers with
the rest of backward, as on a real network. Config keys as the pipeline: dist.emulate_bandwidth_gbps (null = none),
dist.emulate_latency_ms. `bench.native_reference: true` also times plain DDP (no hook) at full speed, which
shows the cost of the hook itself.

    torchrun --standalone --nproc_per_node=2 -m timeprest.n5k.bench_comm_ddp --config n5k/configs/n5k_bench_comm_ddp.yaml

Writes <output.dir>/bench_comm_ddp.csv and bench_comm_ddp.json on rank 0; prints N5K_BENCH_DDP_JSON.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import shutil
import threading
import time

import torch
import torch.distributed as dist

from .. import utils
from ..config import _load_yaml, apply_overrides, resolve
from .data import load_datasets
from .ddp_train import N5kDDPTrainer, init_ddp


class EmulatedAllReduce:
    """DDP comm hook: mean all-reduce of each bucket behind an emulated link (see the module docstring)."""

    def __init__(self, bandwidth_gbps: float | None, latency_ms: float, world: int, device: torch.device):
        self.bytes_per_s = None if not bandwidth_gbps else bandwidth_gbps * 1e9 / 8
        self.latency_s = latency_ms / 1000.0
        self.world, self.device = world, device
        self.free_at = 0.0
        self.stats = {"buckets": 0, "bytes": 0, "link_busy_s": 0.0}
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def hold_s(self, nbytes: int) -> tuple[float, float]:
        """(link occupancy, extra latency) of one ring all-reduce of nbytes."""
        k = self.world - 1
        tx = 2 * k / self.world * nbytes / self.bytes_per_s if self.bytes_per_s else 0.0
        return tx, 2 * k * self.latency_s          # latency also at "inf", as the pipeline's EmulatedLink

    def reset_stats(self):
        self.stats = {"buckets": 0, "bytes": 0, "link_busy_s": 0.0}

    def submit(self, bucket) -> torch.futures.Future:
        buf = bucket.buffer()
        fut = torch.futures.Future()
        self.q.put((buf, fut, time.perf_counter()))
        return fut

    def _loop(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        while True:
            buf, fut, t_post = self.q.get()
            try:
                nbytes = buf.numel() * buf.element_size()
                tx, lat = self.hold_s(nbytes)
                start = max(t_post, self.free_at)
                self.free_at = start + tx
                delay = self.free_at + lat - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                buf.div_(self.world)
                dist.all_reduce(buf, async_op=True).wait()
                self.stats["buckets"] += 1
                self.stats["bytes"] += nbytes
                self.stats["link_busy_s"] += tx
                fut.set_result(buf)
            except Exception as e:  # noqa: BLE001  (surface the error in the training thread)
                fut.set_exception(e)


def emulated_allreduce_hook(state, bucket):
    # no annotations: DDP checks them, and `from __future__ import annotations` would turn them into strings
    return state.submit(bucket)


class N5kSlowNetDDPTrainer(N5kDDPTrainer):
    """`N5kDDPTrainer` with the emulated-network comm hook; logs the all-reduced volume per epoch."""

    def __init__(self, cfg: dict, out_dir: str, data=None, verbose: bool = True):
        super().__init__(cfg, out_dir, data=data, verbose=verbose)
        d = cfg.get("dist", {})
        self.link = EmulatedAllReduce(d.get("emulate_bandwidth_gbps"), float(d.get("emulate_latency_ms", 0.0)),
                                      self.world, self.device)
        self.ddp.register_comm_hook(self.link, emulated_allreduce_hook)

    def run_epoch(self, epoch: int) -> dict:
        self.link.reset_stats()
        row = super().run_epoch(epoch)
        s = self.link.stats
        row.update({"allreduce_buckets": s["buckets"], "allreduce_mb": round(s["bytes"] / 2**20, 1),
                    "link_busy_s": round(s["link_busy_s"], 3)})
        return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="n5k/configs/n5k_bench_comm_ddp.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    raw = apply_overrides(_load_yaml(args.config), args.set)
    bench = raw.pop("bench")
    base_cfg = resolve(copy.deepcopy(raw))
    rank, W, _ = init_ddp(base_cfg)
    base_dir = base_cfg["output"]["dir"]
    if rank == 0:
        os.makedirs(base_dir, exist_ok=True)
        print("env:", utils.env_info(), "| code", utils.code_hash(), flush=True)
    data = load_datasets(base_cfg["data"], print if rank == 0 else (lambda *_: None))
    rows = []
    variants = ([("native", None)] if bench.get("native_reference") else []) + [("ddp", bw) for bw in bench["bandwidths_gbps"]]
    for system, bw in variants:
        r = copy.deepcopy(raw)
        r.setdefault("dist", {})["emulate_bandwidth_gbps"] = bw
        r["dist"]["emulate_latency_ms"] = bench.get("latency_ms", 0.0)
        r.setdefault("training", {})["epochs"] = bench["epochs"]
        r["name"] = f"bench_{system}_bw{bw if bw else 'inf'}"
        cfg = resolve(r)
        out = os.path.join(base_dir, cfg["name"])
        if rank == 0:
            shutil.rmtree(out, ignore_errors=True)
        dist.barrier()
        utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
        cls = N5kDDPTrainer if system == "native" else N5kSlowNetDDPTrainer
        hist = cls(cfg, out, data, verbose=False).fit()
        last = hist[-1]                          # later epochs: no first-epoch warm-up effects
        if rank == 0:
            row = {"system": system, "bandwidth_gbps": bw,
                   "latency_ms": 0.0 if system == "native" else bench.get("latency_ms", 0.0),
                   "epoch_time_s": last["epoch_time_s"], "allreduce_mb": last.get("allreduce_mb"),
                   "allreduce_buckets": last.get("allreduce_buckets"), "link_busy_s": last.get("link_busy_s", 0.0),
                   "peak_mem_mb": last["peak_mem_mb"], "val_mae_kcal": last["val_mae_kcal"], "epochs_timed": len(hist)}
            rows.append(row)
            print(f"[bw {str(bw or 'inf'):>5} Gbit/s] {system:6s} epoch {row['epoch_time_s']:7.2f}s all-reduce "
                  f"{row['allreduce_mb']} MB in {row['allreduce_buckets']} buckets, link busy {row['link_busy_s']:.1f}s",
                  flush=True)
            utils.append_csv(row, os.path.join(base_dir, "bench_comm_ddp.csv"))
    if rank == 0:
        utils.save_json({"code_hash": utils.code_hash(), "config": args.config, "overrides": args.set,
                         "bench": bench, "env": utils.env_info(), "rows": rows},
                        os.path.join(base_dir, "bench_comm_ddp.json"))
        print("N5K_BENCH_DDP_JSON " + json.dumps(rows), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
