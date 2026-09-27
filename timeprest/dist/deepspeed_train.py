"""DeepSpeed pipeline baseline (synchronous 1F1B with a pipeline flush per mini-batch), paper_notes §26.

Same data order, model, partition, initial weights, optimizer and LR schedule as the TiMePReSt /
PipeDream runs (timeprest.dist.train); only the pipeline runtime is DeepSpeed's PipelineEngine:
each mini-batch of M samples is split into N micro-batches, all of them are forwarded and
backwarded with the same weights, gradients are accumulated and the weights are updated once
(no staleness, pipeline bubbles at every flush). No ZeRO / offload (the paper gives no DeepSpeed
configuration, §13).

    torchrun --standalone --nproc_per_node=2 -m timeprest.dist.deepspeed_train --config configs/tin_deepspeed.yaml --resume

Writes metrics.csv / summary.json / config.json in output.dir with the columns of the phase-2
runs, so timeprest.report and timeprest.plot read them. Evaluation uses whole mini-batches of the
test set (10 000 -> 9 984 images on Tiny-ImageNet).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import utils
from ..config import config_hash, load_config
from ..data import build_datasets, epoch_order
from ..engine import _upcast, lr_lambda_factory
from ..models import build_stages


def micro_batch_indices(n: int, seed: int, epoch: int, M: int, N: int) -> list[list[int]]:
    """Micro-batches of one epoch: the epoch order of the phase-2 runs (timeprest.data.epoch_order),
    cut into floor(n / M) mini-batches of M samples, each split into N micro-batches the way
    torch.tensor_split does (as the TiMePReSt runtime splits a mini-batch)."""
    order = epoch_order(n, seed, epoch)
    out = []
    for k in range(n // M):
        mb = order[k * M:(k + 1) * M]
        out += [c.tolist() for c in torch.tensor_split(mb, N)]
    return out


def eval_micro_batches(n: int, M: int, N: int) -> list[list[int]]:
    """Test set in order, whole mini-batches only (a DeepSpeed pipeline batch is always N micro-batches)."""
    idx = torch.arange((n // M) * M)
    return [c.tolist() for k in range(n // M) for c in torch.tensor_split(idx[k * M:(k + 1) * M], N)]


class LossMeter(nn.Module):
    """loss_fn of the last pipeline stage: mean cross-entropy of a micro-batch (DeepSpeed averages
    the N micro-batch losses = sum CE / M, the loss of the other runs) and running statistics."""

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.loss_sum, self.correct1, self.correct5, self.n = 0.0, 0, 0, 0

    def forward(self, out, y):
        out = _upcast(out)
        loss = F.cross_entropy(out, y)
        with torch.no_grad():
            top = out.topk(min(5, out.shape[1]), 1).indices
            self.loss_sum += loss.item() * y.shape[0]
            self.correct1 += (top[:, 0] == y).sum().item()
            self.correct5 += (top == y[:, None]).any(1).sum().item()
            self.n += y.shape[0]
        return loss

    def stats(self) -> dict:
        n = max(1, self.n)
        return {"loss": self.loss_sum / n, "acc1": self.correct1 / n * 100.0, "acc5": self.correct5 / n * 100.0, "n": self.n}


def epoch_row(epoch: int, lr: float, wall: float, train: dict, test: dict, peak_mb: list, cum: float) -> dict:
    """metrics.csv row with the columns of timeprest.dist.train (unknown per-op fields are nan)."""
    na = "|".join("nan" for _ in peak_mb)
    return {
        "epoch": epoch + 1, "train_loss": round(train["loss"], 5), "train_acc1": round(train["acc1"], 3),
        "test_loss": round(test["loss"], 5), "test_acc1": round(test["acc1"], 3), "test_acc5": round(test["acc5"], 3),
        "gap_loss": round(test["loss"] - train["loss"], 5), "lr": lr, "epoch_time_s": round(wall, 3),
        "busy_frac": na, "wait_time_s": na, "peak_mem_mb": "|".join(f"{m:.1f}" for m in peak_mb),
        "mb_sent": na, "msgs_sent": na, "recompute_micro": "|".join("0" for _ in peak_mb),
        "max_snapshots": "|".join("0" for _ in peak_mb), "bwd_overlap_minibatches": None,
        "samples": train["n"], "cum_time_s": round(cum, 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    import deepspeed
    from deepspeed.pipe import PipelineModule

    for k, v in (cfg.get("dist", {}).get("env") or {}).items():
        if v is not None:
            os.environ.setdefault(k, str(v))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    pc, tc, out_dir = cfg["pipeline"], cfg["training"], cfg["output"]["dir"]
    if world != pc["num_stages"]:
        raise ValueError(f"world size {world} != pipeline.num_stages {pc['num_stages']}")
    M, N = tc["batch_size"], pc["num_microbatches"]
    log = (lambda msg: print(msg, flush=True)) if rank == 0 else (lambda msg: None)
    log(f"env: {json.dumps(utils.env_info())} | code {utils.code_hash()} | deepspeed {deepspeed.__version__}")

    train_ds, test_ds = build_datasets(cfg["data"], cfg["model"]["num_classes"], cfg["seed"])
    sample = train_ds[0][0].unsqueeze(0)
    torch.manual_seed(cfg["seed"])                     # same initial weights as timeprest.dist.train
    stages, bounds = build_stages(cfg["model"], pc["num_stages"], pc["partition"], sample)
    meter = LossMeter()
    net = PipelineModule(layers=list(stages), num_stages=world, loss_fn=meter, partition_method="uniform",
                         activation_checkpoint_interval=0)
    K = len(train_ds) // M
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=tc["lr"], momentum=tc["momentum"],
                          weight_decay=tc["weight_decay"], nesterov=tc.get("nesterov", False))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(tc["epochs"] * K, int(tc.get("warmup_epochs", 0) * K), tc["lr_schedule"]))
    ds_cfg = {"train_batch_size": M, "train_micro_batch_size_per_gpu": M // N, "gradient_accumulation_steps": N,
              "steps_per_print": 10 ** 9, "wall_clock_breakdown": False, "fp16": {"enabled": False},
              "bf16": {"enabled": False}, "zero_optimization": {"stage": 0}}
    if M % N:
        raise ValueError("DeepSpeed needs equal micro-batches: batch_size must be divisible by num_microbatches")
    engine, _, _, _ = deepspeed.initialize(model=net, optimizer=opt, lr_scheduler=sched, config=ds_cfg)
    device = torch.device("cuda", local_rank)

    csv_path, meta_path = os.path.join(out_dir, "metrics.csv"), os.path.join(out_dir, "meta.json")
    start, cum = 0, 0.0
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        utils.save_json(cfg, os.path.join(out_dir, "config.json"))
    if args.resume and os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        if meta["config_hash"] != config_hash(cfg):
            raise RuntimeError("checkpoint was written with a different config")
        engine.load_checkpoint(out_dir, tag="last")
        start = meta["epoch"]
        if rank == 0 and os.path.exists(csv_path):
            rows = [r for r in utils.read_csv(csv_path) if int(r["epoch"]) <= start]
            os.remove(csv_path)
            for r in rows:
                utils.append_csv(r, csv_path)
            cum = float(rows[-1]["cum_time_s"]) if rows else 0.0
        log(f"[resume] from epoch {start}")
    elif rank == 0 and os.path.exists(meta_path):
        raise SystemExit(f"[refuse] {out_dir} has a checkpoint; pass --resume or change output.dir")
    log(f"[deepspeed] W={world} M={M} N={N} (synchronous 1F1B, flush per mini-batch) partition={bounds} "
        f"params/stage={[round(sum(p.numel() for p in s.parameters()) / 1e6, 2) for s in stages]} lr={tc['lr']}")

    nw = cfg["data"]["num_workers"]
    ev_batches = eval_micro_batches(len(test_ds), M, N)
    hist = []
    for epoch in range(start, tc["epochs"]):
        loader = DataLoader(train_ds, batch_sampler=micro_batch_indices(len(train_ds), cfg["seed"], epoch, M, N),
                            num_workers=nw, pin_memory=True)
        it = iter(loader)
        lr = opt.param_groups[0]["lr"]
        meter.reset()
        engine.train()
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(K):
            engine.train_batch(data_iter=it)
        torch.cuda.synchronize(device)
        dist.barrier()
        wall = time.perf_counter() - t0
        train_stats = meter.stats()
        meter.reset()
        ev_it = iter(DataLoader(test_ds, batch_sampler=ev_batches, num_workers=nw, pin_memory=True))
        for _ in range(len(ev_batches) // N):
            engine.eval_batch(data_iter=ev_it)
        test_stats = meter.stats()
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
        allst = [None] * world
        dist.all_gather_object(allst, {"train": train_stats, "test": test_stats, "peak": peak})
        last = allst[-1]                                 # the loss is computed on the last stage
        cum += wall
        row = epoch_row(epoch, lr, wall, last["train"], last["test"], [a["peak"] for a in allst], cum)
        hist.append(row)
        if rank == 0:
            utils.append_csv(row, csv_path)
            eta = (tc["epochs"] - epoch - 1) * wall / 3600
            log(f"ep {row['epoch']:>3}/{tc['epochs']} | train {row['train_loss']:.4f} / {row['train_acc1']:.2f}% | "
                f"test {row['test_loss']:.4f} / top1 {row['test_acc1']:.2f}% top5 {row['test_acc5']:.2f}% | "
                f"lr {lr:.2e} | epoch {wall:.1f}s peak[{row['peak_mem_mb']}]MB | ETA {eta:.2f}h")
            if not math.isfinite(row["train_loss"]):
                raise RuntimeError("training diverged")
        engine.save_checkpoint(out_dir, tag="last")
        dist.barrier()
        if rank == 0:
            utils.save_json({"epoch": epoch + 1, "config_hash": config_hash(cfg), "code_hash": utils.code_hash()},
                            meta_path)
        dist.barrier()
    if rank == 0:
        rows = utils.read_csv(csv_path)
        best = max(rows, key=lambda r: float(r["test_acc1"]))
        utils.save_json({"name": cfg["name"], "system": cfg["system"], "epochs_done": len(rows),
                         "best_test_acc1": float(best["test_acc1"]), "best_epoch": int(best["epoch"]),
                         "mean_epoch_time_s": sum(float(r["epoch_time_s"]) for r in rows) / len(rows),
                         "final": rows[-1], "env": utils.env_info(), "code_hash": utils.code_hash()},
                        os.path.join(out_dir, "summary.json"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
