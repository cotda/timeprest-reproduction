"""Nutrition training with data parallelism (DDP baseline, n5k/PLAN.md mục 7, B5).

Every rank holds the whole model; each mini-batch of M samples is the SAME mini-batch as in the single-GPU
trainer (same shuffle of `timeprest.data.train_loader`), split into W contiguous shards of M/W samples.
Each rank backpropagates loss_sum(shard) / (M/W) and DDP averages the gradients, so one step = SGD on the
mean loss of the M samples, with the optimizer and lr schedule of the engine (`lr_lambda_factory`, one
scheduler step per mini-batch). Difference to the single GPU: BatchNorm statistics are computed per shard
(M/W samples, no SyncBN, as usual for DDP).
Validation on rank 0 every epoch, best checkpoint by val MAE kcal, test metrics of the best checkpoint;
same metrics.csv / summary.json layout as the other n5k trainers.

    torchrun --standalone --nproc_per_node=2 -m timeprest.n5k.ddp_train --config n5k/configs/n5k_ddp.yaml [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from .. import utils
from ..config import config_hash, load_config
from ..data import eval_loader, train_loader
from ..engine import lr_lambda_factory
from ..models import build_stages
from .common import SELECT_KEY, eval_cols, prepare, summary_line
from .train import checks_ok, write_summary


def init_ddp(cfg: dict):
    """Process group (nccl on GPUs, gloo on CPU; file init for the local Windows/CPU launcher)."""
    if not dist.is_initialized():
        use_cuda = cfg["runtime"]["device"] == "cuda" and torch.cuda.is_available()
        backend = "nccl" if use_cuda else "gloo"
        init_file = os.environ.get("TIMEPREST_INIT_FILE")
        if init_file:
            dist.init_process_group(backend, init_method="file:///" + init_file.replace("\\", "/").lstrip("/"),
                                    rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        else:
            from datetime import timedelta
            dist.init_process_group(backend, timeout=timedelta(seconds=cfg.get("dist", {}).get("timeout_s", 300)))
    rank, world = dist.get_rank(), dist.get_world_size()
    use_cuda = cfg["runtime"]["device"] == "cuda" and torch.cuda.is_available()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    return rank, world, device


class _Index(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def shard_batches(n: int, M: int, seed: int, epoch: int, rank: int, world: int) -> list[list[int]]:
    """Index lists of this rank's shards: the mini-batches of `train_loader(ds, M, seed, epoch)` (taken from
    that loader itself, so the order is exactly the single-GPU one) cut into `world` contiguous parts."""
    if M % world:
        raise ValueError(f"batch_size {M} is not divisible by the {world} ranks")
    m = M // world
    return [b[rank * m:(rank + 1) * m].tolist() for b in train_loader(_Index(n), M, seed, epoch, 0)]


class N5kDDPTrainer:
    def __init__(self, cfg: dict, out_dir: str, data=None, verbose: bool = True):
        self.raw_cfg, self.out_dir = cfg, out_dir
        self.rank, self.world, self.device = init_ddp(cfg)
        self.verbose = verbose and self.rank == 0
        self.cfg, (self.train_ds, self.val_ds, self.test_ds), self.meta, self.task = prepare(cfg, data, self.log)
        cfg = self.cfg
        torch.manual_seed(cfg["seed"])                       # same initial weights on every rank
        (model,), _ = build_stages(cfg["model"], 1, "auto", self.train_ds[0][0].unsqueeze(0))
        self.model = model.to(self.device)
        self.ddp = DDP(self.model, device_ids=[self.device.index] if self.device.type == "cuda" else None)
        tc = cfg["training"]
        self.M = tc["batch_size"]
        self.m = self.M // self.world
        self.steps = len(self.train_ds) // self.M
        if tc["optimizer"] != "sgd":
            raise ValueError("only sgd is implemented")
        self.opt = torch.optim.SGD(self.model.parameters(), lr=tc["lr"], momentum=tc["momentum"],
                                   weight_decay=tc["weight_decay"], nesterov=tc.get("nesterov", False))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda_factory(
            tc["epochs"] * self.steps, int(tc.get("warmup_epochs", 0) * self.steps), tc["lr_schedule"]))
        self.start_epoch, self.best = 0, None
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        self.best_path = os.path.join(out_dir, "best.pt")

    def log(self, msg: str):
        if getattr(self, "verbose", False):
            print(msg, flush=True)

    # ------------------------------------------------------------------ checkpoint
    def save(self, epoch_done: int):
        rngs = [None] * self.world
        dist.all_gather_object(rngs, utils.rng_state())
        if self.rank == 0:
            state = {"epoch": epoch_done, "model": self.model.state_dict(), "opt": self.opt.state_dict(),
                     "sched": self.sched.state_dict(), "rng": rngs, "best": self.best,
                     "config_hash": config_hash(self.raw_cfg), "code_hash": utils.code_hash()}
            torch.save(state, self.ckpt_path + ".tmp")
            os.replace(self.ckpt_path + ".tmp", self.ckpt_path)
        dist.barrier()

    def try_resume(self) -> bool:
        if not os.path.exists(self.ckpt_path):
            return False
        st = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if st["config_hash"] != config_hash(self.raw_cfg):
            raise RuntimeError("checkpoint was written with a different config")
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])
        utils.set_rng_state(st["rng"][self.rank])
        self.start_epoch, self.best = st["epoch"], st["best"]
        if self.rank == 0 and os.path.exists(self.csv_path):     # drop rows logged after the checkpoint
            rows = [r for r in utils.read_csv(self.csv_path) if int(r["epoch"]) <= self.start_epoch]
            os.remove(self.csv_path)
            for r in rows:
                utils.append_csv(r, self.csv_path)
        dist.barrier()
        self.log(f"[resume] from epoch {self.start_epoch}")
        return True

    # ------------------------------------------------------------------ train / eval
    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def _evaluate(self, ds) -> dict | None:
        """Metrics on rank 0 (None elsewhere); the other ranks wait at the barrier."""
        res = None
        if self.rank == 0:
            self.model.eval()
            st = self.task.eval_start()
            for x, y in eval_loader(ds, self.cfg["runtime"]["eval_batch_size"], self.cfg["data"]["num_workers"],
                                    self.device.type == "cuda"):
                self.task.eval_update(st, self.model(x.to(self.device)), y.to(self.device))
            self.model.train()
            res = self.task.eval_result(st)
        dist.barrier()
        return res

    def run_epoch(self, epoch: int) -> dict:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        lr = self.sched.get_last_lr()[0]
        batches = shard_batches(len(self.train_ds), self.M, self.cfg["seed"], epoch, self.rank, self.world)
        ld = DataLoader(self.train_ds, batch_sampler=batches, num_workers=self.cfg["data"]["num_workers"],
                        pin_memory=self.device.type == "cuda")
        self.model.train()
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        metric = self.task.train_metric_init(self.device)
        self._sync()
        dist.barrier()
        t0, data_t = time.perf_counter(), 0.0
        td = time.perf_counter()
        for x, y in ld:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            data_t += time.perf_counter() - td
            out = self.ddp(x)
            loss = self.task.loss_sum(out, y)
            self.opt.zero_grad(set_to_none=True)
            (loss / self.m).backward()                       # DDP averages over the ranks -> mean over M
            self.opt.step()
            self.sched.step()
            loss_sum += loss.detach().double()
            metric += self.task.train_metric(out.detach(), y)
            td = time.perf_counter()
        self._sync()
        dist.barrier()
        wall = time.perf_counter() - t0
        tot = torch.stack([loss_sum, metric.double().reshape(())]).to(self.device)
        dist.all_reduce(tot)
        n = self.steps * self.M
        peak = torch.cuda.max_memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else float("nan")
        peaks = [None] * self.world
        dist.all_gather_object(peaks, peak)
        ev = self._evaluate(self.val_ds)
        row = None
        if self.rank == 0:
            row = {"epoch": epoch + 1, "train_loss": round(tot[0].item() / n, 5),
                   **{k: round(v, 3) for k, v in self.task.train_summary(tot[1], n).items()},
                   **eval_cols(ev), "lr": lr, "epoch_time_s": round(wall, 3), "data_time_s": round(data_t, 3),
                   "peak_mem_mb": "|".join(f"{p:.1f}" for p in peaks), "samples": n}
        box = [row]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def fit(self, epochs: int | None = None, stop_after: int | None = None) -> list[dict]:
        epochs = epochs or self.cfg["training"]["epochs"]
        if self.rank == 0:
            os.makedirs(self.out_dir, exist_ok=True)
            utils.save_json(self.cfg, os.path.join(self.out_dir, "config.json"))
        self.log(f"[n5k ddp] W={self.world} M={self.M} ({self.m}/rank) K={self.task.K} device={self.device}")
        prev = utils.read_csv(self.csv_path) if (self.rank == 0 and os.path.exists(self.csv_path)) else []
        cum = float(prev[-1]["cum_time_s"]) if prev else 0.0
        rows, done = [], 0
        for epoch in range(self.start_epoch, epochs):
            row = self.run_epoch(epoch)                     # identical row on every rank (broadcast)
            cum += row["epoch_time_s"]
            row["cum_time_s"] = round(cum, 2)
            score = row["val_" + SELECT_KEY]
            if self.best is None or score < self.best["score"]:
                self.best = {"score": score, "epoch": row["epoch"]}
                if self.rank == 0:
                    torch.save({"model": self.model.state_dict(), "epoch": row["epoch"]}, self.best_path)
            if self.rank == 0:
                utils.append_csv(row, self.csv_path)
                self.log(f"ep {row['epoch']:>3}/{epochs} | train {row['train_loss']:.4f} (MAE kcal "
                         f"{row['train_mae_kcal']:.1f}) | " + summary_line(row) + f" | lr {row['lr']:.2e} | "
                         f"epoch {row['epoch_time_s']:.1f}s peak[{row['peak_mem_mb']}]MB | "
                         f"ETA {(epochs - epoch - 1) * row['epoch_time_s'] / 3600:.2f}h")
            if not math.isfinite(row["train_loss"]):        # same row on every rank: all stop together
                raise RuntimeError(f"training diverged (train loss {row['train_loss']}) at epoch {row['epoch']}")
            self.save(epoch + 1)
            rows.append(row)
            done += 1
            if stop_after is not None and done >= stop_after:
                break
        return rows

    def test_best(self) -> dict | None:
        """Test metrics of the best (validation) checkpoint; dict on rank 0, None elsewhere."""
        dist.barrier()
        st = torch.load(self.best_path, map_location="cpu", weights_only=False) if self.rank == 0 else None
        if self.rank == 0:
            self.model.load_state_dict(st["model"])
        ev = self._evaluate(self.test_ds)
        return {"best_epoch": st["epoch"], **eval_cols(ev, "test_")} if self.rank == 0 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    out_dir = cfg["output"]["dir"]
    rank0 = int(os.environ.get("RANK", 0)) == 0
    if cfg["output"].get("require_checks") and not args.force:
        ok, msg = checks_ok(cfg)
        if not ok:
            if rank0:
                print(f"[refuse] {msg} (use --force to override)")
            sys.exit(2)
    if os.path.exists(os.path.join(out_dir, "checkpoint.pt")) and not args.resume:
        if rank0:
            print(f"[refuse] {out_dir} has a checkpoint; pass --resume or change output.dir")
        sys.exit(2)
    utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    tr = N5kDDPTrainer(cfg, out_dir)
    if tr.rank == 0:
        print("env:", json.dumps(utils.env_info()), "| code", utils.code_hash(), "| ranks", tr.world, flush=True)
    if args.resume:
        tr.try_resume()
    rows = tr.fit(stop_after=args.stop_after)
    last_epoch = rows[-1]["epoch"] if rows else tr.start_epoch
    if last_epoch >= cfg["training"]["epochs"]:
        test = tr.test_best()
        if tr.rank == 0:
            write_summary(cfg, out_dir, test, {"trainer": "ddp", "ranks": tr.world})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
