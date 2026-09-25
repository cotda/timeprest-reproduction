"""Phase-2 training on W real processes/GPUs (one stage each).

    torchrun --standalone --nproc_per_node=2 -m timeprest.dist.train --config configs/kaggle_timeprest.yaml [--resume]
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
from torch.utils.data import DataLoader

from .. import utils
from ..config import config_hash, load_config
from ..data import build_datasets, dataset_targets, epoch_order, eval_loader
from ..models import build_stages
from .runtime import Channels, StageRuntime


def init_dist(cfg: dict):
    # `kill -USR1 <pid>` (or `pkill -USR1 -f timeprest.dist`) prints every thread's stack: hang diagnosis
    import faulthandler
    import signal
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    if os.environ.get("TIMEPREST_STACK_DUMP_S"):   # one-shot stack dump of every thread after N seconds
        faulthandler.dump_traceback_later(float(os.environ["TIMEPREST_STACK_DUMP_S"]), exit=False)
    for k, v in (cfg.get("dist", {}).get("env") or {}).items():   # e.g. NCCL_P2P_DISABLE: "1"; null = unset
        if v is not None:
            os.environ.setdefault(k, str(v))
    if not dist.is_initialized():
        want_cuda = cfg["runtime"]["device"] == "cuda" and torch.cuda.is_available()
        backend = cfg.get("dist", {}).get("backend") or ("nccl" if want_cuda else "gloo")
        if want_cuda:
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        init_file = os.environ.get("TIMEPREST_INIT_FILE")   # local launcher (Windows/CPU tests)
        if init_file:
            dist.init_process_group(backend, init_method="file:///" + init_file.replace("\\", "/").lstrip("/"),
                                    rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        else:
            from datetime import timedelta
            kw = {"timeout": timedelta(seconds=cfg.get("dist", {}).get("timeout_s", 300))}
            if backend == "nccl":
                # eager NCCL init: communicators (also of later new_group calls) are created up front
                # instead of lazily at the first send/recv
                kw["device_id"] = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
            try:
                dist.init_process_group(backend, **kw)
            except TypeError:   # older torch without device_id
                kw.pop("device_id", None)
                dist.init_process_group(backend, **kw)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != cfg["pipeline"]["num_stages"]:
        raise ValueError(f"world size {world} != pipeline.num_stages {cfg['pipeline']['num_stages']}")
    use_cuda = cfg["runtime"]["device"] == "cuda" and torch.cuda.is_available()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}") if use_cuda else torch.device("cpu")
    return rank, world, device


def channels_from_cfg(cfg: dict, rank: int, world: int) -> Channels:
    d = cfg.get("dist", {})
    return Channels(rank, world, d.get("p2p_backend", "gloo"),
                    emulate_bandwidth_gbps=d.get("emulate_bandwidth_gbps"),
                    emulate_latency_ms=d.get("emulate_latency_ms") or 0.0)


def boundary_features(stages, sample) -> list[tuple]:
    """Per-sample feature shape leaving every stage (computed on CPU with one sample)."""
    feats, x = [], sample
    with torch.no_grad():
        for m in stages:
            was = m.training
            m.eval()
            x = m(x)
            m.train(was)
            feats.append(tuple(x.shape[1:]))
    return feats


class DistTrainer:
    def __init__(self, cfg: dict, out_dir: str, datasets=None, verbose: bool = True):
        self.cfg, self.out_dir, self.verbose = cfg, out_dir, verbose
        self.rank, self.world, self.device = init_dist(cfg)
        self.channels = channels_from_cfg(cfg, self.rank, self.world)
        torch.manual_seed(cfg["seed"])
        if datasets is None:
            datasets = build_datasets(cfg["data"], cfg["model"]["num_classes"], cfg["seed"])
        self.train_ds, self.test_ds = datasets
        sample = self.train_ds[0][0].unsqueeze(0)
        pc = cfg["pipeline"]
        torch.manual_seed(cfg["seed"])
        stages, self.bounds = build_stages(cfg["model"], pc["num_stages"], pc["partition"], sample)
        feats = boundary_features(stages, sample.float())
        self.M = cfg["training"]["batch_size"]
        self.steps = len(self.train_ds) // self.M
        s = self.rank
        self.rt = StageRuntime(s, self.world, stages[s], pc, cfg["training"], self.device, self.steps,
                               in_feat=feats[s - 1] if s > 0 else tuple(sample.shape[1:]),
                               out_feat=feats[s] if s < self.world - 1 else None,
                               channels=self.channels, batch_size=self.M)
        self.rt.profile_ops = cfg["runtime"].get("profile_ops", True)
        self.n_params = [sum(p.numel() for p in m.parameters()) for m in stages]
        self.train_targets = dataset_targets(self.train_ds)
        self.test_targets = dataset_targets(self.test_ds)
        self.start_epoch = 0
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.meta_path = os.path.join(out_dir, "meta.json")

    def log(self, msg: str):
        if self.verbose and self.rank == 0:
            print(msg, flush=True)

    # ---------------------------------------------------------------- data per rank
    def _train_inputs(self, epoch: int):
        order = epoch_order(len(self.train_ds), self.cfg["seed"], epoch)
        K = self.steps
        batches = [order[k * self.M:(k + 1) * self.M].tolist() for k in range(K)]
        xs = labels = None
        if self.rank == 0:
            g = torch.Generator().manual_seed(self.cfg["seed"] * 7919 + epoch)
            ld = DataLoader(self.train_ds, batch_sampler=batches, num_workers=self.cfg["data"]["num_workers"],
                            pin_memory=self.device.type == "cuda", generator=g)
            xs = (x for x, _ in ld)
        if self.rank == self.world - 1:
            labels = [self.train_targets[torch.as_tensor(b)] for b in batches]
        return K, xs, labels

    def _eval_inputs(self):
        bs = self.cfg["runtime"]["eval_batch_size"]
        n = len(self.test_ds)
        nb = math.ceil(n / bs)
        xs = labels = None
        if self.rank == 0:
            xs = (x for x, _ in eval_loader(self.test_ds, bs, self.cfg["data"]["num_workers"],
                                            self.device.type == "cuda"))
        if self.rank == self.world - 1:
            labels = [self.test_targets[b * bs:(b + 1) * bs] for b in range(nb)]
        return nb, xs, labels

    # ---------------------------------------------------------------- checkpoint
    def save_checkpoint(self, epoch_done: int):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"stage{self.rank}.pt")
        torch.save({"epoch": epoch_done, "runtime": self.rt.state_dict(), "rng": utils.rng_state()}, path + ".tmp")
        os.replace(path + ".tmp", path)
        dist.barrier()
        if self.rank == 0:
            utils.save_json({"epoch": epoch_done, "config_hash": config_hash(self.cfg),
                             "code_hash": utils.code_hash()}, self.meta_path)
        dist.barrier()

    def try_resume(self) -> bool:
        if not os.path.exists(self.meta_path):
            return False
        meta = json.load(open(self.meta_path, encoding="utf-8"))
        if meta["config_hash"] != config_hash(self.cfg):
            raise RuntimeError("checkpoint was written with a different config")
        st = torch.load(os.path.join(self.out_dir, f"stage{self.rank}.pt"), map_location="cpu", weights_only=False)
        if st["epoch"] != meta["epoch"]:
            raise RuntimeError("stage checkpoint and meta.json disagree")
        self.rt.load_state_dict(st["runtime"])
        utils.set_rng_state(st["rng"])
        self.start_epoch = meta["epoch"]
        if self.rank == 0 and os.path.exists(self.csv_path):
            rows = [r for r in utils.read_csv(self.csv_path) if int(r["epoch"]) <= self.start_epoch]
            os.remove(self.csv_path)
            for r in rows:
                utils.append_csv(r, self.csv_path)
        self.log(f"[resume] from epoch {self.start_epoch}")
        return True

    # ---------------------------------------------------------------- epoch
    def run_epoch(self, epoch: int) -> dict:
        K, xs, labels = self._train_inputs(epoch)
        self.rt.warm_up_kernels()   # once; kept out of the epoch's time and peak memory
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        lr = self.rt.opt.param_groups[0]["lr"]
        dist.barrier()
        t0 = time.perf_counter()
        st = self.rt.run_epoch(K, xs, labels)
        dist.barrier()
        wall = time.perf_counter() - t0
        nb, exs, elabels = self._eval_inputs()
        ev = self.rt.evaluate(exs, elabels, nb)
        dist.barrier()
        allst = [None] * self.world
        dist.all_gather_object(allst, {"st": st, "ev": ev})
        return self._row(epoch, lr, wall, [a["st"] for a in allst], allst[-1]["ev"])

    def _row(self, epoch, lr, wall, sts, ev) -> dict:
        last = sts[-1]
        # measured version difference: backward of mini-batch i starts on the last stage before
        # stage 0 finished the backward of i-1  -> the two backward passes overlap (v > 1)
        overlap = None
        if all(s["ops"] for s in sts):
            b_last = {i: a for k, i, _, a, _ in sts[-1]["ops"] if k == "B"}
            b_first_end = {i: b for k, i, _, _, b in sts[0]["ops"] if k == "B"}
            overlap = sum(1 for i in b_last if i > 0 and b_last[i] < b_first_end.get(i - 1, -1))
        row = {
            "epoch": epoch + 1,
            "train_loss": round(last["loss_sum"] / max(1, last["samples"]), 5),
            "train_acc1": round(last["correct"] / max(1, last["samples"]) * 100.0, 3),
            "test_loss": round(ev["loss"], 5), "test_acc1": round(ev["acc1"], 3), "test_acc5": round(ev["acc5"], 3),
            "gap_loss": round(ev["loss"] - last["loss_sum"] / max(1, last["samples"]), 5),
            "lr": lr,
            "epoch_time_s": round(wall, 3),
            "busy_frac": "|".join(f"{(s['busy_s'] or 0) / wall:.3f}" for s in sts),
            "wait_time_s": "|".join(f"{s['wait_time_s']:.2f}" for s in sts),
            "peak_mem_mb": "|".join(f"{s['peak_mem_mb']:.1f}" if s["peak_mem_mb"] is not None else "nan" for s in sts),
            "mb_sent": "|".join(f"{s['bytes_sent'] / 2**20:.1f}" for s in sts),
            "msgs_sent": "|".join(str(s["msgs_sent"]) for s in sts),
            "recompute_micro": "|".join(str(s["recompute_micro"]) for s in sts),
            "max_snapshots": "|".join(str(s["max_snaps"]) for s in sts),
            "bwd_overlap_minibatches": overlap,
            "samples": last["samples"],
        }
        self._last_ops = [s["ops"] for s in sts]
        return row

    def fit(self, epochs: int | None = None, stop_after: int | None = None) -> list[dict]:
        epochs = epochs or self.cfg["training"]["epochs"]
        if self.rank == 0:
            os.makedirs(self.out_dir, exist_ok=True)
            utils.save_json(self.cfg, os.path.join(self.out_dir, "config.json"))
        pc = self.cfg["pipeline"]
        self.log(f"[dist] W={self.world} system={self.cfg['system']} schedule={pc['schedule']} N={pc['num_microbatches']} "
                 f"order={pc['order']} vertical_sync={pc['vertical_sync']} backward={pc['backward_version']} "
                 f"backward_rule={pc['backward_rule']} partition={self.bounds} params/stage="
                 + ",".join(f"{n / 1e6:.2f}M" for n in self.n_params) + f" device={self.device}")
        prev = utils.read_csv(self.csv_path) if (self.rank == 0 and os.path.exists(self.csv_path)) else []
        cum = float(prev[-1]["cum_time_s"]) if prev else 0.0
        rows, done = [], 0
        for epoch in range(self.start_epoch, epochs):
            row = self.run_epoch(epoch)
            cum += row["epoch_time_s"]
            row["cum_time_s"] = round(cum, 2)
            if self.rank == 0:
                utils.append_csv(row, self.csv_path)
                if epoch == self.start_epoch:
                    utils.save_json({"ops_per_stage": self._last_ops},
                                    os.path.join(self.out_dir, f"op_trace_epoch{epoch + 1}.json"))
                eta = (epochs - epoch - 1) * row["epoch_time_s"] / 3600
                self.log(f"ep {row['epoch']:>3}/{epochs} | train {row['train_loss']:.4f} / {row['train_acc1']:.2f}% | "
                         f"test {row['test_loss']:.4f} / top1 {row['test_acc1']:.2f}% top5 {row['test_acc5']:.2f}% | "
                         f"lr {row['lr']:.2e} | epoch {row['epoch_time_s']:.1f}s busy[{row['busy_frac']}] "
                         f"peak[{row['peak_mem_mb']}]MB sent[{row['mb_sent']}]MB recompute[{row['recompute_micro']}] "
                         f"bwd_overlap={row['bwd_overlap_minibatches']} | ETA {eta:.2f}h")
                if not math.isfinite(row["train_loss"]):
                    raise RuntimeError("training diverged")
            self.save_checkpoint(epoch + 1)
            rows.append(row)
            done += 1
            if stop_after is not None and done >= stop_after:
                break
        return rows


def checks_ok(cfg: dict) -> tuple[bool, str]:
    path = cfg["output"]["checks_file"]
    saved = os.path.join(cfg["output"]["dir"], "checks.json")
    if not os.path.exists(path) and os.path.exists(saved):
        path = saved
    if not os.path.exists(path):
        return False, f"{path} not found - run the phase-2 checks first"
    res = json.load(open(path, encoding="utf-8"))
    if not res.get("all_pass"):
        return False, "phase-2 checks did not all PASS"
    if res.get("code_hash") != utils.code_hash():
        return False, "checks were run on a different code version"
    return True, "ok"


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
    rank = int(os.environ.get("RANK", 0))
    if cfg["output"].get("require_checks") and not args.force:
        ok, msg = checks_ok(cfg)
        if not ok:
            if rank == 0:
                print(f"[refuse] {msg} (use --force to override)")
            sys.exit(2)
    if os.path.exists(os.path.join(out_dir, "meta.json")) and not args.resume:
        if rank == 0:
            print(f"[refuse] {out_dir} has a checkpoint; pass --resume or change output.dir")
        sys.exit(2)
    utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    tr = DistTrainer(cfg, out_dir)
    if rank == 0:
        print("env:", json.dumps(utils.env_info()), "| code", utils.code_hash(),
              "| gpus", torch.cuda.device_count(),
              "| p2p", cfg["dist"].get("p2p_backend", "gloo"), flush=True)
        if cfg["output"].get("require_checks") and os.path.exists(cfg["output"]["checks_file"]):
            import shutil
            os.makedirs(out_dir, exist_ok=True)
            shutil.copyfile(cfg["output"]["checks_file"], os.path.join(out_dir, "checks.json"))
    if args.resume:
        tr.try_resume()
    tr.fit(stop_after=args.stop_after)
    if rank == 0:
        hist = utils.read_csv(tr.csv_path)
        best = max(hist, key=lambda r: float(r["test_acc1"]))
        summary = {"name": cfg["name"], "system": cfg["system"], "epochs_done": int(hist[-1]["epoch"]),
                   "best_test_acc1": float(best["test_acc1"]), "best_epoch": int(best["epoch"]),
                   "mean_epoch_time_s": sum(float(r["epoch_time_s"]) for r in hist[1:] or hist) / max(1, len(hist[1:] or hist)),
                   "final": hist[-1], "env": utils.env_info(), "code_hash": utils.code_hash()}
        utils.save_json(summary, os.path.join(out_dir, "summary.json"))
        print("SUMMARY", json.dumps({k: summary[k] for k in ("name", "epochs_done", "best_test_acc1",
                                                              "best_epoch", "mean_epoch_time_s")}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
