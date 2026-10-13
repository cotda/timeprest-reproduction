"""Nutrition training on the real W-GPU pipeline (one process per stage), systems timeprest /
pipedream (n5k/PLAN.md mục 6–7). Same runtime as timeprest.dist.train with the nutrition task;
validation every epoch, best checkpoint by val MAE kcal, test metrics of the best checkpoint.

    torchrun --standalone --nproc_per_node=2 -m timeprest.n5k.dist_train --config n5k/configs/n5k_timeprest.yaml [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.distributed as dist

from .. import utils
from ..config import load_config
from ..data import eval_loader
from ..dist.train import DistTrainer
from .common import SELECT_KEY, eval_cols, prepare, summary_line
from .train import checks_ok, write_summary


def _rank0() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


class N5kDistTrainer(DistTrainer):
    def __init__(self, cfg: dict, out_dir: str, data=None, verbose: bool = True):
        built, (train, val, test), self.meta, self.task = prepare(
            cfg, data, log=print if (verbose and _rank0()) else (lambda *_: None))
        super().__init__(built, out_dir, datasets=(train, val), verbose=verbose)
        self.rt.task = self.task
        self.train_targets, self.test_targets = train.targets, val.targets   # float label vectors
        self.final_test_ds = test
        self.best = None
        self.best_path = os.path.join(out_dir, f"best_stage{self.rank}.pt")

    def try_resume(self) -> bool:
        ok = super().try_resume()
        if ok and os.path.exists(self.csv_path):
            rows = utils.read_csv(self.csv_path) if self.rank == 0 else []
            best = min(rows, key=lambda r: float(r["val_" + SELECT_KEY])) if rows else None
            b = [None]
            if self.rank == 0 and best:
                b = [{"score": float(best["val_" + SELECT_KEY]), "epoch": int(best["epoch"])}]
            dist.broadcast_object_list(b, src=0)
            self.best = b[0]
        return ok

    def _row(self, epoch, lr, wall, sts, ev) -> dict:
        last = sts[-1]
        overlap = None
        if all(s["ops"] for s in sts):
            b_last = {i: a for k, i, _, a, _ in sts[-1]["ops"] if k == "B"}
            b_first_end = {i: b for k, i, _, _, b in sts[0]["ops"] if k == "B"}
            overlap = sum(1 for i in b_last if i > 0 and b_last[i] < b_first_end.get(i - 1, -1))
        n = max(1, last["samples"])
        row = {
            "epoch": epoch + 1,
            "train_loss": round(last["loss_sum"] / n, 5),
            "train_mae_kcal": round(last["abs_err_kcal"] / n, 3),
            **eval_cols(ev),
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
        self.log(f"[n5k dist] W={self.world} system={self.cfg['system']} N={pc['num_microbatches']} M={self.M} "
                 f"backward={pc['backward_version']} rule={pc['backward_rule']} partition={self.bounds} "
                 f"K={self.task.K} params/stage=" + ",".join(f"{n / 1e6:.2f}M" for n in self.n_params)
                 + f" device={self.device}")
        prev = utils.read_csv(self.csv_path) if (self.rank == 0 and os.path.exists(self.csv_path)) else []
        cum = float(prev[-1]["cum_time_s"]) if prev else 0.0
        rows, done = [], 0
        for epoch in range(self.start_epoch, epochs):
            row = self.run_epoch(epoch)           # identical row on every rank (all_gather)
            cum += row["epoch_time_s"]
            row["cum_time_s"] = round(cum, 2)
            score = row["val_" + SELECT_KEY]
            if self.best is None or score < self.best["score"]:
                self.best = {"score": score, "epoch": row["epoch"]}
                torch.save({"module": self.rt.module.state_dict(), "epoch": row["epoch"]}, self.best_path)
            if self.rank == 0:
                utils.append_csv(row, self.csv_path)
                every = self.cfg["runtime"].get("op_trace_every")
                if epoch == self.start_epoch or (every and (epoch + 1) % every == 0):
                    utils.save_json({"ops_per_stage": self._last_ops},
                                    os.path.join(self.out_dir, f"op_trace_epoch{epoch + 1}.json"))
                eta = (epochs - epoch - 1) * row["epoch_time_s"] / 3600
                self.log(f"ep {row['epoch']:>3}/{epochs} | train {row['train_loss']:.4f} (MAE kcal "
                         f"{row['train_mae_kcal']:.1f}) | " + summary_line(row) + f" | lr {row['lr']:.2e} | "
                         f"epoch {row['epoch_time_s']:.1f}s busy[{row['busy_frac']}] peak[{row['peak_mem_mb']}]MB "
                         f"sent[{row['mb_sent']}]MB bwd_overlap={row['bwd_overlap_minibatches']} | ETA {eta:.2f}h")
            if not math.isfinite(row["train_loss"]):   # same row on every rank: all stop together
                raise RuntimeError(f"training diverged (train loss {row['train_loss']}) at epoch {row['epoch']}")
            self.save_checkpoint(epoch + 1)
            rows.append(row)
            done += 1
            if stop_after is not None and done >= stop_after:
                break
        return rows

    def test_best(self) -> dict | None:
        """Test metrics of the best (validation) checkpoint; dict on the last rank, None elsewhere."""
        st = torch.load(self.best_path, map_location="cpu", weights_only=False)
        self.rt.module.load_state_dict(st["module"])
        bs = self.cfg["runtime"]["eval_batch_size"]
        n = len(self.final_test_ds)
        nb = math.ceil(n / bs)
        xs = labels = None
        if self.rank == 0:
            xs = (x for x, _ in eval_loader(self.final_test_ds, bs, self.cfg["data"]["num_workers"],
                                            self.device.type == "cuda"))
        if self.rank == self.world - 1:
            labels = [self.final_test_ds.targets[b * bs:(b + 1) * bs] for b in range(nb)]
        ev = self.rt.evaluate(xs, labels, nb)
        res = [None] * self.world
        dist.all_gather_object(res, None if ev is None else {"best_epoch": st["epoch"], **eval_cols(ev, "test_")})
        return res[-1]


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
    if cfg["output"].get("require_checks") and not args.force:
        ok, msg = checks_ok(cfg)
        if not ok:
            if _rank0():
                print(f"[refuse] {msg} (use --force to override)")
            sys.exit(2)
    if os.path.exists(os.path.join(out_dir, "meta.json")) and not args.resume:
        if _rank0():
            print(f"[refuse] {out_dir} has a checkpoint; pass --resume or change output.dir")
        sys.exit(2)
    utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    tr = N5kDistTrainer(cfg, out_dir)
    if tr.rank == 0:
        print("env:", json.dumps(utils.env_info()), "| code", utils.code_hash(), "| gpus", torch.cuda.device_count(),
              "| p2p", cfg["dist"].get("p2p_backend", "gloo"), flush=True)
    if args.resume:
        tr.try_resume()
    rows = tr.fit(stop_after=args.stop_after)
    last_epoch = rows[-1]["epoch"] if rows else tr.start_epoch
    if last_epoch >= cfg["training"]["epochs"]:
        test = tr.test_best()
        if tr.rank == 0:
            write_summary(cfg, out_dir, test)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
