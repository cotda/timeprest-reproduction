"""Nutrition training on ONE process with the pipeline engine (n5k/PLAN.md mục 6–7):
  pipeline.num_stages: 1  -> ordinary single-GPU training (the "1 GPU" reference)
  pipeline.num_stages: 2  -> W-stage pipeline simulated on one GPU (exact versions; used for the lr sweep)
Validation every epoch, best checkpoint by val MAE kcal, test metrics of the best checkpoint at the end.

    python -m timeprest.n5k.train --config n5k/configs/n5k_single.yaml [--resume] [--set training.lr=0.005]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

from .. import utils
from ..config import config_hash, load_config
from ..data import eval_loader, train_loader
from ..engine import PipelineEngine
from ..models import build_stages
from .common import SELECT_KEY, eval_cols, prepare, summary_line


class EngineTrainer:
    def __init__(self, cfg: dict, device, out_dir: str, data=None, verbose: bool = True):
        self.raw_cfg, self.device, self.out_dir, self.verbose = cfg, torch.device(device), out_dir, verbose
        self.cfg, (self.train_ds, self.val_ds, self.test_ds), self.meta, self.task = prepare(cfg, data, self.log)
        cfg = self.cfg
        torch.manual_seed(cfg["seed"])
        sample = self.train_ds[0][0].unsqueeze(0)
        pc = cfg["pipeline"]
        stages, self.bounds = build_stages(cfg["model"], pc["num_stages"], pc["partition"], sample)
        self.M = cfg["training"]["batch_size"]
        self.steps = len(self.train_ds) // self.M
        self.engine = PipelineEngine(stages, pc, cfg["training"], self.device, self.steps,
                                     profile_ops=cfg["runtime"].get("profile_ops", False), task=self.task)
        nw, pin = cfg["data"]["num_workers"], self.device.type == "cuda"
        bs = cfg["runtime"]["eval_batch_size"]
        self.val_ld = eval_loader(self.val_ds, bs, nw, pin)
        self.test_ld = eval_loader(self.test_ds, bs, nw, pin)
        self.start_epoch, self.best = 0, None
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        self.best_path = os.path.join(out_dir, "best.pt")

    def log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def save(self, epoch_done: int):
        state = {"epoch": epoch_done, "engine": self.engine.state_dict(), "rng": utils.rng_state(), "best": self.best,
                 "config_hash": config_hash(self.raw_cfg), "code_hash": utils.code_hash()}
        torch.save(state, self.ckpt_path + ".tmp")
        os.replace(self.ckpt_path + ".tmp", self.ckpt_path)

    def try_resume(self) -> bool:
        if not os.path.exists(self.ckpt_path):
            return False
        st = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if st["config_hash"] != config_hash(self.raw_cfg):
            raise RuntimeError("checkpoint was written with a different config")
        self.engine.load_state_dict(st["engine"])
        utils.set_rng_state(st["rng"])
        self.start_epoch, self.best = st["epoch"], st["best"]
        if os.path.exists(self.csv_path):     # drop rows logged after the checkpoint
            rows = [r for r in utils.read_csv(self.csv_path) if int(r["epoch"]) <= self.start_epoch]
            os.remove(self.csv_path)
            for r in rows:
                utils.append_csv(r, self.csv_path)
        self.log(f"[resume] from epoch {self.start_epoch}")
        return True

    def run_epoch(self, epoch: int) -> dict:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        lr = self.engine.lrs()[0]
        ld = train_loader(self.train_ds, self.M, self.cfg["seed"], epoch, self.cfg["data"]["num_workers"],
                          True, self.device.type == "cuda")
        st = self.engine.run_epoch(ld)
        ev = self.engine.evaluate(self.val_ld)
        return {"epoch": epoch + 1, "train_loss": round(st["train_loss"], 5),
                "train_mae_kcal": round(st["train_mae_kcal"], 3), **eval_cols(ev), "lr": lr,
                "wall_time_s": round(st["wall_time_s"], 3), "data_time_s": round(st["data_time_s"], 3),
                "peak_mem_mb": round(torch.cuda.max_memory_allocated(self.device) / 2**20, 1)
                if self.device.type == "cuda" else None,
                "max_weight_versions": "|".join(str(v) for v in st["max_weight_versions"]),
                "samples": st["samples"]}

    def fit(self, epochs: int | None = None, stop_after: int | None = None) -> list[dict]:
        epochs = epochs or self.cfg["training"]["epochs"]
        os.makedirs(self.out_dir, exist_ok=True)
        utils.save_json(self.cfg, os.path.join(self.out_dir, "config.json"))
        pc = self.cfg["pipeline"]
        self.log(f"[n5k] system={self.cfg['system']} W={pc['num_stages']} N={pc['num_microbatches']} "
                 f"M={self.M} partition={self.bounds} K={self.task.K} device={self.device}")
        prev = utils.read_csv(self.csv_path) if os.path.exists(self.csv_path) else []
        cum = float(prev[-1]["cum_wall_s"]) if prev else 0.0
        rows, done = [], 0
        for epoch in range(self.start_epoch, epochs):
            t0 = time.perf_counter()
            row = self.run_epoch(epoch)
            cum += row["wall_time_s"]
            row["cum_wall_s"] = round(cum, 2)
            score = row["val_" + SELECT_KEY]
            if self.best is None or score < self.best["score"]:
                self.best = {"score": score, "epoch": row["epoch"]}
                torch.save({"engine": self.engine.state_dict(), "epoch": row["epoch"]}, self.best_path)
            utils.append_csv(row, self.csv_path)
            self.save(epoch + 1)
            self.log(f"ep {row['epoch']:>3}/{epochs} | train {row['train_loss']:.4f} (MAE kcal {row['train_mae_kcal']:.1f}) | "
                     + summary_line(row) + f" | lr {row['lr']:.2e} | {row['wall_time_s']:.1f}s peak {row['peak_mem_mb']}MB | "
                     f"ETA {(epochs - epoch - 1) * (time.perf_counter() - t0) / 3600:.2f}h")
            if not math.isfinite(row["train_loss"]):
                raise RuntimeError("training diverged (NaN/Inf loss)")
            rows.append(row)
            done += 1
            if stop_after is not None and done >= stop_after:
                break
        return rows

    def test_best(self) -> dict:
        """Test metrics of the best (validation) checkpoint."""
        st = torch.load(self.best_path, map_location="cpu", weights_only=False)
        for m, d in zip(self.engine.stages, st["engine"]["stages"]):
            m.load_state_dict(d)
        ev = self.engine.evaluate(self.test_ld)
        return {"best_epoch": st["epoch"], **eval_cols(ev, "test_")}


def checks_ok(cfg: dict) -> tuple[bool, str]:
    path = cfg["output"]["checks_file"]
    if not os.path.exists(path):
        return False, f"{path} not found - run python -m timeprest.n5k.checks first"
    res = json.load(open(path, encoding="utf-8"))
    if not res.get("all_pass"):
        return False, "n5k checks did not all PASS"
    if res.get("code_hash") != utils.code_hash():
        return False, "checks were run on a different code version"
    return True, "ok"


def write_summary(cfg: dict, out_dir: str, test: dict, extra: dict | None = None) -> dict:
    hist = utils.read_csv(os.path.join(out_dir, "metrics.csv"))
    times = [float(r.get("epoch_time_s") or r.get("wall_time_s")) for r in hist]
    summary = {"name": cfg["name"], "system": cfg["system"], "epochs_done": int(hist[-1]["epoch"]),
               **test, "mean_epoch_time_s": sum(times[1:] or times) / max(1, len(times[1:] or times)),
               "final": hist[-1], "env": utils.env_info(), "code_hash": utils.code_hash(), **(extra or {})}
    utils.save_json(summary, os.path.join(out_dir, "summary.json"))
    print("SUMMARY", json.dumps({k: summary[k] for k in ("name", "system", "epochs_done", "best_epoch",
                                                          "test_mae_kcal", "test_pmae_kcal", "test_ingr_f1",
                                                          "mean_epoch_time_s")}), flush=True)
    return summary


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
            print(f"[refuse] {msg} (use --force to override)")
            sys.exit(2)
    if os.path.exists(os.path.join(out_dir, "checkpoint.pt")) and not args.resume:
        print(f"[refuse] {out_dir} has a checkpoint; pass --resume or change output.dir")
        sys.exit(2)
    utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    device = utils.resolve_device(cfg["runtime"]["device"])
    print("env:", json.dumps(utils.env_info()), "| code", utils.code_hash(), flush=True)
    tr = EngineTrainer(cfg, device, out_dir)
    if args.resume:
        tr.try_resume()
    tr.fit(stop_after=args.stop_after)
    if int(utils.read_csv(tr.csv_path)[-1]["epoch"]) >= cfg["training"]["epochs"]:
        write_summary(cfg, out_dir, tr.test_best())


if __name__ == "__main__":
    main()
