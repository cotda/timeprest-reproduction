"""Build engine/data from a resolved config and run epochs with logging + checkpointing."""
from __future__ import annotations

import os
import time

import torch

from . import utils
from .config import config_hash
from .data import build_datasets, eval_loader, train_loader
from .engine import PipelineEngine
from .models import build_stages
from .schedule import format_grid, render_grid, version_difference_formula
from .timing import estimate_pipeline_time, summarize_durations


def build(cfg: dict, device: torch.device, datasets=None):
    """Return (engine, train_ds, test_ds, partition_bounds). Model init uses cfg.seed."""
    torch.manual_seed(cfg["seed"])
    if datasets is None:
        datasets = build_datasets(cfg["data"], cfg["model"]["num_classes"], cfg["seed"])
    train_ds, test_ds = datasets
    sample = train_ds[0][0].unsqueeze(0)
    pc = cfg["pipeline"]
    stages, bounds = build_stages(cfg["model"], pc["num_stages"], pc["partition"], sample)
    steps = _steps_per_epoch(cfg, len(train_ds))
    engine = PipelineEngine(stages, pc, cfg["training"], device, steps,
                            profile_ops=cfg["runtime"].get("profile_ops", False))
    return engine, train_ds, test_ds, bounds


def _steps_per_epoch(cfg: dict, n: int) -> int:
    M = cfg["training"]["batch_size"]
    return n // M if cfg["training"].get("drop_last", True) else -(-n // M)


def make_train_loader(cfg: dict, ds, epoch: int, device: torch.device):
    return train_loader(ds, cfg["training"]["batch_size"], cfg["seed"], epoch,
                        cfg["data"]["num_workers"], cfg["training"].get("drop_last", True),
                        pin_memory=device.type == "cuda")


def run_one_epoch(engine: PipelineEngine, cfg: dict, train_ds, test_ds, epoch: int,
                  device: torch.device, eval_ld=None) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    lr = engine.lrs()[0]
    stats = engine.run_epoch(make_train_loader(cfg, train_ds, epoch, device))
    ev = engine.evaluate(eval_ld or eval_loader(test_ds, cfg["runtime"]["eval_batch_size"],
                                                cfg["data"]["num_workers"], device.type == "cuda"))
    rt = cfg["runtime"]
    row = {
        "epoch": epoch + 1,
        "train_loss": round(stats["train_loss"], 5),
        "train_acc1": round(stats["train_acc1"], 3),
        "test_loss": round(ev["loss"], 5),
        "test_acc1": round(ev["acc1"], 3),
        "test_acc5": round(ev["acc5"], 3),
        "gap_loss": round(ev["loss"] - stats["train_loss"], 5),
        "lr": lr,
        "wall_time_s": round(stats["wall_time_s"], 3),
        "data_time_s": round(stats["data_time_s"], 3),
        "est_pipe_time_s": None,
        "est_pipe_time_comm_s": None,
        "peak_mem_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if device.type == "cuda" else None,
        "stage_mem_mb": "|".join(f"{b / 2**20:.1f}" for b in stats["stage_mem_accounted_bytes"]),
        "max_weight_versions": "|".join(str(v) for v in stats["max_weight_versions"]),
        "v_max": stats["v_max"],
        "samples": stats["samples"],
    }
    if stats["op_durations_s"] is not None:
        W, N = engine.W, engine.N
        row["est_pipe_time_s"] = round(estimate_pipeline_time(
            engine.last_ops, stats["op_durations_s"], stats["comm_bytes"], W, N), 3)
        row["est_pipe_time_comm_s"] = round(estimate_pipeline_time(
            engine.last_ops, stats["op_durations_s"], stats["comm_bytes"], W, N,
            rt["comm_bandwidth_gbps"] * 1e9, rt["comm_latency_ms"] / 1000.0), 3)
        row["_op_means"] = summarize_durations(engine.last_ops, stats["op_durations_s"])
    return row


def describe(cfg: dict, engine: PipelineEngine, bounds) -> str:
    pc = cfg["pipeline"]
    W, N = engine.W, engine.N
    lines = [
        f"system={cfg['system']} schedule={pc['schedule']} W={W} N={N} "
        f"vertical_sync={pc['vertical_sync']} backward_version={pc['backward_version']}",
        f"partition blocks={bounds} params/stage="
        + ",".join(f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M" for m in engine.stages),
    ]
    if W >= 2 and N >= 2:
        lines.append(f"paper formula v=floor((W+N-2)/N)={version_difference_formula(W, N)}")
    return "\n".join(lines)


def schedule_preview(engine: PipelineEngine, slots: int = 24) -> str:
    if not engine.last_ops:
        return ""
    return format_grid(render_grid(engine.last_ops, engine.W, slots))


class Trainer:
    """Full training loop with CSV logging, epoch checkpoints (after the pipeline drains) and resume."""

    def __init__(self, cfg: dict, device: torch.device, out_dir: str, datasets=None, verbose=True):
        self.cfg, self.device, self.out_dir, self.verbose = cfg, device, out_dir, verbose
        self.engine, self.train_ds, self.test_ds, self.bounds = build(cfg, device, datasets)
        self.start_epoch = 0
        self.ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.eval_ld = eval_loader(self.test_ds, cfg["runtime"]["eval_batch_size"],
                                   cfg["data"]["num_workers"], device.type == "cuda")

    def log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def save_checkpoint(self, epoch_done: int, rows: list):
        state = {"epoch": epoch_done, "engine": self.engine.state_dict(), "rng": utils.rng_state(),
                 "config_hash": config_hash(self.cfg), "code_hash": utils.code_hash()}
        tmp = self.ckpt_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)

    def try_resume(self) -> bool:
        if not os.path.exists(self.ckpt_path):
            return False
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if state.get("config_hash") != config_hash(self.cfg):
            raise RuntimeError("checkpoint was produced with a different config; "
                               "use a new output.dir or the same config")
        if state.get("code_hash") != utils.code_hash():
            self.log("[warn] code changed since the checkpoint was written")
        self.engine.load_state_dict(state["engine"])
        utils.set_rng_state(state["rng"])
        self.start_epoch = state["epoch"]
        # drop CSV rows written after the checkpoint (crash between log and checkpoint)
        if os.path.exists(self.csv_path):
            rows = [r for r in utils.read_csv(self.csv_path) if int(r["epoch"]) <= self.start_epoch]
            os.remove(self.csv_path)
            for r in rows:
                utils.append_csv(r, self.csv_path)
        self.log(f"[resume] from epoch {self.start_epoch} ({self.ckpt_path})")
        return True

    def history(self) -> list[dict]:
        return utils.read_csv(self.csv_path) if os.path.exists(self.csv_path) else []

    def fit(self, epochs: int | None = None, stop_after: int | None = None) -> list[dict]:
        cfg = self.cfg
        epochs = epochs or cfg["training"]["epochs"]
        os.makedirs(self.out_dir, exist_ok=True)
        utils.save_json(cfg, os.path.join(self.out_dir, "config.json"))
        self.log(describe(cfg, self.engine, self.bounds))
        rows = []
        prev = self.history()
        cum_wall = float(prev[-1]["cum_wall_s"]) if prev else 0.0
        cum_est = float(prev[-1]["cum_est_pipe_s"]) if prev and prev[-1].get("cum_est_pipe_s") else 0.0
        rising = 0
        last_test, last_train = (float(prev[-1]["test_loss"]), float(prev[-1]["train_loss"])) if prev else (None, None)
        done = 0
        for epoch in range(self.start_epoch, epochs):
            t0 = time.perf_counter()
            row = run_one_epoch(self.engine, cfg, self.train_ds, self.test_ds, epoch, self.device, self.eval_ld)
            op_means = row.pop("_op_means", None)
            cum_wall += row["wall_time_s"]
            cum_est += row["est_pipe_time_s"] or 0.0
            row["cum_wall_s"] = round(cum_wall, 2)
            row["cum_est_pipe_s"] = round(cum_est, 2)
            if last_test is not None and row["test_loss"] > last_test and row["train_loss"] < last_train:
                rising += 1
            else:
                rising = 0
            last_test, last_train = row["test_loss"], row["train_loss"]
            row["overfit_warn"] = int(rising >= 3)
            utils.append_csv(row, self.csv_path)
            self.save_checkpoint(epoch + 1, rows)
            if epoch == self.start_epoch:
                self.log("schedule (first slots):\n" + schedule_preview(self.engine))
                if op_means:
                    utils.save_json({"op_mean_s": op_means}, os.path.join(self.out_dir, "op_timing.json"))
                    self.log("mean op time (s): " + ", ".join(f"{k}={v:.4f}" for k, v in op_means.items()))
            elapsed = time.perf_counter() - t0
            remaining = (epochs - epoch - 1) * elapsed
            self.log(
                f"ep {row['epoch']:>3}/{epochs} | train {row['train_loss']:.4f} / {row['train_acc1']:.2f}% | "
                f"test {row['test_loss']:.4f} / top1 {row['test_acc1']:.2f}% top5 {row['test_acc5']:.2f}% | "
                f"gap {row['gap_loss']:+.3f} | lr {row['lr']:.4f} | 1gpu {row['wall_time_s']:.1f}s "
                f"est{self.engine.W}gpu {row['est_pipe_time_s']}s | peak {row['peak_mem_mb']}MB "
                f"stage[{row['stage_mem_mb']}]MB ver[{row['max_weight_versions']}] v={row['v_max']} | "
                f"ETA {remaining / 3600:.2f}h")
            if row["overfit_warn"]:
                self.log("[WARNING] test loss rose 3 epochs in a row while train loss fell (overfitting?)")
            if not (row["train_loss"] == row["train_loss"]) or row["train_loss"] > 1e4:
                raise RuntimeError("training diverged (NaN/Inf loss)")
            rows.append(row)
            done += 1
            if stop_after is not None and done >= stop_after:
                break
        return rows
