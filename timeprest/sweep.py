"""Short learning-rate sweep over several systems (settings in configs/sweep_lr.yaml).

    python -m timeprest.sweep --config configs/sweep_lr.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import shutil

from . import utils
from .config import _load_yaml, apply_overrides, resolve
from .runner import Trainer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/sweep_lr.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    raw = apply_overrides(_load_yaml(args.config), args.set)
    sweep = raw.pop("sweep")
    base_dir = raw.get("output", {}).get("dir", "results/sweep")
    device = utils.resolve_device(raw.get("runtime", {}).get("device", "cuda"))
    print("env:", utils.env_info(), "| code", utils.code_hash())
    datasets, rows = None, []
    for lr in sweep["lrs"]:
        for name, over in sweep["variants"].items():
            r = apply_overrides(_deep_copy(raw), [f"{k}={v}" for k, v in over.items()] + [f"training.lr={lr}"])
            r["name"] = f"{name}_lr{lr}"
            r.setdefault("output", {})["dir"] = os.path.join(base_dir, r["name"])
            cfg = resolve(r)
            shutil.rmtree(cfg["output"]["dir"], ignore_errors=True)
            utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
            tr = Trainer(cfg, device, cfg["output"]["dir"], datasets, verbose=False)
            datasets = (tr.train_ds, tr.test_ds)
            hist = tr.fit()
            warm = int(math.ceil(cfg["training"]["warmup_epochs"]))
            losses = [h["train_loss"] for h in hist]
            row = {
                "variant": name, "lr": lr,
                "train_loss_by_epoch": " ".join(f"{x:.2f}" for x in losses),
                "max_train_loss_after_warmup": max(losses[warm:]) if len(losses) > warm else None,
                "final_train_loss": losses[-1],
                "final_test_loss": hist[-1]["test_loss"],
                "final_test_acc1": hist[-1]["test_acc1"],
                "final_test_acc5": hist[-1]["test_acc5"],
                "epoch_1gpu_s": round(sum(h["wall_time_s"] for h in hist) / len(hist), 2),
                "epoch_est_2gpu_s": round(sum(h["est_pipe_time_s"] or 0 for h in hist) / len(hist), 2),
            }
            rows.append(row)
            print(f"[{name:18s} lr={lr:<5}] train {row['train_loss_by_epoch']} | "
                  f"test top1 {row['final_test_acc1']:.2f}% loss {row['final_test_loss']:.3f}", flush=True)
            utils.append_csv(row, os.path.join(base_dir, "summary.csv"))
    keys = list(rows[0].keys())
    print("\n| " + " | ".join(keys) + " |\n|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")


def _deep_copy(d):
    import copy
    return copy.deepcopy(d)


if __name__ == "__main__":
    main()
