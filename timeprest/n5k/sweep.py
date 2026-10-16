"""lr sweep on the single-process engine (n5k/PLAN.md mục 6): every (system, lr) pair is one short
`timeprest.n5k.train` run; runs are spread over the given GPUs (one run per GPU at a time). Then a
table of the best validation metrics per run is printed; each system takes its own best lr.

    python -m timeprest.n5k.sweep --config n5k/configs/n5k_sweep.yaml --systems single pipedream timeprest \
        --lrs 0.01 0.005 0.002 --gpus 0 1
    python -m timeprest.n5k.sweep --config n5k/configs/n5k_sweep.yaml --summary-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from .. import utils
from ..config import load_config

SYSTEM_CONFIG = {"single": "n5k_single.yaml", "pipedream": "n5k_pipedream.yaml", "timeprest": "n5k_timeprest.yaml",
                 "timeprest_n4": "n5k_timeprest_n4.yaml"}


def run_name(system: str, lr: float, tag: str = "") -> str:
    return f"sweep_{system}{'_' + tag if tag else ''}_lr{lr:g}"


def jobs(args) -> list[tuple[str, float, list[str]]]:
    """(system, lr, command) of every run; the system config is combined with the sweep config's
    training/output overrides."""
    sweep = load_config(args.config)
    tc, out = sweep["training"], sweep["output"]
    cfg_dir = os.path.dirname(args.config)
    res = []
    for system in args.systems:
        for lr in args.lrs:
            name = run_name(system, lr, getattr(args, 'tag', ''))
            sets = [f"name={name}", f"training.lr={lr}", f"training.epochs={tc['epochs']}",
                    f"training.warmup_epochs={tc['warmup_epochs']}", f"output.dir={out['dir'].replace(sweep['name'], name)}",
                    *args.set]
            cmd = [sys.executable, "-m", "timeprest.n5k.train", "--config",
                   os.path.join(cfg_dir, SYSTEM_CONFIG[system]), "--set", *sets]
            res.append((system, lr, cmd))
    return res


def run_all(args):
    todo = jobs(args)
    running: dict = {}          # gpu -> (proc, system, lr, log file)
    os.makedirs(args.log_dir, exist_ok=True)
    while todo or running:
        for gpu in args.gpus:
            if gpu not in running and todo:
                system, lr, cmd = todo.pop(0)
                log = open(os.path.join(args.log_dir, run_name(system, lr, getattr(args, 'tag', '')) + ".log"), "w", encoding="utf-8")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                running[gpu] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), system, lr, log)
                print(f"[gpu {gpu}] start {run_name(system, lr, getattr(args, 'tag', ''))}", flush=True)
        time.sleep(5)
        for gpu, (p, system, lr, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                print(f"[gpu {gpu}] done {run_name(system, lr, getattr(args, 'tag', ''))} exit={p.returncode}", flush=True)
                del running[gpu]


def summary(args) -> list[dict]:
    sweep = load_config(args.config)
    rows = []
    for system in args.systems:
        for lr in args.lrs:
            d = sweep["output"]["dir"].replace(sweep["name"], run_name(system, lr, getattr(args, 'tag', '')))
            path = os.path.join(d, "metrics.csv")
            if not os.path.exists(path):
                rows.append({"system": system, "lr": lr, "status": "missing"})
                continue
            hist = utils.read_csv(path)
            best = min(hist, key=lambda r: float(r["val_mae_kcal"]))
            last = hist[-1]
            rows.append({"system": system, "lr": lr, "epochs": int(last["epoch"]),
                         "best_val_mae_kcal": float(best["val_mae_kcal"]), "best_epoch": int(best["epoch"]),
                         "last_val_mae_kcal": float(last["val_mae_kcal"]), "last_val_pmae_kcal": float(last["val_pmae_kcal"]),
                         "last_val_ingr_f1": float(last["val_ingr_f1"]), "last_train_loss": float(last["train_loss"]),
                         "epoch_s": float(last.get("wall_time_s") or last["epoch_time_s"])})  # engine | dist_train
    print(f"{'system':<14}{'lr':>8}{'ep':>4}{'best MAE kcal (ep)':>20}{'last MAE kcal':>15}{'last %':>8}"
          f"{'ingr F1':>9}{'train loss':>12}{'s/epoch':>9}")
    for r in rows:
        if r.get("status") == "missing":
            print(f"{r['system']:<14}{r['lr']:>8g}   missing")
            continue
        print(f"{r['system']:<14}{r['lr']:>8g}{r['epochs']:>4}{r['best_val_mae_kcal']:>14.2f} ({r['best_epoch']:>2})"
              f"{r['last_val_mae_kcal']:>15.2f}{r['last_val_pmae_kcal']:>8.1f}{r['last_val_ingr_f1']:>9.1f}"
              f"{r['last_train_loss']:>12.4f}{r['epoch_s']:>9.1f}")
    best = {}
    for r in rows:
        if "best_val_mae_kcal" in r and (r["system"] not in best or r["best_val_mae_kcal"] < best[r["system"]][1]):
            best[r["system"]] = (r["lr"], r["best_val_mae_kcal"])
    print("best lr per system:", {s: v[0] for s, v in best.items()})
    print("N5K_SWEEP_JSON", json.dumps(rows))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="n5k/configs/n5k_sweep.yaml")
    ap.add_argument("--systems", nargs="*", default=["single", "pipedream", "timeprest"])
    ap.add_argument("--lrs", nargs="*", type=float, default=[0.01, 0.005, 0.002])
    ap.add_argument("--gpus", nargs="*", type=int, default=[0])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", default="", help="name suffix of a variant (e.g. a loss setting passed with --set)")
    ap.add_argument("--log-dir", default="n5k/results/sweep/logs")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.summary_only:
        run_all(args)
    summary(args)


if __name__ == "__main__":
    main()
