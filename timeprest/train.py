"""Training entry point.

    python -m timeprest.train --config configs/full_timeprest.yaml [--resume] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from . import utils
from .config import load_config
from .runner import Trainer


def checks_ok(cfg: dict) -> tuple[bool, str]:
    path = cfg["output"]["checks_file"]
    saved = os.path.join(cfg["output"]["dir"], "checks.json")  # copy kept next to the run (Drive)
    if not os.path.exists(path) and os.path.exists(saved):
        path = saved
    if not os.path.exists(path):
        return False, f"{path} not found - run `python -m timeprest.checks --all` first"
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    failed = [r["name"] for r in res.get("results", []) if r["status"] != "PASS"]
    if failed or not res.get("all_pass"):
        return False, f"checks not all PASS (or not all 7 run): {failed}"
    if res.get("code_hash") != utils.code_hash():
        return False, "checks were run on a different code version - re-run the checks"
    missing = [s for s in [cfg["system"]] if s not in res.get("systems", [])]
    if missing:
        return False, f"checks did not cover system(s) {missing}"
    return True, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override config keys, e.g. training.epochs=2")
    ap.add_argument("--resume", action="store_true", help="resume from output.dir/checkpoint.pt")
    ap.add_argument("--force", action="store_true", help="run even if checks did not all PASS")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after this many epochs (debug)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if cfg["output"].get("require_checks") and not args.force:
        ok, msg = checks_ok(cfg)
        if not ok:
            print(f"[refuse] {msg}\n(use --force to override)")
            sys.exit(2)
    out_dir = cfg["output"]["dir"]
    if os.path.exists(os.path.join(out_dir, "checkpoint.pt")) and not args.resume:
        print(f"[refuse] {out_dir} already has a checkpoint; pass --resume to continue or change output.dir")
        sys.exit(2)

    if cfg["output"].get("require_checks") and os.path.exists(cfg["output"]["checks_file"]):
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(cfg["output"]["checks_file"], os.path.join(out_dir, "checks.json"))

    utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    device = utils.resolve_device(cfg["runtime"]["device"])
    print("env:", json.dumps(utils.env_info()), "| code", utils.code_hash())
    trainer = Trainer(cfg, device, out_dir)
    if args.resume:
        trainer.try_resume()
    rows = trainer.fit(stop_after=args.stop_after)
    hist = trainer.history()
    if hist:
        best = max(hist, key=lambda r: float(r["test_acc1"]))
        summary = {
            "name": cfg["name"], "system": cfg["system"], "epochs_done": int(hist[-1]["epoch"]),
            "final": hist[-1], "best_test_acc1": float(best["test_acc1"]), "best_epoch": int(best["epoch"]),
            "mean_epoch_wall_s": sum(float(r["wall_time_s"]) for r in hist) / len(hist),
            "env": utils.env_info(), "code_hash": utils.code_hash(),
        }
        utils.save_json(summary, os.path.join(out_dir, "summary.json"))
        print("SUMMARY", json.dumps({k: summary[k] for k in
                                     ("name", "epochs_done", "best_test_acc1", "best_epoch", "mean_epoch_wall_s")}))
    return rows


if __name__ == "__main__":
    main()
