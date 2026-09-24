"""Pre-flight checks (CLAUDE.md, "Kiểm tra trước khi chạy dài").

    python -m timeprest.checks --all                      # uses configs/quick.yaml
    python -m timeprest.checks --only 3 4 --config configs/quick.yaml

Writes results/checks/latest.json; `timeprest.train` refuses full runs unless all PASS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .. import utils
from ..config import load_config
from .core import CHECKS, Ctx, run_checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="result JSON (default: output.checks_file of the config)")
    ap.add_argument("--work-dir", default="results/checks/work")
    args = ap.parse_args(argv)
    if not args.all and not args.only:
        ap.error("pass --all or --only <ids>")
    ids = [c[0] for c in CHECKS] if args.all else args.only

    cfg = load_config(args.config, args.set)
    device = utils.resolve_device(cfg["runtime"]["device"])
    env = utils.env_info()
    print("env:", json.dumps(env), "| code", utils.code_hash())
    ctx = Ctx(cfg, device, args.work_dir)
    t0 = time.perf_counter()
    results = run_checks(ctx, ids)

    print("\n" + "=" * 64)
    print(f"{'#':<3}{'check':<26}{'status':<8}{'time':>8}")
    for r in results:
        print(f"{r['id']:<3}{r['name']:<26}{r['status']:<8}{r['seconds']:>7.1f}s")
    all_pass = all(r["status"] == "PASS" for r in results) and len(results) == len(CHECKS)
    print("=" * 64)
    print(f"ALL CHECKS PASS: {all_pass}  (total {time.perf_counter() - t0:.0f}s)")

    out = args.out or cfg["output"]["checks_file"]
    utils.save_json({"code_hash": utils.code_hash(), "config": args.config, "overrides": args.set,
                     "systems": ctx.systems, "env": env, "all_pass": all_pass, "results": results}, out)
    print(f"saved {out}")
    sys.exit(0 if all(r["status"] == "PASS" for r in results) else 1)


if __name__ == "__main__":
    main()
