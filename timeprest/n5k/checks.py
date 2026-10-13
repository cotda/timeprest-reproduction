"""Checks of the nutrition task on the real W-GPU pipeline, before long runs (n5k/PLAN.md B4).

    torchrun --standalone --nproc_per_node=2 -m timeprest.n5k.checks --config n5k/configs/n5k_quick.yaml

  D0 data: cache built, splits non-empty, labels finite, < 2 % of the dishes dropped; prints the data report
  D1 static order == single-GPU engine with the nutrition loss (same versions and parameters)
  D2 dynamic order: no deadlock, pipeline drains, version rules hold, no recompute (graph rule)
  D3 checkpoint/resume is seamless
  D4 short run per system: train loss decreases, val MAE kcal below the "always predict the train
     median" baseline; real epoch time, per-GPU peak memory, estimated full-run time
Writes output.checks_file on rank 0 (the trainers refuse to start without an all-PASS file of this code).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import time
import traceback

import torch
import torch.distributed as dist

from .. import utils
from ..config import load_config
from ..dist.checks import gather, sys_cfg
from ..dist.runtime import StageRuntime
from ..dist.train import boundary_features, channels_from_cfg, init_dist
from ..engine import PipelineEngine
from ..models import build_stages
from .common import first_batches, median_predictor_mae, prepare
from .data import format_report
from .dist_train import N5kDistTrainer

PASS, FAIL = "PASS", "FAIL"


def make_runtime(ctx, cfg, K):
    rank, W, device = ctx["rank"], ctx["W"], ctx["device"]
    torch.manual_seed(cfg["seed"])
    sample = ctx["data"][0][0][0].unsqueeze(0)
    stages, _ = build_stages(cfg["model"], W, cfg["pipeline"]["partition"], sample)
    feats = boundary_features(stages, sample)
    rt = StageRuntime(rank, W, copy.deepcopy(stages[rank]), cfg["pipeline"], cfg["training"], device, K,
                      feats[rank - 1] if rank else tuple(sample.shape[1:]), feats[rank] if rank < W - 1 else None,
                      channels_from_cfg(cfg, rank, W), cfg["training"]["batch_size"], task=ctx["task"])
    return rt, stages


def d0_data(ctx):
    meta, (train, val, test) = ctx["meta"], ctx["data"]
    n_split = sum(len(ctx["meta"]["ids"][s]) for s in ("train", "val", "test")) + len(meta["dropped"])
    finite = all(torch.isfinite(d.targets).all().item() for d in (train, val, test))
    frac = len(meta["dropped"]) / max(1, n_split)
    m = {"n_train_val_test": [len(train), len(val), len(test)], "K": len(meta["ingredients"]),
         "dropped": [f"{d['dish_id']} ({d['split']}: {d['reason']})" for d in meta["dropped"]],
         "dropped_frac": round(frac, 4), "labels_finite": finite, "cache_dir": meta["cache_dir"],
         "build_time_s": meta["build_time_s"]}
    ok = finite and frac < 0.02 and min(len(train), len(val), len(test)) > 0 and m["K"] > 0
    if ctx["rank"] == 0:
        print(format_report(meta), flush=True)
    return (PASS if ok else FAIL), m, "cache, splits, labels"


def d1_static_equals_engine(ctx):
    rank, W, device = ctx["rank"], ctx["W"], ctx["device"]
    M, K = ctx["cc"].get("static_batch", 16), ctx["cc"].get("static_steps", 4)
    batches = first_batches(ctx["data"][0], K, M)
    metrics, ok = {}, True
    for system in ctx["systems"]:
        cfg = sys_cfg(ctx["cfg"], system, **{"training.batch_size": M, "training.epochs": 1,
                                             "pipeline.order": "static"})
        utils.set_seed(cfg["seed"], deterministic=True)
        rt, stages = make_runtime(ctx, cfg, K)
        rt.profile_ops = False
        rt.run_epoch(K, (b[0] for b in batches) if rank == 0 else None,
                     [b[1] for b in batches] if rank == W - 1 else None)
        allp = gather({n: p.detach().cpu() for n, p in rt.named})
        if rank == 0:
            eng = PipelineEngine([copy.deepcopy(m) for m in stages], cfg["pipeline"], cfg["training"], device, K,
                                 task=ctx["task"])
            eng.run_epoch(batches)
            worst = 0.0
            for s in range(W):
                for n, p in eng.stages[s].named_parameters():
                    a, b = allp[s][n].double(), p.detach().cpu().double()
                    worst = max(worst, ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item())
            metrics[system] = {"max_rel_param_diff_vs_engine": worst}
            ok &= worst < 1e-4
        del rt
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    ok = gather(ok)[0]
    return (PASS if ok else FAIL), metrics, "static order on W GPUs vs single-GPU engine, fp32 rel tol 1e-4"


def d2_dynamic(ctx):
    rank, W = ctx["rank"], ctx["W"]
    K = ctx["cc"].get("dynamic_steps", 10)
    metrics, ok = {}, True
    for system in ctx["systems"]:
        cfg = sys_cfg(ctx["cfg"], system, **{"training.epochs": 1})
        batches = first_batches(ctx["data"][0], K, cfg["training"]["batch_size"])
        rt, _ = make_runtime(ctx, cfg, K)
        rt.record_versions = True
        t0 = time.perf_counter()
        st = rt.run_epoch(K, (b[0] for b in batches) if rank == 0 else None,
                          [b[1] for b in batches] if rank == W - 1 else None)
        wall = time.perf_counter() - t0
        logs = gather({"ops": rt.op_log, "st": {k: v for k, v in st.items() if k != "ops"}})
        if rank == 0:
            f = [{(i, j): k for kind, i, j, k, _ in L["ops"] if kind == "F"} for L in logs]
            b = [{i: v for kind, i, _, v, _ in L["ops"] if kind == "B"} for L in logs]
            pc = cfg["pipeline"]
            m = {"wall_s": round(wall, 2), "peak_mem_mb": [round(L["st"]["peak_mem_mb"] or 0, 1) for L in logs],
                 "no_recompute": all(L["st"]["recompute_micro"] == 0 for L in logs)}
            ok &= m["no_recompute"]
            if pc["vertical_sync"]:
                m["forward_vertical_sync"] = all(f[s][key] == f[0][key] for s in range(W) for key in f[s])
                ok &= m["forward_vertical_sync"]
            if pc["backward_version"] == "stashed":
                m["backward_uses_forward_version"] = all(b[s][i] == [f[s][(i, j)] for j in range(len(b[s][i]))]
                                                         for s in range(W) for i in b[s])
                ok &= m["backward_uses_forward_version"]
            else:
                m["backward_uses_live_version"] = all(b[s][i] == [i] * len(b[s][i]) for s in range(W) for i in b[s])
                ok &= m["backward_uses_live_version"]
                m["micro_forwarded_with_older_version"] = sum(1 for (i, j), k in f[0].items() if k < i)
            metrics[system] = m
        del rt
    ok = gather(ok)[0]
    return (PASS if ok else FAIL), metrics, f"dynamic order, {K} mini-batches, version invariants"


def d3_resume(ctx):
    rank = ctx["rank"]
    n = ctx["cc"].get("resume_subset", 256)
    train, val, test = ctx["data"]
    small = (train.subset(range(min(n, len(train)))), val.subset(range(min(64, len(val)))),
             test.subset(range(min(64, len(test)))), ctx["meta"])
    metrics, ok = {}, True
    for system in ctx["systems"]:
        cfg = sys_cfg(ctx["raw_cfg"], system, **{"training.epochs": 2, "pipeline.order": "static",
                                                 "data.num_workers": 0})
        base = os.path.join(ctx["work"], f"resume_{system}")
        if rank == 0:
            shutil.rmtree(base, ignore_errors=True)
        dist.barrier()
        utils.set_seed(cfg["seed"], deterministic=True)
        a = N5kDistTrainer(cfg, os.path.join(base, "a"), small, verbose=False)
        ra = a.fit()
        utils.set_seed(cfg["seed"], deterministic=True)
        b = N5kDistTrainer(cfg, os.path.join(base, "b"), small, verbose=False)
        b.fit(stop_after=1)
        b2 = N5kDistTrainer(cfg, os.path.join(base, "b"), small, verbose=False)
        resumed = b2.try_resume()
        rb = b2.fit()
        dp = max((pa - pb).abs().max().item() for (_, pa), (_, pb) in zip(a.rt.named, b2.rt.named))
        dps = gather(dp)
        if rank == 0:
            dl = abs(ra[-1]["train_loss"] - rb[-1]["train_loss"])
            metrics[system] = {"resumed": resumed, "param_max_diff": max(dps), "last_epoch_loss_diff": dl,
                               "best_after_resume": b2.best}
            ok &= resumed and max(dps) <= 1e-5 and dl <= 1e-5
    ok = gather(ok)[0]
    return (PASS if ok else FAIL), metrics, "2 epochs vs 1 + resume (static order, deterministic), tol 1e-5"


def d4_short(ctx):
    rank = ctx["rank"]
    cc = ctx["cc"]
    train, val, test = ctx["data"]
    sub = cc.get("short_subset")
    data = (train.subset(range(min(sub, len(train)))) if sub else train, val, test, ctx["meta"])
    ref = median_predictor_mae(data[0].targets, val.targets, ctx["task"].K)
    metrics, ok = {}, True
    for system in ctx["systems"]:
        cfg = sys_cfg(ctx["raw_cfg"], system, **{"training.epochs": cc.get("short_epochs", 2),
                                                 "training.warmup_epochs": cc.get("short_warmup_epochs", 0.5)})
        out = os.path.join(ctx["work"], f"short_{system}")
        if rank == 0:
            shutil.rmtree(out, ignore_errors=True)
        dist.barrier()
        utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
        tr = N5kDistTrainer(cfg, out, data, verbose=rank == 0)
        rows = tr.fit()
        if rank == 0:
            r1, r2 = rows[0], rows[-1]
            per_sample = r2["epoch_time_s"] / max(1, r2["samples"])
            m = {"train_loss_by_epoch": [r["train_loss"] for r in rows],
                 "val_mae_kcal_by_epoch": [r["val_mae_kcal"] for r in rows],
                 "median_predictor_val_mae_kcal": round(ref, 2),
                 "val_ingr_f1_by_epoch": [r["val_ingr_f1"] for r in rows],
                 "epoch_time_s": r2["epoch_time_s"], "busy_frac_per_gpu": r2["busy_frac"],
                 "peak_mem_mb_per_gpu": r2["peak_mem_mb"], "mb_sent_per_gpu": r2["mb_sent"],
                 "bwd_overlap_minibatches": r2["bwd_overlap_minibatches"],
                 "est_full_epoch_s": round(per_sample * len(train), 1),
                 "est_full_run_h": round(per_sample * len(train) * ctx["cfg"]["training"]["epochs"] / 3600, 2)}
            metrics[system] = m
            ok &= r2["train_loss"] < r1["train_loss"] and r2["val_mae_kcal"] < ref and math.isfinite(r2["train_loss"])
        del tr
    ok = gather(ok)[0]
    return (PASS if ok else FAIL), metrics, "short run per system on the real pipeline"


CHECKS = [(0, "data", d0_data), (1, "static_equals_engine", d1_static_equals_engine), (2, "dynamic_versions", d2_dynamic),
          (3, "checkpoint_resume", d3_resume), (4, "short_run", d4_short)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="n5k/configs/n5k_quick.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--work-dir", default="n5k/results/checks/work")
    args = ap.parse_args(argv)
    raw = load_config(args.config, args.set)
    rank, W, device = init_dist(raw)
    if rank == 0:
        print("env:", json.dumps(utils.env_info()), "| code", utils.code_hash(), "| gpus", torch.cuda.device_count(),
              flush=True)
    cfg, data, meta, task = prepare(raw, log=print if rank == 0 else (lambda *_: None))
    ctx = {"raw_cfg": raw, "cfg": cfg, "cc": cfg["checks"], "rank": rank, "W": W, "device": device,
           "work": args.work_dir, "systems": cfg["checks"].get("systems", ["pipedream", "timeprest"]),
           "data": data, "meta": meta, "task": task}
    ids = args.only if args.only is not None else [c[0] for c in CHECKS]
    results = []
    for cid, name, fn in CHECKS:
        if cid not in ids:
            continue
        t0 = time.perf_counter()
        if rank == 0:
            print(f"\n=== D{cid}: {name} ===", flush=True)
        try:
            status, metrics, msg = fn(ctx)
        except Exception as e:
            traceback.print_exc()
            status, metrics, msg = FAIL, {}, f"{type(e).__name__}: {e}"
        dt = time.perf_counter() - t0
        if rank == 0:
            print(f"[{status}] {name} ({dt:.1f}s) - {msg}")
            for k, v in metrics.items():
                print(f"    {k}: {v}")
        results.append({"id": cid, "name": name, "status": status, "seconds": round(dt, 1),
                        "message": msg, "metrics": metrics})
    if rank == 0:
        all_pass = all(r["status"] == PASS for r in results) and len(results) == len(CHECKS)
        print("\n" + "=" * 60)
        for r in results:
            print(f"D{r['id']} {r['name']:<24}{r['status']:<6}{r['seconds']:>7.1f}s")
        print(f"ALL N5K CHECKS PASS: {all_pass}")
        utils.save_json({"code_hash": utils.code_hash(), "config": args.config, "overrides": args.set,
                         "systems": ctx["systems"], "env": utils.env_info(), "all_pass": all_pass,
                         "results": results}, cfg["output"]["checks_file"])
        print("saved", cfg["output"]["checks_file"])
        print("N5K_CHECKS_JSON", json.dumps({"all_pass": all_pass, "results": results}, default=str))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
