"""Phase-2 checks on the real multi-GPU pipeline (run under torchrun, one process per stage).

    torchrun --standalone --nproc_per_node=2 -m timeprest.dist.checks --config configs/kaggle_quick.yaml

  D1 static order == single-GPU phase-1 engine (same versions and parameters)
  D2 dynamic order: no deadlock, pipeline drains, version rules hold (vertical sync / stashing)
  D3 checkpoint/resume is seamless
  D4 short run per system: loss decreases, accuracy > chance, real epoch time / per-GPU memory
Writes output.checks_file (default results/checks/dist_latest.json) on rank 0.
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
from ..config import load_config, resolve
from ..data import build_datasets
from ..engine import PipelineEngine
from ..models import build_stages
from .runtime import Channels, StageRuntime
from .train import DistTrainer, boundary_features, init_dist

PASS, FAIL = "PASS", "FAIL"


def sys_cfg(base: dict, system: str, **over) -> dict:
    raw = copy.deepcopy(base)
    raw["system"] = system
    for k in ("schedule", "vertical_sync", "backward_version"):
        raw["pipeline"][k] = None
    raw["pipeline"]["num_microbatches"] = base["checks"].get("num_microbatches", 3)
    for path, v in over.items():
        node = raw
        parts = path.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return resolve(raw)


def gather(obj):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def first_batches(train, K, M):
    xs, ys = zip(*[train[i] for i in range(K * M)])
    x, y = torch.stack(xs), torch.tensor(ys)
    return [(x[k * M:(k + 1) * M], y[k * M:(k + 1) * M]) for k in range(K)]


def make_runtime(cfg, rank, W, device, train, K):
    torch.manual_seed(cfg["seed"])
    sample = train[0][0].unsqueeze(0)
    stages, _ = build_stages(cfg["model"], W, cfg["pipeline"]["partition"], sample)
    feats = boundary_features(stages, sample)
    ch = Channels(rank, W)
    rt = StageRuntime(rank, W, copy.deepcopy(stages[rank]), cfg["pipeline"], cfg["training"], device, K,
                      feats[rank - 1] if rank else tuple(sample.shape[1:]), feats[rank] if rank < W - 1 else None, ch,
                      cfg["training"]["batch_size"])
    return rt, stages


def d1_static_equals_engine(ctx):
    rank, W, device = ctx["rank"], ctx["W"], ctx["device"]
    M, K = ctx["cc"].get("static_batch", 48), ctx["cc"].get("static_steps", 6)
    train = ctx["data"][0]
    batches = first_batches(train, K, M)
    metrics, ok = {}, True
    for system in ctx["systems"]:
        cfg = sys_cfg(ctx["cfg"], system, **{"training.batch_size": M, "training.epochs": 1,
                                             "pipeline.order": "static"})
        utils.set_seed(cfg["seed"], deterministic=True)
        rt, stages = make_runtime(cfg, rank, W, device, train, K)
        rt.profile_ops = False
        rt.run_epoch(K, (b[0] for b in batches) if rank == 0 else None,
                     [b[1] for b in batches] if rank == W - 1 else None)
        mine = {n: p.detach().cpu() for n, p in rt.named}
        allp = gather(mine)
        if rank == 0:
            eng = PipelineEngine([copy.deepcopy(m) for m in stages], cfg["pipeline"], cfg["training"], device, K)
            eng.profile_ops = False
            eng.run_epoch(batches)
            worst = 0.0
            for s in range(W):
                for n, p in eng.stages[s].named_parameters():
                    a, b = allp[s][n].double(), p.detach().cpu().double()
                    worst = max(worst, ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item())
            metrics[system] = {"max_rel_param_diff_vs_engine": worst}
            ok &= worst < 1e-4
        del rt
    ok = gather(ok)[0]
    return (PASS if ok else FAIL), metrics, "static (Fig.2) order on W GPUs vs single-GPU engine, fp32 rel tol 1e-4"


def d2_dynamic(ctx):
    rank, W, device = ctx["rank"], ctx["W"], ctx["device"]
    K = ctx["cc"].get("dynamic_steps", 20)
    train = ctx["data"][0]
    metrics, ok = {}, True
    for system in ctx["systems"]:
        cfg = sys_cfg(ctx["cfg"], system, **{"training.epochs": 1})
        M = cfg["training"]["batch_size"]
        batches = first_batches(train, K, M)
        rt, _ = make_runtime(cfg, rank, W, device, train, K)
        rt.record_versions = True
        t0 = time.perf_counter()
        st = rt.run_epoch(K, (b[0] for b in batches) if rank == 0 else None,
                          [b[1] for b in batches] if rank == W - 1 else None)
        wall = time.perf_counter() - t0
        logs = gather({"ops": rt.op_log, "st": {k: v for k, v in st.items() if k != "ops"}, "wall": wall})
        if rank == 0:
            f = [{(i, j): k for kind, i, j, k, _ in L["ops"] if kind == "F"} for L in logs]
            b = [{i: v for kind, i, _, v, _ in L["ops"] if kind == "B"} for L in logs]
            pc = cfg["pipeline"]
            m = {"wall_s": round(wall, 2), "drained": True,
                 "recompute_micro": [L["st"]["recompute_micro"] for L in logs],
                 "graph_micro": [L["st"]["graph_micro"] for L in logs],
                 "wait_time_s": [round(L["st"]["wait_time_s"], 3) for L in logs]}
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
    n = ctx["cc"].get("resume_subset", 960)
    from torch.utils.data import Subset
    train, test = ctx["data"]
    small = (Subset(train, range(n)), Subset(test, range(256)))
    metrics, ok = {}, True
    for system in ctx["systems"]:
        cfg = sys_cfg(ctx["cfg"], system, **{"training.epochs": 2, "pipeline.order": "static",
                                             "data.num_workers": 0})
        base = os.path.join(ctx["work"], f"resume_{system}")
        if rank == 0:
            shutil.rmtree(base, ignore_errors=True)
        dist.barrier()
        utils.set_seed(cfg["seed"], deterministic=True)
        a = DistTrainer(cfg, os.path.join(base, "a"), small, verbose=False)
        ra = a.fit()
        utils.set_seed(cfg["seed"], deterministic=True)
        b = DistTrainer(cfg, os.path.join(base, "b"), small, verbose=False)
        b.fit(stop_after=1)
        b2 = DistTrainer(cfg, os.path.join(base, "b"), small, verbose=False)
        resumed = b2.try_resume()
        rb = b2.fit()
        dp = max((pa - pb).abs().max().item() for (_, pa), (_, pb) in zip(a.rt.named, b2.rt.named))
        dps = gather(dp)
        if rank == 0:
            dl = abs(ra[-1]["train_loss"] - rb[-1]["train_loss"])
            metrics[system] = {"resumed": resumed, "param_max_diff": max(dps), "last_epoch_loss_diff": dl}
            ok &= resumed and max(dps) <= 1e-5 and dl <= 1e-5
    ok = gather(ok)[0]
    return (PASS if ok else FAIL), metrics, "2 epochs vs 1 + resume (static order, deterministic), tol 1e-5"


def d4_short(ctx):
    rank = ctx["rank"]
    metrics, ok = {}, True
    for system in ctx["systems"]:
        cfg = sys_cfg(ctx["cfg"], system)
        out = os.path.join(ctx["work"], f"short_{system}")
        if rank == 0:
            shutil.rmtree(out, ignore_errors=True)
        dist.barrier()
        utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
        tr = DistTrainer(cfg, out, ctx["data"], verbose=False)
        rows = tr.fit()
        if rank == 0:
            r1, r2 = rows[0], rows[-1]
            full_n = ctx["cc"].get("full_train_samples", 50000)
            m = {"train_loss_by_epoch": [r["train_loss"] for r in rows],
                 "test_acc1_by_epoch": [r["test_acc1"] for r in rows],
                 "epoch_time_s": r2["epoch_time_s"], "busy_frac_per_gpu": r2["busy_frac"],
                 "peak_mem_mb_per_gpu": r2["peak_mem_mb"], "mb_sent_per_gpu": r2["mb_sent"],
                 "msgs_sent_per_gpu": r2["msgs_sent"], "recompute_micro": r2["recompute_micro"],
                 "bwd_overlap_minibatches": r2["bwd_overlap_minibatches"],
                 "est_full_epoch_s": round(r2["epoch_time_s"] / max(1, r2["samples"]) * full_n, 1),
                 "est_full_run_h": round(r2["epoch_time_s"] / max(1, r2["samples"]) * full_n
                                         * ctx["cc"].get("full_epochs", 160) / 3600, 2)}
            metrics[system] = m
            ok &= r2["train_loss"] < r1["train_loss"] and r2["test_acc1"] > ctx["cc"].get("short_min_acc1", 2.0)
    ok = gather(ok)[0]
    return (PASS if ok else FAIL), metrics, "short run per system on the real pipeline"


CHECKS = [(1, "static_equals_engine", d1_static_equals_engine), (2, "dynamic_versions", d2_dynamic),
          (3, "checkpoint_resume", d3_resume), (4, "short_run", d4_short)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/kaggle_quick.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--work-dir", default="results/checks/dist_work")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    rank, W, device = init_dist(cfg)
    ctx = {"cfg": cfg, "cc": cfg["checks"], "rank": rank, "W": W, "device": device, "work": args.work_dir,
           "systems": cfg["checks"].get("systems", ["pipedream", "timeprest"])}
    if rank == 0:
        print("env:", json.dumps(utils.env_info()), "| code", utils.code_hash(), "| gpus", torch.cuda.device_count(),
              flush=True)
    ctx["data"] = build_datasets(cfg["data"], cfg["model"]["num_classes"], cfg["seed"])
    ids = args.only or [c[0] for c in CHECKS]
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
        print(f"ALL PHASE-2 CHECKS PASS: {all_pass}")
        utils.save_json({"code_hash": utils.code_hash(), "config": args.config, "overrides": args.set,
                         "systems": ctx["systems"], "env": utils.env_info(), "all_pass": all_pass,
                         "results": results}, cfg["output"]["checks_file"])
        print("saved", cfg["output"]["checks_file"])
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
