"""A (n5k/added/README.md): the n5k pipeline trainer (`timeprest.n5k.dist_train`) plus an EMA of every stage's weights.

Training is unchanged (bit-identical to dist_train): the EMA is updated by an optimizer step post-hook and never
feeds back, and the global RNG state is restored after the extra evaluations. Every epoch the live
weights are evaluated as before (columns val_*), then each EMA decay (columns ema<d>_val_*). Each gets its own best
checkpoint by val MAE kcal (best_stage<r>.pt, best_ema<d>_stage<r>.pt) and its own test metrics in summary.json
("ema": {tag: test metrics}). The EMA update runs inside the training step, so epoch_time_s includes its small cost.
No --resume (the EMA is not in the epoch checkpoint); a run is one kernel step.

    torchrun --standalone --nproc_per_node=2 -m timeprest.n5k.added.ema_train --config n5k/added/configs/added_timeprest.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

from ... import utils
from ...config import load_config
from ..common import SELECT_KEY, eval_cols
from ..dist_train import N5kDistTrainer, _rank0
from ..train import checks_ok, write_summary
from .ema import WeightEMA, tag


class EmaDistTrainer(N5kDistTrainer):
    def __init__(self, cfg: dict, out_dir: str, data=None, verbose: bool = True):
        super().__init__(cfg, out_dir, data=data, verbose=verbose)
        ec = cfg["added"]["ema"]
        self.ema = WeightEMA(self.rt.module, ec["decays"], ec.get("warmup", True))
        self.rt.opt.register_step_post_hook(lambda *_: self.ema.update(self.rt.module))
        self.ema_best = {d: None for d in self.ema.decays}

    def ema_path(self, d: float) -> str:
        return os.path.join(self.out_dir, f"best_{tag(d)}_stage{self.rank}.pt")

    def run_epoch(self, epoch: int) -> dict:
        row = super().run_epoch(epoch)
        rng = utils.rng_state()     # each eval loader draws a base seed from the global RNG: keep training's draws
        for d in self.ema.decays:
            with self.ema.applied(self.rt.module, d):
                nb, exs, elabels = self._eval_inputs()
                ev = self.rt.evaluate(exs, elabels, nb)
            dist.barrier()
            res = [None] * self.world
            dist.all_gather_object(res, ev)
            cols = eval_cols(res[-1], f"{tag(d)}_val_")
            row.update(cols)
            score = cols[f"{tag(d)}_val_{SELECT_KEY}"]
            if self.ema_best[d] is None or score < self.ema_best[d]["score"]:
                self.ema_best[d] = {"score": score, "epoch": row["epoch"]}
                torch.save({"module": self.ema.state_dict(d), "epoch": row["epoch"]}, self.ema_path(d))
        utils.set_rng_state(rng)
        if self.rank == 0:
            self.log(f"   ema val MAE kcal: " + " ".join(f"{tag(d)} {row[f'{tag(d)}_val_{SELECT_KEY}']:.1f}"
                                                        for d in self.ema.decays))
        return row

    def test_all(self) -> tuple[dict | None, dict]:
        """Test metrics of the live best checkpoint and of each EMA best checkpoint (last rank; None elsewhere)."""
        live = self.test_best()
        ema, keep = {}, self.best_path
        for d in self.ema.decays:
            self.best_path = self.ema_path(d)
            ema[tag(d)] = self.test_best()
        self.best_path = keep
        return live, ema


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    out_dir = cfg["output"]["dir"]
    if cfg["output"].get("require_checks") and not args.force:
        ok, msg = checks_ok(cfg)
        if not ok:
            if _rank0():
                print(f"[refuse] {msg} (use --force to override)")
            sys.exit(2)
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        if _rank0():
            print(f"[refuse] {out_dir} already has a run (no --resume for the EMA trainer); change output.dir")
        sys.exit(2)
    utils.set_seed(cfg["seed"], cfg["runtime"]["deterministic"])
    tr = EmaDistTrainer(cfg, out_dir)
    if tr.rank == 0:
        print("env:", json.dumps(utils.env_info()), "| code", utils.code_hash(), "| gpus", torch.cuda.device_count(),
              "| p2p", cfg["dist"].get("p2p_backend", "gloo"), "| ema", tr.ema.decays, flush=True)
    rows = tr.fit()
    if rows and rows[-1]["epoch"] >= cfg["training"]["epochs"]:
        live, ema = tr.test_all()
        if tr.rank == 0:
            write_summary(cfg, out_dir, live, {"ema": ema, "ema_best_val": {tag(d): b for d, b in tr.ema_best.items()}})
            print("N5K_EMA_JSON " + json.dumps({"name": cfg["name"], "live": live[f"test_{SELECT_KEY}"],
                                                **{k: v[f"test_{SELECT_KEY}"] for k, v in ema.items()}}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
