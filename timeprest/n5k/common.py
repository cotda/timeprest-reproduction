"""Shared setup of the n5k trainers and checks: data, model config, task, logging rows."""
from __future__ import annotations

import copy

import torch

from . import model as _model  # noqa: F401  (registers "n5k_resnet50" in timeprest.models.MODEL_BUILDERS)
from .data import load_datasets
from .model import priors_from_meta
from .task import TOTALS, NutritionTask

# validation metrics logged every epoch, in this order (test set: same names with "test_")
EVAL_KEYS = ["loss"] + [f"mae_{t}" for t in TOTALS] + ["pmae_kcal", "mae_gram_present", "ingr_f1", "ingr_map"]
SELECT_KEY = "mae_kcal"     # model selection on the validation set (lower is better)


def prepare(cfg: dict, data=None, log=print):
    """Return (cfg_built, (train, val, test), meta, task). cfg_built = cfg with the model section
    completed from the data (num_ingredients, head priors), so every rank builds the same model."""
    if data is None:
        train, val, test, meta = load_datasets(cfg["data"], log)
    else:
        train, val, test, meta = data
    cfg = copy.deepcopy(cfg)
    cfg["model"]["num_ingredients"] = len(meta["ingredients"])
    cfg["model"].update(priors_from_meta(meta, train.targets))
    task = NutritionTask.from_meta(meta, cfg.get("loss"))
    return cfg, (train, val, test), meta, task


def first_batches(ds, K: int, M: int):
    """The first K*M samples as K mini-batches (inputs and float label vectors stacked)."""
    xs, ys = zip(*[ds[i] for i in range(K * M)])
    x, y = torch.stack(xs), torch.stack(ys)
    return [(x[k * M:(k + 1) * M], y[k * M:(k + 1) * M]) for k in range(K)]


def eval_cols(ev: dict, prefix: str = "val_") -> dict:
    return {prefix + k: round(ev[k], 5) for k in EVAL_KEYS}


def summary_line(row: dict, prefix: str = "val_") -> str:
    return (f"{prefix}loss {row[prefix + 'loss']:.4f} | MAE kcal {row[prefix + 'mae_kcal']:.1f} "
            f"({row[prefix + 'pmae_kcal']:.1f}%) mass {row[prefix + 'mae_mass']:.1f} fat {row[prefix + 'mae_fat']:.2f} "
            f"carb {row[prefix + 'mae_carb']:.2f} prot {row[prefix + 'mae_protein']:.2f} | "
            f"ingr F1 {row[prefix + 'ingr_f1']:.1f} mAP {row[prefix + 'ingr_map']:.1f}")


def median_predictor_mae(train_targets: torch.Tensor, eval_targets: torch.Tensor, K: int) -> float:
    """kcal MAE of always predicting the train median (reference for the short-run check)."""
    med = train_targets[:, K].median()
    return (eval_targets[:, K] - med).abs().mean().item()
