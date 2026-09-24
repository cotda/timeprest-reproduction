"""YAML config loading with `base:` inheritance, `--set key=value` overrides and system presets."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

import yaml

# Presets map a system name to the pipeline mechanism it uses (paper §3, §4.10).
# Keys set explicitly in `pipeline:` override the preset (used for ablations).
SYSTEM_PRESETS = {
    # PipeDream baseline: 1F1B, whole mini-batch per op, horizontal stashing
    # (backward uses the forward's version) + vertical sync of the forward version.
    "pipedream": {"schedule": "1F1B", "num_microbatches": 1,
                  "vertical_sync": True, "backward_version": "stashed"},
    # TiMePReSt: nF1B, backward uses the latest version committed across the pipeline
    # (no horizontal stashing), forward keeps vertical sync.
    "timeprest": {"schedule": "nF1B", "vertical_sync": True, "backward_version": "committed"},
    # Ablation Variant 1 (§4.10): nF1B but keep weight stashing.
    "variant1": {"schedule": "nF1B", "vertical_sync": True, "backward_version": "stashed"},
    # Ablation Variant 2 (§4.10): 1F1B without weight stashing.
    "variant2": {"schedule": "1F1B", "num_microbatches": 1,
                 "vertical_sync": True, "backward_version": "committed"},
}

DEFAULTS: dict[str, Any] = {
    "name": "run",
    "seed": 0,
    "system": "timeprest",
    "data": {
        "dataset": "cifar100",      # cifar100 | cifar10 | synthetic
        "root": "./data",
        "train_subset": None,       # int -> first-k of a fixed permutation
        "test_subset": None,
        "augment": True,
        "num_workers": 2,
        "download": True,
    },
    "model": {"name": "vgg16_bn_cifar", "num_classes": 100, "width": 1.0},
    "pipeline": {
        "num_stages": 2,
        "num_microbatches": 3,
        "partition": "auto",        # "auto" (balanced MACs) or list of block boundaries
        "max_inflight": "pipedream",  # "pipedream" (W - s per stage), int, or null
        "schedule": None, "vertical_sync": None, "backward_version": None,
    },
    "training": {
        "epochs": 160,
        "batch_size": 192,          # mini-batch M (same for all systems, paper §4.5)
        "optimizer": "sgd",
        "lr": 0.1,
        "momentum": 0.9,
        "nesterov": False,
        "weight_decay": 5e-4,
        "lr_schedule": "cosine",    # cosine | constant
        "warmup_epochs": 0,
        "drop_last": True,
    },
    "runtime": {
        "device": "cuda",
        "deterministic": False,
        "profile_ops": True,        # per-op timing for the 2-GPU time estimate
        "eval_batch_size": 500,
        "comm_bandwidth_gbps": 10.0,  # GB/s used by the communication model (estimate only)
        "comm_latency_ms": 0.05,
    },
    "output": {"dir": "results/runs/{name}", "require_checks": False,
               "checks_file": "results/checks/latest.json"},
    "checks": {},
}


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    if base:
        base_path = os.path.join(os.path.dirname(path), base)
        merged = _load_yaml(base_path)
        return _deep_update(merged, raw)
    return raw


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def resolve(cfg: dict) -> dict:
    """Fill defaults, apply the system preset and validate."""
    out = _deep_update(copy.deepcopy(DEFAULTS), cfg)
    system = out["system"]
    if system not in SYSTEM_PRESETS:
        raise ValueError(f"unknown system {system!r}; choose from {list(SYSTEM_PRESETS)}")
    pipe = out["pipeline"]
    user_pipe = cfg.get("pipeline", {}) or {}
    for k, v in SYSTEM_PRESETS[system].items():
        # num_microbatches from the preset wins for 1F1B systems; other keys only if unset
        if k == "num_microbatches" or user_pipe.get(k) is None:
            pipe[k] = v
    if pipe["schedule"] not in ("1F1B", "nF1B"):
        raise ValueError("pipeline.schedule must be 1F1B or nF1B")
    if pipe["backward_version"] not in ("stashed", "committed", "latest"):
        raise ValueError("pipeline.backward_version must be stashed|committed|latest")
    if pipe["schedule"] == "1F1B" and pipe["num_microbatches"] != 1:
        raise ValueError("1F1B uses num_microbatches=1")
    M, N = out["training"]["batch_size"], pipe["num_microbatches"]
    if M < N:
        raise ValueError("batch_size must be >= num_microbatches")
    out["output"]["dir"] = out["output"]["dir"].format(name=out["name"])
    return out


def load_config(path: str | None, overrides: list[str] | None = None) -> dict:
    raw = _load_yaml(path) if path else {}
    apply_overrides(raw, overrides)
    return resolve(raw)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
