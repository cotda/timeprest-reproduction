"""Seeding, environment info, code hashing and small IO helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random

import numpy as np
import torch

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": None,
        "gpu_mem_gb": None,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_mem_gb"] = round(p.total_memory / 2**30, 2)
    return info


def code_hash() -> str:
    """Hash of all .py files of the package except tests (checks are tied to this)."""
    h = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(PKG_DIR)):
        dirs[:] = sorted(d for d in dirs if d not in ("tests", "__pycache__"))
        for f in sorted(files):
            if f.endswith(".py"):
                path = os.path.join(root, f)
                h.update(os.path.relpath(path, PKG_DIR).replace("\\", "/").encode())
                with open(path, "rb") as fh:
                    h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def append_csv(row: dict, path: str):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_bytes(n: float) -> str:
    return f"{n / 2**20:.1f}MB"
