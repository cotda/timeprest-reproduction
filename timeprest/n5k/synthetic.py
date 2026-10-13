"""Tiny synthetic nutrition problem (float64) for CPU tests and the 2-process self-test:
same label/output layout as the real task (timeprest.n5k.task), random ingredients and grams."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..models import mlp_blocks, resnet50_blocks
from .task import TOTALS, NutritionTask

K = 4   # ingredients


def make_task(seed: int = 0, **weights) -> NutritionTask:
    g = torch.Generator().manual_seed(seed)
    per_gram = torch.rand(K, len(TOTALS), generator=g, dtype=torch.float64) * 2
    per_gram[:, 1] = 1.0                                   # mass per gram
    gram_scale = 20 + 80 * torch.rand(K, generator=g, dtype=torch.float64)
    total_std = torch.tensor([150.0, 100.0, 10.0, 20.0, 12.0], dtype=torch.float64)
    return NutritionTask(per_gram, gram_scale, total_std, **weights)


def make_labels(task: NutritionTask, M: int, g: torch.Generator) -> torch.Tensor:
    present = torch.rand(M, K, generator=g) < 0.5
    grams = torch.rand(M, K, generator=g, dtype=torch.float64) * 2 * task.gram_scale * present
    totals = grams @ task.per_gram + torch.randn(M, len(TOTALS), generator=g, dtype=torch.float64)
    return torch.cat([grams, totals.clamp_min(0)], 1)


def make_model(kind: str, W: int) -> list[nn.Sequential]:
    torch.manual_seed(0)
    blocks = mlp_blocks(12, 16, 2 * K, 4) if kind == "mlp" else resnet50_blocks(2 * K, width=1 / 16)
    cut = [round(len(blocks) * s / W) for s in range(W + 1)]
    return [nn.Sequential(*blocks[a:b]).double() for a, b in zip(cut[:-1], cut[1:])]


def make_batches(kind: str, task: NutritionTask, Kb: int, M: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    shape = (M, 12) if kind == "mlp" else (M, 3, 32, 32)
    return [(torch.randn(*shape, generator=g, dtype=torch.float64), make_labels(task, M, g)) for _ in range(Kb)]
