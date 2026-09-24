"""Independent reference computations used by the unit tests and by `timeprest.checks`.
They use plain nn.Modules + autograd (no functional_call, no schedule) on purpose."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import lr_lambda_factory

# Paper Fig.2a-e (p.4), first 18 slots, transcribed by hand from the figure.
FIG2 = {
    (4, 2): """1A 1B 2A 2B 3A 3B 4A 4B 1 5A 5B 2 6A 6B 3 7A 7B 4
               . 1A 1B 2A 2B 3A 3B 1 4A 4B 2 5A 5B 3 6A 6B 4 7A
               . . 1A 1B 2A 2B 1 3A 3B 2 4A 4B 3 5A 5B 4 6A 6B
               . . . 1A 1B 1 2A 2B 2 3A 3B 3 4A 4B 4 5A 5B 5""",
    (4, 4): """1A 1B 1C 1D 2A 2B 2C 2D 3A 3B 1 3C 3D 4A 4B 2 4C 4D
               . 1A 1B 1C 1D 2A 2B 2C 2D 1 3A 3B 3C 3D 2 4A 4B 4C
               . . 1A 1B 1C 1D 2A 2B 1 2C 2D 3A 3B 2 3C 3D 4A 4B
               . . . 1A 1B 1C 1D 1 2A 2B 2C 2D 2 3A 3B 3C 3D 3""",
    (3, 2): """1A 1B 2A 2B 3A 3B 1 4A 4B 2 5A 5B 3 6A 6B 4 7A 7B
               . 1A 1B 2A 2B 1 3A 3B 2 4A 4B 3 5A 5B 4 6A 6B 5
               . . 1A 1B 1 2A 2B 2 3A 3B 3 4A 4B 4 5A 5B 5 6A""",
    (5, 2): """1A 1B 2A 2B 3A 3B 4A 4B 5A 5B 1 6A 6B 2 7A 7B 3 8A
               . 1A 1B 2A 2B 3A 3B 4A 4B 1 5A 5B 2 6A 6B 3 7A 7B
               . . 1A 1B 2A 2B 3A 3B 1 4A 4B 2 5A 5B 3 6A 6B 4
               . . . 1A 1B 2A 2B 1 3A 3B 2 4A 4B 3 5A 5B 4 6A
               . . . . 1A 1B 1 2A 2B 2 3A 3B 3 4A 4B 4 5A 5B""",
    (5, 3): """1A 1B 1C 2A 2B 2C 3A 3B 3C 4A 4B 1 4C 5A 5B 2 5C 6A
               . 1A 1B 1C 2A 2B 2C 3A 3B 3C 1 4A 4B 4C 2 5A 5B 5C
               . . 1A 1B 1C 2A 2B 2C 3A 1 3B 3C 4A 2 4B 4C 5A 3
               . . . 1A 1B 1C 2A 2B 1 2C 3A 3B 2 3C 4A 4B 3 4C
               . . . . 1A 1B 1C 1 2A 2B 2C 2 3A 3B 3C 3 4A 4B""",
}


def fig2_grid(W: int, N: int) -> list[list[str]]:
    return [row.split() for row in FIG2[(W, N)].strip().splitlines()]


def plain_training(stages, batches, N: int, training_cfg: dict, total_steps: int) -> nn.Sequential:
    """Ordinary (non-pipelined) training of the concatenated model with the same optimizer,
    LR schedule and micro-batch loss weighting (sum over chunks of CE_sum / M)."""
    model = nn.Sequential(*[copy.deepcopy(s) for s in stages])
    opt = torch.optim.SGD(model.parameters(), lr=training_cfg["lr"], momentum=training_cfg["momentum"],
                          weight_decay=training_cfg["weight_decay"],
                          nesterov=training_cfg.get("nesterov", False))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total_steps, 0, training_cfg["lr_schedule"]))
    model.train()
    for x, y in batches:
        opt.zero_grad()
        for xc, yc in zip(torch.tensor_split(x, N), torch.tensor_split(y, N)):
            (F.cross_entropy(model(xc), yc, reduction="sum") / x.shape[0]).backward()
        opt.step()
        sch.step()
    return model


def _loaded_copy(module: nn.Module, params: dict) -> nn.Module:
    m = copy.deepcopy(module)
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(params[n])
    return m


def full_grad_at(stages, version_params: dict, versions: list[int], x, y, N: int) -> list[dict]:
    """Gradient of the mini-batch loss of the whole model with stage s at version versions[s]."""
    mods = [_loaded_copy(m, version_params[(s, versions[s])]) for s, m in enumerate(stages)]
    model = nn.Sequential(*mods).train()
    for xc, yc in zip(torch.tensor_split(x, N), torch.tensor_split(y, N)):
        (F.cross_entropy(model(xc), yc, reduction="sum") / x.shape[0]).backward()
    return [{n: p.grad.clone() for n, p in m.named_parameters()} for m in mods]


def mixed_rule_grads(stages, version_params: dict, fwd_versions: list[int], bwd_version: int,
                     x, y, N: int) -> list[dict]:
    """Declared TiMePReSt rule for any W: micro-batch j is forwarded with version fwd_versions[j]
    on every stage (vertical sync); each stage then computes its local VJP at bwd_version on the
    stage input it received, using the gradient passed back by the next stage."""
    W = len(stages)
    grads = [{n: torch.zeros_like(p) for n, p in m.named_parameters()} for m in stages]
    for j, (xc, yc) in enumerate(zip(torch.tensor_split(x, N), torch.tensor_split(y, N))):
        inputs = [xc]
        with torch.no_grad():
            for s in range(W - 1):
                inputs.append(_loaded_copy(stages[s], version_params[(s, fwd_versions[j])]).train()(inputs[-1]))
        upstream = None
        for s in reversed(range(W)):
            m = _loaded_copy(stages[s], version_params[(s, bwd_version)]).train()
            inp = inputs[s].clone().requires_grad_(s > 0)
            out = m(inp)
            if s == W - 1:
                (F.cross_entropy(out, yc, reduction="sum") / x.shape[0]).backward()
            else:
                out.backward(upstream)
            for n, p in m.named_parameters():
                grads[s][n] += p.grad
            upstream = inp.grad if s > 0 else None
    return grads


def max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
