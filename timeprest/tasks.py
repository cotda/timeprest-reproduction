"""Loss and metrics of a training task, used by the pipeline engine (phase 1) and the stage runtime
(phase 2). The schedule and weight-version logic never look at labels; only the last stage calls
the task. `ClassificationTask` is the default and computes exactly what the engine/runtime computed
before tasks existed (summed cross-entropy, top-1/top-5). Other tasks (e.g. `timeprest.n5k.task`)
implement the same methods.

Contract (labels `y` are the per-sample targets, split along dim 0 like the inputs):
  loss_sum(out, y)        -> scalar tensor, loss summed over the samples (the runtime divides by M)
  train_metric_init(dev)  -> accumulator tensor for a training-time metric
  train_metric(out, y)    -> tensor added to that accumulator (out is detached)
  train_summary(acc, n)   -> dict merged into the engine's epoch statistics
  runtime_stats(acc)      -> dict merged into the runtime's (raw, per-stage) epoch statistics
  eval_start() / eval_update(state, out, y) / eval_result(state) -> evaluation metrics dict
  dummy_labels(out)       -> labels of the right shape/dtype for kernel warm-up
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _upcast(t: torch.Tensor) -> torch.Tensor:
    """Compute losses in at least fp32 (keeps fp64 in the float64 tests)."""
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


class ClassificationTask:
    name = "classification"

    def loss_sum(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(_upcast(out), y, reduction="sum")

    def train_metric_init(self, device) -> torch.Tensor:
        return torch.zeros((), device=device, dtype=torch.long)

    def train_metric(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (out.argmax(1) == y).sum()

    def train_summary(self, acc: torch.Tensor, n: int) -> dict:
        return {"train_acc1": acc.item() / max(1, n) * 100.0}

    def runtime_stats(self, acc: torch.Tensor) -> dict:
        return {"correct": int(acc.item())}

    def eval_start(self) -> dict:
        return {"loss": 0.0, "c1": 0, "c5": 0, "n": 0}

    def eval_update(self, st: dict, out: torch.Tensor, y: torch.Tensor):
        out = _upcast(out)
        st["loss"] += F.cross_entropy(out, y, reduction="sum").item()
        top = out.topk(min(5, out.shape[1]), 1).indices
        st["c1"] += (top[:, 0] == y).sum().item()
        st["c5"] += (top == y[:, None]).any(1).sum().item()
        st["n"] += y.shape[0]

    def eval_result(self, st: dict) -> dict:
        n = st["n"]
        return {"loss": st["loss"] / n, "acc1": st["c1"] / n * 100.0, "acc5": st["c5"] / n * 100.0}

    def dummy_labels(self, out: torch.Tensor) -> torch.Tensor:
        return torch.randint(0, out.shape[1], (out.shape[0],), device=out.device)
