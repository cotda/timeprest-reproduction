"""A (n5k/added/README.md): exponential moving average of the weights of one module (one pipeline stage).

Float entries of the state dict (parameters and BN running statistics) follow
    shadow <- d * shadow + (1 - d) * live,   d = min(decay, (1 + t) / (10 + t)) when warmup (t = updates so far),
as in tf.train.ExponentialMovingAverage(num_updates); integer entries (BN num_batches_tracked) are copied.
Several decays are kept side by side; `applied(module, decay)` swaps one in for evaluation and restores the live
weights afterwards (in place, so references to the module's tensors stay valid).
"""
from __future__ import annotations

import contextlib

import torch


def tag(decay: float) -> str:
    return f"ema{decay:g}"


class WeightEMA:
    def __init__(self, module: torch.nn.Module, decays, warmup: bool = True):
        self.decays = [float(d) for d in decays]
        self.warmup = bool(warmup)
        self.updates = 0
        self.shadows = {d: {k: v.detach().clone() for k, v in module.state_dict().items()} for d in self.decays}

    def effective(self, decay: float) -> float:
        return min(decay, (1 + self.updates) / (10 + self.updates)) if self.warmup else decay

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        live = module.state_dict()
        for d, sh in self.shadows.items():
            e = self.effective(d)
            for k, v in live.items():
                if v.dtype.is_floating_point:
                    sh[k].mul_(e).add_(v.detach(), alpha=1.0 - e)
                else:
                    sh[k].copy_(v)
        self.updates += 1

    def state_dict(self, decay: float) -> dict:
        return {k: v.clone() for k, v in self.shadows[float(decay)].items()}

    @contextlib.contextmanager
    def applied(self, module: torch.nn.Module, decay: float):
        live = {k: v.detach().clone() for k, v in module.state_dict().items()}
        module.load_state_dict(self.shadows[float(decay)])
        try:
            yield module
        finally:
            module.load_state_dict(live)
