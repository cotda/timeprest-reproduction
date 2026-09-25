"""Backward on the forward graph with another weight version (PipeDream mechanism, paper_notes §22.5).

PipeDream keeps the autograd graph of every in-flight forward and, before running its backward,
copies the chosen weight version into the module in place (`load_old_params`, then
`torch.autograd.backward` on the stored graph; pipedream/runtime/image_classification/
main_with_runtime.py:410-413, runtime.py:606). Every weight the graph saved (e.g. a conv weight
needed for the input gradient, a BN scale) is then read at that version, while activations and BN
batch statistics stay those of the original forward. Without horizontal stashing (TiMePReSt) the
chosen version is simply the live one.

`GraphForward` gives the same semantics without in-place copies: while the forward runs, every
saved tensor that is (a view of) a stage parameter is replaced by its name and layout, and at
backward time it is rebuilt from the parameter dict set with `use()`. The graph therefore holds no
reference to the forward's weight values.
"""
from __future__ import annotations

import torch
from torch.autograd.graph import saved_tensors_hooks


def _storage_ptr(t: torch.Tensor):
    try:
        return t.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError):   # sparse / wrapper tensors: never a parameter
        return None


class GraphForward:
    """Run `fn()` (a forward using the tensors of `params`) with its weights saved by name."""

    def __init__(self, params: dict[str, torch.Tensor]):
        self.params = params
        self._by_ptr = {}
        for n, p in params.items():
            ptr = _storage_ptr(p)
            if ptr is not None and ptr != 0:
                self._by_ptr[ptr] = n
        self.seen: set[str] = set()
        self._bwd: dict[str, torch.Tensor] | None = None

    def _pack(self, t: torch.Tensor):
        n = self._by_ptr.get(_storage_ptr(t))
        if n is None:
            return t
        p = self.params[n]
        if t.dtype != p.dtype:
            raise RuntimeError(f"saved tensor aliases parameter {n!r} with another dtype")
        self.seen.add(n)
        return ("__param__", n, tuple(t.shape), tuple(t.stride()), t.storage_offset() - p.storage_offset())

    def _unpack(self, obj):
        if isinstance(obj, tuple) and len(obj) == 5 and obj[0] == "__param__":
            if self._bwd is None:
                raise RuntimeError("GraphForward: backward run before use(params)")
            _, n, shape, stride, off = obj
            src = self._bwd[n].detach()
            return src.as_strided(shape, stride, src.storage_offset() + off)
        return obj

    def run(self, fn):
        # a weight used through a non-view copy (autocast to fp16 casts every weight) would be
        # saved as that copy and keep its forward value silently
        if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
            raise RuntimeError("GraphForward does not support autocast (weights would be saved as copies)")
        with saved_tensors_hooks(self._pack, self._unpack):
            return fn()

    def use(self, params: dict[str, torch.Tensor]):
        """Weight version read by the backward of this graph."""
        for n, p in self.params.items():
            q = params[n]
            if q.shape != p.shape or q.stride() != p.stride():
                raise RuntimeError(f"parameter {n!r}: backward version has another layout")
        self._bwd = params


def release_storage(params: dict[str, torch.Tensor]):
    """Free the memory of a weight version no later op will read. A graph may still hold these
    tensors as gradient leaves; the leaves' values are never read by the backward."""
    for p in params.values():
        p.untyped_storage().resize_(0)
