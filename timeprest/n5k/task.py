"""Nutrition task (n5k/PLAN.md mục 4): ingredient detection + grams per ingredient + dish totals.

Layouts (K = number of ingredients known from the train split, T = len(TOTALS)):
  model output  out [B, 2K] = [presence logits (K) | raw grams (K)]
  labels        y   [B, K + T] = [grams per ingredient (K, 0 = absent) | dish totals (T)]
Predicted grams = softplus(raw) * gram_scale; predicted totals = grams @ per_gram, where per_gram[k]
holds (kcal, mass, fat, carb, protein) per gram of ingredient k (mass per gram = 1).

Per-sample loss = w_presence * mean_k BCE(logit_k, gram_k > 0)
                + w_gram     * mean_k SmoothL1(pred_gram_k / scale_k, gram_k / scale_k)
                + w_total    * mean_t SmoothL1((pred_total_t - total_t) / total_std_t, 0)
`loss_sum` returns its sum over the samples (the engine/runtime divide by the mini-batch size M).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..tasks import _upcast

TOTALS = ("kcal", "mass", "fat", "carb", "protein")


class NutritionTask:
    name = "nutrition"

    def __init__(self, per_gram, gram_scale, total_std, w_presence: float = 1.0, w_gram: float = 1.0,
                 w_total: float = 1.0):
        self.per_gram = torch.as_tensor(per_gram, dtype=torch.float64)        # [K, T]
        self.gram_scale = torch.as_tensor(gram_scale, dtype=torch.float64)    # [K]
        self.total_std = torch.as_tensor(total_std, dtype=torch.float64)      # [T]
        K, T = self.per_gram.shape
        if T != len(TOTALS) or self.gram_scale.shape != (K,) or self.total_std.shape != (T,):
            raise ValueError("per_gram [K, 5], gram_scale [K], total_std [5] expected")
        if (self.gram_scale <= 0).any() or (self.total_std <= 0).any():
            raise ValueError("gram_scale and total_std must be > 0")
        self.K = K
        self.w = (float(w_presence), float(w_gram), float(w_total))
        self._cache: dict = {}

    @classmethod
    def from_meta(cls, meta: dict, loss_cfg: dict | None = None) -> "NutritionTask":
        """meta: the dataset statistics written by timeprest.n5k.data (per_gram, gram_scale, total_std)."""
        lc = loss_cfg or {}
        return cls(meta["per_gram"], meta["gram_scale"], meta["total_std"], lc.get("w_presence", 1.0),
                   lc.get("w_gram", 1.0), lc.get("w_total", 1.0))

    # ------------------------------------------------------------------ helpers
    def _consts(self, ref: torch.Tensor):
        key = (ref.device, ref.dtype)
        if key not in self._cache:
            self._cache[key] = tuple(t.to(ref.device, ref.dtype) for t in
                                     (self.per_gram, self.gram_scale, self.total_std))
        return self._cache[key]

    def split_labels(self, y: torch.Tensor):
        return y[:, :self.K], y[:, self.K:]

    def predict(self, out: torch.Tensor) -> dict:
        """Decoded predictions: presence logits [B, K], grams [B, K], totals [B, T]."""
        out = _upcast(out)
        if out.shape[1] != 2 * self.K:
            raise ValueError(f"model output has {out.shape[1]} columns, expected 2K = {2 * self.K}")
        per_gram, scale, _ = self._consts(out)
        logits, raw = out[:, :self.K], out[:, self.K:]
        grams = F.softplus(raw) * scale
        return {"logits": logits, "grams": grams, "totals": grams @ per_gram}

    def per_sample_loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = self.predict(out)
        _, scale, tstd = self._consts(p["grams"])
        y = y.to(p["grams"].dtype)
        grams, totals = self.split_labels(y)
        bce = F.binary_cross_entropy_with_logits(p["logits"], (grams > 0).to(grams.dtype), reduction="none").mean(1)
        gram = F.smooth_l1_loss(p["grams"] / scale, grams / scale, reduction="none").mean(1)
        tot = F.smooth_l1_loss((p["totals"] - totals) / tstd, torch.zeros_like(totals), reduction="none").mean(1)
        wp, wg, wt = self.w
        return wp * bce + wg * gram + wt * tot

    # ------------------------------------------------------------------ task contract (timeprest.tasks)
    def loss_sum(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.per_sample_loss(out, y).sum()

    def train_metric_init(self, device) -> torch.Tensor:
        return torch.zeros((), device=device, dtype=torch.float64)

    def train_metric(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        kcal = self.predict(out)["totals"][:, 0]
        return (kcal - y[:, self.K].to(kcal.dtype)).abs().sum().double()

    def train_summary(self, acc: torch.Tensor, n: int) -> dict:
        return {"train_mae_kcal": acc.item() / max(1, n)}

    def runtime_stats(self, acc: torch.Tensor) -> dict:
        return {"abs_err_kcal": acc.item()}

    def eval_start(self) -> dict:
        return {"loss": 0.0, "abs_err": torch.zeros(len(TOTALS), dtype=torch.float64),
                "sum_true": torch.zeros(len(TOTALS), dtype=torch.float64), "n": 0,
                "gram_err": 0.0, "gram_n": 0, "scores": [], "present": []}

    @torch.no_grad()
    def eval_update(self, st: dict, out: torch.Tensor, y: torch.Tensor):
        p = self.predict(out)
        y = y.to(p["grams"].dtype)
        grams, totals = self.split_labels(y)
        st["loss"] += self.per_sample_loss(out, y).sum().item()
        st["abs_err"] += (p["totals"] - totals).abs().sum(0).double().cpu()
        st["sum_true"] += totals.sum(0).double().cpu()
        st["n"] += y.shape[0]
        mask = grams > 0
        st["gram_err"] += (p["grams"] - grams).abs()[mask].sum().item()
        st["gram_n"] += int(mask.sum().item())
        st["scores"].append(torch.sigmoid(p["logits"]).float().cpu())
        st["present"].append(mask.cpu())

    def eval_result(self, st: dict) -> dict:
        n = max(1, st["n"])
        mae = st["abs_err"] / n
        res = {"loss": st["loss"] / n}
        for k, name in enumerate(TOTALS):
            res[f"mae_{name}"] = mae[k].item()
        mean_kcal = st["sum_true"][0].item() / n
        res["pmae_kcal"] = mae[0].item() / mean_kcal * 100.0 if mean_kcal > 0 else float("nan")
        res["mae_gram_present"] = st["gram_err"] / max(1, st["gram_n"])
        scores, present = torch.cat(st["scores"]), torch.cat(st["present"])
        res["ingr_f1"] = micro_f1(scores >= 0.5, present)
        res["ingr_map"] = mean_average_precision(scores, present)
        return res

    def dummy_labels(self, out: torch.Tensor) -> torch.Tensor:
        return out.new_zeros(out.shape[0], self.K + len(TOTALS))


def micro_f1(pred: torch.Tensor, true: torch.Tensor) -> float:
    tp = (pred & true).sum().item()
    fp = (pred & ~true).sum().item()
    fn = (~pred & true).sum().item()
    return 2 * tp / (2 * tp + fp + fn) * 100.0 if tp + fp + fn else float("nan")


def mean_average_precision(scores: torch.Tensor, true: torch.Tensor) -> float:
    """Macro AP (%) over the ingredients with at least one positive sample (ties broken by order)."""
    aps = []
    for k in range(scores.shape[1]):
        t = true[:, k]
        if not t.any():
            continue
        order = torch.argsort(scores[:, k], descending=True, stable=True)
        hits = t[order].double()
        precision = hits.cumsum(0) / torch.arange(1, len(hits) + 1, dtype=torch.float64)
        aps.append((precision * hits).sum().item() / hits.sum().item())
    return sum(aps) / len(aps) * 100.0 if aps else float("nan")
