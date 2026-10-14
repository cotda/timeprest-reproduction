"""Nutrition model (n5k/PLAN.md mục 4): ResNet-50 (`timeprest.models.resnet50_blocks`, stem
"imagenet") with ImageNet weights from torchvision, a 4-channel first conv (RGB + depth) and a
CAM-style head. Registered as model name "n5k_resnet50" so `build_stages` can partition it.

Head on the 2048×h×w feature map: two 1×1 convs give K presence maps and K gram maps; global
average pooling turns them into presence logits and raw grams -> output [B, 2K] (task layout).
The presence maps are the class activation maps used for the calorie map (B7).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..models import MODEL_BUILDERS, Bottleneck, resnet50_blocks

NAME = "n5k_resnet50"


class N5kHead(nn.Module):
    """Output [B, 2K] = [presence logits | raw grams]; with direct_totals [B, 2K + T] adds raw dish totals
    (task: totals = softplus(raw) * train mean). Every output is the spatial mean of a 1x1-conv map (CAM)."""

    def __init__(self, cin: int, K: int, presence_prior=None, gram_prior=None, direct_totals: bool = False,
                 total_mean=None, T: int = 5):
        super().__init__()
        self.K = K
        self.presence = nn.Conv2d(cin, K, 1)
        self.gram = nn.Conv2d(cin, K, 1)
        self.total = nn.Conv2d(cin, T, 1) if direct_totals else None
        for conv in (self.presence, self.gram, self.total):
            if conv is not None:
                nn.init.normal_(conv.weight, 0, 0.01)
                nn.init.zeros_(conv.bias)
        with torch.no_grad():
            # start at the train prevalence p_k (logit) and at E[grams_k / scale_k] = gram_prior_k
            if presence_prior is not None:
                p = torch.as_tensor(presence_prior, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
                self.presence.bias.copy_(torch.log(p / (1 - p)))
            if gram_prior is not None:
                m = torch.as_tensor(gram_prior, dtype=torch.float32).clamp_min(1e-4)
                self.gram.bias.copy_(torch.log(torch.expm1(m)))          # softplus^-1
            if self.total is not None:                                    # start at the train mean
                self.total.bias.fill_(math.log(math.expm1(1.0)))

    def maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Presence (CAM) and gram maps [B, K, h, w]."""
        return self.presence(x), self.gram(x)

    def total_map(self, x: torch.Tensor) -> torch.Tensor:
        """Raw dish-total maps [B, T, h, w] (direct_totals only)."""
        return self.total(x)

    def forward(self, x):
        p, g = self.maps(x)
        outs = [p.mean((2, 3)), g.mean((2, 3))]
        if self.total is not None:
            outs.append(self.total(x).mean((2, 3)))
        return torch.cat(outs, 1)


def torchvision_key_map(blocks: list[nn.Module]) -> dict:
    """our parameter/buffer name -> torchvision resnet50 name, for every block except the head."""
    out = {"0.0.weight": "conv1.weight"}
    for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
        out[f"0.1.{k}"] = f"bn1.{k}"
    b = 1
    for layer, n in zip(range(1, 5), (3, 4, 6, 3)):
        for i in range(n):
            blk = blocks[b]
            assert isinstance(blk, Bottleneck)
            src = f"layer{layer}.{i}"
            for name in blk.state_dict():
                if name.startswith("shortcut."):
                    tv = name.replace("shortcut.", "downsample.")
                else:
                    tv = name
                out[f"{b}.{name}"] = f"{src}.{tv}"
            b += 1
    return out


def load_torchvision_weights(blocks: list[nn.Module], tv_state: dict, depth_init: str = "rgb_mean"):
    """Copy torchvision ResNet-50 weights into the blocks. With a 4-channel first conv the RGB
    filters are copied and the depth filter is the mean of the three RGB filters (mục 4)."""
    flat = nn.Sequential(*blocks[:-1])
    own = flat.state_dict()
    new = {}
    for ours, tv in torchvision_key_map(blocks).items():
        w = tv_state[tv]
        if ours == "0.0.weight" and own[ours].shape[1] == 4 and w.shape[1] == 3:
            if depth_init != "rgb_mean":
                raise ValueError(f"unknown depth_init {depth_init!r}")
            w = torch.cat([w, w.mean(1, keepdim=True)], 1)
        if own[ours].shape != w.shape:
            raise ValueError(f"{ours} <- {tv}: shape {tuple(w.shape)} != {tuple(own[ours].shape)}")
        new[ours] = w
    missing = set(own) - set(new)
    if missing:
        raise ValueError(f"no torchvision weight for {sorted(missing)[:5]}")
    flat.load_state_dict(new)


def build_n5k_resnet50(model_cfg: dict) -> list[nn.Module]:
    """model_cfg: num_ingredients (K), in_channels (4), pretrained (torchvision weights name, e.g.
    IMAGENET1K_V2, or null), presence_prior / gram_prior (train statistics, injected by the trainer)."""
    K = int(model_cfg["num_ingredients"])
    blocks = resnet50_blocks(num_classes=1, stem="imagenet")
    cin = int(model_cfg.get("in_channels", 4))
    if cin != 3:
        old = blocks[0][0]
        blocks[0][0] = nn.Conv2d(cin, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        nn.init.kaiming_normal_(blocks[0][0].weight, mode="fan_out", nonlinearity="relu")
    if model_cfg.get("pretrained"):
        import torchvision
        tv = torchvision.models.resnet50(weights=model_cfg["pretrained"])
        load_torchvision_weights(blocks, tv.state_dict(), model_cfg.get("depth_init", "rgb_mean"))
    blocks[-1] = N5kHead(2048, K, model_cfg.get("presence_prior"), model_cfg.get("gram_prior"),
                         bool(model_cfg.get("direct_totals", False)), model_cfg.get("total_mean"))
    return blocks


def priors_from_meta(meta: dict, train_targets: torch.Tensor) -> dict:
    """Head initialisation from the train labels: prevalence p_k, E[grams_k] / scale_k and the mean dish
    totals (decoding scale of the direct-totals branch)."""
    K = len(meta["ingredients"])
    grams = train_targets[:, :K].double()
    scale = torch.as_tensor(meta["gram_scale"], dtype=torch.float64)
    return {"presence_prior": (grams > 0).double().mean(0).tolist(),
            "gram_prior": (grams / scale).mean(0).tolist(),
            "total_mean": train_targets[:, K:].double().mean(0).clamp_min(1e-3).tolist()}


MODEL_BUILDERS[NAME] = build_n5k_resnet50
