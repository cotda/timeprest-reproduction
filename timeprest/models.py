"""Models expressed as a flat list of blocks so they can be cut into pipeline stages."""
from __future__ import annotations

import torch
import torch.nn as nn

# VGG-16 layout (13 conv + classifier). CIFAR-style: 32x32 input, single Linear head, no dropout.
# Implementation choice (paper does not specify the variant, see paper_notes.md §8.3).
VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Head(nn.Module):
    def __init__(self, cin: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(x, 1))


def vgg16_bn_blocks(num_classes: int = 100, width: float = 1.0) -> list[nn.Module]:
    blocks: list[nn.Module] = []
    cin = 3
    for v in VGG16_CFG:
        if v == "M":
            blocks.append(nn.MaxPool2d(2, 2))
        else:
            cout = max(4, int(round(v * width)))
            blocks.append(ConvBNReLU(cin, cout))
            cin = cout
    blocks.append(Head(cin, num_classes))
    _init(blocks)
    return blocks


def mlp_blocks(in_dim: int = 12, hidden: int = 16, num_classes: int = 5, depth: int = 4) -> list[nn.Module]:
    """Tiny model without BN/dropout used by CPU unit tests."""
    blocks: list[nn.Module] = [nn.Flatten()]
    d = in_dim
    for _ in range(depth - 1):
        blocks.append(nn.Sequential(nn.Linear(d, hidden), nn.Tanh()))
        d = hidden
    blocks.append(nn.Linear(d, num_classes))
    return blocks


def _init(blocks):
    for b in blocks:
        for m in b.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def build_blocks(model_cfg: dict) -> list[nn.Module]:
    name = model_cfg["name"]
    if name == "vgg16_bn_cifar":
        return vgg16_bn_blocks(model_cfg["num_classes"], model_cfg.get("width", 1.0))
    if name == "mlp":
        return mlp_blocks(model_cfg.get("in_dim", 12), model_cfg.get("hidden", 16),
                          model_cfg["num_classes"], model_cfg.get("depth", 4))
    raise ValueError(f"unknown model {name!r}")


def block_macs(blocks: list[nn.Module], sample: torch.Tensor) -> list[int]:
    """Multiply-accumulates per block for one sample (conv + linear only)."""
    macs = []
    x = sample
    with torch.no_grad():
        for b in blocks:
            count = [0]

            def hook(m, inp, out):
                if isinstance(m, nn.Conv2d):
                    k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
                    count[0] += out.numel() // out.shape[0] * k
                elif isinstance(m, nn.Linear):
                    count[0] += m.in_features * m.out_features

            hs = [m.register_forward_hook(hook) for m in b.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            was_training = b.training
            b.eval()
            x = b(x)
            b.train(was_training)
            for h in hs:
                h.remove()
            macs.append(count[0])
    return macs


def balanced_partition(costs: list[int], num_stages: int) -> list[int]:
    """Boundaries [0, c1, ..., len] splitting costs into contiguous parts minimising the max part."""
    n = len(costs)
    if num_stages > n:
        raise ValueError("more stages than blocks")
    prefix = [0]
    for c in costs:
        prefix.append(prefix[-1] + c)
    INF = float("inf")
    # dp[k][i] = best max-cost splitting first i blocks into k parts
    dp = [[INF] * (n + 1) for _ in range(num_stages + 1)]
    cut = [[0] * (n + 1) for _ in range(num_stages + 1)]
    dp[0][0] = 0
    for k in range(1, num_stages + 1):
        for i in range(k, n + 1):
            for j in range(k - 1, i):
                val = max(dp[k - 1][j], prefix[i] - prefix[j])
                if val < dp[k][i]:
                    dp[k][i], cut[k][i] = val, j
    bounds = [n]
    i = n
    for k in range(num_stages, 0, -1):
        i = cut[k][i]
        bounds.append(i)
    return bounds[::-1]


def build_stages(model_cfg: dict, num_stages: int, partition="auto",
                 sample: torch.Tensor | None = None) -> tuple[list[nn.Sequential], list[int]]:
    blocks = build_blocks(model_cfg)
    if partition == "auto":
        if sample is None:
            sample = torch.zeros(1, 3, 32, 32)
        # weight each block's MACs; 1 extra unit keeps parameter-free blocks from being empty stages
        costs = [c + 1 for c in block_macs(blocks, sample)]
        bounds = balanced_partition(costs, num_stages)
    else:
        bounds = list(partition)
        if bounds[0] != 0 or bounds[-1] != len(blocks) or len(bounds) != num_stages + 1:
            raise ValueError(f"partition must be [0, ..., {len(blocks)}] with {num_stages + 1} entries")
    stages = [nn.Sequential(*blocks[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
    return stages, bounds
