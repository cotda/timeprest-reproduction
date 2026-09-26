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
    """Linear classifier. global_pool: average the final feature map first (inputs larger than
    32x32, e.g. Tiny-ImageNet 64x64 -> 2x2x512), so the head keeps the CIFAR parameter count."""

    def __init__(self, cin: int, num_classes: int, global_pool: bool = False):
        super().__init__()
        self.global_pool = global_pool
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        if self.global_pool:
            x = x.mean((2, 3))
        return self.fc(torch.flatten(x, 1))


def vgg16_bn_blocks(num_classes: int = 100, width: float = 1.0, global_pool: bool = False) -> list[nn.Module]:
    blocks: list[nn.Module] = []
    cin = 3
    for v in VGG16_CFG:
        if v == "M":
            blocks.append(nn.MaxPool2d(2, 2))
        else:
            cout = max(4, int(round(v * width)))
            blocks.append(ConvBNReLU(cin, cout))
            cin = cout
    blocks.append(Head(cin, num_classes, global_pool))
    _init(blocks)
    return blocks


class Bottleneck(nn.Module):
    """ResNet-50 bottleneck (torchvision v1.5 layout: stride on the 3x3 conv, projection shortcut
    when the shape changes)."""
    expansion = 4

    def __init__(self, cin: int, planes: int, stride: int = 1):
        super().__init__()
        cout = planes * self.expansion
        self.conv1 = nn.Conv2d(cin, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        identity = x if self.shortcut is None else self.shortcut(x)
        return self.relu(out + identity)       # out-of-place add: x may be saved by the kept graph


class ResNetHead(nn.Module):
    def __init__(self, cin: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(x.mean((2, 3)))          # global average pooling (= AdaptiveAvgPool2d(1))


def resnet50_blocks(num_classes: int = 200, width: float = 1.0, stem: str = "conv3_pool") -> list[nn.Module]:
    """ResNet-50 [3, 4, 6, 3] as a flat list: stem, 16 bottlenecks, head (paper does not give the
    variant, paper_notes §25). stem "conv3_pool": 3x3 conv stride 1 + 2x2 max-pool (64x64 input ->
    32x32 into layer1, 4x4 into the head); "imagenet": 7x7 stride-2 conv + 3x3 stride-2 max-pool."""
    c = lambda v: max(4, int(round(v * width)))
    w0 = c(64)
    if stem == "conv3_pool":
        blocks: list[nn.Module] = [nn.Sequential(nn.Conv2d(3, w0, 3, padding=1, bias=False), nn.BatchNorm2d(w0),
                                                 nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))]
    elif stem == "imagenet":
        blocks = [nn.Sequential(nn.Conv2d(3, w0, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(w0),
                                nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))]
    else:
        raise ValueError(f"unknown ResNet stem {stem!r}")
    cin = w0
    for planes, n, stride in ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)):
        for k in range(n):
            blocks.append(Bottleneck(cin, c(planes), stride if k == 0 else 1))
            cin = c(planes) * Bottleneck.expansion
    blocks.append(ResNetHead(cin, num_classes))
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
    if name in ("vgg16_bn_cifar", "vgg16_bn"):
        return vgg16_bn_blocks(model_cfg["num_classes"], model_cfg.get("width", 1.0),
                               model_cfg.get("global_pool", False))
    if name == "resnet50":
        return resnet50_blocks(model_cfg["num_classes"], model_cfg.get("width", 1.0), model_cfg.get("stem", "conv3_pool"))
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
