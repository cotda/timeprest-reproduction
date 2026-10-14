"""B3 (n5k/PLAN.md mục 9): torchvision ResNet-50 weights mapped into `resnet50_blocks`, 4-channel
first conv, CAM head. Uses a randomly initialised torchvision model (no download in tests)."""
import torch
import torch.nn as nn
import torchvision

from timeprest.models import build_stages
from timeprest.n5k.model import N5kHead, build_n5k_resnet50, load_torchvision_weights, priors_from_meta
from timeprest.n5k.task import NutritionTask

TOL = dict(rtol=1e-5, atol=1e-5)   # float32, same op order up to the conv implementations


def tv_model(seed=0):
    torch.manual_seed(seed)
    m = torchvision.models.resnet50(weights=None)
    with torch.no_grad():    # non-trivial BN statistics so a wrong buffer mapping would show
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.1, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
                mod.weight.uniform_(0.5, 1.5)
    return m.eval()


def tv_features(m, x):
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    return x


def test_rgb_features_match_torchvision():
    tv = tv_model()
    blocks = build_n5k_resnet50({"num_ingredients": 5, "in_channels": 3, "pretrained": None})
    load_torchvision_weights(blocks, tv.state_dict())
    body = nn.Sequential(*blocks[:-1]).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(body(x), tv_features(tv, x), **TOL)


def test_depth_channel_is_rgb_mean_and_zero_depth_gives_rgb_model():
    tv = tv_model(1)
    b3 = build_n5k_resnet50({"num_ingredients": 5, "in_channels": 3})
    b4 = build_n5k_resnet50({"num_ingredients": 5, "in_channels": 4})
    load_torchvision_weights(b3, tv.state_dict())
    load_torchvision_weights(b4, tv.state_dict())
    w = b4[0][0].weight
    torch.testing.assert_close(w[:, 3], w[:, :3].mean(1))
    x = torch.randn(2, 3, 64, 64)
    x4 = torch.cat([x, torch.zeros(2, 1, 64, 64)], 1)
    with torch.no_grad():
        torch.testing.assert_close(nn.Sequential(*b4[:-1]).eval()(x4), nn.Sequential(*b3[:-1]).eval()(x), **TOL)


def test_head_layout_and_initial_priors():
    K = 6
    p = [0.05, 0.2, 0.5, 0.01, 0.3, 0.9]
    g = [0.02, 0.1, 0.4, 0.005, 0.2, 0.8]
    head = N5kHead(16, K, p, g)
    feat = torch.zeros(3, 16, 7, 7)               # zero features: output = biases
    out = head(feat)
    assert out.shape == (3, 2 * K)
    torch.testing.assert_close(torch.sigmoid(out[0, :K]), torch.tensor(p))
    torch.testing.assert_close(nn.functional.softplus(out[0, K:]), torch.tensor(g))
    pm, gm = head.maps(torch.randn(3, 16, 7, 7))
    assert pm.shape == gm.shape == (3, K, 7, 7)


def test_priors_from_meta_and_initial_loss_is_moderate():
    meta = {"ingredients": [{"id": "a"}, {"id": "b"}], "gram_scale": [100.0, 10.0]}
    y = torch.tensor([[100.0, 0, 0, 0, 0, 0, 0], [50.0, 10.0, 0, 0, 0, 0, 0], [0.0, 20.0, 0, 0, 0, 0, 0],
                      [0.0, 0.0, 0, 0, 0, 0, 0]])
    pr = priors_from_meta(meta, y)
    torch.testing.assert_close(torch.tensor(pr["presence_prior"]), torch.tensor([0.5, 0.5]), check_dtype=False)
    torch.testing.assert_close(torch.tensor(pr["gram_prior"]), torch.tensor([150 / 400, 30 / 40]), check_dtype=False)
    head = N5kHead(8, 2, **pr)
    task = NutritionTask([[1, 1, 0, 0, 0], [2, 1, 0, 0, 0]], meta["gram_scale"], [100, 50, 1, 1, 1])
    loss = task.loss_sum(head(torch.zeros(4, 8, 7, 7)), y) / 4
    assert 0 < loss.item() < 3     # BCE at the prior ~ ln 2; regression terms O(1)


def test_registered_model_partitions_with_4_channel_sample():
    cfg = {"name": "n5k_resnet50", "num_ingredients": 4, "in_channels": 4, "pretrained": None}
    stages, bounds = build_stages(cfg, 2, "auto", torch.zeros(1, 4, 64, 64))
    assert bounds[0] == 0 and bounds[-1] == 18 and 0 < bounds[1] < 18
    x = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        out = stages[1].eval()(stages[0].eval()(x))
    assert out.shape == (2, 8)


def test_direct_totals_head_starts_at_train_mean():
    K, mean = 3, [500.0, 300.0, 20.0, 40.0, 25.0]
    head = N5kHead(16, K, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], direct_totals=True, total_mean=mean)
    out = head(torch.zeros(2, 16, 7, 7))
    assert out.shape == (2, 2 * K + 5)
    assert head.total_map(torch.randn(2, 16, 7, 7)).shape == (2, 5, 7, 7)
    task = NutritionTask(torch.rand(K, 5).tolist(), [10.0] * K, [100, 50, 1, 1, 1], total_scale=mean)
    torch.testing.assert_close(task.predict(out)["totals"][0], torch.tensor(mean), check_dtype=False)
    cfg = {"name": "n5k_resnet50", "num_ingredients": 4, "in_channels": 4, "pretrained": None,
           "direct_totals": True, "total_mean": mean}
    stages, _ = build_stages(cfg, 2, [0, 8, 18], torch.zeros(1, 4, 64, 64))
    with torch.no_grad():
        assert stages[1].eval()(stages[0].eval()(torch.randn(2, 4, 64, 64))).shape == (2, 13)
