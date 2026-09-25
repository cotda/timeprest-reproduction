"""Tiny-ImageNet-200 loader on a miniature copy of the original folder layout."""
import numpy as np
import torch
from PIL import Image

from timeprest.data import build_datasets, dataset_targets
from timeprest.models import build_blocks, build_stages


def make_tree(root, wnids=("n02", "n01"), per_class=3, val=(("v0.JPEG", "n01"), ("v1.JPEG", "n02"))):
    d = root / "kaggle_input" / "some-dataset" / "tiny-imagenet-200"
    (d / "val" / "images").mkdir(parents=True)
    (d / "wnids.txt").write_text("\n".join(wnids) + "\n")
    rng = np.random.default_rng(0)
    for w in wnids:
        (d / "train" / w / "images").mkdir(parents=True)
        (d / "train" / w / f"{w}_boxes.txt").write_text("")
        for k in range(per_class):
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / "train" / w / "images" / f"{w}_{k}.JPEG")
    lines = []
    for name, w in val:
        Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(d / "val" / "images" / name)  # grayscale
        lines.append(f"{name}\t{w}\t0\t0\t63\t63")
    (d / "val" / "val_annotations.txt").write_text("\n".join(lines) + "\n")
    return root / "kaggle_input"


def test_tinyimagenet_layout_labels_and_cache(tmp_path):
    root = make_tree(tmp_path)
    cfg = {"dataset": "tinyimagenet", "root": str(root), "cache_dir": str(tmp_path / "cache"), "augment": True}
    train, test = build_datasets(cfg, 2, seed=0)
    assert len(train) == 6 and len(test) == 2
    # class index = position in sorted wnids ("n01" -> 0, "n02" -> 1), val labels from the annotations
    assert dataset_targets(train).tolist() == [0, 0, 0, 1, 1, 1]
    assert dataset_targets(test).tolist() == [0, 1]
    x, y = train[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32 and y == 0
    xt, _ = test[0]                                   # grayscale val image converted to RGB
    assert xt.shape == (3, 64, 64)
    assert torch.equal(test[1][0], test[1][0])        # no augmentation on the evaluation split
    # second call reads the .npy cache (the image folder may disappear)
    import shutil
    shutil.rmtree(root / "some-dataset" / "tiny-imagenet-200" / "train")
    train2, _ = build_datasets(cfg, 2, seed=0)
    assert torch.equal(train2.x, train.x)


def test_vgg_global_pool_head_on_64px():
    cfg = {"name": "vgg16_bn", "num_classes": 200, "width": 1 / 16, "global_pool": True}
    stages, bounds = build_stages(cfg, 2, "auto", torch.zeros(1, 3, 64, 64))
    x = torch.zeros(2, 3, 64, 64)
    for m in stages:
        x = m.eval()(x)
    assert x.shape == (2, 200)
    n64 = sum(p.numel() for p in build_blocks(cfg)[-1].parameters())
    n32 = sum(p.numel() for p in build_blocks(dict(cfg, global_pool=False))[-1].parameters())
    assert n64 == n32                                 # same head size as the CIFAR model


def test_channel_stats_match_full_computation():
    from timeprest.data import _channel_stats
    x = np.random.default_rng(1).integers(0, 256, (37, 8, 8, 3), dtype=np.uint8)
    mean, std = _channel_stats(x, chunk=5)
    f = x.reshape(-1, 3).astype(np.float64) / 255.0
    np.testing.assert_allclose(mean, f.mean(0), rtol=1e-12)
    np.testing.assert_allclose(std, f.std(0), rtol=1e-9)
