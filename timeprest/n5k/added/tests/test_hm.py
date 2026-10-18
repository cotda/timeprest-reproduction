"""HM (n5k/added/README.md, 2026-10-17 22:25 / 22:31): training only on the dishes with HandMasks. The data are our
splits restricted to those dishes (labels untouched for HM1), HM2 swaps the ingredient labels for the HandMasks
groups (grams = region masses, fallbacks, per-gram medians), and both train and locate end to end on the fake
Nutrition5k folder, on 1 process (engine) and on 2 (pipeline)."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from timeprest.config import load_config
from timeprest.n5k.added import hm
from timeprest.n5k.data import load_datasets
from timeprest.n5k.tests.test_train import INGR, overrides, write_fake

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DISHES = [f"dish_train_{i}" for i in range(10)] + [f"dish_val_{i}" for i in range(4)] + [f"dish_test_{i}" for i in range(5)]


def write_handmasks(root, hw=(40, 40)):
    """Two regions per dish (left gA: 10 g, 30 kcal; right gB: 10 g, 10 kcal); dish_train_1's gB has no mass."""
    (root / "masks").mkdir(parents=True)
    rows = []
    for d in DISHES + ["dish_elsewhere"]:
        m = np.zeros(hw, np.uint8)
        m[:, : hw[1] // 2] = 1
        m[:, hw[1] // 2:] = 2
        Image.fromarray(m).save(root / "masks" / f"{d}.png")
        nb = 0 if d == "dish_train_1" else 1
        rows += [[d, "x", f"{d}a", "gA", 1, "", "", 0, 1, 10.0, 30.0, 1.0, 2.0, 3.0, ""],
                 [d, "x", f"{d}b", "gB", 2, "", "", 0, nb, 10.0, 10.0, 0.5, 1.0, 0.5, ""]]
    with open(root / "ingredient_nutrition.csv", "w", newline="") as f:
        csv.writer(f).writerows([["dish_id", "split", "ann_id", "annotation_name", "label_value", "mask", "rgb_image",
                                  "mask_pixels", "has_nutrition", "mass", "calories", "fat", "carb", "protein",
                                  "cutout"]] + rows)


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("n5k_hm")
    write_fake(tmp)
    write_handmasks(tmp / "hm")
    (tmp / "groups.yaml").write_text(yaml.safe_dump({"groups": {k: {"group": ("gA", "gB", None)[j % 3]}
                                                                for j, k in enumerate(INGR)}}))
    return tmp


def hm_overrides(fake, tmp_path, extra=()):
    return overrides(fake, ("training.lr=0.001", "training.epochs=1", "data.image_size=64",
                            f"data.cache_dir={fake / 'cache'}", f"handmasks.root_search=[{fake / 'hm'}]",
                            "handmasks.mask_hw=[40,40]", f"handmasks.groups_file={fake / 'groups.yaml'}",
                            f"handmasks.dir={tmp_path / 'loc' / '{name}'}", f"output.dir={tmp_path / 'runs' / '{name}'}",
                            *extra))


def cfg(fake, tmp_path, system, extra=()):
    return load_config(os.path.join(ROOT, "n5k", "added", "configs", f"hm_{system}.yaml"),
                       hm_overrides(fake, tmp_path, extra))


def test_hm1_keeps_the_labels_of_the_handmasks_dishes(fake, tmp_path):
    c = cfg(fake, tmp_path, "single")
    full = load_datasets(c["data"], log=lambda *_: None)
    train, val, test, meta = hm.hm_data(c, log=lambda *_: None)
    assert meta["hm"]["dishes"] == {"train": 10, "val": 4, "test": 5}
    assert meta["ids"]["test"] == [f"dish_test_{i}" for i in range(5)] and len(meta["ingredients"]) == len(INGR)
    rows = [full[3]["ids"]["train"].index(d) for d in meta["ids"]["train"]]
    torch.testing.assert_close(train.targets, full[0].targets[rows])


def test_hm2_group_labels(fake, tmp_path):
    c = cfg(fake, tmp_path, "single", ["added.hm.labels=groups"])
    train, val, test, meta = hm.hm_data(c, log=lambda *_: None)
    K = len(INGR)
    full = load_datasets(c["data"], log=lambda *_: None)
    assert [d["id"] for d in meta["ingredients"]] == ["gA", "gB"] and train.targets.shape[1] == 2 + 5
    i1 = meta["ids"]["train"].index("dish_train_1")
    others = [i for i in range(len(train)) if i != i1]
    assert (train.targets[others, :2] == 10.0).all()                  # summed region masses
    # dish_train_1: gB has no mass -> grams of the ingredients mapped to gB (ingr_2, ingr_5), else the train median
    k_b = [j for j, k in enumerate(INGR) if j % 3 == 1]
    src = full[0].targets[full[3]["ids"]["train"].index("dish_train_1")]
    expect = float(src[k_b].sum()) or 10.0
    assert float(train.targets[i1, 1]) == pytest.approx(expect)
    torch.testing.assert_close(train.targets[:, 2:], full[0].targets[[full[3]["ids"]["train"].index(d)
                                                                      for d in meta["ids"]["train"]], K:])
    np.testing.assert_allclose(meta["per_gram"], [[3.0, 1.0, 0.1, 0.2, 0.3], [1.0, 1.0, 0.05, 0.1, 0.05]])
    assert meta["gram_scale"][0] == pytest.approx(10.0)


@pytest.mark.parametrize("labels", ["ingredients", "groups"])
def test_hm_engine_train_and_locate(fake, tmp_path, labels):
    c = cfg(fake, tmp_path, "single", [f"added.hm.labels={labels}"])
    p = subprocess.run([sys.executable, "-m", "timeprest.n5k.added.hm", "train", "--force", "--config",
                        os.path.join("n5k", "added", "configs", "hm_single.yaml"),
                        "--set", *hm_overrides(fake, tmp_path, [f"added.hm.labels={labels}"])],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    s = json.load(open(os.path.join(c["output"]["dir"], "summary.json")))
    assert s["hm"]["labels"] == labels and s["hm"]["dishes"]["test"] == 5
    res = hm.locate_run(hm.locate_cfg(c), c["output"]["dir"], torch.device("cpu"), data=hm.hm_data(c, log=lambda *_: None),
                        log=lambda *_: None)
    assert res["dish_counts"]["test"] == 5 and res["test"]["n_dishes"] == 5
    assert set(res["test"]["m3"]["per_group"]) == {"gA", "gB"}


def test_hm2_pipeline_two_processes(fake, tmp_path):
    c = cfg(fake, tmp_path, "timeprest", ["added.hm.labels=groups"])
    cmd = [sys.executable, "-m", "timeprest.dist.launch_local", "--nproc", "2", "-m", "timeprest.n5k.added.hm", "train",
           "--force", "--config", os.path.join("n5k", "added", "configs", "hm_timeprest.yaml"),
           "--set", *hm_overrides(fake, tmp_path, ["added.hm.labels=groups"])]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, (p.stdout + p.stderr)[-4000:]
    s = json.load(open(os.path.join(c["output"]["dir"], "summary.json")))
    assert s["system"] == "timeprest" and s["hm"]["labels"] == "groups" and s["epochs_done"] == 1
    res = hm.locate_run(hm.locate_cfg(c), c["output"]["dir"], torch.device("cpu"), data=hm.hm_data(c, log=lambda *_: None),
                        log=lambda *_: None)
    assert res["test"]["n_dishes"] == 5 and res["test"]["m3"]["n"] == 10
