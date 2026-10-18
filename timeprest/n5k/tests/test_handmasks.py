"""B7b (n5k/PLAN.md mục 10, 2026-10-17 21:32 / 21:40): HandMasks location metrics. A map placed on the right region
scores best and a uniform map scores the area baseline (M1, M2), the pooled CAM peak hits its group (M3); end to end
on the fake Nutrition5k folder with fake HandMasks, our train dishes are skipped and never evaluated."""
import csv

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from timeprest.n5k import handmasks
from timeprest.n5k.train import EngineTrainer

from .test_train import INGR, cfg_for, write_fake

RULES = {"m1_margin": 0.05, "m2_rel_gain": 0.10, "m3_margin": 0.10, "alpha": 0.01}


def two_region_mask(h=48, w=64):
    m = np.zeros((h, w), np.uint8)
    m[: h // 2, : w // 2] = 1                    # region 1: top-left quadrant
    m[h // 2:, w // 2:] = 2                      # region 2: bottom-right quadrant
    return m


def test_occupancy_and_metrics_on_a_synthetic_dish():
    occ = handmasks.cell_occupancy(two_region_mask(), [1, 2], 32, (2, 2))
    assert occ.shape == (3, 2, 2)
    np.testing.assert_allclose(occ[0], [[1, 0], [0, 0]])
    np.testing.assert_allclose(occ[1], [[0, 0], [0, 1]])
    np.testing.assert_allclose(occ[2], [[1, 0], [0, 1]])
    regions = [{"label": 1, "group": "gA", "kcal": 100.0, "mass": 50.0},
               {"label": 2, "group": "gB", "kcal": 10.0, "mass": 50.0}]
    right = np.array([[100.0, 0.0], [0.0, 10.0]])
    uniform = np.full((2, 2), 27.5)
    cam = np.zeros((3, 2, 2))
    cam[0, 0, 0] = 1.0                           # ingredient 0 (group gA) peaks on region 1
    cam[1, 1, 1] = 1.0                           # ingredient 1 (group gB) peaks on region 2
    cam[2] = 5.0                                 # ingredient 2 has no group: ignored
    r = handmasks.dish_metrics({"cmap": right, "tmap": uniform}, occ, regions, cam, np.ones(3), ["gA", "gB", None])
    assert r["area_food"] == pytest.approx(0.5)
    assert r["cmap_m1_in"] == pytest.approx(1.0) and r["tmap_m1_in"] == pytest.approx(0.5)
    assert r["cmap_m2_mae"] == pytest.approx(0.0)
    assert r["tmap_m2_mae"] == pytest.approx(r["area_m2_mae"]) == pytest.approx(abs(0.5 - 100 / 110))
    assert [(h["group"], h["hit"], h["chance"]) for h in r["m3"]] == [("gA", True, 0.5), ("gB", True, 0.5)]


def test_summarize_rules_pass_for_the_right_map_only():
    occ = handmasks.cell_occupancy(two_region_mask(), [1, 2], 32, (2, 2))
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(12):
        a, b = rng.uniform(50, 150), rng.uniform(5, 20)
        regions = [{"label": 1, "group": "gA", "kcal": a, "mass": 1.0}, {"label": 2, "group": "gB", "kcal": b, "mass": 1.0}]
        maps = {"cmap": np.array([[a, 1.0], [1.0, b]]), "tmap": np.full((2, 2), 1.0)}
        rows.append(handmasks.dish_metrics(maps, occ, regions, None, None, None))
    s = handmasks.summarize(rows, RULES)
    assert s["cmap_m1"]["pass"] and s["cmap_m2"]["pass"] and s["cmap_m2"]["n"] == 12
    assert not s["tmap_m1"]["pass"] and not s["tmap_m2"]["pass"]
    assert "m3" not in s


def test_groups_file_must_cover_every_ingredient(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump({"groups": {"ingr_1": {"group": "gA"}}}))
    with pytest.raises(KeyError):
        handmasks.load_groups(str(p), [{"id": "ingr_1"}, {"id": "ingr_2"}], {"gA"})
    p.write_text(yaml.safe_dump({"groups": {"ingr_1": {"group": "nope"}, "ingr_2": {"group": None}}}))
    with pytest.raises(ValueError):
        handmasks.load_groups(str(p), [{"id": "ingr_1"}, {"id": "ingr_2"}], {"gA"})


def write_fake_handmasks(root, dish_ids, hw=(40, 40)):
    """HandMasks-like folder: two regions per dish (left half gA, right half gB), true kcal per region."""
    (root / "masks").mkdir(parents=True)
    rows = []
    for d in dish_ids:
        m = np.zeros(hw, np.uint8)
        m[:, : hw[1] // 2] = 1
        m[:, hw[1] // 2:] = 2
        Image.fromarray(m).save(root / "masks" / f"{d}.png")
        rows += [[d, "x", f"{d}a", "gA", 1, "", "", 0, 1, 10.0, 30.0, 0, 0, 0, ""],
                 [d, "x", f"{d}b", "gB", 2, "", "", 0, 1, 10.0, 10.0, 0, 0, 0, ""]]
    with open(root / "ingredient_nutrition.csv", "w", newline="") as f:
        csv.writer(f).writerows([["dish_id", "split", "ann_id", "annotation_name", "label_value", "mask", "rgb_image",
                                  "mask_pixels", "has_nutrition", "mass", "calories", "fat", "carb", "protein",
                                  "cutout"]] + rows)


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("n5k_handmasks")
    write_fake(tmp)
    return tmp


def test_handmasks_end_to_end_skips_our_train_dishes(fake, tmp_path):
    hm = tmp_path / "hm"
    write_fake_handmasks(hm, [f"dish_train_{i}" for i in range(3)] + [f"dish_val_{i}" for i in range(4)]
                         + [f"dish_test_{i}" for i in range(5)] + ["dish_elsewhere"])
    groups = tmp_path / "groups.yaml"
    groups.write_text(yaml.safe_dump({"groups": {k: {"group": ("gA", "gB", None)[j % 3]} for j, k in enumerate(INGR)}}))
    extra = ["training.lr=0.001", "training.epochs=1", "data.image_size=64", f"data.cache_dir={tmp_path / 'cache'}",
             f"handmasks.root_search=[{hm}]", "handmasks.mask_hw=[40,40]", f"handmasks.groups_file={groups}",
             f"handmasks.dir={tmp_path / 'out' / '{name}'}"]
    cfg = cfg_for(fake, "handmasks", extra)
    tr = EngineTrainer(cfg, "cpu", str(tmp_path / "run"), verbose=False)
    tr.fit()
    res = handmasks.run(cfg, str(tmp_path / "run"), torch.device("cpu"), log=lambda *_: None)
    assert res["dish_counts"] == {"train": 3, "val": 4, "test": 5, "not_in_our_splits": 1}
    assert res["val"]["n_dishes"] == 4 and res["test"]["n_dishes"] == 5
    assert np.isfinite(res["test"]["mae_kcal"]) and res["test"]["mae_kcal"] >= 0
    for split in ("val", "test"):
        assert set(res[split]) >= {"n_dishes", "cmap_m1", "tmap_m1", "cmap_m2", "m3"}
        assert res[split]["cmap_m2"]["n"] == res[split]["n_dishes"]
        assert res[split]["cmap_m1"]["area"] == pytest.approx(1.0)       # the fake masks cover the whole image
        assert set(res[split]["m3"]["per_group"]) == {"gA", "gB"}
    out = tmp_path / "out" / "handmasks"
    assert (out / "handmasks.json").exists() and (out / "handmasks_test_dishes.json").exists()
