"""B2 (n5k/PLAN.md mục 9): cache, labels, dropped dishes and dataset items on a fake Nutrition5k
folder with the real layout (CSV columns, imagery/realsense_overhead/<dish>/{rgb,depth_raw}.png)."""
import csv
import json
import math
import os

import numpy as np
import pytest
import torch
from PIL import Image

from timeprest.n5k import data as D
from timeprest.n5k.task import NutritionTask

S = 16   # cached image size
META = {"ingr_1": ("rice", [1.3, 0.003, 0.28, 0.027]), "ingr_2": ("egg", [1.5, 0.11, 0.016, 0.1]),
        "ingr_4": ("oil", [8.8, 1.0, 0.0, 0.0])}          # ingr_3 missing from the metadata on purpose
PG3 = [0.5, 0.01, 0.1, 0.02]                              # per-gram values of ingr_3 in dish_ingredients
# dish -> ingredients {id: grams}
DISHES = {
    "train": {"d_t1": {"ingr_1": 100, "ingr_2": 50}, "d_t2": {"ingr_2": 80, "ingr_3": 40},
              "d_t3": {"ingr_1": 200}, "d_t4": {"ingr_3": 60, "ingr_1": 30}, "d_t5": {"ingr_2": 20}},
    "val": {"d_v1": {"ingr_1": 120}, "d_v2": {"ingr_2": 40, "ingr_3": 10}},
    "test": {"d_s1": {"ingr_1": 50, "ingr_4": 10}, "d_s2_broken_depth": {"ingr_2": 60},
             "d_s3_no_nutrition": {"ingr_1": 70}},
}


def per_gram(i):
    return META[i][1] if i in META else PG3


def write_fake(root):
    base = root / "nutrition5k-dataset"
    img = base / "imagery" / "realsense_overhead"
    rng = np.random.default_rng(0)
    nut_rows, ingr_rows = [], []
    for split, dishes in DISHES.items():
        for d, ingr in dishes.items():
            kcal = sum(g * per_gram(i)[0] for i, g in ingr.items())
            fat, carb, prot = (sum(g * per_gram(i)[c] for i, g in ingr.items()) for c in (1, 2, 3))
            if "no_nutrition" not in d:
                nut_rows.append([d, kcal, sum(ingr.values()), fat, carb, prot])
            for i, g in ingr.items():
                p = per_gram(i)
                ingr_rows.append([d, i, META.get(i, ("tomato",))[0], g, g * p[0], g * p[1], g * p[2], g * p[3]])
            folder = img / d
            folder.mkdir(parents=True)
            Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(folder / "rgb.png")
            depth = rng.integers(3000, 5000, (48, 64)).astype(np.uint16)
            depth[:4, :] = 0                                         # no measurement
            if "broken" in d:                                        # as dish_1558722871 on Kaggle
                (folder / "depth_raw.png_.gstmp").write_bytes(b"")
            else:
                Image.fromarray(depth).save(folder / "depth_raw.png")
    with open(base / "dish_nutrition_values.csv", "w", newline="") as f:
        csv.writer(f).writerows([["dish_id", "calories", "mass", "fat", "carb", "protein"]] + nut_rows)
    with open(base / "dish_ingredients.csv", "w", newline="") as f:
        csv.writer(f).writerows([["dish_id", "ingr_id", "ingr_name", "grams", "calories", "fat", "carb", "protein"]]
                                + ingr_rows)
    with open(base / "ingredients_metadata.csv", "w", newline="") as f:
        csv.writer(f).writerows([["ingr_name", "ingr_id", "cal/g", "fat(g)", "carb(g)", "protein(g)"]]
                                + [[n, i, *v] for i, (n, v) in META.items()])
    splits = root / "splits"
    splits.mkdir()
    for split, dishes in DISHES.items():
        (splits / f"{split}.txt").write_text("\n".join(dishes) + "\n")
    return {"root": str(root), "split_dir": str(splits), "cache_dir": str(root / "cache"), "image_size": S,
            "augment": True}


@pytest.fixture()
def built(tmp_path):
    cfg = write_fake(tmp_path)
    tr, va, te, meta = D.load_datasets(cfg, log=lambda *_: None)
    return cfg, tr, va, te, meta


def test_split_counts_and_dropped_reasons(built):
    _, tr, va, te, meta = built
    assert (len(tr), len(va), len(te)) == (5, 2, 1)
    reasons = {d["dish_id"]: d["reason"] for d in meta["dropped"]}
    assert reasons == {"d_s2_broken_depth": "missing depth_raw.png", "d_s3_no_nutrition": "no row in dish_nutrition_values.csv"}
    assert meta["ids"]["test"] == ["d_s1"]


def test_vocabulary_labels_and_per_gram(built):
    _, tr, _, te, meta = built
    ids = [i["id"] for i in meta["ingredients"]]
    assert ids == ["ingr_1", "ingr_2", "ingr_3"]                        # from train only, sorted
    np.testing.assert_allclose(meta["per_gram"][0], [1.3, 1.0, 0.003, 0.28, 0.027])   # metadata, mass/g = 1
    np.testing.assert_allclose(meta["per_gram"][2], [PG3[0], 1.0, *PG3[1:]])          # fallback from dish rows
    assert meta["report"]["per_gram_fallback"] == ["ingr_3"]
    y = tr.targets[meta["ids"]["train"].index("d_t2")]
    np.testing.assert_allclose(y[:3].numpy(), [0, 80, 40])
    np.testing.assert_allclose(y[3:].numpy(), [80 * 1.5 + 40 * 0.5, 120, 80 * 0.11 + 40 * 0.01,
                                               80 * 0.016 + 40 * 0.1, 80 * 0.1 + 40 * 0.02], rtol=1e-5)
    # test dish with an unknown ingredient: totals keep its calories, grams of known ones only
    assert meta["report"]["test"]["dishes_with_unknown_ingredient"] == 1
    assert math.isclose(meta["report"]["test"]["unknown_ingredient_kcal_share"], 88 / (65 + 88), rel_tol=1e-9)
    np.testing.assert_allclose(te.targets[0, :3].numpy(), [50, 0, 0])
    # consistent fake tables: totals recomputed from the grams match exactly in train
    for t in ("kcal", "mass", "fat", "carb", "protein"):
        assert meta["report"]["train"][f"{t}_from_grams_mae"] < 1e-3
    assert meta["report"]["test"]["kcal_from_grams_mae"] > 80        # the unknown oil is missing


def test_normalisation_stats(built):
    _, tr, _, _, meta = built
    present = tr.targets[:, :3] > 0
    np.testing.assert_allclose(meta["gram_scale"], [(100 + 200 + 30) / 3, (50 + 80 + 20) / 3, (40 + 60) / 2])
    np.testing.assert_allclose(meta["total_std"], tr.targets[:, 3:].double().std(0, unbiased=False).numpy(), rtol=1e-6)
    assert 3000 <= meta["depth_lo"] < meta["depth_hi"] <= 5000
    task = NutritionTask.from_meta(meta)
    out = torch.randn(4, 2 * task.K)
    assert torch.isfinite(task.loss_sum(out, tr.targets[:4])) and present.any()


def test_items_shape_normalisation_and_joint_flip(built):
    cfg, tr, _, _, meta = built
    tr.augment = False
    x, y = tr[0]
    assert x.shape == (4, S, S) and x.dtype == torch.float32 and y.shape == (3 + 5,)
    rgb = tr.rgb[0].permute(2, 0, 1).float() / 255
    torch.testing.assert_close(x[:3], (rgb - torch.tensor(D.IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(D.IMAGENET_STD).view(3, 1, 1))
    zero = (0 - meta["depth_mean"]) / meta["depth_std"]
    assert torch.allclose(x[3, 0], torch.full((S,), zero), atol=1e-3)          # top rows had no depth
    tr.augment = True
    seen = set()
    for seed in range(12):
        torch.manual_seed(seed)
        xa, _ = tr[0]
        for name, f in (("none", lambda t: t), ("h", lambda t: t.flip(2)), ("v", lambda t: t.flip(1)),
                        ("hv", lambda t: t.flip(1).flip(2))):
            if torch.equal(xa, f(x)):          # rgb and depth always flipped together
                seen.add(name)
                break
        else:
            raise AssertionError("augmented item is not a joint flip of the original")
    assert seen == {"none", "h", "v", "hv"}


def test_cache_is_reused_and_found_by_search(built, monkeypatch, tmp_path):
    cfg, *_ = built
    monkeypatch.setattr(D, "_build", lambda *a: (_ for _ in ()).throw(AssertionError("rebuilt")))
    D.load_datasets(cfg, log=lambda *_: None)                                  # same cache_dir: no rebuild
    import shutil
    shutil.copytree(cfg["cache_dir"], tmp_path / "kaggle_input" / "n5k-cache")
    other = dict(cfg, cache_dir=str(tmp_path / "empty"), cache_search=str(tmp_path / "kaggle_input"))
    *_, meta = D.load_datasets(other, log=lambda *_: None)
    assert meta["cache_dir"].endswith("n5k-cache")


def test_cache_of_another_image_size_is_never_used(built, monkeypatch, tmp_path):
    cfg, *_ = built
    monkeypatch.setattr(D, "_build", lambda *a: (_ for _ in ()).throw(AssertionError("rebuilt")))
    with pytest.raises(RuntimeError, match="px"):                              # same cache_dir, other size
        D.load_datasets(dict(cfg, image_size=cfg["image_size"] * 2), log=lambda *_: None)
    import shutil
    shutil.copytree(cfg["cache_dir"], tmp_path / "kaggle_input" / "n5k-cache")
    other = dict(cfg, cache_dir=str(tmp_path / "empty"), cache_search=str(tmp_path / "kaggle_input"))
    assert D._find_cache(other["cache_search"], cfg["image_size"]) is not None
    assert D._find_cache(other["cache_search"], cfg["image_size"] * 2) is None   # searched cache: skipped


def test_failed_build_releases_the_lock(tmp_path):
    cfg = write_fake(tmp_path)
    os.remove(os.path.join(tmp_path, "nutrition5k-dataset", "dish_ingredients.csv"))
    with pytest.raises(FileNotFoundError):
        D.load_datasets(cfg, log=lambda *_: None)
    assert not os.path.exists(os.path.join(cfg["cache_dir"], ".building"))


def test_report_and_cli_json(built, capsys):
    *_, meta = built
    text = D.format_report(meta)
    assert "d_s2_broken_depth" in text and "K=3" in text
    json.dumps(meta)    # meta must stay JSON-serialisable (written as meta.json)
