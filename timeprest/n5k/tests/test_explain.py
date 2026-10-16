"""B7 (n5k/PLAN.md mục 10, 2026-10-16): calorie maps, cell masks, checkpoint loading and the deletion test
end to end on the fake Nutrition5k folder (64 px images -> 2 x 2 head grid)."""
import csv
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from timeprest.models import build_stages
from timeprest.n5k import explain
from timeprest.n5k.common import prepare
from timeprest.n5k.train import EngineTrainer

from .test_train import cfg_for, write_fake


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("n5k_explain")
    write_fake(tmp)
    return tmp


def test_merge_stage_states_rebuilds_the_whole_model():
    torch.manual_seed(0)
    mcfg = {"name": "mlp", "in_dim": 12, "hidden": 16, "num_classes": 5, "depth": 4}
    stages, bounds = build_stages(mcfg, 2, [0, 3, 5])
    whole = nn.Sequential(*[b for s in stages for b in s])
    other = nn.Sequential(*[b for s in build_stages(mcfg, 1, [0, 5])[0] for b in s])
    other.load_state_dict(explain.merge_stage_states([s.state_dict() for s in stages], bounds))
    x = torch.randn(3, 12)
    torch.testing.assert_close(other(x), whole(x), rtol=0, atol=0)


def test_ingredient_calorie_map_sums_to_the_kcal_of_present_ingredients():
    torch.manual_seed(0)
    cam = torch.randn(2, 4, 3, 3)
    cam[1, 2] = -1.0                                   # ReLU(CAM) = 0 everywhere -> spread uniformly
    grams = torch.rand(2, 4) * 100
    present = torch.tensor([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    kpg = torch.tensor([1.0, 2.0, 0.5, 3.0])
    C, A = explain.ingredient_calorie_map(cam, grams, present, kpg)
    torch.testing.assert_close(A.sum((2, 3)), torch.ones(2, 4))
    torch.testing.assert_close(A[1, 2], torch.full((3, 3), 1 / 9))
    torch.testing.assert_close(C.sum((1, 2)), (grams * kpg * present).sum(1))
    assert (C >= 0).all()


def test_mask_cells_zeroes_exactly_the_chosen_cells():
    x = torch.ones(4, 14, 21)
    y = explain.mask_cells(x, [0, 5], (2, 3))           # cells of 7 x 7 px; 5 = row 1, column 2
    assert (y == 0).sum() == 4 * 2 * 49 and x.all()
    assert (y[:, :7, :7] == 0).all() and (y[:, 7:, 14:] == 0).all() and (y[:, :7, 7:] == 1).all()
    assert explain.top_cells(torch.tensor([[0.1, 0.9], [0.5, -1.0]]), 2) == [1, 2]


def test_pick_dishes_quantiles_then_random():
    err = np.arange(100, dtype=float)[::-1]             # index 0 has the largest error
    p = explain.pick_dishes(err, 8, seed=0)
    assert len(p) == len(set(p)) == 8
    assert p[:4] == [100 - 1 - round(q * 99) for q in (0.125, 0.375, 0.625, 0.875)]


@pytest.mark.parametrize("name", ["single", "timeprest"])
def test_explain_end_to_end(fake, tmp_path, name):
    extra = ["training.lr=0.001", "training.epochs=1", "data.image_size=64", f"data.cache_dir={tmp_path / 'cache'}",
             "explain.presence_threshold=0.0", "explain.cells=[1,2]", "explain.random_draws=2", "explain.rule_cells=1",
             "explain.figure_dishes=3", f"explain.dir={tmp_path / 'explain'}"]
    cfg = cfg_for(fake, name, extra)
    tr = EngineTrainer(cfg, "cpu", str(tmp_path / "run"), verbose=False)
    tr.fit()
    ref = tr.test_best()
    res = explain.run(cfg, str(tmp_path / "run"), torch.device("cpu"), log=lambda *_: None)
    # the loaded checkpoint is the trainer's best one: same test MAE
    assert res["best_epoch"] == ref["best_epoch"] and abs(res["test_mae_kcal"] - ref["test_mae_kcal"]) < 1e-3
    out = tmp_path / "explain"
    assert len(list((out / "figures").glob("dish_*.png"))) == 3 and len(json.load(open(out / "figures.json"))) == 3
    pairs = list(csv.DictReader(open(out / "deletion_pairs.csv")))
    dishes = list(csv.DictReader(open(out / "deletion_dishes.csv")))
    assert len(dishes) == 2 * 16 and {int(r["c"]) for r in pairs} == {1, 2}
    assert res["n_pairs"] == len(pairs) // 2 > 0         # threshold 0: every true ingredient is a pair
    assert set(res["faithful"]) == set(explain.RULES)
    for kind, m in explain.RULES.values():
        assert res["stats"][kind][f"{m}@1"]["n"] > 0


def test_pipeline_checkpoint_files_load_like_the_engine_one(fake, tmp_path):
    """best_stage<r>.pt of dist_train (one state dict per stage) gives the same model as the engine's best.pt."""
    cfg = cfg_for(fake, "timeprest", ["training.lr=0.001", "training.epochs=1"])
    tr = EngineTrainer(cfg, "cpu", str(tmp_path / "run"), verbose=False)
    tr.fit()
    st = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    (tmp_path / "pipe").mkdir()
    for r, sd in enumerate(st["engine"]["stages"]):
        torch.save({"module": sd, "epoch": st["epoch"]}, tmp_path / "pipe" / f"best_stage{r}.pt")
    built, (_, _, test), _, _ = prepare(cfg, log=lambda *_: None)
    a, ea = explain.load_model(built, str(tmp_path / "run"), test[0][0].unsqueeze(0))
    b, eb = explain.load_model(built, str(tmp_path / "pipe"), test[0][0].unsqueeze(0))
    assert len(st["engine"]["stages"]) == 2 and ea == eb == st["epoch"]
    x = torch.stack([test[i][0] for i in range(4)])
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
