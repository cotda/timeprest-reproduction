"""B4 (n5k/PLAN.md mục 9): trainers and checks end to end on CPU, on a fake Nutrition5k folder
(random images, 32 px, ResNet-50 without pretrained weights)."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from timeprest.config import load_config
from timeprest.n5k.train import EngineTrainer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INGR = {f"ingr_{k}": (f"food{k}", [0.5 + k, 0.01 * k, 0.1, 0.05]) for k in range(1, 6)}


def write_fake(root, sizes=(48, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    base = root / "nutrition5k-dataset"
    nut, rows, split = [], [], {}
    for s, n in zip(("train", "val", "test"), sizes):
        split[s] = [f"dish_{s}_{i}" for i in range(n)]
        for d in split[s]:
            ids = rng.choice(list(INGR), size=rng.integers(1, 4), replace=False)
            grams = {i: float(rng.integers(10, 200)) for i in ids}
            tot = [sum(g * INGR[i][1][c] for i, g in grams.items()) for c in range(4)]
            nut.append([d, tot[0], sum(grams.values()), tot[1], tot[2], tot[3]])
            rows += [[d, i, INGR[i][0], g, *(g * v for v in INGR[i][1])] for i, g in grams.items()]
            folder = base / "imagery" / "realsense_overhead" / d
            folder.mkdir(parents=True)
            Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(folder / "rgb.png")
            Image.fromarray(rng.integers(3000, 5000, (40, 40)).astype(np.uint16)).save(folder / "depth_raw.png")
    for name, header, data in (
            ("dish_nutrition_values.csv", ["dish_id", "calories", "mass", "fat", "carb", "protein"], nut),
            ("dish_ingredients.csv", ["dish_id", "ingr_id", "ingr_name", "grams", "calories", "fat", "carb", "protein"], rows),
            ("ingredients_metadata.csv", ["ingr_name", "ingr_id", "cal/g", "fat(g)", "carb(g)", "protein(g)"],
             [[n, i, *v] for i, (n, v) in INGR.items()])):
        with open(base / name, "w", newline="") as f:
            csv.writer(f).writerows([header] + data)
    (root / "splits").mkdir()
    for s, ids in split.items():
        (root / "splits" / f"{s}.txt").write_text("\n".join(ids) + "\n")


def overrides(tmp, extra=()):
    return [f"data.root={tmp}", f"data.split_dir={tmp / 'splits'}", f"data.cache_dir={tmp / 'cache'}",
            "data.cache_search=null", "data.image_size=32", "data.num_workers=0", "model.pretrained=null",
            "training.batch_size=8", "training.epochs=2", "training.warmup_epochs=0.5", "runtime.device=cpu",
            "runtime.eval_batch_size=8", "runtime.profile_ops=false", f"output.dir={tmp / 'runs' / '{name}'}",
            f"output.checks_file={tmp / 'checks.json'}", "checks.static_batch=4", "checks.static_steps=2",
            "checks.dynamic_steps=4", "checks.resume_subset=16", *extra]


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("n5k")
    write_fake(tmp)
    return tmp


def cfg_for(fake, name, extra=()):
    return load_config(os.path.join(ROOT, "n5k", "configs", f"n5k_{name}.yaml"), overrides(fake, extra))


@pytest.mark.parametrize("name", ["single", "timeprest"])
def test_engine_trainer_resume_and_test_metrics(fake, tmp_path, name):
    # lr 1e-3: on this random-init ResNet-50 with micro-batches of 4, TiMePReSt's graph rule diverges
    # at lr 1e-2 (stage-0 gradients grow ~x5 per step; PipeDream / recompute stay stable), n5k/PLAN.md
    cfg = cfg_for(fake, name, ["training.lr=0.001"])
    a = EngineTrainer(cfg, "cpu", str(tmp_path / "a"), verbose=False)
    ra = a.fit()
    b = EngineTrainer(cfg, "cpu", str(tmp_path / "b"), verbose=False)
    b.fit(stop_after=1)
    b2 = EngineTrainer(cfg, "cpu", str(tmp_path / "b"), verbose=False)
    assert b2.try_resume() and b2.start_epoch == 1
    rb = b2.fit()
    pa = torch.cat([p.detach().flatten() for m in a.engine.stages for p in m.parameters()])
    pb = torch.cat([p.detach().flatten() for m in b2.engine.stages for p in m.parameters()])
    torch.testing.assert_close(pa, pb, rtol=0, atol=0)
    assert ra[-1]["train_loss"] == rb[-1]["train_loss"] and b2.best == a.best
    t = a.test_best()
    assert t["best_epoch"] == a.best["epoch"] and np.isfinite(t["test_mae_kcal"])
    assert len(a.engine.stages) == cfg["pipeline"]["num_stages"]


def run_module(module, fake, cfg_name, extra=(), args=()):
    cmd = [sys.executable, "-m", "timeprest.dist.launch_local", "--nproc", "2", "-m", module,
           "--config", os.path.join("n5k", "configs", f"n5k_{cfg_name}.yaml"), *args,
           "--set", *overrides(fake, extra)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200)
    return r


def test_checks_on_two_processes(fake):
    r = run_module("timeprest.n5k.checks", fake, "quick", extra=("training.lr=0.001",),
                   args=("--work-dir", str(fake / "work")))
    out = r.stdout + r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("N5K_CHECKS_JSON")]
    assert line, out[-4000:]
    res = {x["name"]: x for x in json.loads(line[0].split(" ", 1)[1])["results"]}
    for name in ("data", "static_equals_engine", "dynamic_versions", "checkpoint_resume"):
        assert res[name]["status"] == "PASS", (name, res[name], out[-3000:])
    # D4 ran on both systems (its PASS needs real images: random pixels carry no signal)
    assert set(res["short_run"]["metrics"]) == {"pipedream", "timeprest"}, out[-3000:]
    for m in res["short_run"]["metrics"].values():
        assert len(m["train_loss_by_epoch"]) == 2 and m["est_full_epoch_s"] > 0
    st = res["dynamic_versions"]["metrics"]["timeprest"]
    assert st["backward_uses_live_version"] and st["forward_vertical_sync"]


def test_dist_train_writes_summary(fake):
    r = run_module("timeprest.n5k.dist_train", fake, "timeprest", extra=("training.epochs=1", "training.lr=0.001"),
                   args=("--force",))
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    out_dir = fake / "runs" / "n5k_timeprest_s0"
    s = json.load(open(out_dir / "summary.json"))
    assert s["best_epoch"] == 1 and np.isfinite(s["test_mae_kcal"]) and "test_ingr_map" in s
    row = next(csv.DictReader(open(out_dir / "metrics.csv")))
    assert {"val_mae_kcal", "train_mae_kcal", "busy_frac", "peak_mem_mb"} <= set(row)


@pytest.mark.parametrize("name", ["single", "timeprest"])
def test_two_stream_model_trains_and_tests(fake, tmp_path, name):
    """PLAN mục 10 (2026-10-15): the two-stream model runs through the same trainers unchanged."""
    cfg = cfg_for(fake, name, ["training.lr=0.001", "training.epochs=1", "model.name=n5k_resnet50_2s"])
    a = EngineTrainer(cfg, "cpu", str(tmp_path / "a"), verbose=False)
    rows = a.fit()
    assert np.isfinite(rows[-1]["train_loss"]) and np.isfinite(a.test_best()["test_mae_kcal"])
    assert type(a.engine.stages[0][0]).__name__ == "DualBlock"
