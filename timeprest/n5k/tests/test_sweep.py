import argparse
import os

from timeprest import utils
from timeprest.n5k import sweep

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_jobs_use_system_configs_and_sweep_values():
    a = argparse.Namespace(config=os.path.join(ROOT, "n5k", "configs", "n5k_sweep.yaml"),
                           systems=["single", "timeprest"], lrs=[0.01, 0.002], set=["data.num_workers=0"])
    js = sweep.jobs(a)
    assert [(s, lr) for s, lr, _ in js] == [("single", 0.01), ("single", 0.002), ("timeprest", 0.01), ("timeprest", 0.002)]
    cmd = js[-1][2]
    assert cmd[cmd.index("--config") + 1].endswith("n5k_timeprest.yaml")
    assert {"training.lr=0.002", "training.epochs=8", "training.warmup_epochs=1", "data.num_workers=0",
            "output.dir=n5k/results/sweep/sweep_timeprest_lr0.002"} <= set(cmd)


def test_summary_picks_best_lr_per_system(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "s.yaml"
    cfg.write_text(f"base: {os.path.join(ROOT, 'n5k', 'configs', 'n5k_sweep.yaml')}\n"
                   f"output:\n  dir: {tmp_path.as_posix()}/{{name}}\n")
    for lr, maes in ((0.01, [90, 80]), (0.002, [95, 70])):
        for ep, m in enumerate(maes, 1):
            utils.append_csv({"epoch": ep, "val_mae_kcal": m, "val_pmae_kcal": m / 2, "val_ingr_f1": 50,
                              "train_loss": 1.0, "wall_time_s": 10}, str(tmp_path / f"sweep_single_lr{lr:g}" / "metrics.csv"))
    a = argparse.Namespace(config=str(cfg), systems=["single", "timeprest"], lrs=[0.01, 0.002])
    rows = sweep.summary(a)
    out = capsys.readouterr().out
    assert "best lr per system: {'single': 0.002}" in out
    assert [r.get("status") for r in rows if r["system"] == "timeprest"] == ["missing", "missing"]
