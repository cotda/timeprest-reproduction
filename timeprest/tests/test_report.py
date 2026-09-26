import json

from timeprest.report import render, summarize_run
from timeprest.utils import append_csv


def test_summarize_and_render(tmp_path):
    run = tmp_path / "run1_x" / "kaggle_a"
    run.mkdir(parents=True)
    json.dump({"system": "timeprest", "pipeline": {"schedule": "nF1B", "num_microbatches": 3,
                                                    "backward_version": "committed", "backward_rule": "graph"},
               "training": {"batch_size": 192, "lr": 0.02}}, open(run / "config.json", "w"))
    for e, (acc, t) in enumerate([(50.0, 10.0), (66.0, 12.0), (71.0, 11.0)]):
        append_csv({"epoch": e + 1, "test_acc1": acc, "epoch_time_s": t, "cum_time_s": 10.0 * (e + 1),
                    "peak_mem_mb": "800.0|400.0", "busy_frac": "0.9|0.8", "recompute_micro": "0|0",
                    "bwd_overlap_minibatches": 3}, str(run / "metrics.csv"))
    r = summarize_run(str(run))
    assert r["group"] == "run1_x" and r["rule"] == "graph" and r["best"] == 71.0 and r["epoch_s"] == 11.0
    assert r["reach"][60.0] == (2, 20.0 / 3600) and r["reach"][72.0] == (None, None)
    bench = {"bench": {"bandwidths_gbps": [None, 1], "latency_ms": 0.1},
             "rows": [{"system": "timeprest", "bandwidth_gbps": None, "epoch_time_s": 10.0},
                      {"system": "pipedream", "bandwidth_gbps": None, "epoch_time_s": 8.0},
                      {"system": "timeprest", "bandwidth_gbps": 1, "epoch_time_s": 20.0},
                      {"system": "pipedream", "bandwidth_gbps": 1, "epoch_time_s": 25.0}]}
    text = render([r], [("b", bench)])
    assert "| run1_x | kaggle_a |" in text and "| 2 / 2 / 3 / - |" in text and "| 1 | 20.00 | 25.00 | 0.800 |" in text and "1.250" in text


def test_long_run_refuses_checks_of_another_workload(tmp_path):
    from timeprest import utils
    from timeprest.config import load_config
    from timeprest.dist.train import checks_ok
    cfg = load_config("configs/tin_r50_timeprest.yaml", [f"output.checks_file={tmp_path / 'c.json'}",
                                                         f"output.dir={tmp_path / 'run'}"])
    vgg = load_config("configs/tin_quick.yaml")
    base = {"code_hash": utils.code_hash(), "all_pass": True, "systems": ["pipedream", "timeprest"]}
    json.dump(dict(base, model=vgg["model"], dataset="tinyimagenet"), open(tmp_path / "c.json", "w"))
    ok, msg = checks_ok(cfg)
    assert not ok and "vgg16_bn" in msg                # VGG checks do not unlock a ResNet run
    r50 = load_config("configs/tin_r50_quick.yaml")
    json.dump(dict(base, model=r50["model"], dataset="tinyimagenet"), open(tmp_path / "c.json", "w"))
    assert checks_ok(cfg) == (True, "ok")
    json.dump(dict(base, model=r50["model"], dataset="tinyimagenet", systems=["pipedream"]), open(tmp_path / "c.json", "w"))
    assert not checks_ok(cfg)[0]                       # system not covered by the checks
