# Kaggle kernel, n5k run 4 (n5k/PLAN.md mục 10, "ưu tiên cả 2"): checks D0-D4 of the new code with partition
# [0,8,18]; D4 on the 2-GPU pipeline with the direct-totals branch + sum loss (TiMePReSt stability on real
# data); then short single-GPU runs (8 epochs) of: direct+sum x lr {0.05, 0.02}, direct+w10, w30, w100 at lr 0.05,
# mean at lr 0.1. Attaches the cache built by run 0.
# Pushed by the agent: `kaggle kernels push -p run4 --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

PY = sys.executable
DIRECT_SUM = ["model.direct_totals=true", "loss.ingredient_reduce=sum"]
SWEEP = [PY, "-m", "timeprest.n5k.sweep", "--config", "n5k/configs/n5k_sweep.yaml", "--systems", "single"]
VARIANTS = [("direct_sum", ["0.05", "0.02"], DIRECT_SUM),
            ("direct_w10", ["0.05"], ["model.direct_totals=true", "loss.w_presence=10", "loss.w_gram=10"]),
            ("w30", ["0.05"], ["loss.w_presence=30", "loss.w_gram=30"]),
            ("w100", ["0.05"], ["loss.w_presence=100", "loss.w_gram=100"]),
            ("mean", ["0.1"], [])]
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "timeprest.n5k.checks",
                     "--config", "n5k/configs/n5k_quick.yaml"], 30),
         ("d4_direct_sum", ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "timeprest.n5k.checks",
                            "--config", "n5k/configs/n5k_quick.yaml", "--only", "4",
                            "--work-dir", "n5k/results/d4_direct_sum", "--set", *DIRECT_SUM, "checks.short_epochs=4",
                            "output.checks_file=n5k/results/d4_direct_sum.json"], 20)]
for tag, lrs, sets in VARIANTS:
    steps.append((f"loss_{tag}", SWEEP + ["--lrs", *lrs, "--gpus", "0", "1", "--tag", tag]
                  + (["--set", *sets] if sets else []), 30))
for tag, lrs, _ in VARIANTS:
    steps.append((f"summary_{tag}", SWEEP + ["--lrs", *lrs, "--tag", tag, "--summary-only"], 5))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=170)
