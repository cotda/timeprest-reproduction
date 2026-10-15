# Kaggle kernel, n5k run 8 (n5k/PLAN.md mục 10, 2026-10-15): two-stream model (RGB ResNet-50 + depth ResNet-50,
# `model.name=n5k_resnet50_2s`) against the one-stream model, single GPU (GPU0 only), lr 0.02, 8 epochs, seeds 0/1,
# alternating one-stream / two-stream so the machine slowing down over time does not favour one model.
#   1. checks D0-D4 of this code (the trainers need an all-PASS checks file of the same code)
#   2. checks D0-D3 of the two-stream model on the 2-GPU pipeline (static batch; own checks file; information only)
#   3. the 4 single-GPU runs + summaries (decision rule: PLAN mục 10)
# Pushed by the agent: `kaggle kernels push -p run8_2s --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

PY = sys.executable
TORCHRUN = ["torchrun", "--standalone", "--nproc_per_node=2"]
TWO = "model.name=n5k_resnet50_2s"
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml"], 30),
         ("checks_2s", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml",
                                   "--only", "0", "1", "2", "3", "--work-dir", "n5k/results/checks/work_2s",
                                   "--set", TWO, "output.checks_file=n5k/results/checks/two_stream.json"], 20)]
SWEEP = [PY, "-m", "timeprest.n5k.sweep", "--config", "n5k/configs/n5k_sweep.yaml", "--systems", "single",
         "--lrs", "0.02"]
VARIANTS = []
for s in (0, 1):
    VARIANTS += [(f"m1s{s}", [f"seed={s}"], 25), (f"m2s{s}", [f"seed={s}", TWO], 40)]
for tag, sets, lim in VARIANTS:
    steps.append((f"single_{tag}", SWEEP + ["--gpus", "0", "--tag", tag, "--set", *sets], lim))
for tag, _, _ in VARIANTS:
    steps.append((f"summary_{tag}", SWEEP + ["--tag", tag, "--summary-only"], 5))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=180)
