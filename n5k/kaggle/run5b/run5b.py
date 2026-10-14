# Kaggle kernel, n5k run 5b (n5k/PLAN.md mục 10): the 320px half of run 5 (run 5a holds the 224px half). Attaches the
# 224px cache (run 0, for the checks) and the 320px cache (run 0c, or run 0b if that one finishes first); the data
# code picks the cache of the configured image size.
#   1. D4 (4 epochs, pipedream + timeprest) at 320px with partitions [0,5,18], [0,4,18], [0,6,18]: time, memory
#   2. checks D0-D4 (224px; the trainers of this kernel need an all-PASS checks file)
#   3. single GPU, lr 0.02, 8 epochs, seeds 0/1, at 320px (image-size decision, against run 5a at 224px)
# Pushed by the agent: `kaggle kernels push -p run5b --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
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
R320 = "data.image_size=320"
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15)]
for k in (5, 4, 6):
    steps.append((f"p{k}_320", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml",
                                           "--only", "4", "--work-dir", f"n5k/results/partition320/p{k}",
                                           "--set", R320, f"pipeline.partition=[0,{k},18]", "checks.short_epochs=4",
                                           f"output.checks_file=n5k/results/partition320/p{k}.json"], 25))
steps.append(("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml"], 30))
SWEEP = [PY, "-m", "timeprest.n5k.sweep", "--config", "n5k/configs/n5k_sweep.yaml", "--systems", "single",
         "--lrs", "0.02"]
VARIANTS = [(f"r320s{s}", [R320, f"seed={s}"]) for s in (0, 1)]
for tag, sets in VARIANTS:
    steps.append((f"single_{tag}", SWEEP + ["--gpus", "0", "--tag", tag, "--set", *sets], 25))
for tag, _ in VARIANTS:
    steps.append((f"summary_{tag}", SWEEP + ["--tag", tag, "--summary-only"], 5))
kernel_lib.run(steps=steps, stop_if_failed={"weights"}, budget_min=180)
