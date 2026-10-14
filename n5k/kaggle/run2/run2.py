# Kaggle kernel, n5k run 2 (n5k/PLAN.md mục 10, điểm mở (b)): checks D0-D4 of the new code (loss terms and
# gradient shares in the eval metrics), then the loss-variant sweep on the single-GPU trainer:
# mean (w=1) / w10 (w_presence = w_gram = 10) / sum (sum over the K ingredients) x lr {0.05, 0.02, 0.01}, 8 epochs.
# Attaches the cache built by run 0. Pushed by the agent: `kaggle kernels push -p run2 --accelerator NvidiaTeslaT4`.
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

PY = sys.executable
SWEEP = [PY, "-m", "timeprest.n5k.sweep", "--config", "n5k/configs/n5k_sweep.yaml", "--systems", "single",
         "--lrs", "0.05", "0.02", "0.01"]
VARIANTS = {"mean": [], "w10": ["loss.w_presence=10", "loss.w_gram=10"], "sum": ["loss.ingredient_reduce=sum"]}
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "timeprest.n5k.checks",
                     "--config", "n5k/configs/n5k_quick.yaml"], 30)]
for tag, sets in VARIANTS.items():
    steps.append((f"loss_{tag}", SWEEP + ["--gpus", "0", "1", "--tag", tag] + (["--set", *sets] if sets else []), 40))
for tag in VARIANTS:
    steps.append((f"summary_{tag}", SWEEP + ["--tag", tag, "--summary-only"], 5))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=170)
