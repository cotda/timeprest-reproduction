# Kaggle kernel, n5k run 3 (n5k/PLAN.md mục 6 + 10): final lr sweep with the chosen setup (partition [0,8,18],
# direct-totals branch, sum over ingredients): checks D0-D4, then single / pipedream / timeprest x
# lr {0.05, 0.02, 0.01}, 8 epochs, one run per GPU at a time. Attaches the cache built by run 0.
# Pushed by the agent: `kaggle kernels push -p run3 --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

PY = sys.executable
SWEEP = [PY, "-m", "timeprest.n5k.sweep", "--config", "n5k/configs/n5k_sweep.yaml",
         "--systems", "single", "pipedream", "timeprest", "--lrs", "0.05", "0.02", "0.01"]
kernel_lib.run(
    steps=[("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
           ("checks", ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "timeprest.n5k.checks",
                       "--config", "n5k/configs/n5k_quick.yaml"], 30),
           ("sweep", SWEEP + ["--gpus", "0", "1"], 60),
           ("summary", SWEEP + ["--summary-only"], 5)],
    stop_if_failed={"weights", "checks"},
    budget_min=130,
)
