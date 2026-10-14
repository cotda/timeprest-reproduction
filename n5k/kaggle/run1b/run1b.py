# Kaggle kernel, n5k run 1b (n5k/PLAN.md mục 6): lr sweep single / pipedream / timeprest x {0.01, 0.005, 0.002},
# 8 epochs, one run per GPU at a time. Attaches the outputs of run 0 (n5k_cache/, found by data.cache_search)
# and run 1a (n5k_results/checks: the trainers refuse to start without an all-PASS checks file of the same code).
# Pushed by the agent: `kaggle kernels push -p run1b --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

PY = sys.executable
kernel_lib.run(
    steps=[
        ("sweep", [PY, "-m", "timeprest.n5k.sweep", "--config", "n5k/configs/n5k_sweep.yaml",
                   "--systems", "single", "pipedream", "timeprest", "--lrs", "0.01", "0.005", "0.002",
                   "--gpus", "0", "1"], 150),
        ("summary", [PY, "-m", "timeprest.n5k.sweep", "--config", "n5k/configs/n5k_sweep.yaml", "--summary-only",
                     "--systems", "single", "pipedream", "timeprest", "--lrs", "0.01", "0.005", "0.002"], 5),
    ],
    budget_min=180,
    restore=True,
)
