# Kaggle kernel, n5k run 1a (n5k/PLAN.md B4): ImageNet weights, data report from the cache built by run 0
# (attached kernel output), checks D0-D4 on 2 GPUs. Its n5k_results/checks is attached to run 1b (lr sweep).
# Pushed by the agent: `kaggle kernels push -p run1a --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
# The runner (time limits, logs, RUN_SUMMARY.json) is n5k/kaggle/kernel_lib.py of the code dataset.
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
        ("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
        ("data", [PY, "-m", "timeprest.n5k.data", "--config", "n5k/configs/n5k_quick.yaml"], 20),
        ("checks", ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "timeprest.n5k.checks",
                    "--config", "n5k/configs/n5k_quick.yaml"], 120),
    ],
    stop_if_failed={"weights", "data"},
    budget_min=330,
)
