# Kaggle kernel, n5k run 7 (B6, n5k/PLAN.md mục 11): why is the single-GPU epoch 23-27 s in the long runs (run 6,
# `python -m timeprest.n5k.train`, both GPUs visible, after the checks) but 21.5 s in the sweeps (CUDA_VISIBLE_DEVICES=0,
# no torchrun before it on that kernel)? Same config, 4 epochs each, on one machine:
#   both GPUs visible / GPU 0 only, before and after the checks (torchrun) step.
# Pushed by the agent: `kaggle kernels push -p run7_timing --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
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


def single(tag, gpu0_only):
    cmd = [PY, "-m", "timeprest.n5k.train", "--config", "n5k/configs/n5k_single.yaml", "--force",
           "--set", "training.epochs=4", f"name=timing_{tag}", f"output.dir=n5k/results/timing/{tag}"]
    return (f"single_{tag}", (["env", "CUDA_VISIBLE_DEVICES=0"] if gpu0_only else []) + cmd, 10)


steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         single("both_pre", False), single("gpu0_pre", True),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml"], 30),
         single("both_post", False), single("gpu0_post", True)]
kernel_lib.run(steps=steps, stop_if_failed={"weights"}, budget_min=75)
