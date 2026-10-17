# Kaggle kernel, n5k run 12 (n5k/PLAN.md mục 10, 2026-10-17): lr of TiMePReSt at 30 epochs. B6 used lr 0.01 for
# TiMePReSt (8-epoch sweep) vs 0.02 for PipeDream, and TiMePReSt oscillated in epochs 6-18. Same config as B6
# (n5k_timeprest.yaml: one-stream, N=2, partition [0,8,18], 30 epochs), lr {0.02, 0.005}, seeds 0/1/2 (lr 0.01 = B6),
# alternating order: s0 .02, s0 .005, s1 .005, s1 .02, s2 .02, s2 .005.
# Pushed by the agent: `kaggle kernels push -p run12_tp_lr --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
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
C = "n5k/configs/"
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", C + "n5k_quick.yaml"], 30)]
for seed, lr in [(0, 0.02), (0, 0.005), (1, 0.005), (1, 0.02), (2, 0.02), (2, 0.005)]:
    name = f"n5k_timeprest_lr{lr:g}_s{seed}"
    steps.append((name, TORCHRUN + ["-m", "timeprest.n5k.dist_train", "--config", C + "n5k_timeprest.yaml",
                                    "--set", f"seed={seed}", f"training.lr={lr}", f"name={name}"], 25))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=170)
