# Kaggle kernel, n5k run 14 (n5k/PLAN.md mục 10, 2026-10-17): 60-epoch schedule for TiMePReSt and PipeDream, seed 0.
# Same configs as B6 (one-stream, partition [0,8,18], TiMePReSt lr 0.01, PipeDream lr 0.02); only
# training.epochs=60 (warmup 2, cosine over 60 epochs). Runs on the second account (ktanvn): pushed with
# `python n5k/kaggle/accounts.py push-kernel ktanvn n5k/kaggle/run14_60ep` (ids kaggleolap/... -> ktanvn/...).
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
for system in ("timeprest", "pipedream"):
    name = f"n5k_{system}_ep60_s0"
    steps.append((name, TORCHRUN + ["-m", "timeprest.n5k.dist_train", "--config", C + f"n5k_{system}.yaml",
                                    "--set", "seed=0", "training.epochs=60", f"name={name}"], 45))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=150)
