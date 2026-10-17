# Kaggle kernel, n5k run 15 (n5k/PLAN.md mục 10, 2026-10-17): TiMePReSt at its best 30-epoch lr (0.02, run 12) with
# the 60-epoch schedule, seed 0 (run 14 keeps the B6 lr 0.01). n5k_timeprest.yaml + training.lr=0.02,
# training.epochs=60 (warmup 2, cosine over 60). Second account: pushed with
# `python n5k/kaggle/accounts.py push-kernel ktanvn n5k/kaggle/run15_tp60_lr02` (ids kaggleolap/... -> ktanvn/...).
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
name = "n5k_timeprest_lr0.02_ep60_s0"
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", C + "n5k_quick.yaml"], 30),
         (name, TORCHRUN + ["-m", "timeprest.n5k.dist_train", "--config", C + "n5k_timeprest.yaml",
                            "--set", "seed=0", "training.lr=0.02", "training.epochs=60", f"name={name}"], 45)]
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=100)
