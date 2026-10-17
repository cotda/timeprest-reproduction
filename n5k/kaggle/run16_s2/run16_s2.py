# Kaggle kernel, n5k run 16, seed 2 (n5k/PLAN.md mục 10, 2026-10-17): 60-epoch schedule, seeds 1/2 to complete
# run 14 / run 15 (seed 0). B6 configs (one-stream, partition [0,8,18]) + training.epochs=60 (warmup 2, cosine over
# 60): TiMePReSt lr 0.01, TiMePReSt lr 0.02, PipeDream lr 0.02, one after the other on this machine (order alternates
# between the two seeds). Seed 1 on kaggleolap (`kaggle kernels push -p run16_s1 --accelerator NvidiaTeslaT4`), seed 2
# on ktanvn (`python n5k/kaggle/accounts.py push-kernel ktanvn n5k/kaggle/run16_s2`).
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

SEED = 2
PY = sys.executable
TORCHRUN = ["torchrun", "--standalone", "--nproc_per_node=2"]
C = "n5k/configs/"
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", C + "n5k_quick.yaml"], 30)]
for system, lr in [("timeprest", 0.02), ("pipedream", 0.02), ("timeprest", 0.01)]:
    name = f"n5k_{system}_lr{lr:g}_ep60_s{SEED}"
    steps.append((name, TORCHRUN + ["-m", "timeprest.n5k.dist_train", "--config", C + f"n5k_{system}.yaml",
                                    "--set", f"seed={SEED}", f"training.lr={lr}", "training.epochs=60", f"name={name}"], 45))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=170)
