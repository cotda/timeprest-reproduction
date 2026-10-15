# Kaggle kernel, n5k run 6, seed 0 (B6, n5k/PLAN.md mục 7): the long runs (30 epochs) of every system for one seed,
# one after the other on the same 2 x T4 machine, so their epoch times compare. 224px images, partition [0,8,18]
# (n5k_base.yaml; 320px rejected by run 5). Attaches the 224px cache (run 0).
#   checks D0-D4 (the trainers need an all-PASS checks file), then single GPU, DDP, PipeDream, TiMePReSt N=2,
#   TiMePReSt N=4. Version 2: bench_comm moved to run6_bench (version 1 stayed RUNNING > 3 h with no log).
# Pushed by the agent: `kaggle kernels push -p run6_s0 --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

SEED = 0
BENCH = False
PY = sys.executable
TORCHRUN = ["torchrun", "--standalone", "--nproc_per_node=2"]
C = "n5k/configs/"
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", C + "n5k_quick.yaml"], 30)]
for system, launch, module, limit in [("single", [PY], "train", 60), ("ddp", TORCHRUN, "ddp_train", 45),
                                      ("pipedream", TORCHRUN, "dist_train", 45),
                                      ("timeprest", TORCHRUN, "dist_train", 45),
                                      ("timeprest_n4", TORCHRUN, "dist_train", 50)]:
    steps.append((f"{system}_s{SEED}", launch + ["-m", f"timeprest.n5k.{module}", "--config", C + f"n5k_{system}.yaml",
                                                 "--set", f"seed={SEED}", f"name=n5k_{system}_s{SEED}"], limit))
if BENCH:
    steps.append(("bench_comm", TORCHRUN + ["-m", "timeprest.n5k.bench_comm", "--config", C + "n5k_bench_comm.yaml"], 50))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=330)
