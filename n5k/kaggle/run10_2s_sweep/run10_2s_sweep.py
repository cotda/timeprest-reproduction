# Kaggle kernel, n5k run 10 (n5k/PLAN.md mục 10, 2026-10-16): step 2 of the two-stream model
# (`model.name=n5k_resnet50_2s`): short lr sweep {0.02, 0.01} (8 epochs, warmup 1, seed 0) + epoch time of the
# 4 systems on one machine. PipeDream / TiMePReSt run the real 2-GPU pipeline (dist_train, partition [0,4,18] from
# run 9), not the 1-GPU simulated engine of the old sweep (it holds several mini-batches on one GPU). Single on
# GPU0 only; DDP at the single lr 0.02 only. Runs one after the other, symmetric order (machine slows over time):
#   checks D0-D4 (two-stream, [0,4,18], D4 2 epochs) -> single .02 -> ddp .02 -> pipedream .02 -> timeprest .02
#   -> timeprest .01 -> pipedream .01 -> single .01 -> summary (python -m timeprest.n5k.sweep --summary-only).
# Pushed by the agent: `kaggle kernels push -p run10_2s_sweep --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
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
TWO = "model.name=n5k_resnet50_2s"
PART = "pipeline.partition=[0,4,18]"
TAG = "2s"
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", C + "n5k_quick.yaml",
                                "--work-dir", "n5k/results/checks/work_2s", "--set", TWO, PART], 35)]
LAUNCH = {"single": ["env", "CUDA_VISIBLE_DEVICES=0", PY], "ddp": TORCHRUN, "pipedream": TORCHRUN, "timeprest": TORCHRUN}
MODULE = {"single": "train", "ddp": "ddp_train", "pipedream": "dist_train", "timeprest": "dist_train"}
LIMIT = {"single": 20, "ddp": 15, "pipedream": 20, "timeprest": 20}
ORDER = [("single", 0.02), ("ddp", 0.02), ("pipedream", 0.02), ("timeprest", 0.02),
         ("timeprest", 0.01), ("pipedream", 0.01), ("single", 0.01)]
for system, lr in ORDER:
    name = f"sweep_{system}_{TAG}_lr{lr:g}"                  # = timeprest.n5k.sweep.run_name, so --summary-only finds it
    sets = [TWO, "seed=0", f"name={name}", f"training.lr={lr}", "training.epochs=8", "training.warmup_epochs=1",
            f"output.dir=n5k/results/sweep/{name}"] + ([PART] if system in ("pipedream", "timeprest") else [])
    steps.append((f"{system}_lr{lr:g}", LAUNCH[system] + ["-m", f"timeprest.n5k.{MODULE[system]}",
                                                          "--config", C + f"n5k_{system}.yaml", "--set", *sets],
                  LIMIT[system]))
steps.append(("summary", [PY, "-m", "timeprest.n5k.sweep", "--config", C + "n5k_sweep.yaml", "--summary-only",
                          "--tag", TAG, "--systems", "single", "ddp", "pipedream", "timeprest",
                          "--lrs", "0.02", "0.01"], 5))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=170)
