# Kaggle kernel, n5k run 11 (B7, n5k/PLAN.md mục 10, 2026-10-16): calorie maps, figures and the deletion test.
# B6 kept no checkpoint, so TiMePReSt N=2 (2-GPU pipeline) and single GPU are trained again exactly as in B6 (seed 0,
# 30 epochs, one-stream model, partition [0,8,18]); their best checkpoints are kept in /kaggle/working/b7_ckpt/.
# Then `python -m timeprest.n5k.explain` on each (n5k_results/explain/<name>/: figures, deletion csv/json).
#   checks D0-D4 -> timeprest s0 -> single s0 -> keep checkpoints -> explain timeprest -> explain single
# Pushed by the agent: `kaggle kernels push -p run11_b7 --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
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
RUNS = [("timeprest", TORCHRUN, "dist_train"), ("single", [PY], "train")]
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", C + "n5k_quick.yaml"], 30)]
for system, launch, module in RUNS:
    steps.append((f"{system}_s0", launch + ["-m", f"timeprest.n5k.{module}", "--config", C + f"n5k_{system}.yaml",
                                            "--set", "seed=0", f"name=b7_{system}_s0"], 35))
steps.append(("keep_ckpt", [PY, "-c", "import glob, os, shutil\n"
                            "for p in glob.glob('n5k/results/runs/b7_*/best*.pt'):\n"
                            "    d = os.path.join('/kaggle/working/b7_ckpt', os.path.basename(os.path.dirname(p)))\n"
                            "    os.makedirs(d, exist_ok=True); shutil.copy(p, d); print('kept', p)"], 5))
for system, _, _ in RUNS:
    steps.append((f"explain_{system}", [PY, "-m", "timeprest.n5k.explain", "--config", C + f"n5k_{system}.yaml",
                                        "--set", "seed=0", f"name=b7_{system}_s0"], 25))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=170)
