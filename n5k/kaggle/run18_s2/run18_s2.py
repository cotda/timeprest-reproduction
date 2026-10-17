# Kaggle kernel, n5k run 18, seed 2 (n5k/PLAN.md mục 10, 2026-10-17 20:27): B6 again with loss.w_total = 3 in the main
# config (and TiMePReSt lr 0.02, as every other system): the long runs (30 epochs) of every system for one seed, one
# after the other on the same 2 x T4 machine. Run names n5k_<system>_w3_s2. Attaches the 224px cache (run 0).
#   checks D0-D4, then single GPU, DDP, PipeDream, TiMePReSt N=2, TiMePReSt N=4
#   seed 0 only (B7): keep the best checkpoints of single / PipeDream / TiMePReSt N=2, then timeprest.n5k.explain on each
# Pushed by the agent: `python n5k/kaggle/accounts.py push-kernel <user> n5k/kaggle/run18_s2`.
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

SEED = 2
B7 = ("single", "pipedream", "timeprest") if SEED == 0 else ()
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
                                                 "--set", f"seed={SEED}", f"name=n5k_{system}_w3_s{SEED}"], limit))
if B7:
    steps.append(("keep_ckpt", [PY, "-c", "import glob, os, shutil\n"
                                f"for s in {list(B7)}:\n"
                                f"    for p in glob.glob(f'n5k/results/runs/n5k_{{s}}_w3_s{SEED}/best*.pt'):\n"
                                "        d = os.path.join('/kaggle/working/b7_ckpt', os.path.basename(os.path.dirname(p)))\n"
                                "        os.makedirs(d, exist_ok=True); shutil.copy(p, d); print('kept', p)"], 5))
for system in B7:
    steps.append((f"explain_{system}", [PY, "-m", "timeprest.n5k.explain", "--config", C + f"n5k_{system}.yaml",
                                        "--set", f"seed={SEED}", f"name=n5k_{system}_w3_s{SEED}"], 25))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=330)
