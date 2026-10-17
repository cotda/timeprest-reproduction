# Kaggle kernel, n5k run 13 (n5k/PLAN.md mục 10, 2026-10-17): PipeDream evaluated like TiMePReSt, so either can be
# the main method. (1) B7 for PipeDream: retrain n5k_pipedream.yaml seed 0 (lr 0.02, as B6), keep the best checkpoint
# (/kaggle/working/b7_ckpt/), run timeprest.n5k.explain. (2) 30-epoch lr sweep, mirror of run 12: lr {0.01, 0.005},
# seeds 0/1/2 (lr 0.02 = B6), alternating order: s0 .01, s0 .005, s1 .005, s1 .01, s2 .01, s2 .005.
# Pushed by the agent: `kaggle kernels push -p run13_pd --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
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
PD = ["-m", "timeprest.n5k.dist_train", "--config", C + "n5k_pipedream.yaml", "--set"]
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", C + "n5k_quick.yaml"], 30),
         ("pipedream_s0", TORCHRUN + PD + ["seed=0", "name=b7_pipedream_s0"], 25),
         ("keep_ckpt", [PY, "-c", "import glob, os, shutil\n"
                        "for p in glob.glob('n5k/results/runs/b7_*/best*.pt'):\n"
                        "    d = os.path.join('/kaggle/working/b7_ckpt', os.path.basename(os.path.dirname(p)))\n"
                        "    os.makedirs(d, exist_ok=True); shutil.copy(p, d); print('kept', p)"], 5),
         ("explain_pipedream", [PY, "-m", "timeprest.n5k.explain", "--config", C + "n5k_pipedream.yaml",
                                "--set", "seed=0", "name=b7_pipedream_s0"], 25)]
for seed, lr in [(0, 0.01), (0, 0.005), (1, 0.005), (1, 0.01), (2, 0.01), (2, 0.005)]:
    name = f"n5k_pipedream_lr{lr:g}_s{seed}"
    steps.append((name, TORCHRUN + PD + [f"seed={seed}", f"training.lr={lr}", f"name={name}"], 25))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=180)
