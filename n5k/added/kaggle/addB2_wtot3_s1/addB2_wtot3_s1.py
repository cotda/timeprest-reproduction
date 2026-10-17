# Kaggle kernel, n5k added B2 (n5k/added/README.md; THU NGHIEM, chua phai dong gop): loss.w_total = 3, seed 1,
# TiMePReSt and PipeDream, lr 0.02, 30 epochs, EMA trainer as in A/B. Push (kaggleolap): `kaggle kernels push -p n5k/added/kaggle/addB2_wtot3_s1 --accelerator NvidiaTeslaT4`.
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

SEED = 1
PY = sys.executable
TORCHRUN = ["torchrun", "--standalone", "--nproc_per_node=2"]
A = "n5k/added/configs/"
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml"], 30)]
for system in ("timeprest", "pipedream"):
    name = f"added_{system}_wtot3_s{SEED}"
    steps.append((name, TORCHRUN + ["-m", "timeprest.n5k.added.ema_train", "--config", A + f"added_{system}.yaml",
                                    "--set", f"seed={SEED}", "loss.w_total=3", f"name={name}"], 30))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=100)
