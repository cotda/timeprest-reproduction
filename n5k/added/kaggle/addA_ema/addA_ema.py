# Kaggle kernel, n5k added A (n5k/added/README.md; THU NGHIEM, chua phai dong gop): EMA of the weights at evaluation.
# TiMePReSt and PipeDream, lr 0.02, 30 epochs, seeds 0/1/2, with timeprest.n5k.added.ema_train (training identical
# to dist_train; live + EMA decays 0.98/0.99/0.995 evaluated every epoch). Order alternates between seeds.
# Push: `kaggle kernels push -p n5k/added/kaggle/addA_ema --accelerator NvidiaTeslaT4` (kaggleolap).
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
A = "n5k/added/configs/"
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml"], 30)]
for seed, order in [(0, ("timeprest", "pipedream")), (1, ("pipedream", "timeprest")), (2, ("timeprest", "pipedream"))]:
    for system in order:
        name = f"added_{system}_ema_s{seed}"
        steps.append((name, TORCHRUN + ["-m", "timeprest.n5k.added.ema_train", "--config", A + f"added_{system}.yaml",
                                        "--set", f"seed={seed}", f"name={name}"], 30))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=170)
