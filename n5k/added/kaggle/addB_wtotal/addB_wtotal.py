# Kaggle kernel, n5k added B (n5k/added/README.md; THU NGHIEM, chua phai dong gop): weight of the dish-total loss
# term, loss.w_total in {0.3, 3} (default 1), TiMePReSt and PipeDream, lr 0.02, 30 epochs, seed 0, EMA columns
# logged as in A. Push: `python n5k/kaggle/accounts.py push-kernel ktanvn n5k/added/kaggle/addB_wtotal`.
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
for w, system in [(0.3, "timeprest"), (0.3, "pipedream"), (3, "pipedream"), (3, "timeprest")]:
    name = f"added_{system}_wtot{w:g}_s0"
    steps.append((name, TORCHRUN + ["-m", "timeprest.n5k.added.ema_train", "--config", A + f"added_{system}.yaml",
                                    "--set", "seed=0", f"loss.w_total={w}", f"name={name}"], 30))
kernel_lib.run(steps=steps, stop_if_failed={"weights", "checks"}, budget_min=170)
