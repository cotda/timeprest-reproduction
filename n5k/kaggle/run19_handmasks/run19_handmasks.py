# Kaggle kernel, n5k run 19 (B7b, n5k/PLAN.md mục 10, 2026-10-17 21:32 / 21:40): location of the maps against the
# HandMasks ingredient masks (Hand et al. 2026, dataset n5k-handmasks) for the B7 checkpoints of run18_s0
# (w_total = 3, seed 0; attached kernel output, b7_ckpt/n5k_<system>_w3_s0/). CPU only: ~311 val/test images per model.
#   handmasks single -> handmasks pipedream -> handmasks timeprest
# Pushed by the agent: `python n5k/kaggle/accounts.py push-kernel ktanvn n5k/kaggle/run19_handmasks --cpu`.
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

PY = sys.executable
C = "n5k/configs/"
steps = []
for system in ("single", "pipedream", "timeprest"):
    name = f"n5k_{system}_w3_s0"
    ckpt = sorted(glob.glob(f"/kaggle/input/**/b7_ckpt/{name}", recursive=True))
    print(name, "checkpoint dirs:", ckpt, flush=True)
    steps.append((f"handmasks_{system}", [PY, "-m", "timeprest.n5k.handmasks", "--config", C + "n5k_handmasks.yaml",
                                          "--set", f"name={name}", "runtime.device=cpu", "--ckpt",
                                          ckpt[0] if ckpt else "MISSING"], 30))
kernel_lib.run(steps=steps, stop_if_failed=set(), budget_min=100)
