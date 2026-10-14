# Kaggle kernel, n5k run 0 (CPU only, no GPU quota): build the Nutrition5k cache (n5k_cache/: uint8 RGB +
# uint16 depth .npy, labels, meta.json with the data report). The GPU kernels attach this kernel's output
# instead of the 23 GB / 175k-file source dataset (attaching it stalled at "Adding data 99.9%").
# Pushed by the agent: `kaggle kernels push -p run0` (from n5k/kaggle/, no accelerator).
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

kernel_lib.run(
    steps=[("data", [sys.executable, "-m", "timeprest.n5k.data", "--config", "n5k/configs/n5k_quick.yaml"], 300)],
    budget_min=330,
)
