# Kaggle kernel, n5k run 0b (CPU only, no GPU quota): build the 320px Nutrition5k cache (n5k_cache_320/: uint8 RGB +
# uint16 depth .npy, labels, meta.json with the data report). The GPU kernels attach this kernel's output
# instead of the 23 GB / 175k-file source dataset (attaching it stalled at "Adding data 99.9%").
# Pushed by the agent: `kaggle kernels push -p run0b` (from n5k/kaggle/, no accelerator).
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

kernel_lib.run(
    steps=[("data", [sys.executable, "-m", "timeprest.n5k.data", "--config", "n5k/configs/n5k_quick.yaml",
                     "--set", "data.image_size=320", "data.cache_dir=/kaggle/working/n5k_cache_320"], 300)],
    budget_min=330,
)
