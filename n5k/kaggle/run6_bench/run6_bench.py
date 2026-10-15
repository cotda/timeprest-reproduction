# Kaggle kernel, n5k run 6 bench (B6, n5k/PLAN.md mục 7): epoch time vs emulated stage-to-stage bandwidth
# (inf / 2 / 1 Gbit/s), TiMePReSt and PipeDream on the same 2 x T4 machine, 224px, partition [0,8,18].
# Kept apart from the long runs (run6_s0 version 1, which had it as its last step, stayed RUNNING > 3 h with no log).
# Pushed by the agent: `kaggle kernels push -p run6_bench --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
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
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("bench_comm", TORCHRUN + ["-m", "timeprest.n5k.bench_comm", "--config", "n5k/configs/n5k_bench_comm.yaml"],
          45)]
kernel_lib.run(steps=steps, stop_if_failed={"weights"}, budget_min=75)
