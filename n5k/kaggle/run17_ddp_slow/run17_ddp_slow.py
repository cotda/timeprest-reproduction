# Kaggle kernel, n5k run 17 (n5k/PLAN.md mục 10, 2026-10-17 18:21): DDP on a slow network, on the same 2 x T4 machine as
# the pipeline bench: DDP bench (native + hook at inf / 2 / 1 Gbit/s) -> pipeline bench_comm (TiMePReSt, PipeDream;
# the B6 config) -> DDP bench again (machine drift). Push (ktanvn): `python n5k/kaggle/accounts.py push-kernel ktanvn
# n5k/kaggle/run17_ddp_slow`.
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
DDP = TORCHRUN + ["-m", "timeprest.n5k.bench_comm_ddp", "--config", "n5k/configs/n5k_bench_comm_ddp.yaml"]
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("bench_ddp_1", DDP + ["--set", "output.dir=n5k/results/bench_comm_ddp/rep1"], 25),
         ("bench_pipeline", TORCHRUN + ["-m", "timeprest.n5k.bench_comm", "--config", "n5k/configs/n5k_bench_comm.yaml"], 30),
         ("bench_ddp_2", DDP + ["--set", "output.dir=n5k/results/bench_comm_ddp/rep2"], 25)]
kernel_lib.run(steps=steps, stop_if_failed={"weights"}, budget_min=100)
