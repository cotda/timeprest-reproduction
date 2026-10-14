# Kaggle kernel, n5k run 1c (n5k/PLAN.md mục 10, 2026-10-14): pipeline partition on 2 GPUs. Check D4 (short
# real-pipeline run, pipedream + timeprest, 4 epochs) with partitions [0,7,18], [0,8,18], [0,9,18]; per-epoch
# time, GPU busy fraction, peak memory and MB sent are in n5k_results/partition/p<k>/short_<system>/metrics.csv.
# Attaches the cache built by run 0. Pushed by the agent: `kaggle kernels push -p run1c --accelerator NvidiaTeslaT4`.
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

steps = [("weights", [sys.executable, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15)]
for k in (8, 7, 9):
    steps.append((f"p{k}", ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "timeprest.n5k.checks",
                            "--config", "n5k/configs/n5k_quick.yaml", "--only", "4",
                            "--work-dir", f"n5k/results/partition/p{k}",
                            "--set", f"pipeline.partition=[0,{k},18]", "checks.short_epochs=4",
                            f"output.checks_file=n5k/results/partition/p{k}.json"], 25))
kernel_lib.run(steps=steps, stop_if_failed={"weights"}, budget_min=110)
