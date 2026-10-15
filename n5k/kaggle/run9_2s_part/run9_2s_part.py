# Kaggle kernel, n5k run 9 (n5k/PLAN.md mục 10, 2026-10-15): pipeline partition of the two-stream model
# (`model.name=n5k_resnet50_2s`; [0,8,18] ran out of memory on GPU0 in run 8). Full checks D0-D4 of the two-stream
# model (D4: 4 epochs, pipedream + timeprest) with partitions [0,5,18], [0,4,18], [0,6,18]; one checks file each.
# Per-epoch time, GPU busy fraction, peak memory and MB sent: n5k_results/partition2s/p<k>/short_<system>/metrics.csv.
# Pushed by the agent: `kaggle kernels push -p run9_2s_part --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

TORCHRUN = ["torchrun", "--standalone", "--nproc_per_node=2"]
steps = [("weights", [sys.executable, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15)]
for k in (5, 4, 6):
    steps.append((f"p{k}_2s", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml",
                                          "--work-dir", f"n5k/results/partition2s/p{k}",
                                          "--set", "model.name=n5k_resnet50_2s", f"pipeline.partition=[0,{k},18]",
                                          "checks.short_epochs=4",
                                          f"output.checks_file=n5k/results/partition2s/p{k}.json"], 30))
kernel_lib.run(steps=steps, stop_if_failed={"weights"}, budget_min=120)
