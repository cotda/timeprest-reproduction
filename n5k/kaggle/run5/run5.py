# Kaggle kernel, n5k run 5 (n5k/PLAN.md mục 10): 320px images. Attaches the 224px cache (run 0) and the 320px
# cache (run 0b); the data code picks the cache of the configured image size.
#   1. D4 (4 epochs, pipedream + timeprest) at 320px with partitions [0,6,18], [0,5,18], [0,4,18]: time, memory
#   2. checks D0-D4 of this code with the current config (224px; the trainers need an all-PASS checks file)
#   3. single GPU, lr 0.02, 8 epochs, seeds 0/1, at 224px and 320px (image-size decision)
#   4. DDP smoke run on the 2 GPUs (NCCL, 1 epoch, 224px)
# Pushed by the agent: `kaggle kernels push -p run5 --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
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
R320 = "data.image_size=320"
steps = [("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15)]
for k in (5, 4, 6):
    steps.append((f"p{k}_320", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml",
                                           "--only", "4", "--work-dir", f"n5k/results/partition320/p{k}",
                                           "--set", R320, f"pipeline.partition=[0,{k},18]", "checks.short_epochs=4",
                                           f"output.checks_file=n5k/results/partition320/p{k}.json"], 25))
steps.append(("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml"], 30))
SWEEP = [PY, "-m", "timeprest.n5k.sweep", "--config", "n5k/configs/n5k_sweep.yaml", "--systems", "single",
         "--lrs", "0.02"]
VARIANTS = [(f"r{r}s{s}", [f"data.image_size={r}", f"seed={s}"]) for r in (224, 320) for s in (0, 1)]
for tag, sets in VARIANTS:
    steps.append((f"single_{tag}", SWEEP + ["--gpus", "0", "--tag", tag, "--set", *sets], 25))
for tag, _ in VARIANTS:
    steps.append((f"summary_{tag}", SWEEP + ["--tag", tag, "--summary-only"], 5))
steps.append(("ddp_smoke", TORCHRUN + ["-m", "timeprest.n5k.ddp_train", "--config", "n5k/configs/n5k_ddp.yaml",
                                       "--set", "training.epochs=1", "output.dir=n5k/results/ddp_smoke"], 15))
kernel_lib.run(steps=steps, stop_if_failed={"weights"}, budget_min=200)
