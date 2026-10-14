# Kaggle kernel, n5k run 5a (n5k/PLAN.md mục 10): the 224px half of run 5 (the 320px cache kernel run 0b got stuck,
# so run 5 is split; the 320px half is run 5b). Attaches the 224px cache (run 0).
#   1. checks D0-D4 of this code with the current config (the trainers need an all-PASS checks file)
#   2. single GPU, lr 0.02, 8 epochs, seeds 0/1, at 224px (reference of the image-size decision)
#   3. DDP smoke run on the 2 GPUs (NCCL, 1 epoch, 224px)
# Pushed by the agent: `kaggle kernels push -p run5a --accelerator NvidiaTeslaT4` (from n5k/kaggle/).
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
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml"], 30)]
SWEEP = [PY, "-m", "timeprest.n5k.sweep", "--config", "n5k/configs/n5k_sweep.yaml", "--systems", "single",
         "--lrs", "0.02"]
VARIANTS = [(f"r224s{s}", ["data.image_size=224", f"seed={s}"]) for s in (0, 1)]
for tag, sets in VARIANTS:
    steps.append((f"single_{tag}", SWEEP + ["--gpus", "0", "--tag", tag, "--set", *sets], 25))
for tag, _ in VARIANTS:
    steps.append((f"summary_{tag}", SWEEP + ["--tag", tag, "--summary-only"], 5))
steps.append(("ddp_smoke", TORCHRUN + ["-m", "timeprest.n5k.ddp_train", "--config", "n5k/configs/n5k_ddp.yaml",
                                       "--set", "training.epochs=1", "output.dir=n5k/results/ddp_smoke"], 15))
kernel_lib.run(steps=steps, stop_if_failed={"weights"}, budget_min=120)
