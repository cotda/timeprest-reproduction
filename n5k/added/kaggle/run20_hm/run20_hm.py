# Kaggle kernel, n5k added HM (n5k/added/README.md, 2026-10-17 22:25 / 22:31; thử nghiệm, chưa phải đóng góp): train only
# on the Nutrition5k dishes with HandMasks (our splits: 741 / 157 / 154), seed 0, main config (w_total 3, lr 0.02,
# 30 epochs), then locate each model on the same dishes (timeprest.n5k.handmasks M1-M3, kcal MAE on those dishes).
# HandMasks are fetched from GitHub at a pinned commit (sparse checkout of data/, no cutouts): with the HandMasks dataset
# attached next to the code + cache the kernel stayed QUEUED (probes, n5k/added/README.md 2026-10-18 00:30).
#   fetch HandMasks -> checks -> for HM1 (195 ingredient labels), HM2 (40 HandMasks groups): single, PipeDream, TiMePReSt; train, locate
# Pushed by the agent: `python n5k/kaggle/accounts.py push-kernel kaggleolap n5k/added/kaggle/run20_hm`.
import glob
import sys
import time

print(time.strftime("%H:%M:%S"), "kernel start", flush=True)
code = sorted(glob.glob("/kaggle/input/datasets/*/timeprest-code/timeprest_code")
              + glob.glob("/kaggle/input/timeprest-code/timeprest_code"))[0]
sys.path.insert(0, code + "/n5k/kaggle")
import kernel_lib  # noqa: E402

SEED = 0
PY = sys.executable
TORCHRUN = ["torchrun", "--standalone", "--nproc_per_node=2"]
C = "n5k/added/configs/"
HM_REPO = "https://github.com/liamhand/ingredient-level-nutrition-regression.git"
HM_COMMIT = "0997586b5c177fa8f2c0a61eb7f958ab18605362"     # = n5k/handmasks_data/SOURCE.txt
HM_DIR = "/tmp/handmasks_repo"
FETCH = (f"set -e; rm -rf {HM_DIR}; git clone -q --filter=blob:none --no-checkout {HM_REPO} {HM_DIR}; cd {HM_DIR}; "
         "git sparse-checkout set --no-cone /data/masks/ /data/ingredient_nutrition.csv /data/dish_nutrition.csv "
         f"/LICENSE-DATA; git checkout -q {HM_COMMIT}; test \"$(git rev-parse HEAD)\" = {HM_COMMIT}; "
         "n=$(ls data/masks | wc -l); echo masks=$n; test $n -eq 1056")
steps = [("fetch_handmasks", ["bash", "-c", FETCH], 10),
         ("weights", [PY, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"], 15),
         ("checks", TORCHRUN + ["-m", "timeprest.n5k.checks", "--config", "n5k/configs/n5k_quick.yaml"], 30)]
for tag, labels in (("hm1", "ingredients"), ("hm2", "groups")):
    for system, launch in (("single", [PY]), ("pipedream", TORCHRUN), ("timeprest", TORCHRUN)):
        name = f"{tag}_{system}_s{SEED}"
        args = ["--config", C + f"hm_{system}.yaml", "--set", f"seed={SEED}", f"name={name}", f"added.hm.labels={labels}",
                f"handmasks.root_search=[{HM_DIR}/data]"]
        steps.append((name, launch + ["-m", "timeprest.n5k.added.hm", "train", *args], 30))
        steps.append((f"locate_{name}", [PY, "-m", "timeprest.n5k.added.hm", "locate", *args], 15))
kernel_lib.run(steps=steps, stop_if_failed={"fetch_handmasks", "weights", "checks"}, budget_min=200)
