# Kaggle kernel, n5k run 1 (n5k/PLAN.md B4): data cache + report, checks D0-D4 on 2 GPUs, lr sweep.
# Pushed by the agent with `kaggle kernels push -p n5k/kaggle/run1 --accelerator NvidiaTeslaT4`.
# Inputs: dataset kaggleolap/timeprest-code (git archive of the commit) + gillesokhin/nutrition5k-dataset.
# Outputs (/kaggle/working): logs/<step>.log, n5k_results/ (CSV/JSON, no checkpoints), n5k_cache/ (reusable
# as the input of later kernels), RUN_SUMMARY.json (status, time and the *_JSON result line of every step).
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

STEPS = [
    ("weights", [sys.executable, "-c", "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"]),
    ("data", [sys.executable, "-m", "timeprest.n5k.data", "--config", "n5k/configs/n5k_quick.yaml"]),
    ("checks", ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "timeprest.n5k.checks",
                "--config", "n5k/configs/n5k_quick.yaml"]),
    ("sweep", [sys.executable, "-m", "timeprest.n5k.sweep", "--config", "n5k/configs/n5k_sweep.yaml",
               "--systems", "single", "pipedream", "timeprest", "--lrs", "0.01", "0.005", "0.002", "--gpus", "0", "1"]),
]
STOP_IF_FAILED = {"weights", "data", "checks"}

WORK = "/kaggle/working"
CODE = os.path.join(WORK, "code")
LOGS = os.path.join(WORK, "logs")
summary = {"steps": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_summary():
    with open(os.path.join(WORK, "RUN_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=1)


def find(name, top="/kaggle/input", max_depth=6):
    """First file `name` under `top`, without walking the Nutrition5k image folders."""
    base = top.rstrip("/").count("/")
    for root, dirs, files in os.walk(top):
        dirs[:] = sorted(d for d in dirs if d != "imagery") if root.count("/") - base < max_depth else []
        if name in files:
            return os.path.join(root, name)
    return None


# ---- code: the zip of the commit (or an already extracted copy)
os.makedirs(LOGS, exist_ok=True)
z = find("timeprest_code.zip")
if z:
    with zipfile.ZipFile(z) as f:
        f.extractall(CODE)
else:
    src = os.path.dirname(find("pyproject.toml"))
    shutil.copytree(src, CODE)
commit = find("COMMIT.txt")
summary["commit"] = open(commit).read().strip() if commit else None
os.chdir(CODE)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)
env = dict(os.environ, PYTHONUNBUFFERED="1")
print("code", summary["commit"], "| gpus:", subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout,
      flush=True)

for name, cmd in STEPS:
    t0 = time.time()
    print(f"\n===== {name}: {' '.join(cmd)}", flush=True)
    lines = []
    with open(os.path.join(LOGS, f"{name}.log"), "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            lines.append(line)
        p.wait()
    rec = {"step": name, "exit": p.returncode, "minutes": round((time.time() - t0) / 60, 1),
           "json": [l.strip() for l in lines if "_JSON " in l][-1:]}
    summary["steps"].append(rec)
    save_summary()
    if p.returncode != 0 and name in STOP_IF_FAILED:
        print(f"step {name} failed (exit {p.returncode}): stopping", flush=True)
        break

# ---- results without checkpoints
dst = os.path.join(WORK, "n5k_results")
if os.path.isdir("n5k/results"):
    shutil.copytree("n5k/results", dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.pt", "*.pt.tmp", "work"))
shutil.rmtree(CODE, ignore_errors=True)        # keep the kernel output small (code = the input dataset)
summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save_summary()
print("RUN1_DONE", json.dumps(summary["steps"])[:3000], flush=True)
