"""Upload the code to the private Kaggle dataset kaggleolap/timeprest-code (input of the n5k kernels).

Packs timeprest/, n5k/ (without results), configs/ and pyproject.toml from the WORKING TREE (tracked and
untracked, not ignored files) into timeprest_code.zip, writes COMMIT.txt = HEAD (+ "-dirty" when the
packed files differ from HEAD), and runs `kaggle datasets create` (first time) or `version`.

    python n5k/kaggle/push_code.py -m "fix ..."
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = "kaggleolap/timeprest-code"
PATHS = ["timeprest", "n5k", "configs", "pyproject.toml"]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def files() -> list[str]:
    out = git("ls-files", "--cached", "--others", "--exclude-standard", "--", *PATHS).splitlines()
    return sorted(f for f in out if os.path.isfile(os.path.join(REPO, f)) and not f.startswith("n5k/results/"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", default="update")
    args = ap.parse_args()
    head = git("rev-parse", "--short", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--", *PATHS).strip())
    commit = head + ("-dirty" if dirty else "")
    tmp = tempfile.mkdtemp(prefix="kaggle_code_", dir=os.path.join(REPO, "n5k", "kaggle"))
    fl = files()
    with zipfile.ZipFile(os.path.join(tmp, "timeprest_code.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in fl:
            z.write(os.path.join(REPO, f), f)
    with open(os.path.join(tmp, "COMMIT.txt"), "w") as f:
        f.write(commit + "\n")
    with open(os.path.join(tmp, "dataset-metadata.json"), "w") as f:
        json.dump({"title": "timeprest-code", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]}, f)
    rel = os.path.relpath(tmp, os.getcwd())      # the kaggle CLI breaks on absolute Windows paths
    exists = subprocess.run(["kaggle", "datasets", "status", DATASET], capture_output=True, text=True).returncode == 0
    cmd = (["kaggle", "datasets", "version", "-p", rel, "-m", f"{commit}: {args.message}"] if exists
           else ["kaggle", "datasets", "create", "-p", rel])
    print(f"packing {len(fl)} files as {commit}; {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
