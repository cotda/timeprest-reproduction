"""Download only what the n5k cache needs from gillesokhin/nutrition5k-dataset (23 GB, 175k files; attaching
it to a Kaggle session stalls at "Adding data 99.9%"): the 3 CSVs and imagery/realsense_overhead/<dish>/
{rgb.png, depth_raw.png} of the dishes in n5k/splits. The local copy keeps the dataset layout, so
`python -m timeprest.n5k.data --set data.root=<out>` builds the cache from it unchanged.

    python n5k/kaggle/fetch_overhead.py --out D:/University/Research/n5k_data/raw
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = "gillesokhin/nutrition5k-dataset"
CSVS = ["dish_nutrition_values.csv", "dish_ingredients.csv", "ingredients_metadata.csv"]
IMAGES = ["rgb.png", "depth_raw.png"]


def get(api, rel: str, out: str, tries: int = 4) -> str:
    """Download dataset file `rel` to out/rel; returns "ok", "skip" (already there) or the error."""
    dst_dir = os.path.join(out, os.path.dirname(rel))
    dst = os.path.join(out, rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    os.makedirs(dst_dir, exist_ok=True)
    err = ""
    for k in range(tries):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                api.dataset_download_file(DATASET, rel, path=dst_dir, force=True, quiet=True)
            z = dst + ".zip"                      # Kaggle zips some files (large CSVs)
            if os.path.exists(z):
                with zipfile.ZipFile(z) as f:
                    f.extractall(dst_dir)
                os.remove(z)
            if os.path.exists(dst):
                return "ok"
            err = "not written"
        except Exception as e:  # HTTP errors of the Kaggle client have several types
            err = f"{type(e).__name__}: {str(e)[:120]}"
            if "404" in err:
                break
        time.sleep(2 * (k + 1))
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=os.path.join(REPO, "n5k", "splits"))
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    dishes = []
    for s in ("train", "val", "test"):
        with open(os.path.join(args.splits, f"{s}.txt")) as f:
            dishes += [l.strip() for l in f if l.strip()]
    files = CSVS + [f"imagery/realsense_overhead/{d}/{i}" for d in dishes for i in IMAGES]
    print(f"{len(dishes)} dishes, {len(files)} files -> {args.out}", flush=True)
    t0, done, failed = time.time(), 0, {}
    with ThreadPoolExecutor(args.threads) as pool:
        for rel, res in zip(files, pool.map(lambda r: get(api, r, args.out), files)):
            done += 1
            if res not in ("ok", "skip"):
                failed[rel] = res
            if done % 250 == 0 or done == len(files):
                el = time.time() - t0
                print(f"{done}/{len(files)}  {el:.0f}s  eta {el / done * (len(files) - done):.0f}s  failed {len(failed)}",
                      flush=True)
    for rel, err in failed.items():
        print("FAILED", rel, err)
    print(f"done in {time.time() - t0:.0f}s; failed {len(failed)} (data.py drops those dishes with the reason)")
    sys.exit(0)


if __name__ == "__main__":
    main()
