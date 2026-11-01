"""Chạy detector (+ SAM) trên danh sách ảnh, trên 1 GPU. Notebook mở 2 tiến trình (mỗi GPU một tiến trình).

Ví dụ:
  CUDA_VISIBLE_DEVICES=0 python -m foodbox.run_detect --items items.json --models gdino --sam --out preds
  CUDA_VISIBLE_DEVICES=1 python -m foodbox.run_detect --items items.json --models gdino --shard 1 --nshards 2 ...

Kết quả: <out>/preds_<biến thể>_shard<i>.json, biến thể = tên model hoặc '<model>+sam'.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from .data import load_image
from .detectors import SAMRefiner, build_detector
from .postprocess import postprocess


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mb():
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def _save(path, variant, meta, images):
    with open(path, "w") as f:
        json.dump({"variant": variant, "meta": meta, "images": images}, f)
    print(f"[run_detect] đã ghi {path} ({len(images)} ảnh)", flush=True)


def run(items, models, prompts, out, sam=False, sam_top=20, device="cuda", low_thr=None, shard=0, nshards=1,
        log_every=25):
    os.makedirs(out, exist_ok=True)
    items = items[shard::nshards]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[run_detect] shard {shard}/{nshards}: {len(items)} ảnh, models={models}, sam={sam}, device={gpu}", flush=True)
    refiner = None
    for name in models:
        # 1) Detector: nạp model, chạy thử 1 ảnh (khởi động CUDA), rồi đo thời gian từng ảnh
        _reset_peak()
        t0 = time.time()
        det = build_detector(name, prompts, device=device, low_thr=low_thr)
        load_s = time.time() - t0
        if items:
            det.predict(load_image(items[0]["path"]))
        images = {}
        for k, it in enumerate(items):
            try:
                img = load_image(it["path"])
                _sync()
                t = time.time()
                boxes, scores = det.predict(img)
                _sync()
                images[it["key"]] = {"w": img.width, "h": img.height, "boxes": boxes.round(2).tolist(),
                                     "scores": scores.round(5).tolist(), "time_s": time.time() - t}
            except Exception as e:  # ảnh lỗi: ghi lại rồi chạy tiếp
                images[it["key"]] = {"w": 0, "h": 0, "boxes": [], "scores": [], "time_s": 0.0, "error": repr(e)}
                print(f"[run_detect] LỖI {it['key']}: {e!r}", flush=True)
            if (k + 1) % log_every == 0 or k + 1 == len(items):
                print(f"[run_detect] {name}: {k + 1}/{len(items)}", flush=True)
        meta = {"model": name, "prompts": prompts, "load_s": load_s, "peak_vram_mb": _peak_mb(), "gpu": gpu}
        _save(os.path.join(out, f"preds_{name}_shard{shard}.json"), name, meta, images)

        # 2) SAM: lấy top sam_top box (sau NMS) của mỗi ảnh, thay bằng box bao quanh mask
        if sam:
            if refiner is None:
                refiner = SAMRefiner(device=device)
            _reset_peak()
            sam_images = {}
            for k, it in enumerate(items):
                r = images[it["key"]]
                if "error" in r:
                    sam_images[it["key"]] = r
                    continue
                img = load_image(it["path"])
                b, s = postprocess(r["boxes"], r["scores"], r["w"], r["h"], conf=0.0, nms_iou=0.5,
                                   min_area=0.0, max_area=1.0, drop_container=False, max_det=sam_top)
                _sync()
                t = time.time()
                nb = refiner.refine(img, b)
                _sync()
                sam_images[it["key"]] = {"w": r["w"], "h": r["h"], "boxes": nb.round(2).tolist(),
                                         "scores": s.round(5).tolist(), "time_s": r["time_s"] + time.time() - t}
                if (k + 1) % log_every == 0 or k + 1 == len(items):
                    print(f"[run_detect] {name}+sam: {k + 1}/{len(items)}", flush=True)
            meta_sam = dict(meta, model=f"{name}+sam", peak_vram_mb=max(meta["peak_vram_mb"], _peak_mb()))
            _save(os.path.join(out, f"preds_{name}+sam_shard{shard}.json"), f"{name}+sam", meta_sam, sam_images)
        del det
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_preds(out):
    """Gộp các shard: {biến thể: {"meta": [meta mỗi shard], "images": {key: kết quả}}}."""
    res = {}
    for p in sorted(glob.glob(os.path.join(out, "preds_*_shard*.json"))):
        with open(p) as f:
            d = json.load(f)
        r = res.setdefault(d["variant"], {"meta": [], "images": {}})
        r["meta"].append(d["meta"])
        r["images"].update(d["images"])
    return res


def as_arrays(r):
    """Kết quả 1 ảnh -> (boxes Nx4, scores N) numpy."""
    return np.asarray(r["boxes"], float).reshape(-1, 4), np.asarray(r["scores"], float)


def launch(jobs, timeout_min=40, poll_s=30, cwd=None):
    """Chạy song song các tiến trình run_detect, mỗi job trên 1 GPU: jobs = [(gpu_id, [đối số...], file_log)].

    In dòng log cuối của từng job mỗi poll_s giây; quá timeout_min thì dừng hẳn (tránh kernel treo mất quota).
    cwd: thư mục chạy tiến trình con. Ultralytics tải trọng số (.pt, CLIP) vào thư mục hiện tại, nên đặt ngoài
    /kaggle/working để output không phình thêm ~600 MB (lỗi của 01 v1). Đường dẫn trong args phải là tuyệt đối.
    Trả list mã thoát (None = bị dừng do quá giờ).
    """
    import subprocess
    import sys
    if cwd:
        os.makedirs(cwd, exist_ok=True)
    procs = []
    lib_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # để tiến trình con import được foodbox
    for gpu, args, log in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   PYTHONPATH=os.pathsep.join(p for p in [lib_parent, os.environ.get("PYTHONPATH")] if p))
        fh = open(log, "w")
        p = subprocess.Popen([sys.executable, "-m", "foodbox.run_detect", *args], stdout=fh,
                             stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
        procs.append((p, fh, log, gpu))
        print(f"[launch] GPU {gpu}: {' '.join(args)}", flush=True)
    t0, codes = time.time(), [None] * len(procs)
    while any(p.poll() is None for p, *_ in procs):
        if time.time() - t0 > timeout_min * 60:
            print(f"[launch] quá {timeout_min} phút -> dừng các job còn chạy", flush=True)
            for p, *_ in procs:
                if p.poll() is None:
                    os.killpg(p.pid, 9)
            break
        time.sleep(poll_s)
        for p, _, log, gpu in procs:
            with open(log, errors="replace") as f:
                lines = [l for l in f.read().splitlines() if l.strip()]
            print(f"[launch] {time.time() - t0:5.0f}s GPU {gpu}: {lines[-1] if lines else '(chưa có log)'}", flush=True)
    for k, (p, fh, log, gpu) in enumerate(procs):
        codes[k] = p.poll()
        fh.close()
        print(f"[launch] GPU {gpu} mã thoát {codes[k]} (log: {log})", flush=True)
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="file JSON: list {key, path, ...}")
    ap.add_argument("--models", required=True, help="vd. gdino,owlv2,yoloworld")
    ap.add_argument("--prompts", default="food", help="các prompt cách nhau bởi dấu |, vd. 'food|dish of food'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sam", action="store_true", help="thêm biến thể <model>+sam (box từ mask SAM 2.1)")
    ap.add_argument("--sam-top", type=int, default=20)
    ap.add_argument("--low-thr", type=float, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    with open(a.items) as f:
        items = json.load(f)
    run(items, a.models.split(","), [p for p in a.prompts.split("|") if p.strip()], a.out, sam=a.sam,
        sam_top=a.sam_top, device=a.device, low_thr=a.low_thr, shard=a.shard, nshards=a.nshards)


if __name__ == "__main__":
    main()
