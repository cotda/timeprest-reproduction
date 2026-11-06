"""Lệnh foodbox-detect: nhập ảnh (hoặc thư mục ảnh) → ra bounding box.

  foodbox-detect khay_com.jpg --out ketqua/
  foodbox-detect anh1.jpg anh2.jpg thu_muc_anh/ --out ketqua/ --device cpu
  python -m foodbox.cli khay_com.jpg --out ketqua/          (khi chưa cài lệnh)

Mỗi ảnh ra: <tên>_bbox.jpg (khung đánh số 1, 2, ...), <tên>_1.jpg, <tên>_2.jpg ... (cắt đúng từng box), <tên>.json.
Ảnh trong thư mục con giữ nguyên cấu trúc thư mục con trong --out.
"""
import argparse
import json
import os
import sys
import time

from .data import is_image


def collect(inputs):
    """[(đường dẫn ảnh, thư mục con tương đối để ghi kết quả)] từ danh sách file / thư mục."""
    out = []
    for p in inputs:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                rel = os.path.relpath(root, p)
                out += [(os.path.join(root, f), "" if rel == "." else rel) for f in sorted(files) if is_image(f)]
        elif os.path.isfile(p) and is_image(p):
            out.append((p, ""))
        else:
            print(f"[foodbox] bỏ qua (không phải ảnh / không tồn tại): {p}", flush=True)
    return out


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # console Windows: in được tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="foodbox-detect", description="Nhập ảnh món ăn → ra bounding box từng món")
    ap.add_argument("inputs", nargs="+", help="file ảnh và/hoặc thư mục ảnh")
    ap.add_argument("--out", default="foodbox_out", help="thư mục kết quả (mặc định foodbox_out/)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--conf", type=float, default=None, help="ngưỡng box (mặc định 0.35)")
    ap.add_argument("--other-conf", type=float, default=None, help="ngưỡng box phụ (mặc định 0.45)")
    ap.add_argument("--prompt", action="append", default=None, help='câu chữ cho model (mặc định "food"); lặp lại để thêm')
    ap.add_argument("--no-crops", action="store_true", help="không cắt ảnh riêng từng box")
    ap.add_argument("--no-group", action="store_true",
                    help="không gộp các miếng cùng đĩa (mỗi miếng 1 khung; nhanh gấp ~2 vì bỏ lượt tìm đĩa)")
    ap.add_argument("--no-vis", action="store_true", help="không ghi ảnh có khung")
    ap.add_argument("--pad", type=float, default=0.0, help="nới ảnh cắt mỗi phía pad x cạnh box (mặc định 0 = đúng box)")
    a = ap.parse_args(argv)

    items = collect(a.inputs)
    if not items:
        print("[foodbox] không có ảnh nào để chạy")
        return 1
    from .api import FoodBoxDetector  # nạp torch/transformers chỉ khi thật sự chạy
    print("[foodbox] đang nạp model Grounding DINO (lần đầu tự tải ~900 MB từ HuggingFace)...", flush=True)
    t0 = time.time()
    det = FoodBoxDetector(device=a.device, prompts=a.prompt, conf=a.conf, other_conf=a.other_conf,
                          group=False if a.no_group else None)
    print(f"[foodbox] model sẵn sàng trên {det.device} sau {time.time() - t0:.0f} s; {len(items)} ảnh", flush=True)
    summary = []
    for i, (path, sub) in enumerate(items, 1):
        try:
            r = det.detect_and_save(path, os.path.join(a.out, sub), vis=not a.no_vis, crops=not a.no_crops, pad=a.pad)
            print(f"[foodbox] {i}/{len(items)} {path}: {r['n_boxes']} box ({r['seconds']:.1f} s)", flush=True)
            summary.append({"image": path, "n_boxes": r["n_boxes"], "out": os.path.join(a.out, sub)})
        except Exception as e:  # ảnh lỗi: báo rồi chạy tiếp
            print(f"[foodbox] {i}/{len(items)} {path}: LỖI {e!r}", flush=True)
            summary.append({"image": path, "error": repr(e)})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[foodbox] xong: kết quả ở {a.out}/ ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
