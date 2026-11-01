"""Một lệnh: build notebook -> đẩy lên Kaggle (2 x T4) -> chờ chạy xong -> tải output + log về máy.

Dùng (chạy trong thư mục support_bounding):
  python run_kaggle.py kaggle/01_compare
  python run_kaggle.py kaggle/02_auto_label --poll 120
  python run_kaggle.py kaggle/01_compare --no-push      # chỉ chờ + tải kết quả của lần đẩy trước
  python run_kaggle.py kaggle/01_compare --user tenban  # đẩy lên tài khoản khác (đổi chủ trong id kernel)

Kết quả: results/<tên thư mục>/ (output của notebook + log.txt). Cần Kaggle API đã cấu hình
(~/.kaggle/kaggle.json hoặc ~/.kaggle/access_token); script không đọc/in token.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from tools.build_notebook import build  # noqa: E402


def kaggle(*args, check=True):
    exe = shutil.which("kaggle")
    if exe is None:
        sys.exit("Không tìm thấy lệnh 'kaggle' (pip install kaggle)")
    env = dict(os.environ, PYTHONUTF8="1")  # Windows: thiếu biến này thì CLI ghi log rỗng vì lỗi mã hoá
    r = subprocess.run([exe, *args], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "outdated" not in l.lower())
    if check and r.returncode != 0:
        sys.exit(f"[kaggle {' '.join(args)}] lỗi:\n{out}")
    return out


def status(kid):
    m = re.search(r'status "?([\w.]+)"?', kaggle("kernels", "status", kid, check=False))
    return (m.group(1) if m else "unknown").split(".")[-1].lower()


def notebook_log_to_text(path):
    """Log notebook của Kaggle là JSON [{stream_name, time, data}, ...] -> văn bản thường."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        return "".join(e.get("data", "") for e in json.loads(raw))
    except (OSError, ValueError):
        return raw if "raw" in locals() else ""


def download_output(kid, out_dir, pattern=None):
    """Tải TOÀN BỘ output của kernel (mọi trang). `kaggle kernels output` chỉ tải trang đầu (~500 file) rồi in
    'Next page token' (lỗi E7: 02 v2 thiếu 124/200 hình visualized + tmp_preds). Trả số file đã tải."""
    import requests
    from kaggle import api  # import là tự xác thực bằng cấu hình Kaggle API trên máy
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner, slug = kid.split("/")
    rx = re.compile(pattern) if pattern else None
    token, n, page, log = None, 0, 0, None
    with api.build_kaggle_client() as k:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = owner, slug
            if token:
                req.page_token = token
            resp = k.kernels.kernels_api_client.list_kernel_session_output(req)
            page += 1
            for item in resp.files or []:
                if rx and not rx.search(item.file_name):
                    continue
                dst = os.path.join(ROOT, out_dir, *item.file_name.split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                for attempt in range(4):  # thử lại khi mạng lỗi / bị giới hạn tốc độ
                    try:
                        r = requests.get(item.url, timeout=120)
                        r.raise_for_status()
                        break
                    except requests.RequestException:
                        if attempt == 3:
                            raise
                        time.sleep(5 * (attempt + 1))
                with open(dst, "wb") as f:
                    f.write(r.content)
                n += 1
            log = log or resp.log
            print(f"[run_kaggle] trang {page}: đã tải {n} file", flush=True)
            token = resp.next_page_token
            if not token:
                break
    if log:
        with open(os.path.join(ROOT, out_dir, slug + ".log"), "w", encoding="utf-8") as f:
            f.write(log)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel_dir", help="vd. kaggle/01_compare")
    ap.add_argument("--no-push", action="store_true", help="không đẩy, chỉ chờ và tải kết quả")
    ap.add_argument("--no-wait", action="store_true", help="đẩy xong thì thoát")
    ap.add_argument("--poll", type=int, default=90, help="giây giữa hai lần hỏi trạng thái")
    ap.add_argument("--max-wait-min", type=int, default=240)
    ap.add_argument("--cpu", action="store_true", help="chạy không GPU")
    ap.add_argument("--user", help="đổi chủ kernel trong kernel-metadata.json (tài khoản khác)")
    ap.add_argument("--out", help="thư mục lưu kết quả (mặc định results/<tên thư mục kernel>)")
    ap.add_argument("--download-only", action="store_true", help="chỉ tải lại output (không đẩy, không chờ)")
    a = ap.parse_args()

    kdir = os.path.relpath(os.path.abspath(a.kernel_dir), ROOT)
    name = os.path.basename(os.path.normpath(kdir))
    out = os.path.relpath(os.path.abspath(a.out or os.path.join(ROOT, "results", name)), ROOT)
    with open(os.path.join(ROOT, kdir, "kernel-metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if a.user:  # bản sao tạm với id kernel thuộc tài khoản khác (thư mục gốc không đổi)
        meta["id"] = f"{a.user}/{meta['id'].split('/', 1)[1]}"
        tmp = os.path.join("_push_tmp", name)
        shutil.rmtree(os.path.join(ROOT, tmp), ignore_errors=True)
        shutil.copytree(os.path.join(ROOT, kdir), os.path.join(ROOT, tmp))
        with open(os.path.join(ROOT, tmp, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        kdir = tmp
    kid = meta["id"]

    if not a.no_push:
        build(os.path.join(ROOT, kdir))
        args = ["kernels", "push", "-p", kdir] + ([] if a.cpu else ["--accelerator", "NvidiaTeslaT4"])
        print(kaggle(*args))
        print(f"[run_kaggle] đã đẩy {kid}: https://www.kaggle.com/code/{kid}")
        if a.no_wait:
            return
        time.sleep(30)

    t0, last, st = time.time(), None, "complete"
    while not a.download_only:
        st = status(kid)
        if st != last:
            print(f"[run_kaggle] {time.strftime('%H:%M:%S')} {kid}: {st}", flush=True)
            last = st
        if any(w in st for w in ("complete", "error", "cancel")):
            break
        if time.time() - t0 > a.max_wait_min * 60:
            sys.exit(f"[run_kaggle] quá {a.max_wait_min} phút vẫn '{st}'. Xem log trực tiếp trên trang web kernel; "
                     f"hủy trên web nếu treo. Sau đó: python run_kaggle.py {a.kernel_dir} --no-push")
        time.sleep(a.poll)

    os.makedirs(os.path.join(ROOT, out), exist_ok=True)
    n = download_output(kid, out)
    print(f"[run_kaggle] đã tải {n} file output vào {out}/")
    log = os.path.join(ROOT, out, kid.split("/")[1] + ".log")
    text = notebook_log_to_text(log)
    with open(os.path.join(ROOT, out, "log.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"----- 40 dòng cuối log ({out}/log.txt) -----")
    print("\n".join(text.splitlines()[-40:]))
    print(f"[run_kaggle] trạng thái cuối: {st}; kết quả ở {out}/")
    sys.exit(0 if "complete" in st else 1)


if __name__ == "__main__":
    main()
