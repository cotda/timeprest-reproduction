"""Đóng gói module foodbox thành 1 file zip để gửi / đính kèm task (chỉ code + hướng dẫn + test, không có dữ liệu, trọng số).

Dùng (trong support_bounding/):  python tools/make_module_zip.py
Ra: dist/foodbox_module_v<version>.zip, giải nén là thư mục foodbox_module/ → cd vào đó → pip install -e .
"""
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["pyproject.toml", "MODULE_README.md", "tests/test_api.py"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    init = open(os.path.join(ROOT, "foodbox", "__init__.py"), encoding="utf-8").read()
    version = re.search(r'__version__ = "([^"]+)"', init).group(1)
    files = FILES + sorted(f"foodbox/{f}" for f in os.listdir(os.path.join(ROOT, "foodbox")) if f.endswith(".py"))
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    out = os.path.join(ROOT, "dist", f"foodbox_module_v{version}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(ROOT, *f.split("/")), f"foodbox_module/{f}")
    print(f"[zip] {len(files)} file -> {out} ({os.path.getsize(out) / 1024:.0f} KB)")
    for f in files:
        print("   ", f)


if __name__ == "__main__":
    main()
