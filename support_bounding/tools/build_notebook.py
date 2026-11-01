"""Ghép notebook Kaggle (.ipynb) từ file nguồn dạng 'percent' (notebook.py) + thư viện foodbox/.

Trong notebook.py:
  # %% [markdown]   -> ô markdown (bỏ '# ' đầu dòng)
  # %%              -> ô code
  # %% [foodbox]    -> chèn các ô %%writefile foodbox/*.py (để notebook tự chứa đủ code, không cần dataset code riêng)

Dùng: python tools/build_notebook.py kaggle/01_compare   (ghi ra file code_file trong kernel-metadata.json)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "foodbox")


def _cell(kind, src):
    lines = src.strip("\n").split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def foodbox_cells():
    mods = sorted(f for f in os.listdir(LIB) if f.endswith(".py"))
    mods.sort(key=lambda f: f != "__init__.py")
    cells = [_cell("markdown", "## Thư viện `foodbox`\nCác ô dưới ghi code thư viện ra `/kaggle/working/foodbox/` "
                               "(nguồn: `support_bounding/foodbox/`; sửa ở đó rồi build lại, không sửa trong notebook)."),
             _cell("code", "# Tạo thư mục chứa thư viện\nimport os\nos.makedirs('foodbox', exist_ok=True)")]
    for m in mods:
        with open(os.path.join(LIB, m), encoding="utf-8") as f:
            cells.append(_cell("code", f"%%writefile foodbox/{m}\n" + f.read()))
    return cells


def parse(src):
    cells, kind, buf = [], None, []

    def flush():
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n"))
        if kind and text.strip():
            cells.append(_cell(kind, text))

    for line in src.split("\n"):
        if line.startswith("# %%"):
            flush()
            buf = []
            tag = line[4:].strip()
            if tag == "[foodbox]":
                cells.extend(foodbox_cells())
                kind = None
            else:
                kind = "markdown" if tag == "[markdown]" else "code"
        else:
            buf.append(line)
    flush()
    return cells


def build(kernel_dir):
    with open(os.path.join(kernel_dir, "kernel-metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(kernel_dir, "notebook.py"), encoding="utf-8") as f:
        cells = parse(f.read())
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(kernel_dir, meta["code_file"])
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"[build] {out}: {len(cells)} ô")
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        build(d)
