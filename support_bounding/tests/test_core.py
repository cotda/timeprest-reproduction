"""Test CPU cho phần không cần GPU: chuyển nhãn, hậu xử lý, chỉ số, dò thư mục, build notebook."""
import json
import os

import numpy as np
import pytest
from PIL import Image

from foodbox import data, labels, metrics
from foodbox import postprocess as pp


def test_yolo_roundtrip(tmp_path):
    boxes = [(10, 20, 110, 70), (0, 0, 200, 100)]
    p = tmp_path / "a" / "x.txt"
    labels.write_yolo_txt(str(p), [3, 0], boxes, 200, 100)
    first = p.read_text().splitlines()[0].split()
    assert first[0] == "3"
    assert [float(v) for v in first[1:]] == pytest.approx([0.3, 0.45, 0.5, 0.5])
    cls, back = labels.read_yolo_txt(str(p), 200, 100)
    assert cls == [3, 0]
    np.testing.assert_allclose(back, boxes, atol=1e-3)


def test_clip_box():
    assert data.clip_box((-5, 10, 50, 300), 100, 200) == (0.0, 10.0, 50.0, 200.0)
    assert data.clip_box((150, 10, 160, 20), 100, 200) is None


def test_nms_and_containers():
    boxes = [(0, 0, 10, 10), (1, 1, 10, 10), (20, 20, 30, 30)]
    assert pp.nms(boxes, [0.9, 0.8, 0.7], 0.5) == [0, 2]
    # box 0 ôm trọn 2 box con -> bị bỏ (box cả khay); box 1, 2 giữ lại
    tray = [(0, 0, 100, 100), (5, 5, 40, 40), (50, 50, 90, 90)]
    assert pp.drop_containers(tray) == [1, 2]
    assert pp.drop_containers(tray, [0.5, 0.6, 0.7]) == [1, 2]
    # bát phở điểm cao chứa vài box rác điểm thấp -> giữ cả bát
    assert pp.drop_containers(tray, [0.8, 0.1, 0.1]) == [0, 1, 2]


def test_postprocess_area_filter():
    boxes = [(0, 0, 100, 100), (10, 10, 12, 12), (10, 10, 60, 60)]
    b, s = pp.postprocess(boxes, [0.9, 0.8, 0.7], 100, 100, min_area=0.01, max_area=0.95)
    np.testing.assert_allclose(b, [(10, 10, 60, 60)])
    assert list(s) == [0.7]


def test_assign_classes():
    boxes = [(0, 0, 10, 10), (0, 0, 50, 50)]
    assert pp.assign_classes(boxes, [0.9, 0.5], main_id=2, other_id=9, rule="largest") == [9, 2]
    assert pp.assign_classes(boxes, [0.9, 0.5], main_id=2, other_id=9, rule="score") == [2, 9]


def test_metrics_perfect_and_miss():
    gts = {"a": [(True, 1, (0, 0, 10, 10))], "b": [(True, 1, (0, 0, 10, 10)), (False, 2, (20, 20, 30, 30))]}
    perfect = {"a": (np.array([[0, 0, 10, 10]]), np.array([0.9])),
               "b": (np.array([[0, 0, 10, 10], [20, 20, 30, 30]]), np.array([0.8, 0.7]))}
    m = metrics.evaluate(perfect, gts)
    assert m["ap50"] == pytest.approx(1.0) and m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["mean_iou"] == pytest.approx(1.0) and m["main_acc_largest"] == 1.0

    # Thêm 1 box sai điểm cao nhất, bỏ sót 1 box thật
    partial = {"a": (np.array([[50, 50, 60, 60], [0, 0, 10, 10]]), np.array([0.95, 0.9])),
               "b": (np.array([[0, 0, 10, 10]]), np.array([0.8]))}
    m = metrics.evaluate(partial, gts, conf=0.5)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    # PR: (P=0,R=0) -> (1/2, 1/3) -> (2/3, 2/3); AP nội suy = 1/3*2/3 + 1/3*2/3
    assert m["ap50"] == pytest.approx(4 / 9)
    assert m["mean_iou"] == pytest.approx(2 / 3)


def test_metrics_area_ratio():
    gts = {"a": [(True, 1, (0, 0, 10, 10))]}
    m = metrics.evaluate({"a": (np.array([[0, 0, 12, 12]]), np.array([0.9]))}, gts)
    assert m["area_ratio"] == pytest.approx(1.44)


def _img(path, size=(64, 48)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, (200, 100, 50)).save(path)


def test_find_class_roots_and_sample(tmp_path):
    for split in ["train", "valid"]:
        for c in ["pizza", "sushi"]:
            for k in range(3):
                _img(str(tmp_path / "ds" / "data" / split / c / f"{split}{k}.jpg"))
    roots = data.find_class_roots(str(tmp_path))
    assert sorted(os.path.basename(r) for r in roots) == ["train", "valid"]
    ci = data.list_class_images(roots)
    assert {k: len(v) for k, v in ci.items()} == {"pizza": 6, "sushi": 6}
    items = data.sample_images(ci, n_classes=None, per_class=4, seed=0)
    assert len(items) == 8 and items == data.sample_images(ci, n_classes=None, per_class=4, seed=0)


def test_uec_conversion(tmp_path):
    root = tmp_path / "UECFOOD256"
    (root / "1").mkdir(parents=True)
    (root / "2").mkdir()
    (root / "category.txt").write_text("id\tname\n1\trice\n2\tmiso soup\n")
    (root / "1" / "bb_info.txt").write_text("img x1 y1 x2 y2\n10 0 0 32 24\n")
    (root / "2" / "bb_info.txt").write_text("img x1 y1 x2 y2\n10 32 24 70 48\n")  # ảnh 10 có 2 món, box vượt ảnh
    _img(str(root / "1" / "10.jpg"))
    assert data.find_uec_root(str(tmp_path)) == str(root)
    cats = data.read_uec_categories(str(root))
    assert cats == {1: "rice", 2: "miso soup"}
    boxes = data.read_uec_boxes(str(root))
    items = data.sample_images(data.uec_class_images(str(root), [1], cats), n_classes=None, per_class=None)
    gts = data.convert_uec_to_yolo(items, boxes, {"rice": 1}, str(tmp_path / "gt"))
    assert [(m, c) for m, c, _ in gts["rice/10"]] == [(True, 1), (False, 2)]
    assert gts["rice/10"][1][2] == (32.0, 24.0, 64.0, 48.0)  # đã kẹp vào ảnh 64x48
    lines = (tmp_path / "gt" / "rice" / "10.txt").read_text().splitlines()
    assert [l.split()[0] for l in lines] == ["0", "1"]  # rice=0, food_other=1
    assert (tmp_path / "gt" / "classes.txt").read_text().split() == ["rice", "food_other"]


def test_safe_name():
    assert data.safe_name("chicken-'n'-egg on rice") == "chicken-n-egg_on_rice"
    assert data.safe_name("Sour prawn soup") == "sour_prawn_soup"


def test_build_notebook(tmp_path):
    import shutil
    from tools import build_notebook
    src = os.path.join(os.path.dirname(__file__), "..", "kaggle", "01_compare")
    for f in ["kernel-metadata.json", "notebook.py"]:
        shutil.copy(os.path.join(src, f), tmp_path / f)
    out = build_notebook.build(str(tmp_path))
    nb = json.load(open(out, encoding="utf-8"))
    srcs = ["".join(c["source"]) for c in nb["cells"]]
    assert any(s.startswith("%%writefile foodbox/detectors.py") for s in srcs)
    assert srcs.index(next(s for s in srcs if s.startswith("%%writefile foodbox/__init__.py"))) < \
        srcs.index(next(s for s in srcs if s.startswith("%%writefile foodbox/data.py")))


def test_build_dataset(tmp_path):
    from foodbox.autolabel import build_dataset
    items = []
    for c in ["pizza", "sushi"]:
        for k in range(3):
            p = str(tmp_path / "in" / c / f"{k}.jpg")
            _img(p, (100, 80))
            items.append({"key": f"{c}/{k}", "class": c, "path": p, "stem": str(k)})
    results = {it["key"]: {"w": 100, "h": 80, "boxes": [[10, 10, 60, 60], [70, 10, 90, 30]], "scores": [0.9, 0.5]}
               for it in items}
    results["sushi/2"] = {"w": 100, "h": 80, "boxes": [[10, 10, 60, 60]], "scores": [0.1]}  # dưới ngưỡng -> không box
    results["sushi/1"]["error"] = "boom"
    out = str(tmp_path / "out")
    report, failed = build_dataset(items, results, out, conf=0.3, val_frac=0.34)
    assert open(os.path.join(out, "classes.txt")).read().split() == ["pizza", "sushi", "food_other"]
    lines = open(os.path.join(out, "labels", "pizza", "0.txt")).read().splitlines()
    assert [l.split()[0] for l in lines] == ["0", "2"]  # box lớn = món chính, box nhỏ = food_other
    assert sorted((f["image"], f["reason"]) for f in failed) == [("sushi/1", "boom"), ("sushi/2", "no_box")]
    listed = open(os.path.join(out, "train.txt")).read().split() + open(os.path.join(out, "val.txt")).read().split()
    assert sorted(listed) == sorted(f"./images/{k}.jpg" for k in ["pizza/0", "pizza/1", "pizza/2", "sushi/0"])
    assert os.path.exists(os.path.join(out, "visualized", "pizza", "0.jpg"))
    row = next(r for r in report if r["image"] == "sushi/2")
    assert row["needs_review"] == 1 and row["reasons"] == "no_box"


def test_find_class_roots_deep_kaggle_layout(tmp_path):
    # Giống cách Kaggle gắn food101tiny: /kaggle/input/datasets/<owner>/<slug>/data/food-101-tiny/train/<món>
    base = tmp_path / "input" / "datasets" / "msarmi9" / "food101tiny" / "data" / "food-101-tiny" / "train"
    for c in ["ramen", "sushi"]:
        _img(str(base / c / "1.jpg"))
    assert data.find_class_roots(str(tmp_path / "input")) == [str(base)]


def test_build_dataset_other_conf(tmp_path):
    from foodbox.autolabel import build_dataset
    p = str(tmp_path / "in" / "ramen" / "0.jpg")
    _img(p, (100, 80))
    items = [{"key": "ramen/0", "class": "ramen", "path": p, "stem": "0"}]
    # box lớn điểm 0.4 (món chính), box nhỏ 0.42 và 0.6 (food_other)
    res = {"ramen/0": {"w": 100, "h": 80, "boxes": [[0, 0, 80, 70], [85, 0, 99, 10], [85, 60, 99, 79]],
                       "scores": [0.4, 0.42, 0.6]}}
    build_dataset(items, res, str(tmp_path / "out"), conf=0.35, other_conf=0.45)
    lines = open(tmp_path / "out" / "labels" / "ramen" / "0.txt").read().splitlines()
    assert sorted(l.split()[0] for l in lines) == ["0", "1"]  # giữ món chính 0.4 + food_other 0.6, bỏ food_other 0.42


def test_box_colors_unique_per_image():
    from foodbox.autolabel import box_colors
    cols = box_colors([10, 2, 10, 10], main_id=2)  # 1 box món chính (class 2) + 3 box food_other (class 10)
    assert len(set(cols)) == 4
    assert cols[1] == box_colors([2], main_id=2)[0]  # món chính luôn lấy màu đầu


def test_run_detect_checkpoint_resume(tmp_path, monkeypatch):
    """Kernel bị dừng giữa chừng: phần đã lưu theo đợt đọc được, chạy lại thì chỉ làm tiếp ảnh còn thiếu."""
    from foodbox import run_detect

    calls = {"n": 0, "stop_at": 4}

    class Fake:
        def predict(self, img):
            calls["n"] += 1
            if calls["n"] == calls["stop_at"]:
                raise KeyboardInterrupt  # giống bị dừng hẳn (không phải lỗi 1 ảnh)
            return np.array([[1.0, 1.0, 10.0, 10.0]]), np.array([0.9])

    monkeypatch.setattr(run_detect, "build_detector", lambda *a, **k: Fake())
    items = []
    for k in range(5):
        p = str(tmp_path / "in" / "pho" / f"{k}.jpg")
        _img(p)
        items.append({"key": f"pho/{k}", "class": "pho", "path": p, "stem": str(k)})
    out = str(tmp_path / "preds")
    with pytest.raises(KeyboardInterrupt):  # lần 1: 1 lần chạy thử + ảnh 0, 1 (lưu đợt) + dừng ở ảnh 2
        run_detect.run(items, ["gdino"], ["food"], out, device="cpu", ckpt_every=2)
    assert sorted(run_detect.load_preds(out)["gdino"]["images"]) == ["pho/0", "pho/1"]
    calls.update(n=0, stop_at=-1)
    run_detect.run(items, ["gdino"], ["food"], out, device="cpu", ckpt_every=2)
    assert calls["n"] == 1 + 3  # chạy thử + 3 ảnh còn thiếu
    res = run_detect.load_preds(out)["gdino"]
    assert len(res["images"]) == 5 and not any(f.startswith("partial_") for f in os.listdir(out))


def test_crop_box_pad_and_clip():
    from foodbox.crop import crop_box
    img = Image.new("RGB", (100, 80))
    c, (l, t, r, b) = crop_box(img, (10, 10, 50, 30), pad=0.1)  # nới 4 px ngang, 2 px dọc
    assert (l, t, r, b) == (6, 8, 54, 32) and c.size == (48, 24)
    c, box = crop_box(img, (90, 70, 100, 80), pad=0.5)  # sát mép: bị kẹp trong ảnh
    assert box == (85, 65, 100, 80)


def test_crops_build_dataset_and_tool(tmp_path):
    import csv as _csv
    import subprocess
    import sys
    from foodbox.autolabel import build_dataset
    p = str(tmp_path / "in" / "pho" / "0.jpg")
    _img(p, (100, 80))
    items = [{"key": "pho/0", "class": "pho", "path": p, "stem": "0"}]
    res = {"pho/0": {"w": 100, "h": 80, "boxes": [[0, 0, 70, 60], [75, 5, 99, 30], [1, 1, 3, 3]],
                     "scores": [0.8, 0.6, 0.9]}}  # box cuối quá nhỏ: bị lọc diện tích, không cắt
    out = str(tmp_path / "out")
    build_dataset(items, res, out, conf=0.35, crops=True, crop_pad=0.0)
    rows = list(_csv.DictReader(open(os.path.join(out, "crops", "crops.csv"), encoding="utf-8")))
    assert [(r["crop"], r["class"], r["is_main"]) for r in rows] == [("pho/0_0.jpg", "pho", "1"),
                                                                    ("pho/0_1.jpg", "food_other", "0")]
    assert Image.open(os.path.join(out, "crops", "pho", "0_0.jpg")).size == (70, 60)
    # Cắt lại từ nhãn (như sau khi sửa tay): cùng số ảnh cắt
    tool = os.path.join(os.path.dirname(__file__), "..", "tools", "crop_from_labels.py")
    subprocess.run([sys.executable, tool, out, "--out", str(tmp_path / "c2"), "--pad", "0"], check=True)
    rows2 = list(_csv.DictReader(open(tmp_path / "c2" / "crops.csv", encoding="utf-8")))
    assert sorted(r["crop"] for r in rows2) == ["pho/0_0.jpg", "pho/0_1.jpg"]


def test_save_crops_default_is_exact_box(tmp_path):
    """Mặc định cắt ĐÚNG box model khoanh (không nới): kích thước ảnh cắt = kích thước box."""
    from foodbox.crop import save_crops
    img = Image.new("RGB", (200, 150))
    rows = save_crops(img, [(20.4, 30.0, 120.0, 90.0)], ["pho"], str(tmp_path), "pho", "0", "pho/0.jpg")
    r = rows[0]
    assert (r["x1"], r["y1"], r["x2"], r["y2"]) == (20, 30, 120, 90)
    assert (r["crop_w"], r["crop_h"]) == (100, 60)
    assert (r["box_x1"], r["box_y1"], r["box_x2"], r["box_y2"]) == (20.4, 30.0, 120.0, 90.0)


def test_build_dataset_vietnamese_class_names(tmp_path):
    """Tên món có dấu tiếng Việt (dataset của nhóm): ghi được train.txt, classes.txt, nhãn trên Windows."""
    from foodbox.autolabel import build_dataset
    items = []
    for c in ["bánh_bao_chiên", "phở_bò"]:
        for k in range(2):
            p = str(tmp_path / "in" / c / f"00{k}.jpg")
            _img(p, (100, 80))
            items.append({"key": f"{c}/00{k}", "class": c, "path": p, "stem": f"00{k}"})
    res = {it["key"]: {"w": 100, "h": 80, "boxes": [[10, 10, 60, 60]], "scores": [0.8]} for it in items}
    out = str(tmp_path / "out")
    build_dataset(items, res, out, conf=0.35, crops=True)
    assert open(os.path.join(out, "classes.txt"), encoding="utf-8").read().split() == ["bánh_bao_chiên", "phở_bò", "food_other"]
    lines = (open(os.path.join(out, "train.txt"), encoding="utf-8").read().split()
             + open(os.path.join(out, "val.txt"), encoding="utf-8").read().split())
    assert "./images/phở_bò/001.jpg" in lines and len(lines) == 4
    assert os.path.exists(os.path.join(out, "crops", "bánh_bao_chiên", "000_0.jpg"))


def test_group_dishes_pieces_on_one_board():
    """10 miếng bruschetta trên 1 thớt -> 1 box (hợp các miếng, ôm sát thức ăn, không ôm cả thớt)."""
    pieces = [(10 + 30 * i, 20, 40 + 30 * i, 60) for i in range(5)] + [(10 + 30 * i, 70, 40 + 30 * i, 110) for i in range(5)]
    scores = [0.5 + 0.01 * i for i in range(10)]
    board = [(0, 0, 200, 150)]
    b, s = pp.group_dishes(pieces, scores, board, [0.6], 300, 300)
    np.testing.assert_allclose(b, [(10, 20, 160, 110)])
    assert s[0] == pytest.approx(0.59)


def test_group_dishes_tray_keeps_separate_bowls():
    """Khay chứa 2 bát (vật chứa bao >= 2 vật chứa) -> không gộp theo khay; mỗi bát 1 box."""
    food = [(12, 12, 48, 48), (62, 12, 98, 48)]
    containers = [(0, 0, 110, 60), (10, 10, 50, 50), (60, 10, 100, 50)]  # khay, bát 1, bát 2
    b, s = pp.group_dishes(food, [0.6, 0.5], containers, [0.5, 0.5, 0.5], 200, 200)
    assert len(b) == 2
    # Khay mà KHÔNG thấy bát nào: khay chỉ là 1 vật chứa -> 2 món bị gộp (rủi ro đã biết, đo trên UEC ở NOTES v5)
    b2, _ = pp.group_dishes(food, [0.6, 0.5], containers[:1], [0.5], 200, 200)
    assert len(b2) == 1


def test_group_dishes_big_box_replaces_children_and_free_items():
    big = (0, 0, 100, 100)
    kids = [(5, 5, 40, 40), (50, 50, 95, 95), (10, 60, 40, 90)]
    free = (150, 150, 190, 190)  # không nằm trong vật chứa nào: giữ nguyên
    b, s = pp.group_dishes([big] + kids + [free], [0.7, 0.5, 0.5, 0.5, 0.6])
    assert sorted(map(tuple, b.tolist())) == [(0, 0, 100, 100), (150, 150, 190, 190)]
    # 1 box con duy nhất trong box to: không coi là "nhiều miếng" -> giữ cả hai
    b, _ = pp.group_dishes([big, kids[0]], [0.7, 0.5])
    assert len(b) == 2
