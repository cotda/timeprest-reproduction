"""Test module FoodBoxDetector / foodbox-detect bằng model giả (không tải Grounding DINO, chạy CPU)."""
import json
import os

import numpy as np
import pytest
from PIL import Image

import foodbox
from foodbox import detectors


class FakeGDINO:
    """Trả box cố định: 1 box lớn (0.40), 1 box phụ điểm cao (0.60), 1 box phụ điểm thấp (0.40), 1 box trùng, 1 box rất nhỏ."""

    containers = (np.zeros((0, 4)), np.zeros(0))  # lượt tìm vật chứa: mặc định không có đĩa nào

    def __init__(self, prompts=("food",), device="cpu", model_id=None, **kw):
        self.prompts = prompts

    def predict(self, img, prompts=None, extras=None):
        if prompts is not None:  # lượt 2 (vật chứa)
            return FakeGDINO.containers
        b = np.array([[10, 10, 150, 110], [160, 10, 199, 60], [160, 70, 199, 140], [11, 11, 149, 109], [1, 1, 3, 3]],
                     float)
        return b, np.array([0.40, 0.60, 0.40, 0.39, 0.90])


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(detectors, "GroundingDINO", FakeGDINO)


def _img(path, size=(200, 150)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, (120, 80, 40)).save(path)
    return path


def test_detect_rules(fake, tmp_path):
    det = foodbox.FoodBoxDetector(device="cpu")
    boxes = det.detect(_img(str(tmp_path / "a.jpg")))
    # giữ: box phụ 0.60 (>= 0.45) và box lớn nhất 0.40 (luôn giữ); bỏ: box phụ 0.40 (< 0.45), box trùng (NMS), box nhỏ
    assert [(b["id"], b["score"], b["largest"]) for b in boxes] == [(1, 0.6, False), (2, 0.4, True)]
    assert (boxes[1]["x1"], boxes[1]["y1"], boxes[1]["x2"], boxes[1]["y2"]) == (10.0, 10.0, 150.0, 110.0)


def test_detect_and_save_files(fake, tmp_path):
    det = foodbox.FoodBoxDetector(device="cpu")
    r = det.detect_and_save(_img(str(tmp_path / "in" / "khay.jpg")), str(tmp_path / "out"))
    files = sorted(os.listdir(tmp_path / "out"))
    assert files == ["khay.json", "khay_1.jpg", "khay_2.jpg", "khay_bbox.jpg"]
    assert Image.open(tmp_path / "out" / "khay_2.jpg").size == (140, 100)  # cắt đúng box (10,10)-(150,110)
    saved = json.load(open(tmp_path / "out" / "khay.json", encoding="utf-8"))
    assert saved["n_boxes"] == 2 and saved["boxes"] == r["boxes"] and saved["params"]["conf"] == 0.35


def test_cli_folder(fake, tmp_path):
    from foodbox.cli import main
    _img(str(tmp_path / "in" / "a.jpg"))
    _img(str(tmp_path / "in" / "sub" / "b.png"))
    (tmp_path / "in" / "note.txt").write_text("x")
    assert main([str(tmp_path / "in"), "--out", str(tmp_path / "out"), "--device", "cpu", "--no-crops"]) == 0
    assert sorted(os.listdir(tmp_path / "out")) == ["a.json", "a_bbox.jpg", "sub", "summary.json"]
    assert sorted(os.listdir(tmp_path / "out" / "sub")) == ["b.json", "b_bbox.jpg"]


def test_group_pieces_on_board(fake, tmp_path, monkeypatch):
    """Có 1 thớt bao cả 3 box (0.40 lớn, 0.60, 0.40) -> gộp thành 1 khung = hợp 3 box, điểm cao nhất. Gộp làm TRƯỚC
    luật box phụ >= 0.45, nên miếng điểm 0.40 được gộp vào món thay vì bị bỏ. Tắt group thì vẫn 2 khung như cũ."""
    monkeypatch.setattr(FakeGDINO, "containers", (np.array([[5, 5, 199, 149]], float), np.array([0.6])))
    img = _img(str(tmp_path / "a.jpg"))
    boxes = foodbox.FoodBoxDetector(device="cpu").detect(img)
    assert [(b["x1"], b["y1"], b["x2"], b["y2"], b["score"]) for b in boxes] == [(10.0, 10.0, 199.0, 140.0, 0.6)]
    assert len(foodbox.FoodBoxDetector(device="cpu", group=False).detect(img)) == 2
