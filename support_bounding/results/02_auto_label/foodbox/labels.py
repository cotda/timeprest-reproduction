"""Ghi nhãn YOLO (class_id x_center y_center width height, chuẩn hoá 0-1), classes.txt, data.yaml."""
import os

OTHER_CLASS = "food_other"


def xyxy_to_yolo(box, w, h):
    """(x1, y1, x2, y2) pixel -> (xc, yc, bw, bh) chuẩn hoá 0-1."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def yolo_to_xyxy(yolo, w, h):
    xc, yc, bw, bh = yolo
    return ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)


def write_yolo_txt(path, class_ids, boxes, w, h):
    """Một dòng mỗi box. Ảnh không có box vẫn ghi file rỗng (YOLO hiểu là ảnh nền)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for c, b in zip(class_ids, boxes):
            f.write(f"{int(c)} " + " ".join(f"{min(max(v, 0.0), 1.0):.6f}" for v in xyxy_to_yolo(b, w, h)) + "\n")


def read_yolo_txt(path, w, h):
    """Đọc lại file YOLO -> (class_ids, boxes pixel)."""
    cls, boxes = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 5:
                cls.append(int(p[0]))
                boxes.append(yolo_to_xyxy([float(v) for v in p[1:]], w, h))
    return cls, boxes


def class_list(main_classes):
    """Danh sách class: các món theo tên thư mục + 'food_other' ở cuối."""
    return sorted(main_classes) + [OTHER_CLASS]


def write_classes(out_dir, names):
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")


def write_data_yaml(out_dir, names, train="train.txt", val="val.txt"):
    """data.yaml cho Ultralytics: ảnh ở images/<món>/, nhãn ở labels/<món>/ (YOLO tự đổi images -> labels).

    Không ghi 'path': Ultralytics khi đó lấy thư mục chứa data.yaml, nên chép cả thư mục output đi đâu cũng train được.
    """
    lines = ["# Train: yolo detect train data=data.yaml model=yolo11n.pt", f"train: {train}", f"val: {val}",
             f"nc: {len(names)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
