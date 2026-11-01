"""Vẽ box lên ảnh: mỗi class một màu, có nhãn chữ; ghép nhiều ảnh thành lưới."""
import colorsys

from PIL import Image, ImageDraw, ImageFont

GREEN, RED = (0, 200, 0), (230, 0, 0)


def color_for(i):
    """Màu cố định, dễ phân biệt cho class thứ i (bước tỉ lệ vàng trên vòng màu)."""
    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def draw_boxes(img, boxes, labels=None, colors=None, width=None):
    """Trả bản sao ảnh đã vẽ box. labels/colors: list cùng độ dài với boxes (hoặc None)."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    width = width or max(2, round(max(img.size) / 250))
    font = _font(max(12, round(max(img.size) / 40)))
    for k, b in enumerate(boxes):
        c = colors[k] if colors else RED
        x1, y1, x2, y2 = [float(v) for v in b[:4]]
        d.rectangle([x1, y1, x2, y2], outline=c, width=width)
        if labels and labels[k]:
            tb = d.textbbox((x1, y1), labels[k], font=font)
            ty = y1 - (tb[3] - tb[1]) - 4 if y1 - (tb[3] - tb[1]) - 4 > 0 else y1
            tb = d.textbbox((x1, ty), labels[k], font=font)
            d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=c)
            d.text((x1, ty), labels[k], fill=(255, 255, 255), font=font)
    return img


def fit(img, max_side):
    """Thu nhỏ ảnh để cạnh dài = max_side (giữ tỉ lệ)."""
    s = max_side / max(img.size)
    return img if s >= 1 else img.resize((round(img.width * s), round(img.height * s)))


def titled(img, title, size=18):
    """Thêm dải tiêu đề phía trên ảnh."""
    font = _font(size)
    out = Image.new("RGB", (img.width, img.height + size + 8), (255, 255, 255))
    out.paste(img, (0, size + 8))
    ImageDraw.Draw(out).text((4, 2), title, fill=(0, 0, 0), font=font)
    return out


def grid(images, ncols, bg=(255, 255, 255)):
    """Ghép list ảnh thành lưới ncols cột (mỗi ô theo ảnh lớn nhất)."""
    if not images:
        return Image.new("RGB", (1, 1), bg)
    cw, ch = max(i.width for i in images), max(i.height for i in images)
    nrows = (len(images) + ncols - 1) // ncols
    out = Image.new("RGB", (cw * min(ncols, len(images)), ch * nrows), bg)
    for k, im in enumerate(images):
        out.paste(im, ((k % ncols) * cw, (k // ncols) * ch))
    return out
