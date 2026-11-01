# Khảo sát: model tự động vẽ bounding box cho món ăn

Mục tiêu: với ảnh đã biết tên món (theo thư mục), tự vẽ **box ôm sát phần thức ăn** (không ôm cả đĩa/bát/khay),
mỗi món một box, để làm nhãn YOLO. Ràng buộc: chạy trên Kaggle T4 16 GB (không hỗ trợ bf16), cài đặt bằng pip.

## 1. Các hướng chính

1. **Phát hiện theo câu chữ (open-vocabulary detection):** đưa vào ảnh + câu chữ ("food"), model trả box + điểm tin cậy.
   Không cần train. Ví dụ: Grounding DINO, OWLv2, YOLO-World, Florence-2.
2. **Phát hiện + phân đoạn (detector + SAM):** lấy box của detector làm gợi ý cho SAM để ra mask, rồi lấy box bao quanh mask.
   Box thường **ôm sát thức ăn hơn** (đây là ý tưởng của Grounded-SAM). SAM 3 làm cả hai việc chỉ từ câu chữ.
3. **Model train sẵn trên dataset món ăn:** chính xác với đúng các món đã học, nhưng chỉ nhận diện được tập món cố định.
4. **Mô hình thị giác-ngôn ngữ (VLM) có grounding** (Qwen2.5-VL, Florence-2): vừa gọi tên món vừa trả box; nặng và chậm hơn.

## 2. Bảng so sánh

Cột "Độ chính xác" ghi số liệu trong paper gốc (trên COCO/LVIS, không phải món ăn). Số liệu trên món ăn xem kết quả bước 01
(`results/01_compare/results.csv`). VRAM/tốc độ trên T4 là ước lượng, bước 01 đo lại.

| Model | Độ chính xác (paper) | Ôm sát thức ăn hay cả đĩa? | VRAM (T4) | Tốc độ (T4, ước lượng) | License | Cài trên Kaggle | Paper / repo |
|---|---|---|---|---|---|---|---|
| **Grounding DINO** (tiny 172M / base 233M) | 52.5 AP zero-shot trên COCO (bản lớn nhất) | Thường ôm món/bát; ảnh nhiều món hay ra thêm box cả khay (ta lọc được) | ~2–3 GB | ~0.2–0.4 s/ảnh | Apache-2.0 | Dễ: có sẵn trong `transformers` | [arXiv 2303.05499](https://arxiv.org/abs/2303.05499), [repo](https://github.com/IDEA-Research/GroundingDINO) |
| **OWLv2** (base/large) | LVIS rare: 31.2 → 44.6 AP (bản L/14) | Box khá sát; điểm tin cậy thấp (~0.1–0.4) nên khó chọn ngưỡng | ~2–4 GB | ~0.2–0.5 s/ảnh (ảnh đầu vào 960×960) | Apache-2.0 | Dễ: có sẵn trong `transformers` | [arXiv 2306.09683](https://arxiv.org/abs/2306.09683), [scenic](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) |
| **YOLO-World v2** (s→x) | 35.4 AP zero-shot LVIS, 52 FPS trên V100 | Box sát nhưng hay bỏ sót món lạ / nhiều món | ~1–2 GB | ~0.02–0.05 s/ảnh (nhanh nhất) | GPL-3.0 (repo gốc), AGPL-3.0 (Ultralytics) | Dễ: `pip install ultralytics` (tự cài CLIP) | [arXiv 2401.17270](https://arxiv.org/abs/2401.17270), [repo](https://github.com/AILab-CVC/YOLO-World) |
| **Florence-2** (0.23B / 0.77B) | Mạnh ở nhiều tác vụ (caption, OD, grounding) | Tùy prompt; **không trả điểm tin cậy** nên không lọc ngưỡng và không tính được mAP | ~2–4 GB | ~0.3–0.6 s/ảnh | MIT | Trung bình: `trust_remote_code`, hay vướng phiên bản `transformers`/`flash_attn` | [arXiv 2311.06242](https://arxiv.org/abs/2311.06242), [HF](https://huggingface.co/microsoft/Florence-2-large) |
| **Grounded-SAM / Grounded-SAM-2** (Grounding DINO + SAM/SAM 2) | Như Grounding DINO cho box; mask chất lượng cao | **Sát nhất** khi SAM tách được thức ăn khỏi bát; có lúc SAM lấy cả bát | +1–3 GB cho SAM | +0.05–0.2 s/box | Apache-2.0 | Repo gốc khó build; ta tự ghép (HF + Ultralytics SAM 2.1) | [arXiv 2401.14159](https://arxiv.org/abs/2401.14159), [repo](https://github.com/IDEA-Research/Grounded-SAM-2) |
| **SAM 2 / 2.1** (chỉ phân đoạn, cần box/điểm gợi ý) | SOTA phân đoạn theo gợi ý (ảnh + video) | Dùng để siết box từ mask | ~1–3 GB | ~0.05–0.1 s/ảnh | Apache-2.0 | Dễ: `ultralytics` (`sam2.1_b.pt`) hoặc `transformers` | [arXiv 2408.00714](https://arxiv.org/abs/2408.00714), [repo](https://github.com/facebookresearch/sam2) |
| **SAM 3** (nhận câu chữ → mask mọi đối tượng) | Mới (11/2025), mạnh ở phân đoạn theo khái niệm | Hứa hẹn ôm sát nhất (mask thức ăn trực tiếp) | ~4–6 GB (fp32) | ~0.3–1 s/ảnh | SAM License (riêng của Meta) | Khó hơn: checkpoint trên HF phải xin quyền truy cập | [arXiv 2511.16719](https://arxiv.org/abs/2511.16719), [repo](https://github.com/facebookresearch/sam3) |
| **YOLOE** (open-vocab + seg, nối tiếp YOLO-World) | Nhanh, có mask | Mask → box sát | ~1–2 GB | rất nhanh | AGPL-3.0 | Dễ: `ultralytics` | [arXiv 2503.07465](https://arxiv.org/abs/2503.07465) |
| **Model train sẵn trên dữ liệu món ăn** (YOLO trên UEC-256/UECFoodPix, FoodSAM, model trên Roboflow Universe) | Cao với đúng món đã học | Tùy nhãn: UEC-256 ôm cả đĩa; UECFoodPix / FoodSeg103 có mask thức ăn | ~1–3 GB | nhanh | Tùy model (nhiều bản chỉ cho nghiên cứu) | Tùy model | [FoodSAM arXiv 2308.05938](https://arxiv.org/abs/2308.05938), [FoodSeg103 arXiv 2105.05409](https://arxiv.org/abs/2105.05409), [UECFoodPix Complete](https://mm.cs.uec.ac.jp/uecfoodpix/) |
| **VLM có grounding** (Qwen2.5-VL 3B/7B) | Gọi tên món + box dạng JSON | Tùy câu hỏi | 3B ~8 GB; 7B ~16 GB (vượt T4 nếu không lượng tử hóa) | 1–3 s/ảnh | Tùy bản (Apache-2.0 / Qwen license) | Trung bình | [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) |
| **autodistill** (công cụ, không phải model) | — | Bọc sẵn GroundedSAM, GroundingDINO, OWLv2, YOLO-World, Florence-2: gán nhãn cả thư mục rồi train YOLO | như model bên trong | — | Apache-2.0 | Dễ dùng nhưng hay xung đột phiên bản với torch/transformers mới của Kaggle; ít quyền chỉnh hậu xử lý | [repo](https://github.com/autodistill/autodistill) |

Nhận xét chung:
- **Vấn đề "ôm cả đĩa":** detector học từ COCO/Objects365 nên thường coi "bát phở" là một vật. Hai cách xử lý: (1) lấy box
  từ mask SAM (đã cài: biến thể `+sam`, chỉ cho box co lại bên trong box gốc); (2) thử prompt khác ("food on the plate", tên món).
- **Lưu ý khi đánh giá trên UECFOOD-256:** box thật của UEC **ôm cả bát/đĩa** (đã kiểm tra bằng hình), nên ở IoU 0.5 các box
  ôm sát thức ăn (đúng ý mình) bị phạt. Bước 01 báo thêm `ap30` (IoU 0.3) và `area_ratio` (box dự đoán / box thật) để thấy điều này.
- **autodistill** làm đúng luồng ta cần, nhưng ta tự viết (gói `foodbox/`) để kiểm soát phiên bản, hậu xử lý (NMS, lọc diện tích,
  bỏ box cả khay, `food_other`) và đo đạc.

## 3. Đề xuất thử (bước 01)

1. **Grounding DINO base** (+SAM 2.1): mạnh nhất về zero-shot, cài dễ, có điểm tin cậy → ứng viên chính.
2. **OWLv2 base** (+SAM 2.1): box sát, cùng license Apache, để đối chiếu với Grounding DINO.
3. **YOLO-World v2-x** (+SAM 2.1): nhanh gấp ~10 lần, hợp khi dataset lớn hoặc cần chạy trong web app.

Florence-2 không có điểm tin cậy (không tính được mAP, không lọc theo ngưỡng được) nên chưa đưa vào so sánh. SAM 3 đáng thử
ở vòng sau nếu box từ SAM 2.1 vẫn còn dính đĩa (cần xin quyền checkpoint trên HF).

## 4. Kết quả bước 01 (200 ảnh UECFOOD-256 mini, T4; chi tiết và lịch sử ở `NOTES.md`, v1.1)

| Model | AP@0.5 | AP@0.3 | Recall | IoU TB | s/ảnh | VRAM |
|---|---|---|---|---|---|---|
| YOLO-World v2-x | 0.735 | 0.834 | 0.683 | 0.596 | 0.057 | 1.4 GB |
| Grounding DINO base | 0.732 | 0.836 | 0.717 | 0.611 | 0.582 | 2.3 GB |
| OWLv2 base | 0.386 | 0.475 | 0.503 | 0.427 | 0.379 | 0.8 GB |

Thêm SAM 2.1 không tăng điểm (box thật UEC ôm cả bát, detector đã ôm sát thức ăn sẵn) và chậm thêm ~0.3 s/ảnh.
Đề xuất: Grounding DINO (chính xác nhất, recall cao) hoặc YOLO-World (gần bằng, nhanh gấp 10 lần).
