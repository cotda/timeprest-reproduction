# So sánh độ chính xác với các nghiên cứu khác trên Nutrition5k

- Viết lúc: 2026-10-15 12:05 (UTC+8) = 04:05 UTC. Số liệu tài liệu tra cứu ngày 2026-10-15.
- Số liệu của mình: B6 (xem `2026-10-15_1205_B6-chay-dai.md`). Tập test 486 món, trung bình 267,4 kcal/món.
- Sai số calo tính theo % so với giá trị trung bình (cùng cách tính với các bài dưới).

| Công trình | Đầu vào | Sai số calo |
|---|---|---|
| Exp 3 cũ trong repo (ResNet-50) | RGB | 77 kcal |
| Nutrition5k gốc (CVPR 2021), dự đoán trực tiếp | RGB | 70,6 kcal / 26,1% |
| Nutrition5k gốc, depth làm kênh thứ 4 | RGB-D | 47,6 kcal / 18,8% |
| Nutrition5k gốc, ước thể tích từ depth | RGB-D | 41,3 kcal / 16,5% |
| **Mình: single / PipeDream** | RGB-D (4 kênh) | **51,9 kcal / 19,5%** |
| **Mình: TiMePReSt** | RGB-D (4 kênh) | 56,3 kcal / 21,1% |
| DPF-Nutrition (2023) | RGB + depth dự đoán | 14,7% |
| RDINet (2026) | RGB-D | 14,9% |
| IGSMNet (2025) | RGB-D + ngữ nghĩa nguyên liệu | 12,2% |

Sai số khác của mình (single / PipeDream): khối lượng 32–34 g, chất béo 3,7 g, carb 5,2 g, đạm 5,0–5,2 g. Nhận diện nguyên liệu: mAP 87%, F1 86%.

## Kết luận
1. Mô hình gần bằng cách tương ứng trong paper gốc (depth làm kênh thứ 4: 19,5% so với 18,8%), tốt hơn hẳn khi chỉ dùng RGB (26,1%) và tốt hơn Exp 3 cũ.
2. Kém các bài mới nhất 5–7 điểm %. Các bài đó tách RGB và depth thành hai nhánh, có module hợp nhất và backbone lớn hơn. Đây là lý do thử two-stream (xem `2026-10-15_1205_two-stream-quyet-dinh.md`).
3. Không so sánh tuyệt đối được: mỗi bài chia train/test khác nhau (của mình chỉ có 2270 món train), và số liệu là do các bài tự báo cáo.
4. Chưa tìm thấy chuẩn so sánh chung cho nhận diện nguyên liệu và gram từng nguyên liệu.

## Nguồn
- Nutrition5k: https://arxiv.org/pdf/2103.03375
- Image Based Food Energy Estimation With Depth Domain Adaptation (trích số liệu Nutrition5k): https://arxiv.org/pdf/2208.12153
- IGSMNet: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12610319/
- RDINet: https://www.mdpi.com/2076-3417/16/1/454
- OmniFood8K (bảng có DPF-Nutrition): https://arxiv.org/pdf/2604.12356
