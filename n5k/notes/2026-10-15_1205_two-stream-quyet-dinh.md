# Quyết định thử mô hình two-stream (R50 + R50)

- Viết lúc: 2026-10-15 12:05 (UTC+8) = 04:05 UTC.
- Người dùng chọn R50 + R50 ngày 2026-10-15. Đã ghi vào `n5k/PLAN.md` mục 10.
- Trạng thái lúc viết: code xong, test CPU PASS (50 test n5k). Kernel `kaggleolap/n5k-run8-two-stream` (`n5k/kaggle/run8_2s/`) đẩy lúc 2026-10-15 12:06 (04:06 UTC). Kết quả sẽ ghi ở một note riêng.

## Lý do
- Sai số calo 19,5% chỉ ngang mức "depth làm kênh thứ 4" của paper gốc; các bài 12–15% đều tách RGB và depth thành hai nhánh.
- Mô hình nặng hơn cũng là trường hợp mà pipeline song song có lý do để dùng.

## Thiết kế
- Hai ResNet-50 (nhánh RGB 3 kênh, nhánh depth 1 kênh), đều khởi tạo từ ImageNet. conv1 của nhánh depth là trung bình 3 bộ lọc RGB.
- Hợp nhất muộn: nối feature 2048 + 2048 kênh → head CAM như cũ.
- Vẫn là chuỗi 18 khối, mỗi khối chứa cả hai nhánh song song, nên pipeline, DDP và cách chia stage không phải sửa. Model mới tên `n5k_resnet50_2s`; model cũ giữ nguyên.

## Ước tính (chưa đo)
- Tính toán khoảng 2 lần; single khoảng 45–50 s/epoch; peak khoảng 11 GB (vẫn vừa 1 T4); PipeDream GPU0 có thể khoảng 16 GB, nên phải chia lại stage.

## Quy tắc chọn (đặt trước khi có số liệu)
- Chạy thử `run8_2s`: single, chỉ GPU0, lr 0.02, 8 epoch, seed 0 và 1. Chạy xen kẽ một nhánh / two-stream để máy chậm dần không lệch về một bên.
- Lấy best val kcal và mAP ở epoch đó, trung bình 2 seed. Dùng two-stream nếu `(kcal2s/kcal1s − 1) + (1 − mAP2s/mAP1s) < −0,05`. Nếu không thì giữ mô hình một nhánh và ghi kết quả âm tính.
