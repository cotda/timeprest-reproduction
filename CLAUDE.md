# Dự án: Reproduction TiMePReSt

## Mục tiêu
Tái hiện TiMePReSt (Time and memory efficient pipeline parallel DNN training with removed staleness,
FGCS vol.178, 2026) dựa trên PipeDream. Paper: `1-s2.0-S0167739X25005540-main.pdf` (thư mục gốc).
Ghi chú tóm tắt paper: `paper_notes.md` (đọc file này trước, chỉ mở PDF khi cần chi tiết).
Tiến độ được theo dõi qua git: mỗi thay đổi là một commit nhỏ, message rõ ràng (tiếng Việt hoặc Anh).

## Cấu trúc
- `pipedream/` — code gốc PipeDream (msr-fiddle). CHỈ ĐỌC ĐỂ THAM KHẢO, KHÔNG SỬA.
- `timeprest/` — code mới (PyTorch hiện đại), là một package import được.
- `timeprest/checks/` — các bước kiểm tra trước khi chạy dài (xem mục "Kiểm tra trước khi chạy dài").
- `configs/` — file cấu hình YAML (quick.yaml cho kiểm tra; full_timeprest.yaml / full_pipedream.yaml cho chạy dài, kế thừa base_cifar100.yaml).
- `notebooks/` — notebook Colab / Kaggle, chỉ chứa các cell gọi lệnh, không chứa logic.
- `results/` — log CSV/JSON, biểu đồ (không commit checkpoint).

## Quy trình chạy (người dùng tự chạy trên Colab)
Agent viết code ở máy local; người dùng push lên git rồi chạy trên Colab. Agent KHÔNG chạy được Colab,
nên mọi thứ phải chạy được bằng vài lệnh copy-paste, và in kết quả rõ ràng để người dùng dán lại cho agent.
Notebook `notebooks/phase1_colab.ipynb` gồm các cell theo thứ tự:
1. Mount Google Drive, clone/pull repo, `pip install -e .`
2. In môi trường: GPU, VRAM, phiên bản torch/CUDA.
3. `python -m timeprest.checks --all` → in bảng PASS/FAIL từng bước.
4. Chạy ngắn: `python -m timeprest.train --config configs/quick.yaml`
5. Chạy dài: `python -m timeprest.train --config configs/full_timeprest.yaml --resume` (và full_pipedream.yaml)
Checkpoint và log của chạy dài lưu vào Google Drive (Colab có thể ngắt), hỗ trợ `--resume`.
Script chạy dài từ chối chạy nếu file kết quả của bước 3 chưa PASS hết (trừ khi có cờ `--force`).

## Kiểm tra trước khi chạy dài (timeprest/checks)
Mỗi check chạy trong vài giây đến vài phút, in PASS/FAIL kèm số liệu:
1. Môi trường & seed: cùng seed chạy 2 lần cho cùng loss (sai số nhỏ).
2. Loss khởi tạo hợp lý: xấp xỉ ln(số lớp) (vd. ~2.30 với 10 lớp); output đúng shape; không NaN/Inf.
3. Tương đương gradient: pipeline mô phỏng ở cấu hình không staleness (có weight stashing) cho gradient
   khớp huấn luyện thường (`torch.allclose` với tolerance ghi rõ).
4. Kiểm tra cơ chế TiMePReSt: log phiên bản trọng số dùng ở forward/backward từng micro-batch,
   xác nhận đúng mô tả paper (bỏ horizontal stashing, version difference v=1 khi W ≤ N+1).
5. Overfit 1 batch nhỏ: loss train về gần 0 trong vài trăm bước → mô hình và backward học được.
   (Đây là check cố ý overfit; không liên quan tới overfit khi chạy thật.)
6. Chạy ngắn (subset dữ liệu, 1–2 epoch): loss giảm, val accuracy > ngẫu nhiên, đo thời gian/epoch
   và peak memory, ước lượng tổng thời gian chạy dài.
7. Checkpoint/resume: dừng giữa chừng rồi resume, loss tiếp tục liền mạch.
Khi chạy dài: log mỗi epoch train loss/acc, val loss/acc, khoảng cách train–val (cảnh báo overfit
nếu val loss tăng liên tục trong khi train loss giảm), thời gian, peak memory.

## Ràng buộc kỹ thuật
- PipeDream gốc cần PyTorch cũ + `pre_hook.patch` (build lại PyTorch), không làm được trên Colab/Kaggle.
  KHÔNG chạy `pipedream/runtime`; chỉ tham khảo logic (1F1B, weight stashing).
- GĐ1 dùng 1 GPU Colab; GĐ2 dùng `torch.distributed` (NCCL) + `torchrun` trên Kaggle T4 ×2
  (~15GB/GPU, không bf16 → fp32 hoặc fp16 AMP). Viết code GĐ1 sao cho mở rộng sang GĐ2 dễ dàng.
- Dataset/model có thể nhỏ hơn paper (vd. CIFAR-10 thay ImageNet). Mọi sai khác ghi vào mục
  "Sai khác so với paper" trong `paper_notes.md`.

## Các cơ chế cần cài đặt (đối chiếu paper, mục 3)
1. Baseline: PipeDream 1F1B + weight stashing (horizontal + vertical).
2. TiMePReSt: bỏ horizontal weight stashing, giữ vertical synchronization.
3. Lịch nF1B: chia mini-batch thành N micro-batch, forward hết N, lấy loss trung bình, một lượt backward.
4. Điều kiện W ≤ N+1 để version difference v = 1 (W = số worker/GPU).
Khi không chắc chi tiết nào, trích đúng mục/trang trong paper hoặc hỏi lại, không tự suy đoán.

## Giai đoạn (chỉ sang giai đoạn sau khi giai đoạn trước đạt tiêu chí)
- GĐ1 — Colab, 1 GPU: mô phỏng pipeline trên 1 GPU (các stage chạy tuần tự). Tiêu chí: tất cả check PASS,
  chạy dài hội tụ, có số liệu thời gian và bộ nhớ cho baseline và TiMePReSt.
- GĐ2 — Kaggle T4 ×2: pipeline thật 2 stage. Tiêu chí: chạy ổn định, không deadlock, so sánh được
  thời gian/epoch và peak memory giữa baseline và TiMePReSt.
- GĐ3 — Đánh giá: baseline, ablation (có/không weight stashing, thay N), so với số liệu paper;
  báo cáo accuracy, thời gian, bộ nhớ, số epoch để hội tụ.

## Quy tắc làm việc
- Trước khi code lớn: đề xuất kế hoạch, chờ xác nhận.
- Unit test nhỏ trong `timeprest/tests/`, chạy được trên CPU (`pytest`) trước khi đưa lên Colab.
- Cố định seed; mọi tham số nằm trong config, không hard-code trong notebook.
- Khi người dùng dán output từ Colab: phân tích, chỉ ra check nào lỗi và nguyên nhân khả dĩ trước khi sửa code.
