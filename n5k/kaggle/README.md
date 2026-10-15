# Chạy n5k trên Kaggle bằng API (agent tự đẩy và theo dõi)

- Xác thực: token mới (`KGAT_...`) ở `~/.kaggle/access_token` (KHÔNG dùng `kaggle.json`, không commit token). Tài khoản `kaggleolap`.
- Code: `python n5k/kaggle/push_code.py -m "..."` đóng gói `timeprest/ n5k/ configs/ pyproject.toml` từ thư mục làm việc
  vào dataset riêng tư `kaggleolap/timeprest-code` (`COMMIT.txt` = HEAD, thêm `-dirty` nếu có thay đổi chưa commit).
  Kaggle tự giải nén zip thành thư mục `timeprest_code/`.
- Kernel: mỗi lần chạy là một thư mục `runK/` (`kernel-metadata.json` + script). Đẩy từ `n5k/kaggle/`:
  `kaggle kernels push -p run1 --accelerator NvidiaTeslaT4` → 2 × Tesla T4 15 GB (đã kiểm tra 2026-10-14; Python 3.13,
  torch 2.11+cu128, 4 CPU, 31 GB RAM, 20 GB `/kaggle/working`, có Internet).
- Đường dẫn input: `/kaggle/input/datasets/<owner>/<dataset>/` (Nutrition5k: `/kaggle/input/datasets/gillesokhin/nutrition5k-dataset`, 23 GB).
  Không duyệt `imagery/` khi tìm file: walk toàn bộ mất ~6 phút.
- Theo dõi: `kaggle kernels status <id>`; tải kết quả về `n5k/results/kaggle/<run>/`: `PYTHONUTF8=1 kaggle kernels output <id> -p DIR [--file-pattern REGEX]`
  (thiếu `PYTHONUTF8=1` thì CLI trên Windows ghi log rỗng vì lỗi mã hóa). Đường dẫn `-p` nên là tương đối (CLI lỗi với đường dẫn tuyệt đối Windows khi upload).
- Từ lúc đẩy tới lúc script chạy: ~10 phút (dựng máy + gắn dataset 23 GB).
- **Không có log khi đang chạy:** với kernel commit, `kaggle kernels logs`/`files` và trang web (kể cả Active Events) đều trống
  tới khi phiên kết thúc; phiên bị hủy thì không còn log. API cũng không hủy được (cần session id mà API không trả về), phải hủy trên web.
  Vì vậy mỗi kernel phải ngắn và mọi bước có giới hạn thời gian: dùng `kernel_lib.run(steps, budget_min=...)`
  (lệnh chạy qua coreutils `timeout`, ghi `logs/<bước>.log` và `RUN_SUMMARY.json`, tự kết thúc trước `budget_min`).
  Script kernel chỉ tìm thư mục code (`/kaggle/input/datasets/*/timeprest-code/timeprest_code`) rồi gọi `kernel_lib`.
- **Không gắn dataset Nutrition5k gốc vào kernel GPU:** Kaggle tải cả 175.602 file / 12,2 GiB vào phiên
  ("Adding data ...") trước khi chạy script. Trong phiên tương tác đã thấy đứng lâu ở 99,9% (2026-10-14); với kernel commit
  `run0` thì script bắt đầu ~9 phút sau khi đẩy (bước này không phải nguyên nhân chắc chắn của lần 1 chạy 6 giờ, nguyên nhân
  đó chưa rõ). Kernel commit `run0b` (CPU, cache 320, 2026-10-14) gắn dataset gốc và vẫn RUNNING sau hơn 87 phút, trong khi `run0` chỉ cần ~10 phút. Kernel thử lại `run0c`, cùng nội dung và đẩy lúc 04:51, lại chạy script ngay và xong trong ~1 phút; đến 14:04 `run0b` vẫn RUNNING. Vậy bước gắn dataset gốc có lúc bị treo, và đây là nghi phạm chính của lần 1. Nếu một kernel gắn dataset gốc chạy quá 30 phút, đẩy một kernel mới có slug khác thay vì chờ (kernel cũ phải hủy trên web). Gắn cache nhỏ vẫn nhanh và gọn hơn. Tải từng file qua API không được (HTTP 429 sau vài chục file).
  Cách làm: `run0/` (CPU, không tốn quota GPU) gắn dataset gốc và tạo `n5k_cache/`; kernel GPU chỉ gắn output của run0.
- Các lần chạy: `run1/` (đã hủy sau ~6 giờ, không có log, chỉ để lưu vết), `run0/` (CPU: tạo cache),
  `run1a/` (weights + báo cáo dữ liệu + checks, gắn run0), `run1b/` (sweep, gắn run0 + run1a để có cache và `n5k_results/checks`).
  `run0b/`, `run0c/` (CPU: cache 320; run0c là lần thử lại của run0b), `run5a/` (ảnh 224: checks, single, DDP smoke), `run5b/` (ảnh 320: partition, single; gắn run0 + cache 320). `run6_s{0,1,2}/` (B6: chạy dài 30 epoch, mỗi seed một kernel), `run6_bench/` (bench_comm riêng). `run7_timing/` (thời gian single theo thứ tự chạy), `run8_2s/` (two-stream so với một nhánh, single), `run9_2s_part/` (partition của two-stream, checks D0–D4).
- Kaggle cho tối đa **2 phiên GPU batch cùng lúc** ("Maximum batch GPU session count of 2 reached").
- `torchrun` chạy worker trong session riêng, nên `timeout` không dừng được worker. `kernel_lib.run_step` tự dừng mọi tiến trình con và cháu còn sống khi một bước kết thúc, và chép `n5k_results` sau mỗi bước (sửa sau khi `run6_s0` lần 1 treo hơn 3 giờ không có log).
- `fetch_overhead.py`: tải riêng CSV + ảnh overhead về máy; bị giới hạn tốc độ (429), chỉ dùng khi cần vài file.
- Sau khi đẩy code, chờ `kaggle datasets list --mine` hiện `lastUpdated` mới (giờ UTC) rồi mới đẩy kernel.
