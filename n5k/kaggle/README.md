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
  `run0b/`, `run0c/` (CPU: cache 320; run0c là lần thử lại của run0b), `run5a/` (ảnh 224: checks, single, DDP smoke), `run5b/` (ảnh 320: partition, single; gắn run0 + cache 320). `run6_s{0,1,2}/` (B6: chạy dài 30 epoch, mỗi seed một kernel), `run6_bench/` (bench_comm riêng). `run7_timing/` (thời gian single theo thứ tự chạy), `run8_2s/` (two-stream so với một nhánh, single), `run9_2s_part/` (partition của two-stream, checks D0–D4), `run10_2s_sweep/` (two-stream: sweep lr {0.02, 0.01} trên pipeline thật + thời gian 4 hệ cùng máy), `run11_b7/` (B7: huấn luyện lại TiMePReSt/single seed 0, giữ checkpoint, bản đồ calories + deletion test), `run12_tp_lr/` (TiMePReSt lr {0.02, 0.005} × 3 seed, 30 epoch), `run13_pd/` (PipeDream: B7 seed 0 + lr {0.01, 0.005} × 3 seed, 30 epoch), `run14_60ep/` (tài khoản ktanvn: TiMePReSt + PipeDream 60 epoch, seed 0), `run15_tp60_lr02/` (ktanvn: TiMePReSt lr 0.02, 60 epoch, seed 0), `run16_s1/` (kaggleolap) và `run16_s2/` (ktanvn): 60 epoch seed 1/2, TiMePReSt lr 0.01 + 0.02, PipeDream.
- Kaggle cho tối đa **2 phiên GPU batch cùng lúc** ("Maximum batch GPU session count of 2 reached").
- `torchrun` chạy worker trong session riêng, nên `timeout` không dừng được worker. `kernel_lib.run_step` tự dừng mọi tiến trình con và cháu còn sống khi một bước kết thúc, và chép `n5k_results` sau mỗi bước (sửa sau khi `run6_s0` lần 1 treo hơn 3 giờ không có log).
- `fetch_overhead.py`: tải riêng CSV + ảnh overhead về máy; bị giới hạn tốc độ (429), chỉ dùng khi cần vài file.
- Sau khi đẩy code, chờ `kaggle datasets list --mine` hiện `lastUpdated` mới (giờ UTC) rồi mới đẩy kernel.

## Hai tài khoản (2026-10-17)
- Mỗi tài khoản Kaggle chỉ chạy tối đa 2 phiên GPU cùng lúc (mỗi phiên = 1 máy 2×T4). Tài khoản thứ hai: `ktanvn`. Đã kiểm chứng 2026-10-17: run16_s1 (kaggleolap) và run16_s2 (ktanvn) chạy song song, mỗi kernel 3 lần chạy 60 epoch, khoảng 73 phút.
- Token của các tài khoản ở `kaggle_accounts.json` (gốc repo, **đã ignore**, không commit, không in ra). Quản lý bằng `n5k/kaggle/accounts.py`:
  - `python n5k/kaggle/accounts.py list` (token che bớt), `add <user> --token-file <file>`, `whoami <user>`.
  - `python n5k/kaggle/accounts.py run <user> -- kaggle kernels status <user>/<slug>` (mọi lệnh kaggle với token của tài khoản đó).
  - `python n5k/kaggle/accounts.py push-code <user> -m "..."` → dataset `<user>/timeprest-code`.
  - `python n5k/kaggle/accounts.py push-kernel <user> n5k/kaggle/runX [--cpu]`: đẩy bản sao của thư mục kernel, đổi mọi id `kaggleolap/...` trong `kernel-metadata.json` sang `<user>/...` (thư mục trong repo không đổi).
- Tài khoản `kaggleolap` vẫn là mặc định (`~/.kaggle/access_token`); các lệnh `kaggle ...` gọi thẳng vẫn dùng nó.
- `ktanvn` cần dataset code riêng (`ktanvn/timeprest-code`) và cache riêng (`ktanvn/n5k-run0-cache`, kernel `run0` chạy CPU), vì dataset/kernel riêng tư của tài khoản này không gắn được vào tài khoản kia. Mỗi lần đổi code phải `push-code` cho **cả hai** tài khoản.
- Log khi kernel đang chạy (cập nhật 2026-10-17): **trang web của kernel có hiện log trực tiếp**; API (`kaggle kernels logs`, `output`) chỉ trả log sau khi kernel xong. Kernel "RUNNING" mà trên web không có dòng log nào sau khoảng 15 phút thì nhiều khả năng bị treo trước khi script chạy (như run0b, run6_s0 lần 1, run14 trên ktanvn): người dùng hủy trên web, rồi đẩy lại.
- `ktanvn/n5k-run0-cache`: đẩy 17:27 UTC, script mới chạy lúc 18:08 (41 phút gắn dataset gốc 23 GB), tạo cache 1 phút; dữ liệu khớp kaggleolap (2270/486/486, K=195).
