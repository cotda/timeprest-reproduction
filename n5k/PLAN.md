# Nutrition5k + TiMePReSt: nhận diện nguyên liệu, hàm lượng và vị trí — KẾ HOẠCH (chốt 2026-10-14)

> Kế hoạch đã chốt với người dùng ngày 2026-10-14. Mọi bước của task này làm theo đúng file này.
> Muốn đổi lựa chọn nào thì ghi quyết định mới (ngày, lý do) vào mục 10 trước khi sửa code, không sửa ngầm.
> Quy ước tham chiếu: "mục k" = mục trong file này; "PN §x" = mục trong `paper_notes.md` (kết quả tái hiện TiMePReSt).

## 1. Phạm vi task

- Task (mentor nhóm FoodAI xác nhận): kết hợp phương pháp song song đã tìm hiểu (TiMePReSt) với bài Nutrition5k đã chạy (`nutrition5k_distributed_project/`, Exp 1–5).
- Gồm 3 việc trong **một** mô hình: (1) **nhận diện** món ăn có những nguyên liệu nào; (2) **hàm lượng**: gram từng nguyên liệu, rồi suy ra kcal / fat / carb / protein; (3) **vị trí**: vùng nào trên ảnh chứa nguyên liệu đó và mang bao nhiêu calories (bản đồ calories). Nutrition5k chỉ là dataset.
- Huấn luyện bằng pipeline 2 GPU (Kaggle T4×2) với TiMePReSt, so sánh với các hệ ở mục 7.
- Workload này **không có trong paper TiMePReSt**: hồi quy multi-task, backbone pretrained. Mọi lựa chọn dưới đây là lựa chọn triển khai của project trừ khi ghi khác.

## 2. Tổ chức file (tách riêng khỏi phần tái hiện TiMePReSt)

| Thứ | Vị trí |
|---|---|
| Kế hoạch, nhật ký quyết định, ghi chép kết quả | `n5k/PLAN.md` (file này) — **không** ghi vào `paper_notes.md` |
| Code riêng của task (data, model, task loss/metric, train, checks, DDP, vẽ) | `timeprest/n5k/` |
| Unit test CPU | `timeprest/n5k/tests/` |
| Config | `n5k/configs/` |
| Notebook Kaggle | `n5k/notebooks/` |
| Kết quả (CSV/JSON/hình; không commit checkpoint) | `n5k/results/` |

Code cũ (`timeprest/*.py`, `timeprest/dist/*.py`) chỉ được sửa ở mức **móc nối tối thiểu**, hành vi cũ giữ nguyên:
- `engine.py`, `dist/runtime.py`: loss / metric đi qua một đối tượng `task` (mặc định = phân loại, tính đúng như cũ).
- `models.py`: cho phép đăng ký thêm model ngoài (để `build_stages` dựng được model của task này).
- Test cũ phải PASS không đổi sau mỗi lần sửa.

## 3. Bằng chứng từ các thí nghiệm trước dùng để chọn thiết kế

| Phát hiện | Nguồn | Hệ quả |
|---|---|---|
| Chất lượng cuối TiMePReSt ≈ PipeDream (3 seed) | PN §24.5 | Không kỳ vọng MAE cuối tốt hơn |
| nF1B học nhanh hơn ở giữa quá trình (VGG +1.9 điểm, ResNet-50 +3 điểm) | PN §24.5, §25.2 | Luận điểm chính: MAE theo epoch và tại cùng mốc thời gian |
| ResNet-50 cùng máy: TiMePReSt chậm hơn 5–8 %/epoch | PN §25.2 | Cùng máy có thể chậm hơn; cần bench mạng giả lập |
| Nhanh hơn 9–16 % khi mạng 1–2 Gbit/s | PN §23.1, §24.2, §25.2 | Chạy `bench_comm` ∞ / 2 / 1 Gbit/s |
| Bộ nhớ `graph` ≈ PipeDream; pipeline đồng bộ ít hơn ~3 lần | PN §22.10, §26.1 | Không tuyên bố tiết kiệm bộ nhớ |
| `graph` phân kỳ khi lr cao + warmup ngắn | PN §22.7 | Warmup + sweep lr |
| Exp cũ: ResNet18 102 ảnh/s, ResNet50 82 ảnh/s (1 GPU) | `nutrition5k_distributed_project/*/results` | Nghi nghẽn đọc PNG (suy luận, chưa đo): phải cache ảnh trước khi so thời gian |
| Exp 1–2: MAE ResNet18 ~204–243 kcal, ResNet50 77 kcal cùng setup | như trên | Nghi do target không chuẩn hóa (giả thuyết): bắt buộc chuẩn hóa target |

## 4. Mô hình và nhãn

```
RGB-D 4 kênh 224×224 → ResNet-50 (stem imagenet, pretrained ImageNet) → feature 2048×7×7
  → conv 1×1 → K bản đồ CAM (mỗi nguyên liệu một bản đồ)
       ├─ global pool → logit có/không nguyên liệu k                 (nhận diện, BCE)
       ├─ nhánh conv 1×1 + global pool + softplus → gram_k             (hàm lượng)
       │     kcal/fat/carb/protein = Σ_k gram_k × hàm lượng trên mỗi gram của k
       └─ ReLU(CAM_k) chuẩn hóa × kcal_k, cộng theo k → bản đồ calories (vị trí; chỉ tính lúc eval)
```

- K = mọi nguyên liệu xuất hiện trong train (~195; Exp 5 có vocab 197 gồm PAD/UNK). Nguyên liệu chỉ có ở val/test (Exp 5: 6 lần ở val, 3 lần ở test) không dự đoán được; tổng dinh dưỡng vẫn đánh giá với nhãn tổng của món.
- Hàm lượng trên mỗi gram lấy từ `ingredients_metadata.csv` (`cal/g`, `fat(g)`, `carb(g)`, `protein(g)`). Bước B2 phải kiểm tra Σ theo nguyên liệu có khớp `dish_nutrition_values.csv` không và ghi sai lệch.
- Kênh depth của conv1: khởi tạo bằng trung bình trọng số 3 kênh RGB pretrained.
- Loss = BCE (nhận diện) + SmoothL1 (gram, đã chuẩn hóa) + SmoothL1 (tổng kcal/fat/carb/protein/mass, đã chuẩn hóa); hệ số từng thành phần trong config, mặc định 1. Thống kê chuẩn hóa tính trên train.
- Vị trí học kiểu weakly supervised (CAM): Nutrition5k không có box/mask.

## 5. Dữ liệu

- Dataset Kaggle `gillesokhin/nutrition5k-dataset`, `imagery/realsense_overhead/<dish>/{rgb.png, depth_raw.png}`.
- **Split**: dùng lại split cố định của Exp 1–3 (`nutrition5k_distributed_project/exp_00*/*/splits/{train,val,test}.csv`, 3 bản trùng md5; 2270 / 487 / 487). Loại mẫu có depth hỏng và ghi danh sách dish_id bị loại vào log. Hiện biết `dish_1558722871` (thuộc **test**, file `depth_raw.png_.gstmp` 0 byte), nên test còn 486. Notebook `*_ModelParallel` của Exp 4–5 tạo split RGB-D khác, **không** dùng.
- Depth: cắt theo P1 / P99 của train (Exp 4: 3038 / 4860) rồi đưa về [0, 1].
- **Cache**: giải mã một lần ra uint8 `.npy` (~650 MB ở 224×224×4), như Tiny-ImageNet (PN §24.1). Augmentation áp như nhau cho RGB và depth (lật ngang/dọc; không đổi màu kênh depth).

## 6. Cấu hình TiMePReSt và huấn luyện

| Hạng mục | Chọn | Lý do |
|---|---|---|
| W (GPU / stage) | 2 | Kaggle T4×2 |
| M (mini-batch) | **64** | Bộ nhớ: ResNet-50 TIN M=192 tốn 8.4 GB GPU0 (PN §25.2); ở 224×224 activation/ảnh ~3 lần, nên M=64 ~8–9 GB (ước lượng, D4 đo thật). Khớp batch 64 của Exp cũ |
| **N (micro-batch)** | **2 (micro 32)** — người dùng chọn 2026-10-14 | W=2 nên N ≥ 1 đã cho v=1; micro-batch 16 làm BN nhiễu và tăng chi phí cố định mỗi kernel (PN §25.2) |
| Quy tắc backward | `graph` | Mặc định project (PN §22.5–22.6) |
| Precision | fp32 | `graph` không hỗ trợ autocast (PN §22.6) |
| Optimizer | SGD momentum 0.9, weight decay 1e-4 | Code chỉ có SGD (`engine.py`, `dist/runtime.py`); Eq.1 của paper là SGD. Exp cũ dùng Adam 1e-4, không so trực tiếp được |
| lr | Sweep {0.01, 0.005, 0.002} trên engine 1 GPU; mỗi hệ lấy lr tốt nhất của nó (quy tắc PN §22.9) | Fine-tune pretrained |
| Lịch lr | Warmup 2 epoch + cosine | PN §22.7 |
| Số epoch | 30 | Exp cũ 20 epoch MAE còn giảm |
| Partition | `auto` theo MACs | Head rẻ, nằm ở stage cuối; bản đồ CAM chỉ tính lúc eval, không thêm truyền |
| Truyền giữa stage | gloo (PN §21.1) | Như cũ |
| Seed | 0, 1, 2 cho mọi hệ ở mục 7 | Run ngắn (ước lượng 15–25 phút, chưa đo) |
| BN | Train mode như các run cũ | Không đổi cách xử lý BN so với PN §21 |

## 7. Các hệ so sánh và metric

| Hệ | Ghi chú |
|---|---|
| 1 GPU | Mốc tham chiếu (engine, không pipeline) |
| DDP 2 GPU | Batch 32/GPU, cùng data / model / optimizer; script `timeprest.n5k.ddp_train` |
| PipeDream M=64 | 1F1B + weight stashing, không vsync (preset `pipedream`, PN §19.4) |
| **TiMePReSt M=64, N=2** | Hệ chính |
| TiMePReSt M=64, N=4 | Ablation N |
| `bench_comm` ∞ / 2 / 1 Gbit/s | TiMePReSt và PipeDream, cùng máy |

- **Không chạy** PipeDream theo micro-batch (M=32, cách hiểu thứ hai PN §27) — người dùng chọn 2026-10-14.
- Metric: MAE kcal / fat / carb / protein / mass (val mỗi epoch; test ở checkpoint tốt nhất theo val); F1 / mAP nhận diện nguyên liệu; MAE gram và kcal theo từng nguyên liệu; MAE theo epoch và tại cùng mốc thời gian (cách tính PN §24.3); thời gian/epoch; peak memory từng GPU; tỉ lệ bận GPU.
- Vị trí (không có mask, mục 10): (1) hình minh họa bản đồ calories; (2) **deletion test**: che vùng bản đồ của nguyên liệu k xếp hạng cao nhất (theo tỉ lệ diện tích) và so mức giảm xác suất nhận diện / gram dự đoán của k với che vùng ngẫu nhiên cùng diện tích. Kết luận được phép: bản đồ **trung thực với mô hình** (chỉ đúng vùng mô hình dựa vào); **không** được tuyên bố định vị đúng vị trí thật.

## 8. Dự đoán ghi trước khi chạy

- MAE cuối TiMePReSt ≈ PipeDream.
- TiMePReSt giảm MAE nhanh hơn ở giữa quá trình (lặp lại PN §24.5 / §25.2).
- Cùng máy: DDP nhanh nhất; TiMePReSt có thể chậm hơn PipeDream ~5 %. Ở 1–2 Gbit/s TiMePReSt nhanh hơn PipeDream.
- Bộ nhớ TiMePReSt và PipeDream không khác nhau đáng kể.
- Khi báo cáo phải nói rõ: mô hình vừa 1 GPU và 2 GPU cùng máy thì DDP là lựa chọn nhanh nhất; giá trị của bài là đánh giá TiMePReSt trên một workload multi-task mới.

## 9. Các bước thực hiện (theo thứ tự, mỗi bước một commit)

| Bước | Nội dung | Điều kiện xong |
|---|---|---|
| B1 | Móc nối `task` vào `engine.py` / `dist/runtime.py` (thay 8 chỗ `cross_entropy` / `argmax` viết cố định), registry model trong `models.py`; `timeprest/n5k/task.py`: loss + metric dinh dưỡng, nhãn float [B, D] | Test cũ PASS không đổi; test mới: engine với task dinh dưỡng cho gradient khớp train thường (check 3) và đúng version (check 4); runtime 2 tiến trình CPU khớp engine |
| B2 | `timeprest/n5k/data.py`: dataset + script tạo cache; nhãn gram từng nguyên liệu; kiểm tra Σ nguyên liệu so với tổng của món | Test CPU trên dữ liệu giả; log số mẫu bị loại và sai lệch tổng |
| B3 | `timeprest/n5k/model.py`: ResNet-50 pretrained (map key torchvision → khối của `resnet50_blocks`, `downsample` → `shortcut`), conv1 4 kênh, head CAM / gram | Test: output phần RGB trùng torchvision (tolerance ghi rõ) |
| B4 | `timeprest/n5k/checks.py` (D1–D4 cho task này), `n5k/configs/n5k_quick.yaml`; sweep lr trên engine 1 GPU | D1–D4 PASS; peak memory M=64 đo thật |
| B5 | Baseline DDP `timeprest/n5k/ddp_train.py` | Chạy 2 tiến trình CPU/gloo local; cùng định dạng `metrics.csv` / `summary.json` |
| B6 | Configs `n5k/configs/*.yaml`, notebook `n5k/notebooks/n5k_kaggle.ipynb`; chạy dài 3 seed + `bench_comm` | Kết quả vào `n5k/results/runN_*/` |
| B7 | Script vẽ: nguyên liệu + bản đồ calories phủ lên ảnh + bảng hàm lượng; deletion test (mục 7) | Hình cho báo cáo; bảng deletion (vùng bản đồ so với ngẫu nhiên) |
| B8 | Ghi kết quả vào mục 11 của file này, cập nhật báo cáo | — |

## 10. Nhật ký quyết định

- 2026-10-14: phạm vi gồm nhận diện + hàm lượng + vị trí (người dùng). N = 2 (người dùng). Không chạy PipeDream M=32 (người dùng). Các lựa chọn còn lại ở mục 4–7 do agent đề xuất, người dùng chấp nhận cùng lúc.
- 2026-10-14: task có thư mục riêng `n5k/` và code riêng `timeprest/n5k/`; không ghi vào `paper_notes.md`, code cũ chỉ sửa ở mức móc nối (người dùng).
- 2026-10-14: người dùng **không tự gán mask được**. Bỏ mIoU / pointing game trên mask tự gán; thay bằng hình minh họa + deletion test (người dùng đồng ý). FoodSeg103 không dùng (không có depth, phải train thêm bản RGB và ánh xạ nhãn); chỉ xét lại nếu mentor yêu cầu độ chính xác vị trí so với thực tế.
- 2026-10-14: sweep lr gồm `single` (dùng chung cho DDP: cùng SGD trên M=64), `pipedream`, `timeprest`, 8 epoch, warmup 1 (agent đề xuất theo mục 6; chạy ở lần Kaggle 1).
- 2026-10-14: agent tự đẩy và theo dõi kernel Kaggle qua API (người dùng cho phép; `n5k/kaggle/README.md`). Lần 1 (`n5k-run1-checks-sweep`, 1 kernel gồm data + checks + sweep) chạy khoảng 6 giờ mà không xem được log (Kaggle không trả log khi kernel commit đang chạy, và cũng không lưu log của phiên bị hủy). Người dùng chọn hủy, không thu được số liệu nào. Thay đổi kỹ thuật, không đổi thiết kế thí nghiệm: (1) tạo cache bằng 16 luồng đọc song song, có in tiến độ (`data.decode_threads`, `data.progress_every`); (2) tách thành các kernel ngắn, mỗi bước có giới hạn thời gian để luôn kết thúc và lưu được log (`n5k/kaggle/kernel_lib.py`). Thứ tự: `run1a` (weights, data, checks), rồi `run1b` (sweep, gắn output của `run1a` để dùng lại cache và file checks).
- 2026-10-14: người dùng chạy thử bằng notebook tay và thấy phiên đứng ở "Adding data" (đã tải 175.533/175.602 file, 99,9%) của dataset Nutrition5k gốc. Sau đó bản commit của `run0` bắt đầu script ~9 phút sau khi đẩy và tạo cache trong 1 phút, nên nguyên nhân lần 1 chạy 6 giờ **chưa rõ** (có thể ở checks hoặc sweep; `run1a`/`run1b` có giới hạn thời gian từng bước để tìm ra). Thay đổi kỹ thuật: kernel CPU `run0` gắn dataset gốc và tạo cache; các kernel GPU (`run1a`, `run1b`) chỉ gắn output của `run0`. Nội dung cache và code tạo cache không đổi.
- 2026-10-14: D4 cho thấy pipeline lệch tải với partition `auto` `[0,9,18]` (chia theo MACs). GPU bận 0,92/0,53, peak 8,8/1,6 GB. Lý do: các block đầu có độ phân giải cao nên tốn thời gian hơn so với số MACs.
  - Paper không báo cáo độ bận GPU, chỉ nói phân hoạch "tương đối để cân bằng sử dụng GPU" (p.3) và không đưa partition cụ thể. Fig.15 cho thấy bộ nhớ các stage cùng cỡ, chênh nhau khoảng 1,5–2 lần. PipeDream gốc chia theo thời gian đo được cộng chi phí truyền (`pipedream/optimizer/`).
  - Người dùng đồng ý đo bằng kernel `run1c`: chạy D4 (4 epoch, cả hai hệ) với `[0,7,18]`, `[0,8,18]`, `[0,9,18]`.
  - Quy tắc chọn (đặt trước khi có số liệu): lấy partition có thời gian/epoch của **PipeDream** nhỏ nhất, tính trung bình các epoch từ epoch 2 trở đi; nếu chênh lệch dưới 3% thì giữ `[0,9,18]`. Hai hệ dùng chung partition.
  - Nếu đổi partition: sửa `pipeline.partition` trong `n5k_base.yaml`, chạy lại checks và chạy lại sweep lr (partition quyết định layer nào bị trễ version).
  - **Kết quả (`run1c`): chọn `[0,8,18]`** (PipeDream nhanh hơn 9,3%, vượt ngưỡng 3%). Đã sửa `n5k_base.yaml`; `n5k_single.yaml` giữ `auto` (1 stage).
- 2026-10-14 (người dùng xác nhận): sweep lần 1 cho thấy phần nguyên liệu gần như chưa học được (F1 6%, mAP 8%), trong khi tổng dinh dưỡng học tốt (điểm mở (b)). Kế hoạch:
  - (a) Log thêm, không đổi hành vi: `val_loss_{presence,gram,total}` (từng thành phần chưa nhân trọng số) và `val_gshare_*` (tỉ lệ của mỗi thành phần đã nhân trọng số trong chuẩn gradient theo output của model).
  - (b) Thêm tùy chọn `loss.ingredient_reduce: mean|sum` (mặc định `mean`, giữ như cũ). Chạy ngắn 3 biến thể trên `single`: `mean` (w=1), `w10` (mean, w_presence = w_gram = 10), `sum` (≈ w = 195). Mỗi biến thể × lr {0.05, 0.02, 0.01}, 8 epoch.
  - Quy tắc chọn (đặt trước khi có số liệu): lấy biến thể có **val ingr mAP cao nhất** ở lr tốt nhất của nó (lr tốt nhất theo val MAE kcal như cũ), với điều kiện best val MAE kcal không tệ hơn biến thể `mean` quá 5%. Không biến thể nào đạt thì giữ `mean`.
  - Sau đó sweep pipedream/timeprest bằng biến thể đã chọn và partition đã chọn, lr {0.05, 0.02, 0.01}.
- Điểm còn mở: (a) mentor có cần độ chính xác vị trí so với thực tế không (nếu có: xét FoodSeg103); (b) hệ số loss từng thành phần: giữ 1 trừ khi D4 / sweep cho thấy một thành phần lấn át.

## 11. Kết quả

Mỗi bước ghi ngày, commit code, nguồn file kết quả.

### B1 — móc nối task (2026-10-14, chưa commit)

- Code cũ sửa ở mức móc nối:
  - `timeprest/tasks.py` (mới): `ClassificationTask` = loss/metric phân loại cũ, chuyển nguyên văn từ engine/runtime.
  - `engine.py`, `dist/runtime.py`: tham số `task=None` (mặc định `ClassificationTask`); 8 chỗ `cross_entropy` / `argmax` gọi qua `self.task`. Khóa log cũ giữ nguyên (`train_acc1`, `correct`, `loss/acc1/acc5`).
  - `models.py`: `MODEL_BUILDERS` (đăng ký model ngoài). `reference.py`: tham số `loss_sum` (mặc định CE).
  - `pyproject.toml`: thêm `timeprest/n5k/tests` vào `testpaths`.
- Code mới: `timeprest/n5k/task.py` (`NutritionTask`, layout ở mục 4; metric eval: `mae_{kcal,mass,fat,carb,protein}`, `pmae_kcal`, `mae_gram_present`, `ingr_f1`, `ingr_map`), `synthetic.py` (bài toán giả float64), `dist_selftest.py` (runtime 2 tiến trình với task dinh dưỡng).
- Kiểm chứng (CPU, float64, tolerance rtol 1e-9 / atol 1e-12): 108 test cũ PASS không đổi; 16 test mới PASS (`timeprest/n5k/tests/test_task.py`):
  - loss khớp công thức viết tay; dự đoán hoàn hảo cho MAE ≈ 0, F1 = mAP = 100; mAP / F1 khớp giá trị tính tay;
  - check 3: pipeline không trễ (1 mini-batch trong pipeline) = train thường, MLP N=1/2 và ResNet N=2;
  - check 4: TiMePReSt N=2 (`graph`) khớp reference PipeDream-swap, có micro-batch forward bằng version cũ; PipeDream stash khớp gradient toàn model;
  - runtime 2 tiến trình: static trùng engine (tham số và metric eval) cho TiMePReSt MLP/ResNet và PipeDream; dynamic đúng quy tắc với version đã log.

### B2 — dữ liệu (2026-10-14, chưa commit)

- `n5k/splits/{train,val,test}.txt`: dish id lấy từ split Exp 1–3 (2270 / 487 / 487, không trùng nhau).
- `timeprest/n5k/data.py`: đọc 3 CSV bằng `csv` (không cần pandas); ảnh 224×224 (RGB bilinear, depth nearest để giá trị 0 = không đo không bị trộn); cache `.npy` (uint8 RGB, uint16 depth thô, nhãn float32) + `meta.json` ghi cuối (từ vựng K, per-gram, `gram_scale`, `total_mean/std`, depth P1/P99 trên pixel > 0, món bị loại kèm lý do, báo cáo khớp tổng). Khóa `.building` để chỉ một tiến trình giải mã; build lỗi thì gỡ khóa. Tìm cache có sẵn qua `data.cache_search`.
- Per-gram lấy từ `ingredients_metadata.csv`; nguyên liệu không có trong metadata thì lấy tỉ lệ từ `dish_ingredients.csv` (ghi trong `per_gram_fallback`).
- Lệnh: `python -m timeprest.n5k.data --config n5k/configs/n5k_quick.yaml` (in báo cáo + dòng `N5K_DATA_JSON`).
- Test (thư mục Nutrition5k giả, đúng cấu trúc, có món thiếu depth kiểu `.gstmp` 0 byte và món thiếu dòng dinh dưỡng): 7 test PASS — lý do loại đúng, từ vựng chỉ từ train, nhãn và per-gram đúng, tỉ lệ kcal của nguyên liệu lạ, thống kê chuẩn hóa, lật RGB và depth cùng nhau, dùng lại / tìm thấy cache, gỡ khóa khi lỗi.

### B3 — model (2026-10-14, chưa commit)

- `timeprest/n5k/model.py`: model `n5k_resnet50` đăng ký vào `MODEL_BUILDERS`; nạp trọng số torchvision (`IMAGENET1K_V2`) vào `resnet50_blocks(stem="imagenet")`; conv1 4 kênh (kênh depth = trung bình bộ lọc RGB); head `N5kHead` (2 conv 1×1 → K bản đồ CAM + K bản đồ gram, global average pool). Bias của head khởi tạo theo tần suất xuất hiện và gram trung bình trên train (loss ban đầu hợp lý).
- Partition `auto` ở 224×224, 4 kênh: `[0, 9, 18]`, 2.23 / 1.94 GMAC mỗi stage, activation ở biên 1024×14×14 = 0.77 MB/ảnh fp32.
- Test: 5 PASS — feature RGB trùng torchvision (rtol/atol 1e-5, BN có thống kê khác mặc định), depth = 0 cho đúng model 3 kênh, layout head và bias khởi tạo, partition với input 4 kênh.

### Dữ liệu thật (Kaggle, 2026-10-14, người dùng chạy tay `run0` trong Draft Session)
- Cache được tạo trong 1 phút (16 luồng), kích thước 778 MB. Số món train/val/test là 2270/486/486; loại 2 món: `dish_1564159636` (val, `depth_raw.png` rỗng, mới phát hiện) và `dish_1558722871` (test, thiếu depth).
- K = 195 nguyên liệu (có gram > 0 trong train), nhãn có 200 chiều.
- Tổng dinh dưỡng tính lại từ gram nguyên liệu khớp tổng của món (train: sai số tuyệt đối trung bình ≈ 1e-5 cho mọi thành phần). `per_gram_fallback` rỗng.
- Món có nguyên liệu ngoài từ vựng: val 6, test 2, chiếm 0,54% / 0,29% kcal; sai số kcal tính từ gram 1,37 / 0,76 kcal.
- Ảnh mẫu sau chuẩn hóa: trung bình RGB −0,064, depth −0,155. Kết luận: D0 (dữ liệu) đạt.

### Checks D0–D4 trên Kaggle 2×T4 (`run1a` lần 2, 2026-10-14, code hash `0064711c9199`; log ở `n5k/results/kaggle/run1a/`)
- Tất cả PASS; cả kernel mất khoảng 4 phút (checks 2,9 phút).
- D1: static order trên 2 GPU khớp engine 1 GPU (chênh lệch 0.0) cho cả pipedream và timeprest.
- D2: không recompute; TiMePReSt forward theo vertical sync, backward dùng version live; có 11 micro-batch được forward bằng version cũ hơn.
- D3: resume khớp tuyệt đối (chênh lệch 0.0).
- D4: 2 epoch trên toàn tập train.

| hệ | val MAE kcal ep1→ep2 (trung vị: 165,8) | epoch (s) | GPU bận | peak MB / GPU | ước tính 30 epoch |
|---|---|---|---|---|---|
| PipeDream M=64 | 156,0 → 150,3 | 20,0 | 0,92 / 0,53 | 8853 / 1571 | 0,17 h |
| TiMePReSt N=2 | 155,9 → 150,8 | 18,5 | 0,90 / 0,53 | 8770 / 1601 | 0,16 h |

- Nhận xét:
  - Partition [0,9,18] cân bằng theo MACs, nhưng stage 0 giữ phần lớn activation (8,8 GB so với 1,6 GB) và GPU 1 chỉ bận 53% thời gian. Điều này hạn chế lợi ích pipeline; ghi lại để bàn ở B8, chưa đổi partition (mục 6).
  - Thời gian chạy dài rất ngắn (khoảng 10 phút / 30 epoch), nên 3 seed × mọi hệ ở B6 vừa trong một kernel.

### Sweep lr lần 1 (`run1b`, 2026-10-14, partition `[0,9,18]`, 8 epoch, 1 seed; log ở `n5k/results/kaggle/run1b/`)
| hệ | lr 0.01 | lr 0.005 | lr 0.002 |
|---|---|---|---|
| single (best val MAE kcal) | **83,4** | 106,9 | 151,2 |
| PipeDream M=64 | **84,3** | 107,1 | 151,1 |
| TiMePReSt N=2 | **85,5** | 108,6 | 151,6 |
- Cả 3 hệ tốt nhất ở lr 0.01, **nằm ở biên trên của lưới**, nên lr tối ưu có thể còn lớn hơn. Ba hệ chênh nhau khoảng 2 kcal, dưới mức nhiễu của 1 seed. Không hệ nào phân kỳ.
- Ở lr 0.01 (single), tổng dinh dưỡng học tốt: MAE kcal 158 → 83 (pMAE 32,7%), mass 62 g. Phần **nguyên liệu học rất yếu**: F1 6,2%, mAP 7,8%, MAE gram trên nguyên liệu có mặt 26,6 → 25,8 (gần như không đổi).
  - Giả thuyết (chưa kiểm chứng): loss presence/gram lấy trung bình trên K = 195 nguyên liệu, nên mỗi nguyên liệu chỉ nhận gradient khoảng 1/195, trong khi loss tổng chỉ trung bình trên 5 thành phần. Đây là điểm mở (b) ở mục 10.

### Đo partition (`run1c`, 2026-10-14, D4 4 epoch trên 2×T4, cùng máy; log ở `n5k/results/kaggle/run1c/`)
Thời gian/epoch là trung bình epoch 2–4; GPU bận, peak MB và MB truyền lấy ở epoch cuối.

| hệ | partition | s/epoch | so với [0,9,18] | GPU bận | peak MB | MB truyền/epoch |
|---|---|---|---|---|---|---|
| PipeDream | [0,9,18] | 22,09 | — | 0,86 / 0,47 | 9078 / 1520 | 1715 |
| PipeDream | **[0,8,18]** | **20,04** | **−9,3%** | 0,91 / 0,74 | 8360 / 2051 | 3430 |
| PipeDream | [0,7,18] | 20,74 | −6,1% | 0,83 / 0,82 | 7768 / 2345 | 3430 |
| TiMePReSt | [0,9,18] | 19,34 | — | 0,91 / 0,52 | 8708 / 1554 | 1715 |
| TiMePReSt | **[0,8,18]** | **18,51** | **−4,3%** | 0,93 / 0,88 | 8269 / 2058 | 3430 |
| TiMePReSt | [0,7,18] | 18,88 | −2,4% | 0,84 / 0,93 | 7675 / 2353 | 3430 |

- `[0,7,18]` cân bằng độ bận nhất nhưng chậm hơn `[0,8,18]`. Cả hai cùng truyền gấp đôi `[0,9,18]` (biên ở 28×28×512).
- Val MAE ở epoch 4 như nhau giữa các partition (115–118).
- Lưu ý nhiễu giữa các máy: PipeDream `[0,9,18]` đo 20,0 s ở `run1a` nhưng 22,1 s ở đây. So sánh tốc độ giữa các hệ phải đo trên cùng một máy (bench_comm ở B6).

### B4 — trainer, check, sweep, notebook (2026-10-14, chưa commit; chưa chạy trên Kaggle)

- `timeprest/n5k/train.py` (engine 1 tiến trình: `num_stages: 1` = 1 GPU thường, `2` = pipeline mô phỏng cho sweep), `dist_train.py` (pipeline thật, kế thừa `DistTrainer`), `checks.py` (D0 dữ liệu, D1 static = engine, D2 version động, D3 resume, D4 chạy ngắn: loss giảm và MAE kcal val < dự đoán hằng trung vị train), `sweep.py`. Chọn checkpoint tốt nhất theo val MAE kcal, test chỉ đánh giá checkpoint đó.
- Configs `n5k/configs/`: `n5k_base` (giá trị mục 4–6), `n5k_quick`, `n5k_timeprest`, `n5k_pipedream`, `n5k_timeprest_n4`, `n5k_single`, `n5k_sweep` (8 epoch, warmup 1). Notebook `n5k/notebooks/n5k_kaggle.ipynb` (lần 1: dữ liệu + check + sweep; lần 2: chạy dài).
- Test CPU (dữ liệu giả 48/16/16 món, 32 px, ResNet-50 không pretrained): engine trainer resume liền mạch (W=1 và W=2), checks trên 2 tiến trình gloo D0–D3 PASS (D4 chạy được; PASS cần ảnh thật), `dist_train` ghi `summary.json`. Toàn bộ: **142 test PASS** (108 cũ + 34 mới).
- **Phát hiện (dữ liệu giả, chưa chắc đúng với dữ liệu thật):** ResNet-50 khởi tạo ngẫu nhiên, micro-batch 4, lr 0.01: TiMePReSt (`graph`) **phân kỳ** — gradient stage 0 tăng ~×5 mỗi bước (482 → 1321 → 7129 → 3e10 → NaN), trong khi PipeDream, 1 GPU, `timeprest_recompute`, Variant 1/2 và DeepSpeed đều ổn định cùng điều kiện. Check 4 khớp reference chính xác, nên đây là tính chất của quy tắc `graph` (cùng hướng PN §22.7), không phải lỗi cài đặt. Trên dữ liệu thật (pretrained, micro-batch 32, warmup) có thể không xảy ra; D4 và sweep sẽ cho biết. Nếu xảy ra: chọn lr nhỏ hơn theo sweep (quy tắc PN §22.9), không thêm kỹ thuật ngoài paper (gradient clipping) nếu chưa ghi quyết định ở mục 10.
- Chưa làm: B5 (DDP), bench mạng giả lập cho task này (B6), B7.
