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

### B4 — trainer, check, sweep, notebook (2026-10-14, chưa commit; chưa chạy trên Kaggle)

- `timeprest/n5k/train.py` (engine 1 tiến trình: `num_stages: 1` = 1 GPU thường, `2` = pipeline mô phỏng cho sweep), `dist_train.py` (pipeline thật, kế thừa `DistTrainer`), `checks.py` (D0 dữ liệu, D1 static = engine, D2 version động, D3 resume, D4 chạy ngắn: loss giảm và MAE kcal val < dự đoán hằng trung vị train), `sweep.py`. Chọn checkpoint tốt nhất theo val MAE kcal, test chỉ đánh giá checkpoint đó.
- Configs `n5k/configs/`: `n5k_base` (giá trị mục 4–6), `n5k_quick`, `n5k_timeprest`, `n5k_pipedream`, `n5k_timeprest_n4`, `n5k_single`, `n5k_sweep` (8 epoch, warmup 1). Notebook `n5k/notebooks/n5k_kaggle.ipynb` (lần 1: dữ liệu + check + sweep; lần 2: chạy dài).
- Test CPU (dữ liệu giả 48/16/16 món, 32 px, ResNet-50 không pretrained): engine trainer resume liền mạch (W=1 và W=2), checks trên 2 tiến trình gloo D0–D3 PASS (D4 chạy được; PASS cần ảnh thật), `dist_train` ghi `summary.json`. Toàn bộ: **142 test PASS** (108 cũ + 34 mới).
- **Phát hiện (dữ liệu giả, chưa chắc đúng với dữ liệu thật):** ResNet-50 khởi tạo ngẫu nhiên, micro-batch 4, lr 0.01: TiMePReSt (`graph`) **phân kỳ** — gradient stage 0 tăng ~×5 mỗi bước (482 → 1321 → 7129 → 3e10 → NaN), trong khi PipeDream, 1 GPU, `timeprest_recompute`, Variant 1/2 và DeepSpeed đều ổn định cùng điều kiện. Check 4 khớp reference chính xác, nên đây là tính chất của quy tắc `graph` (cùng hướng PN §22.7), không phải lỗi cài đặt. Trên dữ liệu thật (pretrained, micro-batch 32, warmup) có thể không xảy ra; D4 và sweep sẽ cho biết. Nếu xảy ra: chọn lr nhỏ hơn theo sweep (quy tắc PN §22.9), không thêm kỹ thuật ngoài paper (gradient clipping) nếu chưa ghi quyết định ở mục 10.
- Chưa làm: B5 (DDP), bench mạng giả lập cho task này (B6), B7.
