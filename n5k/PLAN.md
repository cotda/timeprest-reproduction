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
  (tùy chọn `model.direct_totals`, 2026-10-14, mục 10)
  → conv 1×1 → 5 bản đồ tổng → global mean + softplus × trung bình train → kcal/mass/fat/carb/protein trực tiếp
       (loss tổng tính trên nhánh này; bản đồ kcal 7×7 của nhánh này là một bản đồ calories trực tiếp)
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
- 2026-10-14 (người dùng xác nhận "ưu tiên cả 2"; làm run4 trước, ảnh lớn hơn ở run5): **thêm nhánh tổng trực tiếp** (`model.direct_totals: true`). Head có thêm conv 1×1 → 5 bản đồ tổng (kcal, mass, fat, carb, protein), lấy global mean rồi softplus × trung bình train.
  - Loss tổng tính trên nhánh này. Kcal tính từ gram nguyên liệu vẫn được báo cáo (`mae_kcal_ingr`).
  - Lý do: một đầu ra (gram) phải phục vụ cả hai mục tiêu gây ra đánh đổi ở `run2`.
  - Mặc định `false` để giữ hành vi cũ.
  - Quy tắc chọn mới (đặt trước run4, thay quy tắc 5% ở trên), lấy trên mọi cấu hình đã thử ở `run2` + `run4`: chọn cấu hình nhỏ nhất theo `(MAE kcal / MAE kcal tốt nhất − 1) + (1 − mAP / mAP tốt nhất)`.
  - run4 (single, 8 epoch): direct + sum × lr {0.05, 0.02}; direct + w10 × lr 0.05; head cũ w30, w100 × lr 0.05; mean × lr 0.1 (lr ở biên).
  - **Kết quả (`run4`): chọn `direct_totals: true` + `ingredient_reduce: sum`** (điểm 0,17, thấp nhất; cấu hình đứng sau có điểm 0,41). Đã sửa `n5k_base.yaml`.
  - Ở run4, lr tốt nhất của biến thể này là 0.02, nằm ở biên dưới của lưới {0.05, 0.02}. Sweep cuối (`run3`) dùng lr {0.05, 0.02, 0.01} cho cả 3 hệ.
  - D4 trên 2 GPU với direct + sum ổn định cho cả PipeDream và TiMePReSt. Phân kỳ chỉ thấy trên dữ liệu giả.
- 2026-10-14: **chốt lr theo sweep cuối (`run3`)**, theo quy tắc PN §22.9 (mỗi hệ lấy lr có val MAE kcal tốt nhất). Kết quả: single / DDP / PipeDream = 0.02, TiMePReSt (cả N=4) = 0.01. Với TiMePReSt, 0.01 và 0.02 gần như bằng nhau về kcal (68,6 và 68,8); mAP ở 0.01 tốt hơn (63,3 so với 61,9), nên hai tiêu chí cùng chọn 0.01. Đã sửa các config. **B4 hoàn tất.**
- 2026-10-14 (người dùng đồng ý: "chia đều hơn rồi nâng ảnh", làm sau run4): **thử ảnh 320** (`run0b` CPU tạo cache 320; `run5` GPU).
  - Ước lượng bộ nhớ (đếm activation, hiệu chỉnh theo số đo ở 224): ở 320, `[0,8,18]` cần khoảng 16,6 GB cho GPU0 nên sẽ thiếu bộ nhớ; `[0,5,18]` khoảng 12,7 / 4,0 GB.
  - Partition ở 320: chạy D4 4 epoch với `[0,6,18]`, `[0,5,18]`, `[0,4,18]`. Chọn partition có thời gian/epoch của PipeDream nhỏ nhất trong số những partition có peak < 14 GB trên **cả hai** GPU.
  - Kích thước ảnh: chạy single GPU, lr 0.02, 8 epoch, seed 0 và 1, ở cả 224 và 320 trong cùng kernel. Dùng 320 nếu, tính trên trung bình 2 seed, `(kcal320/kcal224 − 1) + (1 − mAP320/mAP224) < −0,05` (tốt hơn ít nhất 5% tổng hợp). Nếu không thì giữ 224.
  - Code: cache lưu `image_size` và từ chối cache sai kích thước (trước đây có thể lấy nhầm cache).
  - Thêm một lần chạy thử DDP trên GPU (NCCL, 1 epoch, ảnh 224).
  - Kết quả (`run5a` + `run5b`, xem mục 11): **giữ ảnh 224 và partition `[0,8,18]`**. Điểm +0,044 > −0,05: ảnh 320 kém hơn trên cả kcal và mAP, đồng thời pipeline chậm hơn khoảng 3 lần.
  - Cập nhật 04:51 UTC: `run0b` (đẩy 03:22) vẫn ở RUNNING sau 87 phút, trong khi `run0` xong trong ~10 phút. Nhiều khả năng bị treo ở bước gắn dataset gốc; không xem log hay hủy được bằng API. Vì vậy **tách `run5`** thành hai kernel, nội dung và quy tắc chọn giữ nguyên: `run5a` (ảnh 224: checks, single seed 0/1, DDP smoke; chỉ gắn `run0`) chạy ngay; `run5b` (ảnh 320: D4 với 3 partition, checks, single seed 0/1) chạy khi có cache 320. Đồng thời đẩy `run0c`, cùng nội dung với `run0b`, để thử lại. Chạy 224 và 320 ở hai kernel khác nhau không ảnh hưởng tới phép so sánh chất lượng: metric không phụ thuộc máy, chỉ thời gian thì có.
- 2026-10-15 (người dùng chọn "R50 + R50"): **thử mô hình two-stream** (`model.name: n5k_resnet50_2s`). Lý do: B6 cho MAE kcal 19,5%, ngang mức "depth làm kênh thứ 4" của paper Nutrition5k (18,8%) nhưng kém các bài RGB-D gần đây (12–15%), và các bài đó đều tách RGB và depth thành hai nhánh. Mô hình nặng hơn cũng là trường hợp mà pipeline có lý do để dùng.
  - Kiến trúc: hai ResNet-50 (stem imagenet), nhánh RGB 3 kênh và nhánh depth 1 kênh, đều khởi tạo từ ImageNet; conv1 của nhánh depth là trung bình 3 kênh RGB (giống cách khởi tạo kênh depth hiện tại). Hợp nhất muộn: nối feature 2048 + 2048 kênh rồi đưa vào head CAM như cũ (`N5kHead`, cin 4096). Mỗi khối của chuỗi 18 khối chứa cả hai nhánh song song (input ghép theo kênh, tách ra, tính, ghép lại), nên partition, runtime pipeline và DDP không phải sửa. Dữ liệu không đổi (vẫn tensor 4 kênh). Mô hình cũ `n5k_resnet50` giữ nguyên, các config hiện tại không đổi.
  - Chạy thử `run8_2s` (single GPU, chỉ GPU0, lr 0.02, 8 epoch, seed 0 và 1; chạy xen kẽ 1s/2s/1s/2s để máy chậm dần không lệch về một bên). Mô hình một nhánh chạy lại trong cùng kernel làm mốc. lr giữ 0.02 cho cả hai (chưa sweep riêng two-stream).
  - Quy tắc chọn (đặt trước khi có số liệu, giống quy tắc ảnh 224/320): với best val kcal và mAP tại epoch đó, lấy trung bình 2 seed. Dùng two-stream nếu `(kcal2s/kcal1s − 1) + (1 − mAP2s/mAP1s) < −0,05`. Nếu không thì giữ mô hình một nhánh và ghi kết quả âm tính.
  - Nếu đạt: đo lại partition trên 2×T4 (giống `run1c`, PipeDream ở GPU0 có thể thiếu bộ nhớ), chạy lại checks D0–D4 và sweep lr ngắn, rồi chạy lại B6. Các bước này sẽ đề xuất riêng trước khi chạy.
  - **Kết quả (`run8_2s`, xem mục 11): điểm −0,091 < −0,05, theo quy tắc chọn two-stream.** Chưa đổi `n5k_base.yaml`: two-stream cần partition mới, vì D2 của two-stream với `[0,8,18]` thiếu bộ nhớ trên GPU0. Partition và các bước tiếp theo sẽ đề xuất trước khi chạy.
- 2026-10-15 (người dùng đồng ý "chạy bước 1 đo partition"): **partition cho two-stream** (`run9_2s_part`).
  - Ước lượng: activation của two-stream ở 224 gần bằng mô hình một nhánh ở 320 (gấp khoảng 2 lần). Ở 320 chỉ `[0,4,18]` vừa bộ nhớ (`[0,5,18]` cần hơn 14,2 GB). Vì vậy đo cùng 3 partition `[0,5,18]`, `[0,4,18]`, `[0,6,18]`.
  - Mỗi partition chạy đủ checks D0–D4 của two-stream (D4: 4 epoch, cả PipeDream và TiMePReSt), mỗi partition một file checks riêng.
  - Quy tắc chọn (đặt trước khi có số liệu, giống quyết định ở 320): trong các partition có D0–D3 PASS và peak < 14 GB trên **cả hai** GPU, lấy partition có thời gian/epoch của PipeDream nhỏ nhất. Nếu không partition nào đạt thì báo lại người dùng, không tự đổi mô hình hay batch.
  - Sau bước này mới đề xuất bước 2 (sweep lr) và bước 3 (B6 với two-stream).
  - **Kết quả (`run9_2s_part`, xem mục 11): chọn `[0,4,18]`**, partition duy nhất vừa bộ nhớ (`[0,5,18]` và `[0,6,18]` thiếu bộ nhớ trên GPU0 ở D2). Chưa sửa config: two-stream chưa thành mô hình chính cho đến khi người dùng quyết định bước 2–3.
- 2026-10-15 (người dùng): mỗi kết luận từ thí nghiệm cũng được ghi thành một file riêng, có ghi thời điểm, trong `n5k/notes/` (mục lục ở `n5k/notes/README.md`), để cuối dự án tổng hợp. PLAN vẫn là nơi ghi đầy đủ.
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

### B6 — kết quả chạy dài (`run6_s{0,1,2}` + `run6_bench`, 2026-10-14; log ở `n5k/results/kaggle/run6_*`)
- Cấu hình: ảnh 224, partition `[0,8,18]`, M=64, 30 epoch, lr: single / DDP / PipeDream 0.02, TiMePReSt (N=2, N=4) 0.01. Test lấy ở checkpoint có val MAE kcal tốt nhất. Số liệu dạng trung bình ± độ lệch chuẩn trên 3 seed; mỗi seed là một kernel, các hệ chạy nối tiếp trên cùng máy.

| Hệ | MAE kcal | %kcal | MAE mass | F1 | mAP | MAE gram | s/epoch | Peak MB (GPU lớn nhất) |
|---|---|---|---|---|---|---|---|---|
| Single GPU | 51,9 ± 2,8 | 19,5 | 33,8 | 86,6 | 87,2 | 12,0 | 25,2 ± 2,1 (*) | 5642 |
| DDP 2 GPU | 53,3 ± 1,4 | 20,0 | 34,0 | 86,8 | 87,4 | 12,1 | **13,0 ± 0,9** | 3056 |
| PipeDream | **51,9 ± 0,9** | 19,5 | **31,9** | 86,0 | 86,2 | 12,0 | 19,9 ± 0,6 | 8360 / 2041 |
| TiMePReSt N=2 | 56,3 ± 0,4 | 21,1 | 36,3 | 85,3 | 85,3 | 12,8 | 18,9 ± 0,5 | 8285 / 2045 |
| TiMePReSt N=4 | 57,3 ± 0,6 | 21,5 | 37,1 | 85,0 | 85,2 | 13,5 | 20,5 ± 0,5 | 8298 / 2040 |

- Cùng máy, theo từng seed: TiMePReSt/PipeDream = 0,948 / 0,945 / 0,953, tức **TiMePReSt nhanh hơn khoảng 5%**. N=4/PipeDream = 1,02–1,04. DDP/PipeDream = 0,63–0,68.
- Theo epoch, val MAE kcal ở epoch 10 / 20: PipeDream 70,1 / 51,1; TiMePReSt 82,4 / 57,4; N=4 76,5 / 58,4; single 84,9 / 50,1; DDP 78,9 / 52,6.
- Kcal tính từ gram (nhánh nguyên liệu): PipeDream 94,6, single 97,3, DDP 102,1, TiMePReSt 114,3, N=4 129,3.
- `bench_comm` (`run6_bench`, cùng một máy, 2 epoch, đo epoch cuối; truyền 3430 MB/epoch mỗi chiều):

| Băng thông | TiMePReSt (s) | PipeDream (s) | Tỉ lệ |
|---|---|---|---|
| ∞ (cùng máy) | 17,54 | 18,54 | 0,946 |
| 2 Gbit/s | 29,72 | 32,55 | 0,913 |
| 1 Gbit/s | 44,61 | 48,11 | 0,927 |

- Đối chiếu dự đoán ở mục 8:
  - (1) "MAE cuối TiMePReSt ≈ PipeDream": **không đúng**. TiMePReSt kém 4,3 kcal (+8%), ổn định qua 3 seed (độ lệch chuẩn 0,4–0,9); mAP kém 0,9 điểm.
    - Hai hệ khác lr (0.01 so với 0.02), nhưng ở sweep 8 epoch TiMePReSt với 0.02 cho kcal gần như bằng 0.01 (68,8 so với 68,6). Vì vậy nhiều khả năng khoảng cách đến từ cơ chế: backward dùng trọng số mới hơn forward. Chưa kiểm chứng bằng lần chạy 30 epoch với lr 0.02.
  - (2) "TiMePReSt giảm MAE nhanh hơn ở giữa quá trình": **không đúng**; ở epoch 10 và 20 TiMePReSt đều kém PipeDream.
  - (3) Thời gian:
    - "DDP nhanh nhất": đúng.
    - "TiMePReSt có thể chậm hơn PipeDream khoảng 5%": ngược lại, TiMePReSt nhanh hơn khoảng 5%.
    - "Ở 1–2 Gbit/s TiMePReSt nhanh hơn": đúng (7–9%).
  - (4) "Bộ nhớ TiMePReSt ≈ PipeDream": đúng (−0,9% trên GPU0).
- N=4 so với N=2: chậm hơn 8,5% và MAE kém hơn 1 kcal; không có lợi ở W=2.
- (*) Thời gian của single chưa dùng để so sánh được: các lần chạy dài cho 23–27 s, trong khi sweep cùng config cho 21,5 s. Đã đo nguyên nhân ở `run7_timing` (xem mục dưới): **không phải do số GPU nhìn thấy hay do chạy sau checks**, mà do máy chậm dần theo thời gian chạy. Vì vậy chỉ so sánh tỉ lệ giữa các hệ chạy trên cùng máy (các tỉ lệ ở trên); không so sánh giá trị tuyệt đối của s/epoch giữa các kernel.

### Partition của two-stream (`run9_2s_part`, 2026-10-15, chạy 09:34–10:01 UTC; checks D0–D4 với từng partition; log ở `n5k/results/kaggle/run9_2s_part/`)
- `[0,5,18]` và `[0,6,18]`: D0, D1 PASS; **D2 thiếu bộ nhớ trên GPU0** (14,5 GB), kernel tự dừng bước đó (NCCL timeout rồi SIGABRT, khoảng 6,7 phút mỗi bước).
- `[0,4,18]`: **D0–D4 PASS hết** (D1: khớp engine, chênh lệch 0,0). D4 4 epoch:

| Hệ | s/epoch | GPU bận | Peak MB | MB truyền/epoch | Val kcal ep1→ep4 |
|---|---|---|---|---|---|
| PipeDream | 58,39 | 0,55 / 0,94 | 13474 / 9435 | 13720 | 611 → 75,2 |
| TiMePReSt | 58,12 | 0,57 / 0,95 | 13371 / 9329 | 13720 | 125 → 79,2 |

- Theo quy tắc: **chọn `[0,4,18]`** (partition duy nhất đạt). GPU0 chỉ dư khoảng 0,5 GB.
- Nhận xét quan trọng:
  - Pipeline **không nhanh hơn single GPU**: khoảng 58 s/epoch so với 59 s ở `run8_2s` (khác máy nên chênh lệch khoảng ±15%, nhưng không có dấu hiệu nhanh hơn rõ rệt).
  - Lệch tải: GPU1 bận 94%, GPU0 chỉ 55%. Không cắt muộn hơn được vì GPU0 phải giữ activation của 2 mini-batch đang chạy (pipeline bất đồng bộ W=2), nên thiếu bộ nhớ.
  - Peak GPU0 (13,4 GB) gần bằng single GPU (13,8 GB): ở cấu hình này pipeline bất đồng bộ không giảm được bộ nhớ lớn nhất.
  - Truyền 13,7 GB/epoch mỗi chiều (cắt ở 56×56), gấp 4 lần mô hình một nhánh.
  - TiMePReSt ≈ PipeDream về thời gian (tỉ lệ 0,995).

### Two-stream R50 + R50 (`run8_2s`, 2026-10-15, đẩy 04:06 UTC, chạy 38 phút; single, chỉ GPU0, lr 0.02, 8 epoch; log ở `n5k/results/kaggle/run8_2s/`)
- Checks D0–D4 của code mới (mô hình một nhánh): PASS hết.
- Checks của two-stream trên pipeline 2 GPU (`[0,8,18]`): D0, D1 PASS (static order khớp engine, chênh lệch 0,0 cho cả hai hệ). **D2 thiếu bộ nhớ trên GPU0** (14,5 GB) rồi NCCL timeout. D3 chưa chạy. Two-stream cần partition mới cho pipeline.
- Val ở epoch có val MAE kcal tốt nhất (cách đọc giống quyết định ảnh 224/320):

| Mô hình | seed | MAE kcal | mAP | epoch | Test kcal | Test mAP | s/epoch | Peak MB |
|---|---|---|---|---|---|---|---|---|
| Một nhánh | 0 | 65,53 | 67,81 | 6 | 66,82 | 69,37 | 27,6 | 5642 |
| Two-stream | 0 | 61,75 | 75,42 | 8 | 59,26 | 74,24 | 59,1 | 13822 |
| Một nhánh | 1 | 64,21 | 72,66 | 7 | 61,36 | 71,81 | 27,3 | 5643 |
| Two-stream | 1 | 62,73 | 72,11 | 6 | 58,90 | 73,28 | 59,3 | 13821 |

- Trung bình 2 seed: một nhánh kcal 64,87 / mAP 70,24; two-stream 62,24 / 73,76. **Điểm = (62,24/64,87 − 1) + (1 − 73,76/70,24) = −0,091**, đạt ngưỡng < −0,05.
- Kiểm tra độ vững (không thuộc quy tắc): ở epoch cuối (8), kcal 65,2 → 62,4 và mAP 72,8 → 75,3, điểm −0,077, cũng đạt. Trên test: kcal 64,1 → 59,1 (−7,8%), mAP 70,6 → 73,8. kcal tốt hơn ở cả 2 seed. mAP tại epoch tốt nhất thì seed 1 hơi kém (72,1 so với 72,7); phần lớn chênh lệch mAP đến từ seed 0, nơi mô hình một nhánh đạt kcal tốt nhất sớm (epoch 6) khi mAP còn đang tăng.
- Chi phí: 2,15 lần thời gian (59,2 so với 27,5 s/epoch, cùng máy, chạy xen kẽ) và 2,45 lần bộ nhớ (13,8 so với 5,6 GB). Single GPU với M=64 gần chạm trần T4 (14,56 GB).
- Kcal tính từ gram không cải thiện (114–116 so với 111–115).

### Thời gian của single (`run7_timing`, 2026-10-14, đẩy 20:14 UTC, xếp hàng đến 22:18, chạy 13 phút; log ở `n5k/results/kaggle/run7_timing/`)
- Single, 4 epoch, seed 0, lần lượt trên cùng một máy: thấy 2 GPU → chỉ GPU0 (`CUDA_VISIBLE_DEVICES=0`) → checks D0–D4 → thấy 2 GPU → chỉ GPU0. s/epoch ở epoch 2–4:

| Thứ tự | Trường hợp | s/epoch (ep 2 / 3 / 4) | Peak MB |
|---|---|---|---|
| 1 | 2 GPU, trước checks | 21,3 / 21,9 / 22,6 | 5643 |
| 2 | GPU0, trước checks | 23,6 / 24,0 / 24,4 | 5642 |
| 3 | 2 GPU, sau checks | 24,1 / 24,6 / 24,7 | 5642 |
| 4 | GPU0, sau checks | 24,4 / 24,5 / 24,8 | 5643 |

- Thời gian tăng đều theo thứ tự chạy (cả trong từng lần chạy: 21,3 → 22,6 s), không phụ thuộc trường hợp; data_time chỉ khoảng 0,2 s. Nhiều khả năng GPU T4 nóng lên và hạ xung, nên chạy càng lâu càng chậm (khoảng 15%). Chưa đo xung nhịp/nhiệt độ để khẳng định.
- Hệ quả: sweep (21,5 s) chạy lúc máy còn nguội; các lần chạy dài cho 23–27 s là bình thường. Các tỉ lệ TiMePReSt/PipeDream ở B6 đo trên cùng máy, các hệ chạy liền nhau, nên vẫn dùng được: tỉ lệ khớp nhau qua 3 seed (0,945–0,953) và khớp với `bench_comm` (0,946, chỉ 2 epoch mỗi hệ nên máy ít kịp chậm đi).
- `run7_timing` ở trạng thái RUNNING hơn 2 giờ là do xếp hàng trên Kaggle, không phải treo.
- `run6_s0` lần 2 (đẩy 19:12, xong 20:11) và `run6_bench` (8,2 phút) đều chạy hết mọi bước. Vì bench chỉ cần 8 phút, lần treo của `run6_s0` lần 1 không phải do bench chạy lâu. Nguyên nhân vẫn chưa rõ: có thể kẹt lúc khởi động, hoặc treo ở một bước bất kỳ rồi giữ pipe.

### B6 — chuẩn bị (2026-10-14, chưa commit; chưa chạy)
- `timeprest/n5k/bench_comm.py`: cùng quy trình với `timeprest.dist.bench_comm`, dùng lại `bench_variants` và `plot`; khác ở chỗ dùng trainer pipeline của task dinh dưỡng và chỉ nạp cache một lần. Code cũ không sửa.
  - Config `n5k/configs/n5k_bench_comm.yaml`: TiMePReSt (lr 0.01) và PipeDream (lr 0.02); ∞ / 2 / 1 Gbit/s; latency 0,1 ms; 2 epoch, đo epoch cuối.
  - Test CPU `test_bench_comm.py` (2 tiến trình gloo, dữ liệu giả): mỗi cặp (băng thông, hệ) ra một dòng, và link 0,002 Gbit/s làm epoch chậm hơn. Toàn bộ test n5k: 44 PASS.
- Kernel `n5k/kaggle/run6_s{0,1,2}/`: mỗi seed một kernel 2×T4. Các bước lần lượt: checks → single → DDP → PipeDream → TiMePReSt N=2 → TiMePReSt N=4, đều 30 epoch; riêng seed 0 thêm `bench_comm`. Các hệ chạy nối tiếp trên cùng máy nên thời gian/epoch so được với nhau trong từng kernel. Kích thước ảnh và partition lấy từ `n5k_base.yaml`, sẽ chốt sau `run5b`.
  - Ước tính ở ảnh 224: khoảng 65 phút/kernel, seed 0 khoảng 80 phút; tổng khoảng 3,5 giờ GPU.
  - Người dùng giao (2026-10-14, "tự theo dõi kết quả run5b rồi tự làm luôn"): agent tự chọn theo quy tắc rồi đẩy B6. Vì giữ 224, kernel chỉ gắn cache 224 (`run0`).
  - Đẩy 14:57 UTC: `run6_s0` và `run6_s1` chạy song song. Kaggle giới hạn tối đa 2 phiên GPU batch cùng lúc, nên `run6_s2` được đẩy khi một kernel xong.
  - `run6_s1` xong lúc 16:00 (63 phút; các bước đều ok: checks 3,1, single 15,3, DDP 8,6, PipeDream 12,0, TiMePReSt 11,5, N=4 12,3 phút). `run6_s2` được đẩy lúc 19:00.
  - **`run6_s0` (lần 1) treo:** đến 19:05 vẫn RUNNING, trang web báo "Running for 10901 s", log trống, output 0 B. Theo dự kiến nó phải xong trước khoảng 17:10, vì `bench_comm` có giới hạn 50 phút.
    - Nguyên nhân khả dĩ, chưa chứng minh: `torchrun` chạy mỗi worker trong session riêng (`start_new_session=True`, đã đọc trong mã nguồn torch). `timeout` chỉ gửi tín hiệu tới process group của nó, nên worker còn sống sẽ giữ pipe stdout và `kernel_lib` đọc log mãi không hết.
    - Khả năng khác: kernel kẹt ngay khi khởi động, như `run0b`. Không có log nên chưa phân biệt được hai khả năng.
    - Sửa `kernel_lib`: (1) `run_step` ghi lại mọi tiến trình con và cháu trong lúc chạy, và dừng những tiến trình còn sống khi bước kết thúc hoặc tới hạn cứng; thread đọc log không còn chặn kernel; (2) chép `n5k_results` sau mỗi bước. Test `timeprest/n5k/tests/test_kernel_lib.py` (2 PASS). Trên Windows, Git `timeout` dừng luôn tiến trình cháu nên không tái hiện được lỗi treo; test chỉ xác nhận bản sửa chạy đúng.
    - `bench_comm` được tách khỏi `run6_s0` sang kernel riêng `run6_bench`. `run6_s0` (lần 2) chỉ còn các lần chạy dài, đẩy sau khi lần 1 bị hủy trên web.

### Ảnh 224 hay 320 (`run5a` + `run5b`, 2026-10-14, single GPU, lr 0.02, 8 epoch, seed 0/1; log ở `n5k/results/kaggle/run5a/`, `run5b/`)
- Cách đọc số (định trước khi có kết quả 320): kcal = val MAE kcal tốt nhất; mAP = val mAP nguyên liệu ở chính epoch đó; lấy trung bình 2 seed.
- 224 (`run5a`): seed 0: kcal 62,10 / mAP 72,81 (epoch 7); seed 1: 66,34 / 72,75 (epoch 7). **Trung bình: 64,22 / 72,78.** Test mAP 72,4 / 72,0.
  - Hai seed lệch nhau 4,2 kcal, khớp mức dao động ±3 kcal đã thấy ở run3.
- 320 (`run5b`): seed 0: kcal 63,32 / mAP 71,00 (epoch 6); seed 1: 66,79 / 70,02 (epoch 6). **Trung bình: 65,06 / 70,51.**
- **Điểm = (65,06/64,22 − 1) + (1 − 70,51/72,78) = +0,044**, không đạt ngưỡng < −0,05: ảnh 320 còn kém hơn 224 trên cả hai tiêu chí. **Giữ 224**; config không đổi.
- Chi phí của 320, đo cùng máy:
  - Single: 56,9 s/epoch (224: 21,5 s), peak 11,2 GB (224: 5,6 GB).
  - Partition ở 320 (D4 4 epoch): `[0,5,18]` và `[0,6,18]` thiếu bộ nhớ trên GPU0 (cần hơn 14,2 GB; ước lượng cũ 12,7 GB cho `[0,5,18]` là thấp). Chỉ `[0,4,18]` chạy được: PipeDream 61,0 s/epoch, TiMePReSt 58,8 s; peak 12,0/8,0 GB; bận 0,57/0,94; truyền 14 000 MB/epoch, gấp 4 lần 224 vì cắt sớm hơn ở tầng có bản đồ đặc trưng lớn.
  - Nghĩa là pipeline ở 320 chậm hơn 224 khoảng 3 lần.
- Checks D0–D4 lặp lại ở `run5b` (224, máy khác): tất cả PASS. D4: PipeDream 21,5 s, TiMePReSt 20,6 s.
- Cache 320 (`run0c`): cùng split, cùng K=195 và cùng 2 món bị loại như cache 224; tạo trong 47 s. `run0b` (cùng nội dung) vẫn RUNNING lúc 14:04 UTC, gần 11 giờ sau khi đẩy, trong khi `run0c` bắt đầu chạy script ngay khi được đẩy. Vậy kernel commit gắn dataset gốc có lúc bị treo, nhiều khả năng cũng là nguyên nhân của lần 1.

### B5 — DDP (2026-10-14, chưa commit; chạy thử trên GPU ở `run5a`)
- File `timeprest/n5k/ddp_train.py` và config `n5k/configs/n5k_ddp.yaml`.
- Cách chia batch: mỗi rank giữ cả model. Mỗi mini-batch M=64 **chính là mini-batch của trainer 1 GPU** (thứ tự lấy từ chính `train_loader`), được cắt thành 2 phần 32 mẫu. DDP lấy trung bình gradient, nên tương đương SGD trên trung bình loss của 64 mẫu. Optimizer, lịch lr (bước theo mini-batch) và lr 0.02 giống single.
- Khác single: BatchNorm tính thống kê trên 32 mẫu mỗi rank, không dùng SyncBN (DDP thông thường).
- Val trên rank 0 mỗi epoch; checkpoint/resume lưu RNG của từng rank. `metrics.csv` / `summary.json` cùng định dạng các trainer khác, summary có thêm `trainer: ddp`.
- Test CPU (`timeprest/n5k/tests/test_ddp.py`, 5 test):
  - shard của các rank ghép lại đúng bằng mini-batch 1 GPU (đã kiểm tra cả khi DataLoader dùng 0 và 2 worker);
  - 2 tiến trình gloo chạy hết và ghi summary;
  - 2 epoch liền so với 1 epoch + resume: tham số và metric khớp tuyệt đối.
- Toàn bộ test: 150 PASS.
- Chạy thử trên 2×T4 (`run5a`, NCCL, 1 epoch, ảnh 224): chạy hết, summary có `trainer: ddp`, `ranks: 2`. Thời gian epoch 18,5 s (single cùng máy: 21,5 s), peak 7533 MB mỗi GPU. Val MAE kcal 134 sau 1 epoch là bình thường, vì cả epoch nằm trong warmup (lr cuối epoch 2,9e-4). Thời gian chính thức lấy ở B6.
- Checks D0–D4 của code hiện tại trên 2×T4 (`run5a`): **tất cả PASS**. D4 với `[0,8,18]`: PipeDream 18,8 s/epoch, TiMePReSt 17,9 s; peak 8391/2166 MB. Tương ứng run1c là 20,0 s, chênh nhau khoảng 6%, nằm trong mức nhiễu giữa các máy.

### Sweep lr cuối (`run3`, 2026-10-14, cấu hình chốt: partition [0,8,18], direct + sum, 8 epoch, 1 seed; log ở `n5k/results/kaggle/run3/`)
- Checks: PASS hết. Số liệu ở epoch có val MAE kcal tốt nhất.

| hệ | lr | MAE kcal | mAP | F1 | MAE gram | MAE mass |
|---|---|---|---|---|---|---|
| single | 0.05 | 67,1 | 64,9 | 70,5 | 17,6 | 41,7 |
| single | **0.02** | **64,4** | 74,7 | 77,0 | 16,1 | 39,9 |
| single | 0.01 | 67,0 | 69,9 | 72,9 | 16,8 | 43,0 |
| PipeDream | 0.05 | 69,8 | 61,7 | 69,2 | 17,5 | 42,7 |
| PipeDream | **0.02** | **63,5** | 73,5 | 75,8 | 16,2 | 38,8 |
| PipeDream | 0.01 | 65,8 | 68,9 | 71,8 | 16,9 | 42,4 |
| TiMePReSt | 0.05 | 111,4 | 24,6 | 40,0 | 24,8 | 71,4 |
| TiMePReSt | 0.02 | 68,8 | 61,9 | 69,5 | 18,0 | 45,9 |
| TiMePReSt | **0.01** | **68,6** | 63,3 | 69,6 | 18,0 | 45,4 |

- Single và PipeDream tốt nhất ở lr 0.02, nằm giữa lưới, và gần như bằng nhau.
- TiMePReSt **nhạy với lr lớn hơn**: ở lr 0.05 kém hẳn (111 kcal, mAP 25). Ở lr tốt nhất, sau 8 epoch nó vẫn kém PipeDream khoảng 5 kcal và **khoảng 10 điểm mAP**. Đây là dấu hiệu staleness của quy tắc `graph` ảnh hưởng tới việc học nguyên liệu (gradient mạnh với `sum`). Cần kiểm chứng ở chạy dài 30 epoch với 3 seed (B6).
- TiMePReSt tốt nhất ở 0.01, là biên dưới của lưới; nhưng 0.01 và 0.02 gần bằng nhau, nên khả năng có lr tối ưu nhỏ hơn hẳn là thấp. Có thể thử thêm 0.005 trong một kernel sau (khoảng 4 phút).
- Biến động giữa các lần chạy: single, lr 0.02, cùng cấu hình cho 61,1 ở run4 và 64,4 ở run3 (khoảng ±3 kcal).

### Nhánh tổng trực tiếp (`run4`, 2026-10-14, single GPU, 8 epoch, 1 seed; log ở `n5k/results/kaggle/run4/`)
- Checks với partition `[0,8,18]` và code mới: PASS hết.
- D4 2 GPU với direct + sum, 4 epoch:
  - PipeDream: MAE kcal 82,1, mAP 44,6, 19,1 s/epoch.
  - TiMePReSt: MAE kcal 78,8, mAP 42,0, 17,3 s/epoch.
  - Cả hai ổn định.

Mọi cấu hình của run2 + run4, số liệu ở epoch có val MAE kcal tốt nhất. Mức tốt nhất: MAE kcal 52,2, mAP 73,6. điểm = (kcal/52,2 − 1) + (1 − mAP/73,6).

| cấu hình | lr | MAE kcal | mAP | F1 | MAE gram | MAE mass | kcal từ gram | gshare p/g/t | điểm |
|---|---|---|---|---|---|---|---|---|---|
| **direct + sum** | **0.02** | **61,1** | **73,6** | **76,1** | **16,3** | **39,4** | 117,2 | 0,65/0,30/0,05 | **0,17** |
| direct + sum | 0.05 | 68,3 | 66,4 | 72,0 | 16,6 | 40,7 | 104,8 | 0,67/0,28/0,05 | 0,41 |
| w30 | 0.05 | 67,6 | 63,4 | 69,7 | 18,2 | 49,2 | 67,6 | 0,55/0,21/0,24 | 0,43 |
| direct + w10 | 0.05 | 52,2 | 38,3 | 51,7 | 24,2 | 34,7 | 143,0 | 0,43/0,11/0,46 | 0,48 |
| w100 | 0.05 | 79,0 | 72,5 | 76,2 | 16,5 | 55,3 | 79,0 | 0,61/0,28/0,11 | 0,53 |
| w10 (run2) | 0.05 | 60,4 | 40,2 | 52,2 | 22,9 | 43,3 | — | 0,47/0,12/0,41 | 0,61 |
| sum (run2) | 0.05 | 81,7 | 66,5 | 72,1 | 17,1 | 58,3 | — | 0,66/0,28/0,06 | 0,66 |
| mean | 0.1 | 54,0 | 18,2 | 23,4 | 24,5 | 41,9 | 54,0 | 0,12/0,02/0,86 | 0,79 |
| mean (run2) | 0.05 | 54,9 | 15,4 | 16,8 | 24,7 | 43,0 | — | 0,11/0,02/0,86 | 0,84 |

- Nhánh trực tiếp gỡ được phần lớn đánh đổi: so với `sum` head cũ, kcal giảm 81,7 → 61,1 trong khi mAP tăng 66,5 → 73,6.
- Kcal tính từ gram của nhánh nguyên liệu vẫn kém (104–143). Sai số khi nhận diện nguyên liệu và gram vẫn dồn lại vào tổng kcal; nhánh trực tiếp tránh được việc này.
- MAE kcal của direct + sum vẫn kém mức tốt nhất 17%. Thành phần total chỉ chiếm 5% gradient, nên còn chỗ để tăng `w_total`. Chưa thử, ghi lại để cân nhắc.

### Biến thể loss (`run2`, 2026-10-14, single GPU, 8 epoch, 1 seed; log ở `n5k/results/kaggle/run2/`)
Checks của code mới: PASS hết. Số liệu lấy ở epoch có val MAE kcal tốt nhất. gshare = tỉ lệ chuẩn gradient của presence / gram / total.

| biến thể | lr | MAE kcal | mAP | F1 | MAE gram (có mặt) | MAE mass | gshare p/g/t |
|---|---|---|---|---|---|---|---|
| mean | **0.05** | **54,9** | 15,4 | 16,8 | 24,7 | 43,0 | 0,11 / 0,02 / 0,86 |
| mean | 0.02 | 67,6 | 9,8 | 10,7 | 25,6 | 53,7 | 0,12 / 0,02 / 0,86 |
| mean | 0.01 | 83,0 | 7,7 | 6,5 | 25,8 | 62,7 | 0,11 / 0,02 / 0,87 |
| w10 | **0.05** | 60,4 | 40,2 | 52,2 | 22,9 | 43,3 | 0,47 / 0,12 / 0,41 |
| w10 | 0.02 | 73,2 | 25,2 | 39,1 | 24,1 | 53,3 | 0,44 / 0,09 / 0,47 |
| w10 | 0.01 | 88,1 | 18,9 | 30,1 | 25,7 | 64,0 | 0,50 / 0,08 / 0,42 |
| sum | **0.05** | 81,7 | 66,5 | 72,1 | 17,1 | 58,3 | 0,66 / 0,28 / 0,06 |
| sum | 0.02 | 83,4 | 67,8 | 72,0 | 17,5 | 56,5 | 0,66 / 0,28 / 0,06 |
| sum | 0.01 | 84,4 | 67,7 | 71,9 | 17,3 | 64,1 | 0,67 / 0,27 / 0,06 |

- Giả thuyết được xác nhận: với `mean`, thành phần tổng chiếm 86% gradient, phần gram chỉ 2%.
- Có đánh đổi rõ: phần nguyên liệu càng mạnh thì mAP tăng (15 → 40 → 67) và MAE gram giảm (25 → 17), nhưng MAE kcal tăng (55 → 60 → 82).
- Theo quy tắc đặt trước (MAE kcal không tệ hơn `mean` quá 5%, tức ≤ 57,7): w10 (+9,9%) và sum (+49%) đều không đạt, nên **quy tắc chọn `mean`**.
- `mean` và `w10` lại tốt nhất ở lr 0.05, vẫn là biên trên của lưới.

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
- Chưa làm: bench mạng giả lập cho task này (B6), B7. (B5 xong ngày 2026-10-14, xem mục 11.)
