# TiMePReSt — Paper notes cho nghiên cứu và tái hiện

> Đọc file này trước khi mở PDF hoặc sửa thuật toán. Đây là bản diễn giải có nguồn và bản đồ triển khai, không phải mã nguồn chính thức hay đặc tả đầy đủ do tác giả phát hành. Các chỗ paper chưa nói đủ được đánh dấu để agent không tự biến giả định thành sự thật.

## 0. Cách sử dụng và độ tin cậy

- **Phạm vi:** bản tạp chí 15 trang do người dùng cung cấp; đã đọc nội dung và kiểm tra trực quan Fig. 1–17, trong đó phóng lớn Fig. 15–17 để đọc biểu đồ. Số trang dưới đây là số in trên PDF, cũng trùng thứ tự trang file.
- **Supplementary:** paper dẫn S1–S12 nhưng chưa có file bổ sung để đọc. Truy cập trang ScienceDirect trong lần tra cứu bị HTTP 403. Không khẳng định phần bổ sung không tồn tại; các mô tả về hình S chỉ lấy từ dẫn chiếu trong main paper.
- **Code:** chưa xác minh được repository chính thức của TiMePReSt. PipeDream có code chính thức; dùng nó làm nền tảng là lựa chọn của project tái hiện.
- **Các nhãn bằng chứng:** `[P]` = paper nói rõ; `[F≈]` = đọc xấp xỉ từ hình, không phải raw data; `[D]` = suy ra/kiểm tra toán học; `[I]` = đề xuất triển khai của người soạn; `[?]` = chưa đủ dữ liệu hoặc có mâu thuẫn; `[EXT]` = nguồn ngoài được ghi link.
- **Không dùng `[F≈]` làm số liệu chính xác trong bài báo mới**, làm ngưỡng PASS cứng hay làm bằng chứng tái hiện. Các khoảng ghi trong bảng là khoảng đọc bằng mắt, không phải confidence interval.
- **Khi có mâu thuẫn:** giữ cả phát biểu tổng quát và bằng chứng cụ thể, ưu tiên mô tả thận trọng; không tự sửa số liệu gốc.
- **Phân công tài liệu:** `AGENTS.md` quy định cách làm việc; file này chứa kiến thức paper; config/log chứa quyết định và kết quả thực tế của project.
- **Trạng thái:** bản ghi chú dựa trên tài liệu; không có thí nghiệm tái hiện nào được chạy khi soạn file.

### Tra nhanh theo nhiệm vụ

| Agent cần làm | Đọc mục |
|---|---|
| Hiểu paper giải quyết gì | 1–3 |
| Sửa scheduler/weight stashing | 4–7, 13–15 |
| Chọn model, dữ liệu, hardware | 8–9 |
| So kết quả với paper | 10–12 |
| Biết cấu hình nào không được tự đoán | 13 |
| Viết test và lập kế hoạch triển khai | 14–15 |
| Tìm đúng hình/trang/citation | 16–18 |

## 1. Nhận diện đúng paper

| Trường | Giá trị |
|---|---|
| Tên | TiMePReSt: Time and memory efficient pipeline parallel DNN training with removed staleness |
| Tác giả | Ankita Dutta; Nabendu Chaki; Rajat K. De |
| Đơn vị | Indian Statistical Institute; University of Calcutta |
| Tạp chí | Future Generation Computer Systems |
| Tập/năm/article number | 178 (2026), 108260 |
| DOI | https://doi.org/10.1016/j.future.2025.108260 |
| Received / revised / accepted | 24-10-2024 / 13-11-2025 / 15-11-2025 |
| Available online | 23-11-2025 |
| File nguồn | `1-s2.0-S0167739X25005540-main.pdf` |
| SHA-256 file nguồn | `dc6c291b36913e983f12e4750734ae685de7b17dbbaea769c6dbc40a1f11fea0` |
| Tác giả đầu | `ankitadutta1993@gmail.com` |
| Corresponding author | Rajat K. De, `rajat@isical.ac.in` |

Nguồn: [P, p.1]. Bản arXiv có tại https://arxiv.org/abs/2410.14312, nhưng không dùng các số hình, cấu hình hoặc kết quả ở bản đó để thay thế bản tạp chí một cách âm thầm. V-TiMePReSt và I-TiMePReSt là tên các phương pháp trong một công trình khác; không gộp vào thuật toán của file này.

## 2. Paper làm gì và đóng góp nằm ở đâu?

### 2.1. Bài toán

[P, §1–3, pp.1–5] Huấn luyện DNN trên nhiều accelerator bằng pipeline parallelism: chia model theo các nhóm layer liên tiếp, mỗi stage nằm trên một GPU. Cho nhiều mini-batch đang hoạt động để chồng lấp công việc giữa stage. Khi batch khác cập nhật trọng số, batch đang chạy có thể vẫn dùng phiên bản cũ.

Ba vấn đề gắn với nhau:

1. **Weight staleness:** một pass sử dụng trọng số cũ trong khi đã có cập nhật mới theo cách định nghĩa của tác giả.
2. **Memory overhead:** giữ nhiều phiên bản trọng số và activation trong quá trình nhiều batch cùng hoạt động.
3. **Training time:** chi phí truyền thông, lập lịch và backward/update hạn chế hiệu quả pipeline.

Đây là paper về **hệ thống huấn luyện/lập lịch**, không phải đề xuất kiến trúc VGG/ResNet mới, loss mới cho phân loại ảnh, hoặc mô hình ngôn ngữ mới.

### 2.2. Các đóng góp tác giả đề xuất

| Thành phần | Ý tưởng | Nơi đọc |
|---|---|---|
| Relax weight consistency | Cho forward và backward của một mini-batch dùng các phiên bản khác nhau; bỏ horizontal weight stashing | §3, §3.1, pp.3–5 |
| nF1B | Chia mini-batch thành N micro-batch chạy forward; backward chung dựa trên loss của cả nhóm | §3.1–3.2, pp.3–6 |
| Computation–communication overlap | Chuyển activation phần nhỏ sang stage sau khi phần tiếp theo đang tính | Fig.3, p.5 |
| Multiple sequence analysis | Liên hệ số worker W, micro-batch N và version difference v | §3.4–3.6, pp.6–7 |
| Communication model | Công thức số byte và thời gian truyền, có hệ số overlap | §3.8, p.8 |
| Evaluation | CNN trên ba dataset ảnh; TinyLlama trên UltraChat; cluster 2/3/4 GPU; ablation | §4, pp.8–14 |

### 2.3. Đánh đổi cần nhớ

[P, §4.6–4.10, pp.9–14] TiMePReSt có thể cần **nhiều epoch hơn** để đạt cùng chất lượng, nhưng chạy **mỗi epoch nhanh hơn**, từ đó tốt hơn theo thời gian trong các thí nghiệm báo cáo. Giảm bộ nhớ không có nghĩa luôn đạt chất lượng cao nhất ở cùng số epoch.

Hai ngoại lệ bắt buộc giữ trong mọi tóm tắt:

- §4.7 nói **DeepSpeed hơi tốt hơn TiMePReSt về bộ nhớ**, dù abstract/conclusion khái quát TiMePReSt vượt các baseline.
- Ablation Variant 1 **giữ weight stashing + nF1B** có accuracy/loss tốt hơn TiMePReSt theo cả thời gian và epoch, nhưng tốn bộ nhớ hơn (§4.10).

### 2.4. Không thuộc đóng góp đã chứng minh

[P, đầu p.3 và §5 p.14] Paper không giải quyết thuật toán phân hoạch tối ưu model trên GPU đồng nhất/không đồng nhất. Tác giả nói phân hoạch tương đối để cân bằng sử dụng GPU. Thí nghiệm thực tế chỉ dùng 2, 3, 4 worker; sơ đồ 5 worker không phải benchmark 5 GPU. Không có benchmark Colab hoặc Kaggle T4×2 trong paper.

## 3. Nền tảng và quan hệ với các công trình trước

Các số `[16]`, `[11]`... bên dưới là số reference của PDF, không phải số mục của file này.

| Công trình | Vai trò trong paper | Mức kế thừa có thể khẳng định |
|---|---|---|
| PipeDream [16], SOSP 2019 | 1F1B, nhiều mini-batch đang hoạt động, horizontal/vertical weight stashing, checkpoint; baseline chính | Trực tiếp: §3.1 gọi nF1B là biến thể của 1F1B PipeDream; không chứng minh dùng cùng source code |
| GPipe [11], NeurIPS 2019; đánh giá GPipe [12] | Micro-batching và synchronous pipeline | Nền tảng/đối chiếu liên quan; không phải nguồn code triển khai được xác nhận |
| PipeDream-2BW / PipeDream-Flush [17], ICML 2021 | Đánh đổi bộ nhớ và consistency | Được thảo luận; không có bộ đường cong baseline riêng cho hai biến thể này |
| DeepSpeed [13–15] | 3D parallelism, ZeRO, offload | Baseline thực nghiệm; config cụ thể chưa đầy đủ trong main paper |
| Zero Bubble [18], ICLR 2024 | Pipeline scheduling | Baseline thực nghiệm; exact variant/implementation chưa rõ |
| DualPipe qua DeepSeek-V3 [20] | Bidirectional pipeline | Baseline thực nghiệm; không suy ra cấu hình từ tên phương pháp |
| Chimera [19] | Bidirectional pipeline | Related work |
| XPipe [26] | Micro-batching và xử lý staleness | Related work; không có bằng chứng code TiMePReSt xây trên XPipe |
| DAPPLE [21], BaPipe [22], HetPipe [23] | Các nghiên cứu liên quan phân hoạch/pipeline | Bối cảnh cho phần partitioning nằm ngoài phạm vi |
| MEPipe [27] | Slice-level sequence pipeline scheduling | Related work |
| GraphPipe [28] | Graph pipeline, phụ thuộc DAG và thực thi nhánh độc lập | Related work |
| WeiPipe [29] | Weight-passing để giảm communication | Related work |
| Mario [30] | Activation checkpointing chồng lấp với bubble | Related work; đừng nhầm với checkpoint phục hồi lỗi của §3.3 |

[EXT] Code PipeDream chính thức: https://github.com/msr-fiddle/pipedream. README xác nhận nhánh `pipedream` cho SOSP 2019 và `pipedream_2bw` cho ICML 2021. Runtime dùng PyTorch; repo có profiler, partition optimizer, graph utilities và runtime. Không tự suy ra commit hoặc đường dẫn hàm trong bản clone người dùng từ README trên web.

## 4. Thuật ngữ và ký hiệu

| Ký hiệu/thuật ngữ | Nghĩa trong file |
|---|---|
| W | Số worker/stage/GPU trong pipeline đang xét |
| N | Số micro-batch của một mini-batch; `n` trong nF1B tương ứng N |
| M | Số mẫu trong mini-batch; khi chia đều mỗi micro-batch có M/N mẫu |
| m | Số layer khi mô tả kiến trúc tổng quát |
| mini-batch i | Nhóm dữ liệu dùng cho một lượt backward chung và update theo mô tả của paper |
| micro-batch iA, iB... | Các phần của mini-batch i dùng cho forward |
| stage | Một nhóm layer liên tiếp được gán cho worker |
| F / B | Forward của micro-batch / backward của mini-batch trong sơ đồ TiMePReSt |
| horizontal stashing | Giữ phiên bản để forward và backward của cùng mini-batch dùng trọng số nhất quán |
| vertical synchronization | Duy trì version consistency qua các stage trong cùng một pass theo mô tả paper |
| v | Chênh lệch chỉ số mini-batch kế tiếp trong cùng chuỗi truyền tác động update; không tự đồng nhất với mọi định nghĩa gradient staleness |
| eta | Learning rate trong phương trình update; không có giá trị số được xác định ở phương trình |
| D / data_comm | Số byte của một lần inter-stage communication trong mô hình đơn giản hóa |
| B_net | Network bandwidth; đổi tên từ B trong paper để tránh nhầm backward hoặc batch size |
| alpha | Tỷ lệ thời gian communication được che bởi computation, nằm trong [0,1] |

[?] Paper dùng `n` cho số worker trong Fig.1 và cho số micro-batch ở nF1B; nên code dùng `num_stages` và `num_microbatches` riêng. Phần lý thuyết đặt W>1, N>1; không áp công thức máy móc cho W=1 hoặc N=1.

## 5. Method chi tiết: kiến trúc, scheduling, loss và update

### 5.1. Luồng tính toán giữa stage

[P, §3.1, p.3; Fig.1 p.4]

1. Chia DNN thành các nhóm layer liên tiếp; phân bổ lên worker để cân bằng bộ nhớ/sử dụng tài nguyên theo mô tả của tác giả.
2. Worker đầu chứa/đọc input dataset. Các worker sau nhận output activation của stage trước; không nhất thiết lưu dataset cục bộ.
3. Forward đi từ stage đầu đến stage cuối.
4. Stage cuối tính prediction loss.
5. Backward đi từ stage cuối về stage đầu.

[I] Trong triển khai autograd phân tán, gradient gửi ngược qua biên thường là gradient theo activation đầu vào của stage. Parameter gradient dùng cho update tại stage; không mặc định gửi mọi parameter gradient về stage trước. Main paper gọi chung là gradients nên phải phân biệt bằng tensor shape và phép tính thực tế.

### 5.2. Mini-batch và micro-batch

[P, pp.3–5] Mỗi mini-batch M được chia thành N micro-batch bằng nhau. Forward của phần trước có thể chạy ở stage sau trong khi phần kế tiếp chạy ở stage trước. Khi toàn bộ N micro-batch của mini-batch hoàn thành forward, dùng loss trung bình của nhóm để bắt đầu backward chung.

[D] Với micro-batch bằng kích thước, nếu mỗi loss đã là mean trên mẫu, loss nhóm là:

```text
L_i = (1/N) * sum_j L_ij
```

[I] Nếu implementation hỗ trợ micro-batch không đều, cần mean có trọng số theo số mẫu. Với language model cần xác định denominator là token hợp lệ hay sequence; không tự dùng số micro-batch làm denominator. Paper không mô tả trường hợp không đều/token masking.

### 5.3. Lịch nF1B

[P, §3.2, pp.5–6]

- Lịch có warmup trước backward đầu tiên.
- Sau đó mỗi worker đi theo nhịp N forward rồi một backward khi dependency cho phép.
- Backward được ưu tiên khi đến để tránh bị chặn bởi quá nhiều forward.
- Với N=2 hoặc N=4, bài mô tả backward lặp lại mỗi 3 hoặc 5 time points sau backward đầu tiên.
- Nhiều mini-batch cùng đang hoạt động, không phải hoàn thành cả F/B của batch i rồi mới bắt đầu batch i+1.

**Ví dụ đọc Fig.2c, W=3, N=2** (p.4; bảng chép các ô đầu từ hình, chưa phải đo wall-clock):

| Stage / slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | F1A | F1B | F2A | F2B | F3A | F3B | B1 | F4A | F4B | B2 |
| 2 | idle | F1A | F1B | F2A | F2B | B1 | F3A | F3B | B2 | F4A |
| 3 | idle | idle | F1A | F1B | B1 | F2A | F2B | B2 | F3A | F3B |

Ở đây B1 là backward chung của mini-batch 1 tại stage tương ứng. Stage 1 đã forward mini-batch 3 trước khi B1 đến, nên update từ batch trước có thể xuất hiện giữa forward và backward của batch sau.

[?] Một ô F và một ô B trong hình có cùng bề rộng không chứng minh thời gian thực của chúng bằng nhau. Warmup/drain tổng quát, tie-break giữa các queue và synchronization cụ thể cần được đặc tả khi triển khai.

### 5.4. Weight-version policy

[P, §3 và §3.1, pp.3–5]

- Bỏ yêu cầu giữ cùng trọng số giữa F và B của một mini-batch (**horizontal**).
- Tác giả nói backward dùng trọng số cập nhật mới thay vì phiên bản cũ dùng ở forward.
- Vẫn giữ **vertical synchronization** để nhất quán version trong từng pass.
- Micro-batch của cùng mini-batch có thể bắt đầu bằng version khác nhau. Ví dụ Fig.2b: 3C/3D có thể dùng update từ mini-batch 1, khác 3A/3B.
- Version cũ chỉ cần giữ đến khi forward còn dùng nó đã xong và có version thay thế, thay vì chờ backward của mọi batch dùng nó ở forward.

**Điểm không được giản lược:** “latest” trong mô tả không đủ để kết luận mỗi stage luôn lấy `current_parameters` tại mọi thời điểm, vì paper đồng thời giữ vertical consistency. Cần định nghĩa version theo pass/stage và kiểm chứng bằng trace.

### 5.5. Update equation

[P, Eq.(1), pp.4–5] Đổi ký hiệu vector trọng số thành theta để không nhầm W là số worker:

```text
theta(t+1) = theta(t)
             - eta * grad f(theta_1(t-v+1), ..., theta_W(t-v+1))
```

`f` là loss trên mini-batch, `theta_l` là tham số stage l, `v` là version difference của paper. Với v=1, biểu thức được tác giả trình bày như gradient ở version hiện tại t.

[?] Phương trình này không xác định đầy đủ cách tính gradient khi activation được sinh bởi version forward cũ. Main paper không đưa source/pseudocode chứng minh phép backward trong autograd phù hợp với ký hiệu gradient ở trên. Không được coi `optimizer.step()` trên parameter mới là bằng chứng gradient đã được tính ở parameter mới.

### 5.6. “Một backward” thực sự có nghĩa gì?

[P] Paper gom prediction error/loss của N forward thành một backward mini-batch.

[D/I] Một lần gọi backward trên tổng loss vẫn phải đi qua mọi nhánh graph có đóng góp. Có thể giảm overhead/lần gọi/message nhưng không tự động giảm FLOPs hoặc byte gradient N lần. Nếu có batching/recomputation/fusion để đạt lợi ích đó thì phải mô tả và đo.

Không có căn cứ để thay bằng trung bình logits rồi tính loss một lần; nói chung `loss(mean(logits))` khác `mean(loss(logits))`.

### 5.7. Checkpoint và fault tolerance

[P, §3.3 p.6] Mỗi stage lưu model parameters cục bộ vào cuối epoch, sau backward mini-batch cuối. Khi gián đoạn, tác giả mô tả resume từ epoch hoàn thành gần nhất của các stage; checkpoint không cần gửi tham số sang worker khác.

[?] Chưa có chi tiết lưu optimizer, scheduler, RNG, sampler, global step, commit đồng bộ giữa stage hoặc recovery giữa epoch. Đây là checkpoint phục hồi huấn luyện, không phải activation recomputation/checkpointing để tiết kiệm bộ nhớ.

[I] Khi tái hiện, lưu đủ trạng thái cần thiết và epoch manifest chung tại ranh giới pipeline đã drain. Ghi rõ đây là bổ sung kỹ thuật nếu paper không mô tả.

## 6. Multiple sequence problem và công thức version difference

### 6.1. Bản chất theo paper

[P, §3.4, p.6] Với W=4, N=2 trong Fig.2a, update lan theo hai chuỗi `{1,3,5,7,...}` và `{2,4,6,...}`. Tác giả nói hai chuỗi không tương tác đến cuối epoch nên trọng số cuối chỉ phản ánh một phần các mini-batch. Đây là lý do họ ưu tiên v=1.

Với W=4, N=4 trong Fig.2b, chỉ có chuỗi `{1,2,3,4,...}`. v là khoảng cách chỉ số mini-batch liên tiếp **trong cùng chuỗi update**, không phải micro-batch count hoặc số epoch trễ.

### 6.2. Công thức chính

[P, Eq.(2),(3),(9),(10),(13),(14),(16), pp.6–7]

```text
W <= N + 1  <=>  v = 1
W >  N + 1  <=>  v > 1

v = floor((W + N - 2) / N),  W >= 2, N >= 2
```

| W | N | v từ công thức | Vị trí/ý nghĩa |
|---:|---:|---:|---|
| 4 | 2 | 2 | Fig.2a; cấu hình có multiple sequences |
| 4 | 4 | 1 | Fig.2b |
| 3 | 2 | 1 | Fig.2c |
| 5 | 2 | 2 | Fig.2d; chỉ minh họa schedule |
| 5 | 3 | 2 | Fig.2e; chỉ minh họa schedule |
| 2 | 3 | 1 | Main experiment Cluster A |
| 3 | 3 | 1 | Main experiment Cluster B |
| 4 | 3 | 1 | Main experiment Cluster C |

[D] Với yêu cầu v=1, chọn `N >= max(2, W-1)` trong miền công thức. Điều này không có nghĩa N càng lớn càng tối ưu về tốc độ/bộ nhớ.

### 6.3. Cách tác giả xây công thức và giới hạn

[P, §3.5–3.6]

```text
f1 = W + N - 1                (4)
f2 = f1 + 1 = W + N           (5)
b = W                        (6)
f2 >= f1 + b - N  <=> v = 1   (7)
f2 <  f1 + b - N  <=> v > 1   (8)
```

Thay vào dẫn tới W<=N+1. Sau đó tác giả dùng `v <= (W+N-1)/N`, đặt phần chênh x và giả định x tỷ lệ nghịch N để đi đến biểu thức xấp xỉ rồi lấy floor (Eq.11–16).

[?] Đây là lập luận trên lịch/time-point lý tưởng của bài, không phải chứng minh tổng quát cho mọi runtime bất đồng bộ, thời gian stage không đều hoặc thứ tự event tùy ý. Định nghĩa f2 và phép suy từ bất đẳng thức sang biểu thức chính xác cần được kiểm tra bằng trace, không dùng như định lý độc lập của implementation.

[I] Viết simulator kiểm tra ít nhất các cặp trong bảng. Nếu runtime cập nhật cùng một global weight tuần tự bằng mọi gradient, không tự kết luận nó có các chuỗi độc lập như paper; phải xem cơ chế đọc/ghi version thực tế.

## 7. Mô hình communication

[P, §3.8, p.8] Giả định D gần như bằng nhau cho mỗi lần truyền và mỗi biên stage, bandwidth B_net:

```text
V_naive ≈ 2 * (W-1) * N * D                 (17)
T_naive = V_naive / B_net                   (18)

V_TiMePReSt ≈ (W-1) * (N+1) * D            (19)
T_no_overlap = V_TiMePReSt / B_net          (20)
T_exposed = (1-alpha) * V_TiMePReSt / B_net  (21)

V_TiMePReSt / V_naive = (1 + 1/N) / 2
```

Forward có N hạng truyền; backward có một hạng truyền. Với N=3, tỷ lệ từ công thức là 2/3; với N=2 là 3/4; khi N lớn tiến tới 1/2. Đây là kết quả toán của mô hình giả định, **không phải số đo speedup thực tế**.

[?] Mô tả còn nói giảm communication N lần nhờ backward. Không được áp con số đó cho tổng F+B: chính công thức cho tỷ lệ `(N+1)/(2N)`.

[D] Trong triển khai thông thường, gom N gradient activation có thể khiến message backward lớn N lần. Nếu vậy số message giảm nhưng tổng byte không giảm theo giả định D cố định. Các stage còn có kích thước activation khác nhau. Phải đo actual tensor bytes, latency, bandwidth và overlap; không ép implementation vào công thức bằng cách bỏ dữ liệu gradient.

[?] Alpha là biến mô hình, main paper không báo giá trị đo hoặc thuật toán xác định nó. Công thức tăng tuyến tính với W không tự chứng minh hiệu năng scale tuyến tính; tác giả kết luận cũng nói số worker và training time không có quan hệ nghịch đảo đơn giản.

## 8. Mô hình, dataset và phạm vi thí nghiệm

### 8.1. Phân biệt model với training system

| Loại | Các tên trong bài |
|---|---|
| DNN được huấn luyện/finetune | VGG-16, ResNet-50, AlexNet, TinyLlama |
| Training system được so sánh | TiMePReSt, PipeDream, DeepSpeed, DualPipe, Zero Bubble |
| Dataset | CIFAR-100, Tiny-ImageNet-200, MS-COCO 2017, UltraChat |

Không dùng từ “model TiMePReSt” để suy ra một neural architecture khác VGG-16; trong nhiều đoạn paper dùng model/platform khá rộng.

### 8.2. Mười workload đã được báo cáo

[P, §4.1–4.7, Figs.4–10 và 15–17, pp.8–13]

| ID gợi ý [I] | Model | Dataset | Tác vụ ghi trong bài | Quality metric | Đường cong Cluster A |
|---|---|---|---|---|---|
| vgg16_cifar100 | VGG-16 | CIFAR-100 | Image classification | Top-1, Top-5, loss | Fig.4 p.9 |
| vgg16_tinyimagenet | VGG-16 | Tiny-ImageNet-200 | Image classification | Top-1, Top-5, loss | S4, chỉ được dẫn chiếu |
| vgg16_coco2017 | VGG-16 | MS-COCO 2017 | Image classification | Top-1, Top-5, loss | S5, chỉ được dẫn chiếu |
| resnet50_cifar100 | ResNet-50 | CIFAR-100 | Image classification | Top-1, Top-5, loss | S6, chỉ được dẫn chiếu |
| resnet50_tinyimagenet | ResNet-50 | Tiny-ImageNet-200 | Image classification | Top-1, Top-5, loss | Fig.5 p.10 |
| resnet50_coco2017 | ResNet-50 | MS-COCO 2017 | Image classification | Top-1, Top-5, loss | S7, chỉ được dẫn chiếu |
| alexnet_cifar100 | AlexNet | CIFAR-100 | Image classification | Top-1, Top-5, loss | S8, chỉ được dẫn chiếu |
| alexnet_tinyimagenet | AlexNet | Tiny-ImageNet-200 | Image classification | Top-1, Top-5, loss | S9, chỉ được dẫn chiếu |
| alexnet_coco2017 | AlexNet | MS-COCO 2017 | Image classification | Top-1, Top-5, loss | Fig.6 p.10 |
| tinyllama_ultrachat | TinyLlama | UltraChat | Fine-tuning language model | BLEU, loss | Fig.7 p.10 |

Đường cong bổ sung trong main paper: VGG-16/Tiny-ImageNet trên Cluster B (Fig.8), cùng workload trên Cluster C với 5 systems (Fig.9), TinyLlama/UltraChat trên Cluster C với 5 systems (Fig.10). Fig.15–17 có đủ 10 nhóm workload nhưng không có đầy đủ đường cong accuracy cho mọi workload trên mọi cluster.

### 8.3. Chi tiết model/dataset KHÔNG được phép mặc định

- **VGG-16:** chưa xác định bản có BatchNorm hay không; classifier/head, kích thước input và pretrained weights chưa rõ.
- **ResNet-50:** chưa xác định cách sửa stem cho ảnh nhỏ hoặc resize lên input lớn; pretrained/scratch chưa rõ.
- **AlexNet:** chưa xác định biến thể cho CIFAR/Tiny-ImageNet và thay đổi classifier.
- **COCO 2017:** bài gọi là image classification, không phải object detection. Chưa rõ biến đổi annotation thành single-label/multi-label, có crop object hay không, cách định nghĩa Top-k/loss. Không tự dùng pipeline detection hoặc mặc định 80-way cross-entropy.
- **TinyLlama:** tên checkpoint/revision, tokenizer, sequence length, trainable parameters, full fine-tuning hay adapter, precision/quantization chưa được main paper cung cấp đủ.
- **UltraChat:** chưa xác định version/subset, chat template, prompt masking, packing, train/eval split và generation settings.
- **BLEU:** chưa có chuẩn tính, tokenizer, n-gram/smoothing, corpus-vs-sentence aggregation và bộ reference được xác định đầy đủ.
- Không lấy kích thước dataset chuẩn hoặc train/test split thường dùng rồi ghi là cấu hình tác giả. Cần định danh dataset và effective sample count trong config tái hiện.

## 9. Thiết lập phần cứng và cấu hình được báo cáo

### 9.1. Cluster

[P, Table 1, p.8; chép theo đúng bảng]

| Cluster | Số worker | GPU mỗi worker |
|---|---:|---|
| A | 2 | Worker 1: NVIDIA Quadro RTX 6000 (24 GB); worker 2: NVIDIA GeForce RTX 2080 (12 GB) |
| B | 3 | Worker 1–2: NVIDIA Quadro RTX 6000 (24 GB); worker 3: NVIDIA GeForce RTX 2080 (12 GB) |
| C | 4 | Worker 1–3: NVIDIA Quadro RTX 6000 (24 GB); worker 4: NVIDIA GeForce RTX 2080 (12 GB) |

[?] Dòng “RTX 2080 (12 GB)” ở đây là **nguyên thông tin trong Table 1**, không phải thông số phần cứng đã kiểm chứng độc lập. Nếu cần tái tạo hardware chính xác, xác minh lại tên card/VRAM với tác giả thay vì âm thầm đổi tên GPU.

Mỗi worker có một GPU. Main paper không cho đủ CPU, RAM, mạng/interconnect, bandwidth đo, driver/CUDA/PyTorch, rank mapping và layer split để tái tạo môi trường.

### 9.2. N và batch size

[P, §4.1 p.8; §4.9 pp.11–12]

- Main experiments: **N=3** trên cả A/B/C; tương ứng v=1 trong công thức.
- N sensitivity: Cluster C, **N=2**, với hai batch-size settings: bằng setting N=3 và một mini-batch lớn hơn.
- Batch size khác nhau giữa dataset; cùng dataset dùng cùng mini-batch size giữa các systems để so sánh theo lời tác giả (§4.5).
- **Không có bảng giá trị số mini-batch size theo workload trong main paper đã đọc.** Không đặt batch=32/64/128 rồi gọi là “paper setting”.
- Sơ đồ Fig.2 minh họa N=2/3/4 không thay thế cấu hình main experiments N=3.

### 9.3. Hệ thống so sánh và phạm vi bằng chứng

| Cluster | Đường cong chính đọc được | Biểu đồ hiệu năng/bộ nhớ |
|---|---|---|
| A | Figs.4–7: TiMePReSt vs PipeDream; các workload còn lại được dẫn sang S4–S9 | S10: hardware/throughput; phần bộ nhớ được dẫn tới S11/S12, chưa xem |
| B | Fig.8: VGG-16/Tiny-ImageNet, TiMePReSt vs PipeDream | Fig.16: thời gian/epoch và throughput đủ 10 workload |
| C | Figs.9–10: TiMePReSt, PipeDream, DeepSpeed, DualPipe, Zero Bubble | Fig.15: stage memory; Fig.17: thời gian/epoch và throughput đủ 10 workload, kèm biến thể |

Paper nói so sánh rộng trên các cluster, nhưng không suy ra mọi đường cong của mọi baseline đều có sẵn trong main PDF. Exact baseline commit/config không được xác định chỉ bằng tên system.

## 10. Metric và cách đọc kết quả đúng

### 10.1. Các trục và đơn vị

[P, §4.2–4.7, pp.8–10]

| Metric | Ý nghĩa trong paper | Cách dùng khi tái hiện |
|---|---|---|
| Top-1 / Top-5 accuracy | Chất lượng phân loại CNN | Cần xác định train/val/test split trước khi so |
| BLEU | Chất lượng đầu ra TinyLlama | Không gọi là classification accuracy |
| Loss | Loss đường cong huấn luyện/đánh giá được vẽ | Exact loss definition/reduction/split cần xác minh |
| Hardware efficiency | Thời gian cho một epoch; càng thấp càng tốt | Không đồng nhất với GPU utilization % |
| Statistical efficiency | Epoch cần để đạt chất lượng mục tiêu | Có thể kém đi dù wall-clock tốt lên |
| Throughput | **Epochs/hour** trong paper | Khi đo samples/s hoặc tokens/s phải ghi đơn vị mới |
| Memory footprint | GB, vẽ theo từng stage | Đừng lấy tổng cột stacked làm peak VRAM của một GPU |
| Time points | Đơn vị thời gian chuẩn hóa trong biểu đồ chất lượng | Không phải trực tiếp giờ/phút/epoch |

### 10.2. Time points

[P, §4.5 p.9] Khoảng cách giữa hai time points bằng thời gian một epoch của một system trong phép so sánh; độ dài thay đổi theo workload. Main paper diễn đạt là “any one of the models” mà không định danh rõ hệ quy chiếu của từng hình.

[?] Vì vậy không nhân time points với epoch time tùy ý để xuất số giờ chính xác. Không dùng quy ước từ paper khác hoặc phiên bản khác để điền vào chỗ thiếu. Khi tái hiện, lưu wall-clock seconds thật và ghi rõ nếu tạo thêm trục normalized time.

### 10.3. Training hay test accuracy?

[?] Abstract/conclusion nói “training accuracy”, còn caption thường chỉ ghi “accuracy”. Main paper không mô tả đủ evaluation protocol để mặc định các đường cong là held-out test accuracy. So sánh accuracy Colab trên test set với đường cong này phải ghi giới hạn đó.

### 10.4. Kết quả chính theo tác giả

1. [P, §4.2] Đạt target quality nhanh hơn các baseline được so sánh; không có bảng target-accuracy/time-to-target số chính xác cho từng workload.
2. [P, §4.3–4.4] Accuracy/BLEU cao hơn và loss thấp hơn sau cùng lượng thời gian ở các đoạn thí nghiệm báo cáo. Đây không phải bảo đảm ưu thế tại mọi điểm của mọi đường cong.
3. [P, §4.5–4.6] Epoch nhanh hơn, throughput cao hơn; chấp nhận statistical efficiency kém hơn.
4. [P, §4.5–4.6] A→B: cả TiMePReSt và PipeDream tốt lên về hardware/throughput. B→C: TiMePReSt tiếp tục tốt lên, PipeDream không cải thiện theo mô tả.
5. [P, §4.7] Bộ nhớ TiMePReSt thấp hơn PipeDream/DualPipe/Zero Bubble, nhưng DeepSpeed hơi thấp hơn TiMePReSt.
6. [P, §4.8] Tác giả liên hệ phần tiết kiệm memory của DeepSpeed với offload và phần chậm hơn với overhead. Đây là diễn giải tác giả, không thay thế config hoặc profiling từng baseline.

## 11. Kết quả định lượng đọc từ hình

> Tất cả giá trị trong mục này là `[F≈]`: ước lượng thủ công từ hình đã render, có làm tròn. Không có CSV/raw runs để xác nhận, không được viết thêm chữ số thập phân hoặc gọi là số đo chính xác. Khoảng giá trị dưới đây biểu thị độ thô khi đọc, không phải độ biến thiên qua seed.

### 11.1. Các mốc chất lượng tiêu biểu

| Hình / workload | Trục và vùng đọc | TiMePReSt | PipeDream / biến thể | Ý nghĩa |
|---|---|---|---|---|
| Fig.4a, VGG-16/CIFAR-100, A | Cuối trục khoảng 27 time points | Top-1 khoảng 72–75% | PipeDream khoảng 44–48% | Ưu thế tại cùng normalized time |
| Fig.4c, cùng workload | Khoảng 27 time points | Top-5 khoảng 88–91% | PipeDream khoảng 74–78% | Không phải cùng số epoch |
| Fig.4b, cùng workload | Cuối trục khoảng 160 epoch | Top-1 khoảng 72–75% | PipeDream khoảng 73–77% | Chất lượng theo epoch gần nhau, PipeDream hơi cao hơn |
| Fig.8a, VGG-16/Tiny-ImageNet, B | Cuối trục khoảng 25 time points | Top-1 khoảng 69–73% | PipeDream khoảng 31–35% | Chênh lệch theo thời gian lớn |
| Fig.8b, cùng workload | Khoảng 140–145 epoch | Top-1 khoảng 68–72% | PipeDream khoảng 73–77% | Đánh đổi statistical efficiency |
| Fig.9a, VGG-16/Tiny-ImageNet, C | Khoảng 21 time points | Top-1 khoảng 69–73% | Các baseline khác khoảng 24–35% | 5 systems trong hình |
| Fig.10a, TinyLlama/UltraChat, C | Khoảng 23 time points | BLEU khoảng 68–72 | Các baseline khác khoảng 28–40 | BLEU, không ghi dấu % accuracy |
| Fig.13a, VGG-16/Tiny-ImageNet, C | Khoảng 36 time points | Top-1 khoảng 63–67% | Variant 1 khoảng 78–82%; Variant 2 khoảng 33–37% | Giữ stashing + nF1B cho chất lượng tốt hơn ở hình này |
| Fig.14a, TinyLlama/UltraChat, C | Khoảng 18 time points | BLEU khoảng 37–40 | Variant 1 khoảng 50–53; Variant 2 khoảng 21–24 | Cùng trade-off của ablation |

Lưu ý: không so trực tiếp điểm cuối của Fig.9a với Fig.13a như cùng ngân sách thời gian vì thang time points/cửa sổ vẽ khác nhau. Màu legend có thể đổi giữa panel, đặc biệt Fig.9; luôn đọc tên đường ở chính panel đó.

Figs.5–7 cho xu hướng cùng chiều ở ResNet-50/Tiny-ImageNet, AlexNet/COCO và TinyLlama/UltraChat trên A. Fig.7c có giai đoạn loss của TiMePReSt cao hơn rồi mới thấp hơn về sau; không viết “thấp hơn ở mọi thời điểm”.

### 11.2. Thời gian/epoch trên Cluster B và C

Nguồn: Fig.16a (B) và Fig.17a (C), p.13. Đơn vị phút/epoch. Khoảng đọc được làm tròn rộng, đặc biệt các cột thấp.

| Workload | B: TiMePReSt | B: PipeDream | C: TiMePReSt | C: PipeDream |
|---|---:|---:|---:|---:|
| VGG-16 / CIFAR-100 | ~10–12 | ~60–65 | ~5–8 | ~65–70 |
| VGG-16 / Tiny-ImageNet-200 | ~73–78 | ~128–133 | ~65–70 | ~135–140 |
| VGG-16 / COCO 2017 | ~110–115 | ~165–172 | ~100–105 | ~170–176 |
| ResNet-50 / CIFAR-100 | ~7–10 | ~50–55 | ~5–8 | ~58–63 |
| ResNet-50 / Tiny-ImageNet-200 | ~60–65 | ~120–125 | ~53–58 | ~130–135 |
| ResNet-50 / COCO 2017 | ~95–100 | ~153–160 | ~85–90 | ~162–168 |
| AlexNet / CIFAR-100 | ~7–9 | ~33–37 | ~4–7 | ~40–45 |
| AlexNet / Tiny-ImageNet-200 | ~50–55 | ~100–105 | ~43–48 | ~110–115 |
| AlexNet / COCO 2017 | ~78–83 | ~130–135 | ~65–70 | ~138–143 |
| TinyLlama / UltraChat | ~30–33 | ~44–48 | ~29–33 | ~53–58 |

Fig.17a còn có DeepSpeed, DualPipe, Zero Bubble: trong các nhóm main baseline, TiMePReSt có thời gian/epoch thấp nhất theo biểu đồ. Không suy ra speedup end-to-end tới target chỉ từ tỷ số thời gian/epoch, vì số epoch tới target khác nhau.

### 11.3. Throughput và điểm chưa nhất quán

Nguồn: Fig.16b, Fig.17b, p.13. Một số cột TiMePReSt `[F≈]`:

| Workload | B: epochs/hour đọc từ hình | C: epochs/hour đọc từ hình |
|---|---:|---:|
| VGG-16 / CIFAR-100 | ~6 | ~7.5 |
| ResNet-50 / CIFAR-100 | ~6 | ~7.5 |
| AlexNet / CIFAR-100 | ~7.2 | ~8.5 |
| TinyLlama / UltraChat | ~1.8–2.0 | ~2.3–2.5 |

[D/?] Theo định nghĩa cùng một run, throughput epochs/hour phải gần `60 / minutes_per_epoch`. Các cột trong hai panel không phải lúc nào cũng khớp quan hệ này. Ví dụ B, ResNet-50/COCO: thời gian ~95–100 phút gợi throughput ~0.6–0.63, trong khi cột hình ở khoảng 0.8–0.9. Đây là dấu hiệu cần hỏi raw measurements/phương pháp tính, không phải lý do tự chỉnh số của paper. Khi tái hiện, báo cả hai metric và kiểm tra tính nhất quán của log.

### 11.4. Bộ nhớ trên Cluster C

Nguồn Fig.15 p.13. Các cột được stack theo 4 stage. Bảng dưới đọc **tổng chiều cao cột**, tức tổng các segment stage được vẽ, không phải per-GPU peak và không chắc là tổng peak xảy ra cùng thời điểm.

| Workload | PipeDream tổng GB [F≈] | TiMePReSt tổng GB [F≈] | DeepSpeed tổng GB [F≈] |
|---|---:|---:|---:|
| VGG-16 / CIFAR-100 | ~11 | ~7 | ~6 |
| VGG-16 / Tiny-ImageNet | ~15 | ~10 | ~8 |
| VGG-16 / COCO | ~16 | ~13 | ~12 |
| ResNet-50 / CIFAR-100 | ~13 | ~10–11 | ~9–10 |
| ResNet-50 / Tiny-ImageNet | ~17 | ~12 | ~10 |
| ResNet-50 / COCO | ~21 | ~19–20 | ~18 |
| AlexNet / CIFAR-100 | ~11 | ~10 | ~9 |
| AlexNet / Tiny-ImageNet | ~14 | ~9 | ~8 |
| AlexNet / COCO | ~17 | ~13 | ~12 |
| TinyLlama / UltraChat | ~18–19 | ~15 | ~14 |

Ví dụ VGG-16/Tiny-ImageNet: tổng TiMePReSt ~10 GB không có nghĩa mỗi GPU cần 10 GB. Phải đọc từng màu stage nếu cần so với VRAM của một card. Main paper chưa định nghĩa rõ đây là allocated/reserved/NVML memory, peak hay sample tại một thời điểm.

Không dùng bảng này làm bảo đảm rằng cùng model sẽ vừa Colab GPU: input resolution, batch size, precision, optimizer và partition đều còn thiếu.

## 12. Sensitivity và ablation

### 12.1. Thay số micro-batch và batch size

[P, §4.9 pp.11–12; Figs.11–12 p.12; Fig.17 p.13]

| Setting | W | N | Mini-batch size | v theo Eq.(3) |
|---|---:|---:|---|---:|
| Main | 4 | 3 | M, chưa cho số | 1 |
| N=2 | 4 | 2 | Cùng M với main | 2 |
| N=2 larger batch | 4 | 2 | Lớn hơn M, chưa cho số/hệ số | 2 |

Workload: VGG-16/Tiny-ImageNet-200 và TinyLlama/UltraChat. Tác giả báo N=3 tốt hơn cả hai setting N=2 về quality/loss theo time và epoch; N=2 với mini-batch lớn hơn có kết quả kém nhất. Họ giải thích bằng giảm overlap khi tăng mini-batch với N cố định.

[D] Thí nghiệm N=3→2 cũng đổi v=1→2; ở M cố định còn đổi micro-batch size. Đây không phải phép đo thuần riêng tác động N. Setting larger batch lại đổi số update mỗi epoch và kích thước hiệu dụng; khi tái hiện phải ghi các yếu tố đồng biến này.

[P] Với miền N>=2 và W thuộc {2,3,4}, tác giả chỉ có W=4,N=2 tạo v>1. Điều này không có nghĩa N>3 không được phép, mà các N đó vẫn nằm trong nhóm v=1 của công thức.

### 12.2. Loại từng thành phần

[P, §4.10 pp.12,14; Figs.13–15,17]

| Phiên bản | Scheduling | Horizontal stashing | Kết quả được báo cáo |
|---|---|---|---|
| TiMePReSt | nF1B | Bỏ | Cấu hình chính |
| Variant 1 | nF1B | Giữ | Quality tốt hơn, loss thấp hơn so với main theo cả time/epoch; memory cao hơn |
| Variant 2 | 1F1B | Bỏ | Chậm hơn main theo time; tác giả mô tả kết quả theo epoch tương đối tương đương |

Trong legend, hai biến thể ghi “weight stashing, nF1B” và “no weight stashing, 1F1B”. Trong ngữ cảnh method, thao tác cốt lõi là horizontal stashing; không tự diễn giải legend là bỏ cả vertical.

Workload thực sự thấy trong hình: VGG-16/Tiny-ImageNet (Fig.13), TinyLlama/UltraChat (Fig.14), đều Cluster C. §4.10 có câu chỉ nói VGG-16 nhưng §4.1 và Fig.14 cho biết có TinyLlama; ghi nhận khác biệt mô tả này.

[F≈] Fig.13f vẫn có khoảng cách loss giữa Variant 2 và main, nên không chuyển lời “comparable epoch-wise” thành “hai đường giống nhau”.

### 12.3. Kết luận nào được và không được rút ra?

- Có bằng chứng thực nghiệm trong bài cho lợi ích thời gian của nF1B so với variant 1F1B đã thử.
- Có bằng chứng cho trade-off memory–quality của việc bỏ horizontal stashing.
- Không có cơ sở nói bỏ stashing luôn tăng accuracy hoặc luôn có lợi hơn giữ stashing.
- Không có kiểm chứng mọi hyperparameter/hardware hay chứng minh tối ưu toàn cục.
- Các biểu đồ ablation và đường cong main có cửa sổ time points khác nhau; không ghép điểm cuối thành cùng một thí nghiệm.

## 13. Thiếu thông tin, mâu thuẫn và mức độ ảnh hưởng

### 13.1. Những cấu hình main paper chưa xác định đủ

`UNKNOWN` có nghĩa **chưa tìm thấy thông tin đủ trong main paper đã đọc**, không có nghĩa tác giả chắc chắn không có hoặc supplementary chắc chắn không cung cấp.

| Nhóm | Trạng thái | Ảnh hưởng / việc cần làm |
|---|---|---|
| Global mini-batch size từng workload | UNKNOWN | Không so fair về số update, memory, throughput nếu đoán |
| Micro-batch count | Biết: main N=3; sensitivity N=2 | Giữ tách khỏi micro-batch size |
| Optimizer cụ thể | UNKNOWN | Eq.(1) là dạng update, không đủ để khẳng định training dùng plain SGD |
| LR, momentum, weight decay | UNKNOWN | Phải ghi là lựa chọn tái hiện nếu tự đặt |
| LR scheduler / warmup | UNKNOWN | Quyết định update theo epoch/step cần lưu rõ |
| Epoch budget chính xác từng run | Chỉ thấy miền trục hình | Không coi tick cuối là config hoàn chỉnh |
| Initialization / pretrained weights | UNKNOWN | Ảnh hưởng mạnh chất lượng |
| Model variant và head | UNKNOWN | Cần xác minh architecture trước baseline |
| Resolution, normalization, augmentation | UNKNOWN | Không mặc định CIFAR luôn dùng input 32×32 |
| Dataset version/subset/split/count | UNKNOWN | Lưu manifest của bản tái hiện |
| COCO task conversion và Top-k | UNKNOWN, rủi ro cao | Cần hỏi tác giả hoặc xác định là task adaptation riêng |
| TinyLlama checkpoint/tokenizer | UNKNOWN | Không tự suy ra checkpoint từ tên họ model |
| UltraChat formatting, masks, packing, length | UNKNOWN | Cần định nghĩa rõ token-level workload |
| Fine-tuning full model/LoRA/QLoRA | UNKNOWN | Không đưa quantization/adapter vào rồi gọi đúng cấu hình paper |
| BLEU protocol / decoding settings | UNKNOWN | BLEU khác protocol không so trực tiếp |
| Training/evaluation split cho accuracy/loss | Chưa rõ | Ghi metric provenance |
| Layer partition cho mỗi model/cluster | UNKNOWN | Paper không cung cấp thuật toán partition mới |
| Runtime scheduler đầy đủ | Chỉ có mô tả và Fig.2 | Cần warmup/steady/drain/tie-break/state machine |
| Backward qua activation cũ, weight mới | UNKNOWN về thao tác cụ thể | Nút thắt lớn nhất khi khẳng định faithful reproduction |
| Vertical version propagation | Có mô tả khái niệm; thiếu protocol | Cần version tags, lifetime và event trace |
| Optimizer state khi đổi weight version | UNKNOWN | Momentum/Adam state không được bỏ qua |
| BN buffers, dropout và recomputation RNG | UNKNOWN | Có thể làm lệch gradient dù schedule trông đúng |
| Framework/library versions và backend | UNKNOWN | Repo PipeDream cũ không đồng nghĩa môi trường TiMePReSt |
| Network topology/bandwidth | UNKNOWN | Không thể tái tạo wall-clock chính xác |
| DeepSpeed/DualPipe/Zero Bubble config | UNKNOWN | Không có căn cứ benchmark mặc định là fair |
| Memory measurement protocol | UNKNOWN | Không biết allocated/reserved/device total/peak |
| Time-point reference theo từng hình | Chưa định danh rõ | Không đổi sang giây tùy ý |
| Number of seeds/error bars/raw logs | Không thấy báo cáo đủ | Không suy ra statistical significance |
| Source code và raw experiment data | Chưa truy cập được code chính thức | Có thể liên hệ tác giả; không bịa link |

### 13.2. Các chỗ cần đọc phê bình, không đưa thẳng vào code

| Vấn đề | Bằng chứng | Cách xử lý |
|---|---|---|
| “Zero staleness” vs vertical stashing / “almost zero” | §3 giữ vertical sync; §4.8 dùng cách diễn đạt gần zero | Định nghĩa staleness theo pass/stage, không coi toàn bộ gradient là SGD đồng bộ |
| Abstract/conclusion nói thắng memory toàn bộ baseline | §4.7 nói DeepSpeed tốt hơn chút | Báo ngoại lệ DeepSpeed |
| Giữ stashing cho chất lượng tốt hơn | §4.10 Variant 1 | Không diễn giải bỏ stashing cải thiện accuracy |
| “Giảm communication n lần” | Eq.19 tổng F+B cho tỷ lệ `(N+1)/(2N)` | Tách message count, bytes, overhead và exposed time |
| D cố định cho backward nhóm | §3.8 | Kiểm tra actual tensor sizes; không lược mất gradient |
| Time/epoch và throughput chưa khớp hoàn toàn | Fig.16–17 | Giữ số đọc riêng, yêu cầu raw data |
| So sánh GPipe/data parallel theo epoch | §1 và §3.7 diễn đạt update một lần/epoch | Không dùng làm quy tắc triển khai baseline |
| Memory của PipeDream-2BW/Flush | §4.8 trước nói TiMePReSt kém memory hơn về lý thuyết, sau có câu mô tả ưu thế memory rộng | Đánh dấu không nhất quán; không kết luận TiMePReSt luôn tốt hơn hai biến thể |
| GPU “RTX 2080 (12 GB)” | Table 1 | Giữ đúng nội dung trích bảng, xác minh hardware nếu cần |
| Hình và cross-reference | §4.2 có chỗ dẫn 4a/4b cho top-1/top-5, nhưng Fig.4b là top-1 vs epoch | Ưu tiên caption và trục thực tế |
| Mô tả ablation chỉ nhắc VGG ở một câu | §4.1 và Fig.14 có TinyLlama | Giữ đủ cả hai workload, ghi khác biệt văn bản |

[EXT] Google mô tả GPipe chia **một mini-batch** thành micro-batch và tích lũy gradient giữa chúng, không yêu cầu chờ cả dataset/epoch. Vì vậy lời mô tả GPipe “chỉ update mỗi epoch” trong TiMePReSt không nên được dùng để tạo baseline GPipe sai. Nguồn đối chiếu: https://research.google/blog/introducing-gpipe-an-open-source-library-for-efficiently-training-large-scale-neural-network-models/ và paper GPipe [11].

### 13.3. Khi nào phải mở lại PDF hoặc xin thêm nguồn?

- Khi cần số đo chính xác, raw data, hệ quy chiếu time points hoặc kết quả trong supplementary.
- Khi định nghĩa/sửa scheduler, lifetime phiên bản, vertical sync hay mixed-version backward.
- Khi cần trích nguyên văn, biểu thức với ký hiệu gốc hoặc hình minh họa để xuất bản.
- Khi một test không khớp Fig.2 hoặc phát hiện mâu thuẫn công thức/đường cong.
- Khi muốn tuyên bố tương đương thuật toán hoặc tái hiện thành công kết quả tác giả.

Không cần đọc lại toàn bộ paper chỉ để nhớ danh sách model/dataset/cluster; các bảng phía trên đủ cho bước đó.

## 14. Bản đồ triển khai từ paper sang PipeDream

> Toàn bộ mục 14–15 là `[I]`: đề xuất cho project tái hiện, không phải pseudocode chính thức của tác giả. Chưa audit repo clone của người dùng, nên không bịa tên hàm hoặc số dòng.

### 14.1. Các thành phần cần tìm/sửa

| Thành phần | Cần xác định trong repo | Yêu cầu để tái hiện |
|---|---|---|
| Model partition | Graph/profiler/partition optimizer/generated stage modules | Giữ model semantics; lưu layer split cho từng W |
| Data feeder | Loader, input routing, labels | Mini-batch ID, micro-batch ID, kích thước và thứ tự nhất quán |
| Scheduler | Warmup, steady-state, flush/drain | nF1B đúng cấp F micro/B mini; nhiều mini-batch đang hoạt động |
| Activation cache | Key, tensor, autograd graph, release | Phân biệt activation từ version nào và release khi nào |
| Weight manager | Snapshot, load, stash, reference count | Loại horizontal policy nhưng giữ vertical theo đặc tả đã chọn |
| Backward engine | VJP/autograd, upstream gradient, aggregation | Xác định semantics khi F/B khác version, không chỉ đổi tensor |
| Optimizer | SGD/Adam state, zero_grad, step | Update một lần đúng group; version và state không lẫn |
| Communication | Tensor shapes/dtypes, send/recv order | Truyền đủ activation và activation gradient; không deadlock |
| Checkpoint | Stage state và global progress | Safe checkpoint sau drain; resume nhất quán |
| Metrics | Train/eval, timing, memory, throughput | Đơn vị và split rõ; đo đủ per-stage và end-to-end |

Repo upstream có các thư mục `graph/`, `profiler/`, `optimizer/`, `runtime/`, `scripts/`; tên thư mục `optimizer/` không mặc định là nơi chứa SGD/Adam training optimizer. Agent phải tìm implementation thật trong branch/commit đang làm.

### 14.2. Event contract gợi ý

Lưu event bằng JSONL hoặc bảng với các trường:

```text
run_id, rank, stage_id, event_seq, event_kind,
mini_batch_id, micro_batch_id_or_group,
forward_version, backward_version, optimizer_step,
num_samples_or_valid_tokens, tensor_shape, dtype,
bytes_sent, bytes_received, local_timestamp
```

Timestamp giữa các máy không mặc định đồng bộ; dùng event dependencies để kiểm tra thứ tự. Phase trace cần ít nhất `forward`, `backward`, `update`, `send`, `recv`, `release`. Cho phép tắt trace chi tiết khi đo throughput.

### 14.3. Pseudocode khái niệm, KHÔNG phải code sẵn chạy

```text
initialize stage parameters, optimizer, queues, caches, version ledger

while work remains:
    choose a dependency-ready event according to the audited nF1B schedule

    if event is FORWARD(i, j):
        select forward version using the declared vertical version policy
        execute stage forward for micro-batch (i, j)
        cache inputs/activations + version metadata needed by backward
        send activation to next stage, or store loss contribution at last stage

    if event is BACKWARD_GROUP(i):
        assert all required forward contributions / downstream gradients ready
        select backward version using the declared vertical version policy
        compute group VJP using the explicitly validated mixed-version rule
        send complete input gradients to previous stage
        apply optimizer update once using the declared parameter/state rule
        release only caches and versions no longer referenced

drain all work before epoch-level checkpoint
```

`choose event`, `select version` và `mixed-version rule` là các điểm phải đặc tả/test; không được để stub rồi báo implementation đầy đủ. Không tự dùng một barrier toàn pipeline sau mỗi mini-batch, vì nó sẽ thay đổi inter-batch overlap.

### 14.4. Ba cách backward khác nhau cần phân biệt

| Cách tính | Thực tế | Có tự động đúng TiMePReSt không? |
|---|---|---|
| Giữ graph/snapshot forward cũ | Gradient khớp forward cũ | Không; chưa thực hiện tuyên bố backward dùng version mới |
| Dùng activation/upstream gradient cũ nhưng thay weight trong local VJP | Mixed-version gradient | Cần đặc tả và kiểm chứng; không phải tự động gradient exact của loss ở trọng số mới |
| Recompute toàn bộ forward/loss bằng trọng số mới rồi backward | Gradient exact ở trọng số mới nếu recompute nhất quán | Có thể thay đổi thuật toán, lịch, chi phí và RNG so với paper; phải ghi là lựa chọn tái hiện |

**Ví dụ kiểm tra toán học `[D]`:** model hai layer scalar `a=w1*x`, `y=w2*a`, loss `(y-target)^2`. Nếu forward dùng `w_old`, residual cũ là `r=2*(y_old-target)`. Giữ a_old nhưng local backward dùng w2_new cho stage trước cho:

```text
g_w2 = r * a_old
g_w1 = r * w2_new * x
```

Các giá trị này nói chung không bằng gradient của loss được recompute tại `(w1_new,w2_new)`; residual và activation vẫn cũ. Đây là lý do cần reference độc lập cho policy cụ thể, thay vì kiểm tra chỉ bằng parameter version counter.

### 14.5. Cấu hình ghi nhận tối thiểu

Đây là schema minh họa `[I]`, không phải tên key trong repo upstream. `null` phải được resolve hoặc đánh dấu không áp dụng trước khi chạy benchmark đầy đủ.

```yaml
paper:
  doi: 10.1016/j.future.2025.108260
  version: journal_2026
provenance:
  upstream_commit: null
  implementation_commit: null
  reproduction_level: prototype
workload:
  model: null
  model_variant_or_checkpoint: null
  dataset: null
  dataset_revision_and_split: null
  preprocessing: null
training:
  global_batch_size: null
  num_microbatches: 3  # Paper main setting; not enough to define batch size.
  optimizer: null
  learning_rate: null
  lr_schedule: null
  precision: null
  epochs: null
  seed: null
pipeline:
  num_stages: null
  partitions: null
  schedule: nF1B
  horizontal_stashing: false
  vertical_version_policy: null
  mixed_version_backward_rule: null
evaluation:
  split: null
  metric_protocol: null
  timing_protocol: null
  memory_protocol: null
```

Một config tự chọn phải ghi `implementation-choice` và rationale trong tài liệu run. Không dùng từ `paper_defaults` cho các key UNKNOWN.

## 15. Kiểm chứng và điều kiện hoàn thành

### 15.1. Ma trận kiểm chứng

| Kiểm tra | Cách thực hiện [I] | Chứng minh được / không chứng minh được |
|---|---|---|
| Model split equivalence | Mạng nhỏ, fixed weights; so output của split và full model | Bảo toàn phép forward; chưa kiểm chứng pipeline concurrency |
| Loss aggregation | So mean loss full batch với micro-batch, kể cả phần cuối không đều | Đúng scaling; với BN/dropout phải kiểm soát semantics |
| Fixed-version gradients | So từng parameter gradient và input VJP với reference độc lập | Đúng trường hợp không mismatch; chưa chứng minh mixed-version |
| Mixed-version gradients | Hai layer/tiny network; cố tình update giữa F/B; reference theo công thức đã chọn | Kiểm chứng policy cụ thể, không tự chứng minh policy là của tác giả |
| Schedule | Trace các cặp W,N trong mục 6, so Fig.2 | Thứ tự/dependency/no missing work; slot không chứng minh speedup |
| Version propagation | Trace version F/B/update theo stage, gồm micro-batch cùng nhóm khác version | Có/không giữ vertical consistency theo đặc tả |
| Update count | Đếm mẫu/token, group và optimizer steps | Không bị step N lần ngoài ý muốn hoặc bỏ sample |
| Cache lifecycle | Đếm live cache/version trước và sau drain | Không leak; chưa thay thế đo peak memory |
| Multi-GPU communication | Synthetic batch nhỏ, timeout; so shape/dtype/order/bytes | Không deadlock/sai routing trong test; cần smoke train tiếp |
| Checkpoint resume | Chạy liên tục vs resume sau epoch/drain | Tính nhất quán checkpoint trong phạm vi được hỗ trợ |
| Baseline preservation | Chạy baseline mode trước/sau sửa trên cùng config | Không vô tình thay dữ liệu/optimizer/model |
| Performance | Warmup, CUDA sync ở biên đo, nhiều step/epoch | Wall-clock thực; không bật đồng bộ mỗi event làm mất overlap |

BatchNorm dùng batch statistics và dropout dùng RNG có thể khiến micro-batching khác full-batch ngay cả khi math loss đúng. Trước hết dùng toy model không stateful/stochastic để tách lỗi, sau đó test riêng các layer này. Không đổi BN thành eval trong benchmark chính mà không báo.

### 15.2. Lộ trình phù hợp tài nguyên project

1. **Audit repo và đóng đặc tả:** đọc branch/commit; tìm scheduler, weight manager và runtime; liệt kê UNKNOWN.
2. **CPU/toy model:** kiểm tra loss, graph, mixed-version rule và schedule.
3. **Colab một GPU:** chạy baseline accuracy và simulation stage nếu cần; ghi chính xác phần được mô phỏng. Không báo speedup distributed từ bước này.
4. **Kaggle T4×2 khi được yêu cầu:** chạy pipeline thật W=2; N=3 là main setting paper và cho v=1 theo công thức. Ghi hardware khác paper.
5. **Ablation trong phạm vi phần cứng:** thay scheduling/stashing có kiểm soát; không gọi W=2,N=2 là phép tái hiện multiple-sequence case W=4,N=2.
6. **Đánh giá đầy đủ:** nhiều seed nếu ngân sách cho phép, so time-to-quality/quality-at-time/epoch time/memory; không chỉ báo accuracy cuối.

### 15.3. Các cấp độ tuyên bố kết quả

| Cấp | Điều kiện |
|---|---|
| Smoke test | Code chạy một ít step, không lỗi rõ ràng |
| Mechanism prototype | Có scheduler/version policy và test math/trace; các giả định được ghi |
| Single-GPU simulation | Nhiều stage logical cùng GPU; không xác nhận communication speedup |
| Distributed implementation | Nhiều GPU thật, communication và update được kiểm chứng |
| Partial reproduction | Một số workload/config được tái hiện, còn khác biệt/UNKNOWN được công bố |
| Faithful reproduction | Có đủ cơ sở ánh xạ thuật toán, cấu hình, protocol và bằng chứng kết quả; không còn giả định trọng yếu chưa giải quyết |

Một run đạt Top-1 ~73% không đủ để đi thẳng từ smoke test sang faithful reproduction.

### 15.4. Câu hỏi nên gửi tác giả nếu cần đóng khoảng trống

1. Có source code/revision và supplementary đầy đủ cho bản journal không?
2. Backward sử dụng weight mới với activation cũ theo phép tính cụ thể nào? Có recomputation hay custom backward không?
3. Version consistency qua stage và optimizer state được quản lý ra sao?
4. Có configs (model variant, preprocessing, batch size, optimizer/LR/scheduler, split/seed) cho từng workload không?
5. COCO được chuyển thành task classification như thế nào? TinyLlama/UltraChat/BLEU protocol là gì?
6. Layer partitions, hardware/network và baseline configs/commits là gì?
7. Có raw epoch time, throughput, stage memory và định nghĩa time points cho từng hình không?

File chỉ liệt kê câu hỏi; không tự gửi email khi người dùng chưa yêu cầu.

## 16. Bản đồ trang, hình và phương trình

### 16.1. Main paper

| Vị trí | Nội dung cần mở lại |
|---|---|
| p.1 | Metadata, abstract, email tác giả |
| pp.1–3, §1–2 | Động cơ, related work; nhiều phát biểu so sánh cần đọc phê bình |
| p.3, §3–3.1 | Vertical sync, kiến trúc stage, định nghĩa nF1B |
| p.4, Fig.1 | Model layer partition và hướng F/B |
| p.4, Fig.2a–e | Lịch W,N; đọc chuỗi minibatch và warmup |
| pp.4–5, Eq.1 | Weight update và version difference |
| p.5, Fig.3 | Minh họa computation–communication overlap |
| pp.5–6, §3.2 | Backward priority, nhịp nF1B |
| p.6, §3.3 | Checkpoint cuối epoch |
| pp.6–7, §3.4–3.6 | Multiple sequences; Eq.2–16 |
| pp.7–8, §3.7 | So sánh GPipe và các tuyên bố về memory/synchronization |
| p.8, §3.8, Eq.17–21 | Communication model |
| p.8, Table 1, §4.1 | Cluster, N=3 và thiết lập tổng quát |
| p.9, Fig.4 | VGG-16/CIFAR-100/A; top-1, top-5, loss vs time/epoch |
| p.10, Fig.5 | ResNet-50/Tiny-ImageNet/A; 6 panels |
| p.10, Fig.6 | AlexNet/COCO/A; 6 panels |
| p.10, Fig.7 | TinyLlama/UltraChat/A; BLEU và loss, 4 panels |
| p.11, Fig.8 | VGG-16/Tiny-ImageNet/B; TiMePReSt vs PipeDream |
| p.11, Fig.9 | VGG-16/Tiny-ImageNet/C; 5 systems |
| p.11, Fig.10 | TinyLlama/UltraChat/C; 5 systems |
| p.12, Fig.11 | N sensitivity + larger batch, VGG-16/Tiny-ImageNet/C |
| p.12, Fig.12 | N sensitivity + larger batch, TinyLlama/UltraChat/C |
| p.12, Fig.13 | Stashing/schedule ablation, VGG-16/Tiny-ImageNet/C |
| p.12, Fig.14 | Stashing/schedule ablation, TinyLlama/UltraChat/C |
| p.13, Fig.15 | Stacked stage memory, C, đủ 10 workload và các biến thể |
| p.13, Fig.16 | B: minutes/epoch và epochs/hour, đủ 10 workload |
| p.13, Fig.17 | C: minutes/epoch và epochs/hour, main systems + biến thể |
| p.14, §4.10 | Định nghĩa Variant 1/2 và kết luận trade-off |
| p.14, §5 | Kết luận, giới hạn partitioning/scaling |
| p.14 | Data availability và supplementary statement |
| pp.14–15 | References [1]–[30] |

Với phần lớn hình CNN có 6 panels: a/b = top-1 time/epoch; c/d = top-5 time/epoch; e/f = loss time/epoch. Hình LLM có 4 panels: a/b = BLEU time/epoch; c/d = loss time/epoch. Luôn xác nhận legend tại panel vì màu không thống nhất hoàn toàn.

### 16.2. Supplementary được main paper dẫn tới — CHƯA ĐỌC

| Mục/hình | Nội dung theo dẫn chiếu của main paper |
|---|---|
| Section S1 | Data parallelism |
| S2a / S2b | Tensor parallelism / pipeline parallelism minh họa |
| S3 | PipeDream weight versions/weight stashing |
| S4 | VGG-16/Tiny-ImageNet/A |
| S5 | VGG-16/COCO/A |
| S6 | ResNet-50/CIFAR-100/A |
| S7 | ResNet-50/COCO/A |
| S8 | AlexNet/CIFAR-100/A |
| S9 | AlexNet/Tiny-ImageNet/A |
| S10 | Hardware efficiency và throughput, Cluster A |
| S11 / S12 | Memory footprint; main paper dùng cùng Fig.15 để bàn các cluster, chưa xác nhận từng hình thuộc cluster nào |

Không tạo số liệu cho S4–S12 từ việc main text nói “similar trend”.

## 17. References và source code để agent tra cứu

### 17.1. Tài liệu nền tảng

- **Paper mục tiêu:** Dutta, A., Chaki, N., & De, R. K. (2026). TiMePReSt: Time and memory efficient pipeline parallel DNN training with removed staleness. *Future Generation Computer Systems, 178*, 108260. https://doi.org/10.1016/j.future.2025.108260
- **[16] PipeDream:** Narayanan, D., Harlap, A., Phanishayee, A., Seshadri, V., Devanur, N. R., Ganger, G. R., Gibbons, P. B., & Zaharia, M. (2019). PipeDream: Generalized pipeline parallelism for DNN training. *SOSP 2019*, 1–15. https://doi.org/10.1145/3341301.3359646
- **[17] PipeDream-2BW:** Narayanan, D., Phanishayee, A., Shi, K., Chen, X., & Zaharia, M. (2021). Memory-efficient pipeline-parallel DNN training. *ICML*, 7937–7947. https://arxiv.org/abs/2006.09503
- **[11] GPipe:** Huang, Y., et al. (2019). GPipe: Efficient training of giant neural networks using pipeline parallelism. *NeurIPS 32*. https://arxiv.org/abs/1811.06965
- **[12] GPipe evaluation:** Zhang, P., Lee, B., & Qiao, Y. (2023). Experimental evaluation of the performance of GPipe parallelism. *FGCS, 147*, 107–118. https://doi.org/10.1016/j.future.2023.04.033
- **[18] Zero Bubble:** Qi, P., Wan, X., Huang, G., & Lin, M. (2024). Zero bubble (almost) pipeline parallelism. *ICLR 2024*.
- **[20] DualPipe-related citation:** Liu, A., et al. (2024). DeepSeek-V3 technical report. https://arxiv.org/abs/2412.19437
- **[26] XPipe:** Guan, L., Yin, W., Li, D., & Lu, X. (2019). XPipe: Efficient pipeline model parallelism for multi-GPU DNN training. https://arxiv.org/abs/1911.04610

### 17.2. DeepSpeed references được dùng trong PDF

- [13] Aminabadi et al., *DeepSpeed-inference: Enabling efficient inference of transformer models at unprecedented scale*, SC22, 2022.
- [14] Rajbhandari et al., *ZeRO-Infinity: Breaking the GPU memory wall for extreme scale deep learning*, SC, 2021.
- [15] Ren et al., *ZeRO-Offload: Democratizing billion-scale model training*, USENIX ATC, 2021.

[?] [13] là paper inference; không thể từ citation đó suy ra exact training setup của baseline DeepSpeed ở TiMePReSt. Cần config/implementation thật.

### 17.3. Code đã xác nhận và giới hạn

| Phương pháp | Link | Trạng thái |
|---|---|---|
| PipeDream | https://github.com/msr-fiddle/pipedream | Official repository; nhánh `pipedream`, MIT license |
| PipeDream-2BW | https://github.com/msr-fiddle/pipedream/tree/pipedream_2bw | README official repo xác nhận tên branch |
| GPipe | https://github.com/tensorflow/lingvo/blob/master/lingvo/core/gpipe.py | Code trong Lingvo, được Google Research giới thiệu |
| GPipe PyTorch port | https://github.com/kakaobrain/torchgpipe | Implementation của Kakao Brain, không phải code gốc Google |
| TiMePReSt | Chưa xác minh được repository chính thức | Không thay bằng repo bên thứ ba mà không ghi provenance |

Các nguồn web để kiểm tra lại:

- https://github.com/msr-fiddle/pipedream (README, branch mapping, kiến trúc thư mục).
- https://research.google/blog/introducing-gpipe-an-open-source-library-for-efficiently-training-large-scale-neural-network-models/ (GPipe micro-batch và code).
- https://arxiv.org/abs/2410.14312 (preprint TiMePReSt; không phải bản cấu hình authoritative của file này).
- https://www.sciencedirect.com/science/article/pii/S0167739X25005540 (trang publisher; chưa truy cập được supplementary trong lần soạn).

Main PDF p.14 nói dữ liệu sẽ được cung cấp khi yêu cầu. Đây không phải lời xác nhận source code chắc chắn được cung cấp khi liên hệ. Email công khai nằm ở mục 1; chỉ liên hệ khi được người dùng yêu cầu.

## 18. Tóm tắt thực thi dành cho agent

1. Target là **journal TiMePReSt**, không trộn bản preprint hoặc I-/V-TiMePReSt.
2. Main method = **nF1B + bỏ horizontal stashing + giữ vertical consistency theo mô tả**, không chỉ gradient accumulation.
3. Main setting N=3; W=2/3/4; có 10 workload như mục 8. Không có đủ batch/LR/optimizer config trong main PDF.
4. Đặc tả và test **mixed-version backward** trước; đây là chỗ có thể chạy không lỗi mà vẫn sai method.
5. Preserve baseline PipeDream; mỗi giả định phải có provenance và test riêng.
6. Đánh giá time, quality, memory; không hứa TiMePReSt thắng theo mọi metric.
7. DeepSpeed có memory tốt hơn chút; ablation giữ stashing có quality tốt hơn nhưng memory cao hơn.
8. Số hình ở mục 11 là ước lượng, time points không phải giờ; không tạo exact targets từ chúng.
9. Một GPU kiểm tra prototype/simulation, hai GPU kiểm tra distributed runtime; không đồng nhất hardware với paper.
10. Khi còn UNKNOWN trọng yếu, gọi kết quả là **partial reproduction với giả định công khai**. Cần mở PDF/supplementary hoặc hỏi tác giả để nâng mức xác nhận.

## 19. Sai khác so với paper và lựa chọn triển khai (GĐ1 — thí nghiệm 1: VGG-16 / CIFAR-100)

> Mục này ghi các quyết định của project tái hiện `[I]`, không phải mô tả của tác giả. Code: `timeprest/`, config: `configs/base_cifar100.yaml`.

### 19.1. Mục tiêu so sánh

Đối chiếu Fig.4 (p.9): VGG-16 / CIFAR-100 / Cluster A (W=2), N=3 (§4.1). Giai đoạn 1 chạy **mô phỏng trên 1 GPU**, nên:
- Đường cong theo **epoch** (Fig.4b/d/f) so trực tiếp được: ngữ nghĩa phiên bản trọng số và thứ tự update giống pipeline thật.
- Đường cong theo **thời gian** (Fig.4a/c/e) chỉ là **ước lượng**: đo thời gian từng op trên 1 GPU, rồi mô phỏng sự kiện cho 2 GPU (`timeprest/timing.py`). Số thật chờ GĐ2.

### 19.2. Bảng sai khác / lựa chọn

| Hạng mục | Paper | Project (GĐ1) |
|---|---|---|
| Phần cứng | Quadro RTX 6000 + RTX 2080, 2 máy | 1 GPU Colab; 2 stage chạy tuần tự trên cùng GPU |
| Model | VGG-16, biến thể không nêu | VGG-16-BN kiểu CIFAR (13 conv 3×3 + BN + ReLU, head `Linear(512,100)`, không dropout), Kaiming init |
| Dữ liệu | CIFAR-100, split/augment không nêu | Train 50k / test 10k chuẩn; random crop 32 (pad 4) + flip; normalize. Accuracy báo trên **test set** (paper không rõ train hay test) + train acc chạy dọc |
| Mini-batch M | Không có số | M = 192 cho cả hai hệ; TiMePReSt: N=3 micro-batch × 64; PipeDream: 1F1B với nguyên M |
| Optimizer | Eq.1 chỉ là dạng update | SGD momentum 0.9, lr 0.1, cosine theo step, wd 5e-4, 160 epoch, fp32, seed 0 |
| Phân hoạch | "Cân bằng bộ nhớ" (p.3), không có số | Cân bằng MACs: blocks `[0, 8, 19]`; stage 1 ≈ 1.15M tham số, stage 2 ≈ 13.6M |
| Lịch | Fig.2 + mô tả §3.2 | Slot lý tưởng, backward ưu tiên, stage s giữ ≤ W−s mini-batch đang chạy (NOAM của PipeDream). **Khớp đúng Fig.2a–e** (test `test_matches_paper_fig2`); giới hạn NOAM không bao giờ thay đổi lịch nF1B |
| Phiên bản forward | Vertical sync (p.3), giữ bản cũ tới khi forward dùng nó xong (p.5) | Micro-batch lấy version mới nhất của stage 0 lúc vào pipeline; mọi stage sau dùng đúng version đó |
| Phiên bản backward, TiMePReSt | "latest updated version" (p.4) + vertical sync | `committed` = version đã áp dụng trên **mọi** stage (= version của stage 0) lúc backward bắt đầu ở stage cuối. Khi W ≤ N+1 version này trùng version live ở từng stage (đã kiểm chứng) |
| Phiên bản backward, PipeDream | Horizontal + vertical stashing (p.2) | `stashed` = version của forward + vertical sync (theo CLAUDE.md). PipeDream gốc mặc định **không** bật vertical sync; bật nó làm stage 2 giữ 2 version thay vì 1 |
| Cách tính gradient khi F/B khác version | Không mô tả | Mỗi stage chỉ lưu **input** của stage; khi backward thì chạy lại forward cục bộ bằng version được chọn rồi VJP với gradient từ stage sau. Áp dụng cho **cả hai hệ**, nên khác biệt duy nhất là version. Hệ quả: activation memory là input của stage (giống nhau giữa hai hệ); op B tốn thêm 1 forward |
| "Một backward" | Một backward trên loss trung bình | Một op B mỗi mini-batch/stage; bên trong lặp N chunk để BN dùng đúng batch-stat của từng micro-batch. Về toán học bằng backward trên tổng loss |
| Loss | Trung bình loss N micro-batch | Σ_j CE_sum_j / M (trung bình có trọng số theo số mẫu; bằng trung bình thường khi chia đều) |
| BatchNorm | Không nêu | Batch-stat theo micro-batch (64) cho TiMePReSt, theo 192 cho PipeDream; running stats cập nhật 1 lần/forward, đóng băng khi recompute |
| Biên epoch | Checkpoint cuối epoch (§3.3) | Pipeline drain cuối mỗi epoch; `drop_last`; checkpoint gồm model/optimizer/scheduler/RNG, resume khớp bit-exact (check 7) |
| Bộ nhớ | GB theo stage (Fig.15), phương pháp đo không nêu | (a) `torch.cuda.max_memory_allocated` của cả tiến trình; (b) bộ nhớ "sổ sách" theo stage = tham số × số version đang giữ + activation/gradient đang giữ (chưa tính momentum) |
| Thời gian | Phút/epoch, time points | (a) wall-clock 1 GPU; (b) ước lượng 2 GPU: mô phỏng sự kiện với thời gian op đo được, truyền thông 10 GB/s + 50 µs, không tranh chấp link. Trục "time points" trong plot = thời gian ước lượng tích lũy / thời gian epoch trung bình của TiMePReSt |

### 19.3. Phát hiện khi kiểm chứng công thức (`[D]`, simulator của project)

- Eq.(2) (v = 1 ⇔ W ≤ N+1) đúng trên toàn lưới W = 2..8, N = 2..7.
- Eq.(3) v = ⌊(W+N−2)/N⌋ khớp mọi cặp W ≤ 5 (bao gồm mọi hình của paper). Với lịch của project, nó **khác** ở (W,N) = (6,2): mô phỏng v=2, công thức 3; (8,2): 3 vs 4; (8,3): 2 vs 3. Các cặp này nằm ngoài thí nghiệm của paper, và phép suy diễn Eq.(11)–(16) có dùng xấp xỉ `x ~ 1/N`. Không kết luận paper sai: đây là quan sát trên lịch tổng quát hóa từ Fig.2.
- Số version trọng số cần giữ (trace): PipeDream 1F1B, W=2: [2, 2] khi có vertical sync, [2, 1] khi không (đúng W−s của PipeDream). TiMePReSt W=2, N=3: [1, 2]. Stage 2 vẫn cần tạm giữ version cũ vì vertical sync của forward (micro-batch 2A/2B vào stage 2 sau khi stage 2 đã update).
