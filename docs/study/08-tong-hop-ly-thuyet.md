# 08 — Tổng hợp lý thuyết T-KGAT (từ đầu đến 29/10/2026)

Tài liệu này gom toàn bộ phần lý thuyết cần nắm cho đề tài, theo đúng thứ tự một người mới cần học, và nối từng khái niệm với chỗ nó nằm trong code và trong dữ liệu của nhóm. Bảy ghi chú chủ đề (`docs/theory/theory-01` … `theory-07`) và bảy hướng dẫn học (`docs/study/01` … `07`) đi sâu hơn từng phần; tài liệu này là bản đồ chung và là bản ôn tập.

Số liệu trong tài liệu được lấy từ `results/registry.csv` (qua `tkgat.registry.load()` và `scripts/10_rq3_table.py`), `results/significance.csv`, `docs/data-report-numbers.md` và mục "Known facts" của `CLAUDE.md`. Kết quả KGAT và T-KGAT là **sơ bộ** (chạy trước khi cô duyệt quy trình).

---

## 0. Bức tranh tổng thể trong một trang

**Đề tài:** T-KGAT — Temporal Knowledge Graph Attention Network for Recommendation. Mô hình gốc là KGAT (Wang và cộng sự, KDD 2019). Nhóm thêm thời gian vào attention.

**Câu chuyện của đề tài theo một dòng:** hệ gợi ý dùng lịch sử tương tác (lọc cộng tác) và tri thức về item (Knowledge Graph); KGAT lan truyền thông tin trên đồ thị gộp hai nguồn đó bằng attention; nhưng KGAT coi mọi tương tác và mọi fact như nhau theo thời gian, và cách đánh giá chia ngẫu nhiên của paper cho phép tương lai lọt vào quá khứ. Đề tài (1) dựng lại quy trình đánh giá không rò rỉ, (2) dựng KG đúng theo thời điểm, (3) kiểm tra xem thêm Δt vào attention có giúp không và phần giúp có thật sự đến từ thời gian không.

**Ba câu hỏi nghiên cứu (`docs/plan.md`, mục 4):**

| RQ | Câu hỏi | Trả lời bằng |
|---|---|---|
| RQ1 | Dữ liệu MovieLens và KG liên kết có đặc điểm gì theo thời gian (độ phủ liên kết, tỷ lệ fact có ngày, fact "tương lai")? | EDA, báo cáo dữ liệu |
| RQ2 | Khi đã loại rò rỉ, KG còn giúp KGAT hơn lọc cộng tác thuần không? | KGAT so với KGAT không KG, các biến thể KG |
| RQ3 | Đưa thời gian vào attention có cải thiện thêm không, và cải thiện có đến từ thời gian không? | T-KGAT so với KGAT, kèm đối chứng hoán vị |

**Chuỗi kiến thức cần học (mỗi khối dựa trên khối trước):**

```
Hệ gợi ý + phản hồi ngầm + BPR (1)
        │
Đánh giá offline + rò rỉ + chia theo thời gian (2) ──► Thống kê: seed, kiểm định cặp, Holm (7)
        │
KG + KGE (TransE, TransR) + CKG (3) ◄── Tích hợp dữ liệu, liên kết thực thể, đúng theo thời điểm (6)
        │
KGAT: attention + lan truyền + hai loss (4)
        │
Thời gian: T-KGAT Stage 1, Stage 2, đối chứng hoán vị (5)
```

---

## 1. Hệ gợi ý, phản hồi ngầm và BPR

### 1.1 Bài toán
- **Gợi ý top-K:** với mỗi user, trả về K item chưa tương tác mà user có khả năng thích nhất. Đây là bài toán **xếp hạng**, không phải dự đoán số sao: chỉ cần item user thích đứng trên item không thích.
- **Phản hồi tường minh** (explicit): user cho điểm, ví dụ 1–5 sao. **Phản hồi ngầm** (implicit): chỉ biết user có tương tác (xem, click, mua). Không tương tác **không** có nghĩa là không thích (có thể chưa thấy), nên không có nhãn âm thật và không dùng được sai số bình phương (r − ŷ)².
- **Nhóm chuyển MovieLens sang phản hồi ngầm:** rating ≥ 4 là tương tác dương (`min_rating: 4` trong `configs/data/*.yaml`). Rating thấp không dùng làm dương, nhưng vẫn vào **mask** khi chấm (mục 2.4).

### 1.2 Matrix Factorization (MF)
- Mỗi user u có vector p_u, mỗi item i có vector q_i, cùng độ dài `dim`.
- Điểm: ŷ_ui = p_u · q_i. Tích vô hướng càng lớn thì item càng đứng cao trong danh sách của user.

### 1.3 BPR (Bayesian Personalized Ranking)
- Giả định: user u thích item đã tương tác i hơn item chưa tương tác j. Mỗi mẫu huấn luyện là bộ ba (u, i, j).
- x̂_uij = ŷ_ui − ŷ_uj; loss L = Σ −ln σ(x̂_uij) + λ‖Θ‖².
- Đẳng thức hay bị hỏi: −ln σ(x) = ln(1 + e^(−x)) = softplus(−x). Code dùng `F.softplus(-(x_ui - x_uj))`.
- Ví dụ số: x̂ = 0 ⇒ loss = ln 2 ≈ 0,693; x̂ = 3 ⇒ loss ≈ 0,049. Loss nhỏ khi item dương đứng xa trên item âm.
- **Lấy mẫu âm:** j được lấy ngẫu nhiên trong các item user chưa tương tác **trên train**, không bao giờ dùng val hay test. Trong repo, `NegativeSampler` lấy mẫu trên GPU bằng `torch.Generator`.
- Siêu tham số: `dim` (độ dài embedding), `lr` (bước Adam), `l2` (hệ số phạt chống overfit).
- **Overfit và dừng sớm:** loss train vẫn giảm nhưng Recall@20 trên val ngừng tăng. Repo đo val mỗi 5 epoch, patience 5 cho BPR-MF (10 cho KGAT), giữ epoch tốt nhất trên val.

### 1.4 Baseline
- **MostPop:** cùng một danh sách K item nhiều tương tác dương nhất trong train cho mọi user.
- **MostPop recent (MostPop gần đây):** chỉ đếm N ngày cuối của train; N chọn trên val trong lưới {1, 3, 5, 7, 14, 30} ngày (quyết định 0007). Cửa sổ neo ở timestamp cuối của train, không phải ở t1.
- **BPR-MF:** MF học bằng BPR; lưới (dim, lr, l2), 5 seed.
- Vì sao bắt buộc có MostPop recent: nó là baseline thời gian đơn giản nhất. Mô hình có yếu tố thời gian phải thắng nó, không chỉ thắng MostPop.

### 1.5 Code
`src/tkgat/models/baselines.py` (`MostPop`, `BPRMF`), `src/tkgat/train/bpr.py` (`NegativeSampler`, `fit_bpr`), `configs/model/bprmf.yaml`, `mostpop.yaml`, `mostpop_recent.yaml`.

---

## 2. Đánh giá offline và rò rỉ dữ liệu

### 2.1 Rò rỉ dữ liệu là gì
Rò rỉ là khi thông tin của val/test (hoặc của tương lai) ảnh hưởng tới một quyết định lúc huấn luyện hay tiền xử lý. Hậu quả: điểm test lạc quan, không còn ước lượng được hiệu năng trên dữ liệu chưa thấy.

**Ba lỗi rò rỉ của kết quả cũ (lý do làm lại, quyết định 0001):**
1. Chọn epoch tốt nhất trên test (không có tập val).
2. Lọc k-core trên toàn bộ dữ liệu rồi mới chia (quyết định giữ/bỏ user dùng cả tương tác của test).
3. KG dựng từ metadata hiện tại (tri thức tương lai).
Thêm: chỉ chạy 1 seed, CPU và GPU lệch nhau khoảng 18%, nên chênh lệch nhỏ hơn mức đó bị báo nhầm là cải thiện.

### 2.2 Chia theo mốc thời gian toàn cục (quyết định 0002)
- Hai mốc t1 < t2 là phân vị 0,8 và 0,9 của **mọi** timestamp, áp chung cho mọi user.
- train: ts < t1; val: t1 ≤ ts < t2; test: ts ≥ t2.
- Vì sao không chia theo từng user (vài tương tác cuối của mỗi user làm test): tương lai của user A vẫn lọt vào train qua user B.
- Các bước: rating ≥ 4 là dương → 5-core **chỉ trên train dương** → val/test chỉ giữ user và item đã có trong train (user/item mới bị loại và được đếm) → gộp cặp (user, item) trùng.
- Không huấn luyện lại trên train + val trước khi chấm test (quyết định 2 của `plan.md`): chọn epoch và tham số trên val, chấm test một lần.

**Số của nhóm:**

| | ML-100K | ML-1M |
|---|---|---|
| t1 / t2 | 07/03/1998 / 31/03/1998 | 02/12/2000 / 29/12/2000 |
| Train | 43.018 dương, 745 user, 932 item | — |
| User val / test | 66 / 51 | — / 952 |

### 2.3 Cái giá của chia theo thời gian: cold-start
User MovieLens chấm hàng loạt rồi rời đi: 77,9% rating của ML-100K (72,4% của ML-1M) nằm trong 24 giờ đầu của user. Phần lớn rating sau t1 đến từ user mới, bị loại vì không có trong train. ML-100K chỉ giữ 12,6% số dương ở val và 13,3% ở test, còn 66 và 51 user. ML-1M giữ 75,7% số dương ở test, 952 user. **Vì vậy kết luận dựa chủ yếu vào ML-1M.**

### 2.4 Mask
- Khi xếp hạng cho user u, loại các item u đã tương tác (gợi ý lại item đã xem không có ý nghĩa và chiếm chỗ trong top-K).
- Mask val: mọi tương tác (kể cả rating thấp) có ts < t1. Mask test: mọi tương tác có ts < t2.
- Mask không bao giờ chứa tương tác từ mốc của chính phase đó trở đi, vì đó là đáp án.

### 2.5 Chỉ số
- **Xếp hạng toàn bộ item** rồi tính chỉ số. Không dùng sampled metrics (1 dương + 100 âm) vì chúng có thể đảo thứ tự giữa các mô hình (Krichene & Rendle, 2020).
- Recall@K(u) = |top-K(u) ∩ T_u| / |T_u|.
- DCG@K(u) = Σ_{k=1..K} 1[item hạng k ∈ T_u] / log₂(k+1); NDCG@K = DCG@K / IDCG@K.
- Ví dụ tính tay: xếp hạng [4, 7, 1, 9], đáp án {7, 9}, K = 3. Top-3 có 7 ở hạng 2. Recall@3 = 0,5; DCG@3 = 1/log₂3 ≈ 0,631; IDCG@3 = 1 + 0,631 = 1,631; NDCG@3 ≈ 0,387.
- Recall nói "tìm được bao nhiêu phần đáp án"; NDCG thưởng thêm khi đáp án đứng cao.

### 2.6 `check_split`: 6 kiểm tra (thuộc lòng)
1. Thứ tự thời gian (train < t1 ≤ val < t2 ≤ test).
2. Không cặp (user, item) nào nằm ở hai phần.
3. Val/test chỉ có user và item đã biết lúc train.
4. Train thỏa 5-core khi tính riêng trên train.
5. Mask chỉ dùng quá khứ.
6. Đáp án không trùng mask của chính phase đó.

Ba nguồn rò rỉ `check_split` không bắt, và nơi chặn chúng: chọn mô hình bằng test (chỉ `scripts/run.py` chấm test, sau khi chọn trên val), thống kê dùng tương lai (`MostPop` chỉ đọc train), tri thức tương lai trong KG (`check_kg`). `scripts/02_build_splits.py` từ chối lưu split nếu có vi phạm; CI chạy toàn bộ test mỗi lần đẩy code.

### 2.7 Code
`src/tkgat/data/split.py`, `src/tkgat/data/leakage.py` (`check_split`), `src/tkgat/data/encode.py` (mask), `src/tkgat/eval/metrics.py`, `scripts/run.py`, `tests/test_leakage.py`.

---

## 3. Knowledge Graph và Knowledge Graph Embedding

### 3.1 KG
- Tập triple (h, r, t): thực thể đầu, quan hệ, thực thể đuôi. Ví dụ (Tenet, director, Christopher Nolan).
- KG giúp gợi ý ở hai phía: phim ít rating nhận thêm thông tin qua thuộc tính chung; user ít lịch sử đi qua thuộc tính để tới phim mới.
- Quan hệ 1-N: một phim nhiều diễn viên (`cast_member`).

### 3.2 KGE: TransE và TransR
- **TransE:** quan hệ là phép tịnh tiến, e_h + e_r ≈ e_t; điểm g(h, r, t) = ‖e_h + e_r − e_t‖² (nhỏ là đúng). Yếu với 1-N: mọi tail của cùng (h, r) bị kéo về một điểm.
- **TransR (KGAT dùng):** mỗi quan hệ có ma trận chiếu W_r riêng: g(h, r, t) = ‖W_r e_h + e_r − W_r e_t‖², với e_h, e_t ∈ ℝᵈ, e_r ∈ ℝᵏ, W_r ∈ ℝᵏˣᵈ. Các tail xa nhau trong không gian chung vẫn gần nhau sau khi chiếu.
- **Loss KGE:** L = Σ −ln σ(g(h, r, t′) − g(h, r, t)), với t′ là tail hỏng ngẫu nhiên. Viết g(sai) − g(đúng) vì g là khoảng cách. Ví dụ g(đúng) = 1, g(sai) = 4 ⇒ loss ≈ 0,049; bằng nhau ⇒ ln 2.

### 3.3 CKG (Collaborative Knowledge Graph)
- Gộp tương tác và KG vào một đồ thị: mỗi tương tác dương thành triple (u, Interact, i); item được căn chỉnh với entity trong KG; thêm quan hệ ngược (t, r⁻¹, h) để thông tin đi hai chiều.
- Ví dụ đường đi: user A → Interact → phim B → genre → Hành động ← genre⁻¹ ← phim C.
- **Trong repo:** cạnh tương tác chỉ từ train; KG cho train/val dựng tại t1, KG cho test dựng tại t2 và bỏ triple có đuôi chưa từng xuất hiện ở t1 (entity chưa huấn luyện không được vào điểm test).

### 3.4 Ba họ mô hình dùng KG
Embedding-based (CKE: TransR + MF, không lan truyền), path-based (PER, KPRN), propagation-based (RippleNet, KGCN, KGAT). KGAT = TransR như CKE + lan truyền nhiều bước có attention, nên CKE là baseline KG tự nhiên (giai đoạn 4, chưa làm).

### 3.5 Code
`src/tkgat/kg/ckg.py` (`build_ckg`, `load_kg`), `scripts/06_build_kg.py`.

---

## 4. KGAT

### 4.1 Ba khối
1. **Embedding** của user, item, entity, relation.
2. **Các lớp lan truyền có attention** trên CKG.
3. **Dự đoán** bằng tích vô hướng của vector đã nối qua các lớp.

### 4.2 Attention
- Hàng xóm của node h: mọi cạnh (h, r, t) trong CKG, kể cả cạnh ngược. User có hàng xóm là item đã tương tác; item có hàng xóm là user đã tương tác và thuộc tính KG.
- Điểm trước chuẩn hóa: π(h, r, t) = (W_r e_t)ᵀ tanh(W_r e_h + e_r).
- **Softmax trên toàn bộ hàng xóm của h:** π(h, r, t) ← exp π(h, r, t) / Σ_{(h, r′, t′) ∈ N_h} exp π(h, r′, t′). Thứ quyết định là π **so với** các hàng xóm khác của cùng node.
- Ví dụ số: ba cạnh π = (2, 1, 0) ⇒ trọng số ≈ (0,665; 0,245; 0,090).
- Repo tính attention theo từng nhóm quan hệ, trong `no_grad`; A khởi tạo đều 1/|N_h|; attention được tính lại sau mỗi epoch.

### 4.3 Lan truyền và aggregator Bi-Interaction
- Hàng xóm gộp: e_{N_h} = Σ π(h, r, t) e_t.
- Bi-Interaction: f = LeakyReLU(W₁(e_h + e_{N_h})) + LeakyReLU(W₂(e_h ⊙ e_{N_h})). Nhánh nhân từng phần tử cho mô hình thấy mức giống nhau giữa node và hàng xóm.
- Sau L lớp, node nhận thông tin từ các node cách tối đa L bước. L = 3: user → phim đã xem → đạo diễn → phim khác của đạo diễn.
- Vector cuối là **phép nối** mọi lớp: e*_u = e_u⁽⁰⁾ ‖ … ‖ e_u⁽ᴸ⁾; ŷ(u, i) = e*_uᵀ e*_i. Nối để giữ đặc trưng riêng (lớp 0) lẫn tín hiệu lân cận.

### 4.4 Hai loss và huấn luyện luân phiên
- L = L_KG (TransR) + L_CF (BPR trên ŷ) + λ‖Θ‖².
- Mỗi epoch: các bước CF với Adam riêng, các bước KG với Adam riêng, rồi cập nhật attention.

### 4.5 Cấu hình của nhóm
`configs/model/kgat.yaml`: dim 64, kge_dim 64, ba lớp [64, 32, 16], mess_dropout 0,1, batch CF 1.024, batch KG 2.048, tối đa 300 epoch, đánh giá mỗi 5 epoch, patience 10. Lưới 4 điểm lr ∈ {1e-3, 1e-4} × l2 ∈ {1e-5, 1e-4}, 5 seed (2019–2023), chọn theo trung bình Recall@20 val. `kgat_nokg`: cùng lưới và thiết lập, không có triple KG.

### 4.6 Khác paper và lý do
- Viết lại bằng PyTorch; code TensorFlow 1.x của tác giả chỉ để tham chiếu.
- Chia theo thời gian (paper chia ngẫu nhiên), 5-core (paper 10-core; 10-core chạy kèm để phân tích độ nhạy).
- Số của nhóm không so trực tiếp với số trong paper.
- **Không tái lập từng bit trên GPU:** `torch.sparse.mm` (cuSPARSE) không tất định; chênh lệch 1e-7 ở epoch 1 bị khuếch đại. Hai lần chạy cùng seed có thể lệch 0,02–0,03 Recall@20 val trên ml-1m-u10. Coi đó là nhiễu seed. BPR-MF thì tái lập được.

### 4.7 Code
`src/tkgat/models/kgat.py`, `src/tkgat/train/kgat.py`, `src/tkgat/kg/ckg.py`, `docs/design-kgat.md`.

---

## 5. Thời gian trong gợi ý và T-KGAT

### 5.1 Vì sao thời gian quan trọng
Ba cách: sở thích trôi (user đổi gu), độ phổ biến đổi (*Titanic* hot sau Oscar), tri thức đổi (phần tiếp theo chưa ra, giải chưa trao). Mô hình tĩnh coi mọi tương tác như nhau.

Bốn họ phương pháp: tham số thay đổi theo thời gian (timeSVD++), trọng số suy giảm theo độ cũ, mô hình chuỗi (SASRec, TiSASRec đưa khoảng thời gian vào attention), KG có thời gian hiệu lực. T-KGAT nằm giữa "suy giảm" và "attention có khoảng thời gian".

### 5.2 T-KGAT Stage 1: Δt trên cạnh user–item (mô hình chính; quyết định 0005, 0009)
- π_time(h, r, t) = π(h, r, t) − γ · δ_r · Δt_norm, **trước** softmax.
- δ_r = 1 cho Interact và Interact⁻¹, 0 cho mọi quan hệ KG.
- Δt_norm = max(0, t_ref − ts) / dt_den, với t_ref = timestamp lớn nhất của train (dùng chung cho val và test), dt_den = t_ref − timestamp nhỏ nhất của train. Không dùng timestamp nào của val/test.
- **Tương đương suy giảm mũ:** softmax(π − γΔt) = trọng số cũ × e^(−γΔt), chuẩn hóa lại. Cạnh cũ bị giảm **so với các hàng xóm khác của cùng node**.
- Ví dụ số: hai cạnh cùng π, Δt_norm = 0 và 1, γ = 3 ⇒ trọng số ≈ (0,953; 0,047).
- **γ = 0 cho đúng KGAT** ⇒ so sánh lồng nhau. γ là siêu tham số chọn trên val (lưới {1, 3, 10}), không học bằng gradient. (lr, l2) lấy từ điểm KGAT đã chọn trên val (giao thức hai tầng).
- Nếu mọi cạnh của một node có cùng Δt thì tác dụng triệt tiêu ở node đó.

### 5.3 Đặc điểm dữ liệu quyết định tác dụng của Stage 1
- Timestamp của MovieLens được nhóm **giả định** là lúc nhập rating (README GroupLens chỉ ghi "số giây từ epoch"; quyết định 0008).
- 53,0% user ML-100K có toàn bộ cạnh train trong chưa đầy một giờ; độ lệch chuẩn trung vị theo user chỉ 14 phút ⇒ phía user, Δt gần như bằng nhau ⇒ suy giảm gần như không tác dụng.
- Phía item: độ lệch chuẩn trung vị theo item là 45 ngày (ML-100K), 67 ngày (ML-1M) ⇒ tác dụng (nếu có) đến từ phía item, qua cạnh Interact⁻¹.

### 5.4 T-KGAT Stage 2: Δt trên triple của KG (biến thể so sánh; 0009 mục 7–10)
- Mỗi cạnh KG mang ngày từ đó fact đúng (`kg_t*_dates.tsv`, script 06). δ_r = 1 cho quan hệ KG, 0 cho Interact (so sánh với Stage 1, không cộng dồn).
- Fact không ngày: Δt = 0. Δt âm cắt về 0. **Δt_norm của cạnh KG bị chặn ở 1** vì ngày của fact thường cách t_ref hàng chục năm.
- Hệ quả đo được tối 29/10 (`count_dt.py` trên KG default t1): fact có ngày với Δt_norm < 1 chỉ chiếm **0,7%** (ml-100k) và **0,0%** (ml-1m-u10); 82,6% / 83,3% bị chặn ở 1; 16,7% không có ngày. **Vậy Stage 2 theo thiết kế hiện tại chỉ phân biệt fact có ngày với fact không ngày**, không đo được "fact cũ hay mới". Kết quả Stage 2 phải được đọc là hiệu ứng giảm trọng số fact có ngày, không phải tác dụng của thời gian trên KG.

### 5.5 Đối chứng hoán vị (bắt buộc)
- Stage 1: xáo `train_ts` trên **toàn bộ** cạnh train (không phải trong từng user), mỗi seed một hoán vị, giữ nguyên tập giá trị timestamp.
- Stage 2: xáo ngày của fact trên các triple của KG t1 (kể cả NaN).
- Logic: nếu bản hoán vị tốt bằng bản thật thì phần cải thiện (nếu có) không đến từ thông tin thời gian.
- Kết quả âm là kết quả hợp lệ nếu quy trình đúng (biên bản 07/08).

### 5.6 Code
`src/tkgat/models/kgat.py` (`time_on`, `delta_r`, phần phạt thời gian), `src/tkgat/kg/ckg.py` (`build_ckg`, `permute_train_ts`, `permute_kg_ts`, `load_kg_dates`), `src/tkgat/kg/temporal.py` (`fact_valid_from`, `triple_dates`), configs `tkgat.yaml`, `tkgat_perm.yaml`, `tkgat_sens.yaml`, `tkgat_kg.yaml`, `tkgat_kg_perm.yaml`.

---

## 6. Tích hợp dữ liệu, liên kết thực thể và tính đúng theo thời điểm

Đây là phần gắn đề tài với ngành Hệ thống thông tin: ghép nhiều nguồn, đo chất lượng, bảo đảm dữ liệu "đúng tại thời điểm" như trong data warehouse.

### 6.1 Chuỗi tích hợp
MovieLens (id, tên, năm) → KB4Rec (MovieLens → Freebase) → Wikidata qua P646 (Freebase ID) → fact có ngày. ML-100K không có trong KB4Rec nên đi vòng: ML-100K → ML-1M theo tên + năm → ML-1M dùng chung id với ML-20M → Freebase (quyết định 0003). Mọi khớp không chính xác được kiểm lại.

### 6.2 Làm sạch trước khi nối
18 tiêu đề trùng gộp về một id; item 267 "unknown" bị loại; cặp (user, item) trùng giữ lần đầu; checksum file thô.

### 6.3 Độ phủ khác độ chính xác
- Độ phủ: 97,2% phim (99,5% rating) của ML-100K có liên kết Freebase; ML-1M 98,2% (99,7%).
- Độ chính xác: kiểm mẫu 50 phim ngẫu nhiên (seed 2019): 0/50 sai; khoảng Wilson 95% của tỷ lệ sai là [0,0%; 7,1%]. 40 phim được máy xác nhận bằng hai nguồn độc lập (năm Wikidata và tên YAGO), 10 phim kiểm tay. **Không nói "đúng 100%".**
- Lỗi đã biết: *Clubland* → Cloudland Canyon State Park.
- Công thức Wilson: p = k/n; khoảng = (p + z²/2n ± z·√(p(1−p)/n + z²/4n²)) / (1 + z²/n), z = 1,96. Với k = 0 cận dưới bằng 0; n = 200 thì cận trên còn khoảng 1,9%.

### 6.4 Ngày trong Wikidata (quyết định 0006)
- Statement = (item, property, value) cùng qualifier. Ngày ở qualifier (P585 thời điểm, P580 bắt đầu) hoặc ở entity (P577 công chiếu, P571 thành lập, P569 sinh).
- `timePrecision` 9 = năm, 10 = tháng, 11 = ngày. "1998" độ chính xác 9 lưu là `1998-01-01` nhưng không có nghĩa là 1/1.
- **Quy tắc bảo thủ:** chỉ có năm ⇒ 31/12; chỉ có tháng ⇒ ngày cuối tháng. Chỉ dùng fact khi chắc chắn nó đã xảy ra trước mốc.

### 6.5 KG đúng theo thời điểm (quyết định 0004)
- Fact có ngày: giữ nếu ngày < mốc. Fact trỏ tới thực thể ra đời sau mốc: bỏ. Fact không ngày: giữ nếu hai đầu đã tồn tại (biến thể `default`); biến thể `drop_undated` bỏ hết fact không ngày; `static` là KG không lọc (chỉ làm tham chiếu đo rò rỉ qua KG).
- KG train/val dựng tại t1, test tại t2. `check_kg` từ chối lưu KG có fact sau mốc.
- Ví dụ: 11 giải Oscar của *Titanic* (23/03/1998) nằm giữa t1 và t2 của ML-100K ⇒ chỉ có trong KG test; *Titanic* mất 28 fact ở t1. *Toy Story* `followed_by` *Toy Story 2* (24/11/1999) sau toàn bộ ML-100K.
- Bộ lọc chỉ bỏ 1,3% fact (ML-100K; 1,5% ở ML-1M) nhưng chúng rơi vào phim phổ biến: 10% item nhiều rating nhất giữ 15,3% số fact nhưng 44,3% số fact giải thưởng/đề cử tương lai bị bỏ tại t1 (ML-100K).
- **Giới hạn phải nói:** Wikidata là bản chụp 2026. Ta lọc được *ngày sự kiện*, không lọc được *ngày fact được nhập vào KG*.

### 6.6 Hai nguồn KG hiện tại
- Wikidata (bản chụp 17/10/2026): 119.748 fact, 18 quan hệ, 3.725 trong 3.811 phim ML-1M đã liên kết. Ngày statement gần như chỉ có ở `award_received` (80%) và `nominated_for` (97%).
- Metadata MovieLens (quyết định 0011, từ PR #34): `movielens.genre`, `movielens.release_year`, `movielens.release_decade`, gán ngày là ngày công chiếu (ML-1M chỉ có năm ⇒ 31/12), đi qua cùng bộ lọc.
- Kích thước KG default tại t1: ml-100k 40.210 triple (3.650 là metadata MovieLens), ml-1m-u10 71.553 (6.209).

### 6.7 Code
`src/tkgat/data/clean.py`, `src/tkgat/kg/linking.py`, `scripts/03_link_entities.py`, `scripts/04_fetch_wikidata.py`, `src/tkgat/kg/temporal.py`, `scripts/06_build_kg.py`, `src/tkgat/analysis/report.py` (`wilson_interval`), `notebooks/02_entity_linking.ipynb`, `notebooks/03_kg_temporal.ipynb`.

---

## 7. Kiểm định thống kê

### 7.1 Hai nguồn biến thiên
- **Ngẫu nhiên khi huấn luyện** (khởi tạo, lấy mẫu âm, thứ tự batch): 5 seed (2019–2023); báo cáo trung bình ± **độ lệch chuẩn mẫu** (ddof = 1; numpy mặc định ddof = 0 là sai ở đây, pandas `.std()` là đúng).
- **Mẫu user** (test chỉ là một mẫu): kiểm định cặp theo user.

### 7.2 Kiểm định cặp và Wilcoxon
- Hai mô hình chấm trên cùng user ⇒ xét hiệu d_u = m_A(u) − m_B(u). Kiểm định độc lập bỏ phí thông tin này.
- **Wilcoxon signed-rank** (mặc định): xếp hạng |d_u|, cộng hạng của d_u > 0; không cần giả định chuẩn (Recall theo user lệch, nhiều số 0). User có d_u = 0 bị bỏ ⇒ báo cả `n_pairs` và `n_nonzero`.
- Paired t: t = d̄ / (s_d/√n), cần d_u xấp xỉ chuẩn.
- Điểm theo user = trung bình qua các seed chung rồi mới so cặp; ghi thêm p nhỏ nhất/lớn nhất khi so từng seed.
- Cảnh báo khi dưới 100 user hoặc dưới 30 chênh lệch khác 0: ngưỡng **tự đặt**, không phải chuẩn thống kê.

### 7.3 So sánh nhiều lần và Holm
- m kiểm định ở mức 0,05 ⇒ xác suất có ít nhất một dương tính giả là 1 − 0,95^m (m = 10 ⇒ ≈ 0,40).
- Holm–Bonferroni: sắp p tăng dần; bác bỏ H_(k) nếu p_(j) ≤ α/(m − j + 1) với mọi j ≤ k. Ví dụ p = {0,01; 0,02; 0,04}, m = 3: ngưỡng 0,0167; 0,025; 0,05 ⇒ bác bỏ cả ba (Bonferroni thuần chỉ bác bỏ cái đầu).
- Họ Holm trong repo: cùng bộ dữ liệu, cùng metric. Bảng baseline 3 cặp; bảng RQ3 3 phép cố định (T-KGAT vs KGAT, T-KGAT thật vs hoán vị, KGAT vs không KG); phép chưa chạy tính p = 1. Stage 2 có họ Holm riêng.

### 7.4 Đọc kết quả đúng cách
- p lớn với 51 hay 95 user **không** chứng minh "không khác biệt": kiểm định có ít sức mạnh.
- Không chọn seed đẹp nhất; báo cáo kết quả âm kèm giải thích.
- Nhiều dataset: Demšar (2006) khuyên Friedman + Nemenyi; nhóm chỉ có 2–3 bộ nên báo riêng từng bộ.

### 7.5 Code
`src/tkgat/eval/stats.py`, `src/tkgat/analysis/significance.py`, `scripts/09_significance.py`, `src/tkgat/analysis/rq3.py`, `scripts/10_rq3_table.py`.

---

## 8. Quy trình thực nghiệm và quản lý kết quả

- **Một lệnh cho mỗi thí nghiệm:** `scripts/run.py --data <config> --model <config>` chạy mọi điểm lưới × 5 seed, chọn điểm theo trung bình Recall@20 val, rồi chấm test **một lần** cho điểm đã chọn. `--val-only` không mở file test (dùng cho độ nhạy). `--resume` dùng lại checkpoint (không dùng khi đổi dữ liệu như 0011).
- **Registry chỉ thêm dòng** (`results/registry.csv`): mỗi dòng có thời gian, git commit, dataset, hash split, model, config, seed, phase, epoch, các chỉ số, device. `registry.load()` lấy dòng mới nhất cho mỗi khóa; dòng `-dirty` (chạy khi code có thay đổi chưa commit) bị loại khỏi bảng chính thức.
- **Không số nào gõ tay:** báo cáo, notebook và bảng đều đọc từ registry hoặc file do script ghi.
- **Decision record** (`docs/decisions/NNNN-*.md`): mọi thay đổi giao thức phải có file quyết định (bối cảnh, quyết định, hệ quả); quyết định đã accepted không sửa, chỉ thay thế bằng quyết định mới.
- **Thiết bị:** mọi model và trainer nhận `device`; GPU laptop RTX 3050 4 GB. `seed_all(seed)` đầu mỗi lần chạy.
- **Dữ liệu không vào git** (giấy phép MovieLens/KB4Rec).
- **Tag:** `v0.1-review` (commit `0e194e3`) đóng băng dữ liệu, KG, baseline, báo cáo dữ liệu và là commit của lần chạy lại theo 0011. Chưa có `v0.2-kgat`: việc chạy lại từ clone sạch trên máy khác chưa làm (dời sang giai đoạn 4), nên kết quả KGAT và T-KGAT vẫn là sơ bộ, chưa được kiểm tra tái lập độc lập.

**Các quyết định (đọc tên là phải nhớ nội dung):**

| # | Nội dung |
|---|---|
| 0001 | Làm lại từ đầu vì kết quả cũ rò rỉ |
| 0002 | Chia theo mốc thời gian toàn cục, val/test chỉ user/item đã biết |
| 0003 | Liên kết thực thể ML-100K → ML-1M → Freebase |
| 0004 | Luật KG theo thời gian (fact có ngày, thực thể ra đời sau mốc, fact không ngày) |
| 0005 | Thời gian trong attention |
| 0006 | Xử lý ngày: năm ⇒ 31/12, tháng ⇒ cuối tháng |
| 0007 | Mở rộng lưới cửa sổ MostPop recent thành {1, 3, 5, 7, 14, 30} |
| 0008 | Tập con user (ml-1m-u10, u05), xuất định dạng KGAT gốc, giả định timestamp |
| 0009 | T-KGAT: công thức Δt, t_ref, chuẩn hóa, lưới γ, hoán vị, Stage 2 có điều kiện |
| 0010 | Độ nhạy γ chỉ trên val (proposed) |
| 0011 | Thêm metadata MovieLens vào KG, chạy lại mọi mô hình dùng KG |

---

## 9. Kết quả hiện tại (sơ bộ, KG có metadata MovieLens)

Recall@20 trên test, trung bình ± độ lệch chuẩn mẫu qua 5 seed.

| | ml-100k (51 user) | ml-1m-u10 (95 user) | ML-1M đầy đủ (952 user) |
|---|---|---|---|
| MostPop | 0,1555 | 0,0846 | 0,0902 |
| MostPop recent | **0,3372** (14 ngày) | 0,1007 (3 ngày) | 0,1060 (7 ngày) |
| BPR-MF | 0,1471 ± 0,0295 | 0,1000 ± 0,0069 | 0,1194 ± 0,0031 |
| KGAT | 0,2096 ± 0,0284 | 0,1086 ± 0,0054 | chưa chạy |
| KGAT không KG | 0,2017 ± 0,0232 | 0,1042 ± 0,0101 | chưa chạy |
| T-KGAT Stage 1 | 0,2151 ± 0,0174 (γ = 3) | 0,1113 ± 0,0046 (γ = 1) | chưa chạy |
| T-KGAT hoán vị | 0,1950 ± 0,0156 | 0,1119 ± 0,0049 | chưa chạy |
| T-KGAT Stage 2 | 0,1933 ± 0,0216 (γ = 1) | 0,1145 ± 0,0148 (γ = 1) | chưa chạy |
| Stage 2 hoán vị ngày fact | 0,1971 ± 0,0238 | 0,1164 ± 0,0099 | chưa chạy |

p Holm (Wilcoxon cặp theo user), Recall / NDCG:
- T-KGAT vs KGAT: ml-100k 1,000 / 0,651; ml-1m-u10 0,450 / 0,049.
- T-KGAT thật vs hoán vị: ml-100k 0,944 / 0,651; ml-1m-u10 0,450 / 0,833.
- KGAT vs không KG: ml-100k 1,000 / 0,587; ml-1m-u10 0,143 / 0,049.
- Baseline ML-1M: BPR-MF > MostPop recent (p Holm ≈ 1,1·10⁻⁵); ml-100k: MostPop recent > BPR-MF (p Holm 0,029).

**Đọc:** chưa có bằng chứng thời gian giúp (T-KGAT không khác bản hoán vị); bằng chứng KG giúp yếu hơn so với KG cũ; độ nhạy γ trên val cho hai chiều ngược nhau (ml-100k nghiêng về γ = 0, ml-1m-u10 về γ = 1). Stage 2 cũng không có bằng chứng (xem dưới).

**Stage 2** (Δt trên triple KG, γ = 1 được chọn trên cả hai bộ, là mép dưới lưới): ml-100k 0,1933 ± 0,0216, thấp hơn KGAT nhưng không có ý nghĩa (Recall p Holm 0,162); ml-1m-u10 0,1145 ± 0,0148, cao hơn KGAT nhưng không có ý nghĩa (Recall p Holm 0,189; NDCG 0,065). So với bản hoán vị ngày fact: không khác biệt trên cả hai bộ (p Holm từ 0,155 trở lên), và bản hoán vị còn nhỉnh hơn ở Recall. Vì Δt_norm của cạnh KG bị chặn ở 1 (chỉ 0,7% / 0,0% fact có ngày nằm dưới mức này), Stage 2 thực chất giảm trọng số fact có ngày so với fact không ngày, không đo được "fact cũ hay mới". Đọc: không có bằng chứng; cần thiết kế lại thang Δt cho KG.

---

## 10. Từ điển thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Implicit feedback | Phản hồi ngầm: chỉ biết có tương tác |
| Top-K, full ranking | Xếp hạng toàn bộ item, lấy K đầu |
| Positive / negative | Tương tác dương (rating ≥ 4) / item lấy mẫu làm âm |
| k-core | Giữ user và item có ít nhất k tương tác (lặp đến ổn định), tính trên train |
| Cold-start | User/item mới chưa có lịch sử trong train |
| Mask | Item đã tương tác bị loại khỏi danh sách xếp hạng |
| Recall@K, NDCG@K | Tỷ lệ đáp án lọt top-K; như trên nhưng thưởng thứ hạng cao |
| Early stopping, patience | Dừng khi val không tăng sau một số lần đánh giá |
| Seed | Hạt giống ngẫu nhiên; 5 seed để đo dao động khi huấn luyện |
| Triple (h, r, t) | Một fact trong KG |
| CKG | Đồ thị gộp tương tác và KG |
| TransE / TransR | KGE tịnh tiến / tịnh tiến sau chiếu theo quan hệ |
| Attention, softmax | Trọng số cho hàng xóm, chuẩn hóa tổng bằng 1 |
| Bi-Interaction | Aggregator tổng + nhân từng phần tử |
| Δt, t_ref, dt_den | Độ cũ của cạnh, mốc tham chiếu (max train), độ dài cửa sổ train |
| γ, δ_r | Cường độ suy giảm; 1 nếu quan hệ mang thời gian, 0 nếu không |
| Permutation control | Đối chứng hoán vị timestamp |
| Point-in-time correctness | Chỉ dùng dữ liệu đã biết tại thời điểm dự đoán |
| Coverage / precision (liên kết) | Độ phủ / độ chính xác của liên kết thực thể |
| Wilson interval | Khoảng tin cậy cho một tỷ lệ, dùng được khi k = 0 |
| Wilcoxon signed-rank | Kiểm định cặp phi tham số |
| Holm–Bonferroni | Hiệu chỉnh nhiều phép so sánh |
| Registry | File kết quả chỉ thêm dòng |
| Decision record | File ghi quyết định giao thức |
| `-dirty` | Dòng registry chạy khi code chưa commit sạch |

---

## 11. 30 câu tự kiểm tra tổng hợp (có đáp án ngắn)

1. Vì sao không dùng (r − ŷ)² cho phản hồi ngầm? — Không có nhãn âm thật; ô trống có thể là chưa thấy.
2. Chứng minh −ln σ(x) = softplus(−x). — −ln(1/(1+e^(−x))) = ln(1+e^(−x)).
3. Item âm của BPR lấy từ đâu? — Item user chưa tương tác trên train.
4. Vì sao MostPop recent là baseline bắt buộc? — Là baseline thời gian đơn giản nhất; mô hình thời gian phải thắng nó.
5. Ba lỗi rò rỉ của kết quả cũ? — Chọn epoch trên test; k-core trên toàn bộ dữ liệu; KG từ metadata hiện tại.
6. Vì sao chia theo mốc toàn cục mà không theo từng user? — Theo từng user để tương lai của user khác lọt vào train.
7. k-core tính trên phần nào, vì sao? — Chỉ train; tính trên toàn bộ là dùng tương lai để chọn user.
8. Vì sao test ML-100K chỉ 51 user? — User chấm hàng loạt rồi đi; rating sau t2 chủ yếu của user mới, bị loại.
9. Mask test gồm gì? — Mọi tương tác trước t2, kể cả rating thấp.
10. Kể 6 kiểm tra của `check_split`. — Mục 2.6.
11. Vì sao không dùng sampled metrics? — Có thể đảo thứ tự mô hình.
12. Tính NDCG@3 cho [4, 7, 1, 9], đáp án {7, 9}. — ≈ 0,387.
13. TransR sửa điểm yếu nào của TransE? — Quan hệ 1-N, nhờ chiếu W_r.
14. Loss KGE bằng bao nhiêu khi g(sai) = g(đúng)? — ln 2.
15. CKG gồm những cạnh nào? — Interact (chỉ train), triple KG, và các cạnh ngược.
16. Softmax của attention chuẩn hóa trên tập nào? — Toàn bộ hàng xóm của node h.
17. Vì sao nối embedding mọi lớp? — Giữ đặc trưng riêng và tín hiệu lân cận.
18. Vì sao KGAT không tái lập từng bit trên GPU? — `torch.sparse.mm` không tất định.
19. Công thức T-KGAT Stage 1? — π − γ·δ_r·Δt_norm trước softmax.
20. Vì sao γ = 0 là KGAT? — Thừa số e^(−γΔt) = 1.
21. t_ref và dt_den lấy từ đâu? — Chỉ từ timestamp của train.
22. Nếu mọi cạnh của user có cùng timestamp? — Tác dụng triệt tiêu ở node user.
23. Đối chứng hoán vị cho biết gì? — Phần hơn có đến từ thời gian hay không.
24. Vì sao Stage 2 hiện tại không đo được thời gian? — Δt_norm bị chặn ở 1 nên 99%+ fact có ngày bị phạt như nhau.
25. Độ phủ 97,2% có phải độ chính xác? — Không; độ chính xác đo bằng mẫu 50 phim, Wilson [0%; 7,1%].
26. Ngày chỉ có năm được đọc thế nào, vì sao? — 31/12, để không đưa fact tương lai vào.
27. Quy tắc theo mốc có giải quyết hết rò rỉ qua KG không? — Không, Wikidata là bản chụp 2026.
28. Vì sao std qua seed dùng ddof = 1? — 5 seed là một mẫu; ddof = 0 đánh giá thấp dao động.
29. Holm khác Bonferroni thế nào? — Ngưỡng nới dần theo thứ hạng p, mạnh hơn.
30. p Holm = 0,3 với 51 user nghĩa là gì? — Chưa đủ bằng chứng có khác biệt; chưa chứng minh không khác biệt.
