# 11 — Tổng hợp lý thuyết T-KGAT, bản cho Last-FM (`lastfm-u25` và `lastfm-u25-latest`), cập nhật 13/11/2026

> **Cập nhật 14/11/2026 (trước buổi họp): Stage 2 có lỗi thiết kế chuẩn hóa.** `dt_norm` của cạnh KG chia cho cửa sổ của **tương tác** train (`dt_den`, quyết định 0009 mục 3) rồi bị chặn ở 1 (mục 10), nên fact cũ hơn cửa sổ đều có `dt_norm` = 1. `lastfm-u25` bị một phần (55,7% triple bị chặn ở 1), `lastfm-u25-latest` gần như toàn bộ (80,1%, 0,0% trong cửa sổ). Code làm đúng quyết định 0009; lỗi nằm ở thiết kế, không phải ở dữ liệu. **Hệ quả:** kết quả Stage 2 của cả hai bản không dùng để kết luận "fact cũ hay mới"; **Stage 1 không bị ảnh hưởng** (cạnh tương tác luôn nằm trong cửa sổ). Đề xuất: quyết định mới chuẩn hóa Δt của fact theo ngày của fact, báo là thêm sau khi đã thấy test, rồi chạy lại Stage 2. Các câu cũ bên dưới đã sửa theo ghi chú này. Notebook trình bày: bản rút gọn `notebooks/00_present_lastfm_short.ipynb` (bản đầy đủ `00_present_lastfm.ipynb` để tra khi cô hỏi sâu). PR của bản `latest` là **#44** (thay #43 đã đóng).

Tài liệu này là bản ôn lý thuyết dành riêng cho buổi báo cáo kết quả **Last-FM** với cô (dời sang 15/11/2026 lúc 20:00; lịch cũ 14/11 lúc 13:00). Đây là bản cập nhật của tài liệu `lastfm-u25` ngày 12/11: mục 1 đến 11 giữ nội dung của bản 12/11 (chỉ thêm vài dòng ở mục 0, 2.8, 5.3, 8, 10 và 11 để trỏ sang bản mới), **mục 12 mới** nói về bản dữ liệu `lastfm-u25-latest` của Phôn, và từ điển (mục 10) và các câu tự kiểm (mục 11) có thêm phần của bản mới. Cấu trúc giống `docs/study/08-tong-hop-ly-thuyet.md` (bản 29/10 cho MovieLens): phần lý thuyết chung (mục 1, 3, 4, 7) giữ nguyên ý, các phần gắn với dữ liệu (mục 0, 2, 5, 6, 8, 9, 10, 11) viết lại cho Last-FM. Lý thuyết hôm nay **không đổi** so với bản 29/10; thứ đổi là dữ liệu, số liệu và những gì phải nói thêm.

**Phạm vi:** `lastfm-u25` (25% user của Last-FM, quyết định 0013; mục 0 đến 11) và `lastfm-u25-latest` (cùng user, timestamp là lần nghe **muộn nhất**; mục 12). Hai bản **không được trộn số**: mọi bảng ghi rõ tên run.

**Nguồn số liệu:** `results/registry.csv` (139 dòng của `lastfm-u25`, commit `ce55c0c`, không dòng `-dirty`), `results/significance.csv`, `results/rq3_models.csv`, `results/rq3_pairs.csv`, `docs/rq3-table.md`, `docs/data-report-lastfm.md`, `docs/data-report-lastfm-numbers.md`, `docs/bprmf-training-table.md`, `results/runs/lastfm-u25/*_summary.json`, quyết định 0012, 0013, 0014. Các số ở **mục 5.3**: số cạnh theo tuần và độ trải thời gian theo user, item có trong `docs/data-report-lastfm-numbers.md`; phân bố Δt_norm và tỷ lệ cạnh trong 30, 365 ngày cuối được notebook `notebooks/00_present_lastfm.ipynb` tính lại từ `train.csv` (đã kiểm ngày 13/11). Số của bản `lastfm-u25-latest` (mục 12) đến từ `docs/data-report-lastfm-latest-numbers.md`, `results/rq3_models_lastfm-latest.csv`, `results/rq3_pairs_lastfm-latest.csv` và `results/lastfm-latest-provenance.json` (PR #44, xếp chồng lên PR #42); các dòng registry của bản này **không nằm** trong `results/registry.csv` của repo.

**Trạng thái kết quả:** *sơ bộ* (cả hai bản). PR #42 còn là draft, PR #44 (bản `latest`) xếp chồng lên #42, chưa có tag `v0.2-kgat`, chưa kiểm tra tái lập độc lập trên máy khác. Mọi chỗ trình bày số của KGAT và T-KGAT phải nói chữ "sơ bộ".

---

## 0. Bức tranh tổng thể trong một trang

**Đề tài:** T-KGAT — Temporal Knowledge Graph Attention Network for Recommendation. Mô hình gốc là KGAT (Wang và cộng sự, KDD 2019). Nhóm thêm thời gian vào attention.

**Vì sao có Last-FM:** ngày 01/11 cô nhận xét kết quả MovieLens cho thấy vấn đề nằm ở dữ liệu đầu vào và đề nghị nhóm chuyển sang Last-FM, lặp lại các thí nghiệm của MovieLens (quyết định 0012). Quyết định chuyển được đưa ra **sau khi điểm test MovieLens đã được xem**; nhóm làm theo hướng cô chỉ, không chọn để cải thiện điểm số. Câu này phải nói rõ.

**Protocol không đổi** so với MovieLens: cùng mã chia dữ liệu, cùng mốc phân vị 0,8 / 0,9, 5-core chỉ trên train, cùng `check_split`, cùng luật KG theo thời gian (0004, 0006), cùng lưới, cùng 5 seed. Chỉ dataset khác.

**Ba câu hỏi nghiên cứu (`docs/plan.md`, mục 4)** và câu trả lời hiện có trên `lastfm-u25`:

| RQ | Câu hỏi | Trả lời hiện có (sơ bộ) |
|---|---|---|
| RQ1 | Dữ liệu và KG có đặc điểm gì theo thời gian? | Val/test lớn hơn MovieLens rất nhiều (2.319 / 1.608 user); KG gần như không có fact nằm giữa t1 và t2; 24,4% triple nằm trong cửa sổ Δt (MovieLens: 0,7% và 0,0%) |
| RQ2 | KG còn giúp khi đã loại rò rỉ không? | KGAT hơn KGAT không KG 0,0016 Recall@20, **không có ý nghĩa** (p Holm 0,803) |
| RQ3 | Thời gian trong attention có giúp, và có thật đến từ thời gian không? | T-KGAT không khác KGAT (p Holm 0,420) và không khác bản hoán vị (p Holm 0,803); Stage 2 cũng không |

**Bản `lastfm-u25-latest` (mục 12) trả lời gì thêm:** đổi timestamp thành lần nghe muộn nhất làm hết bước nhảy số cạnh quanh 01/04/2014 nhưng không loại được thiên lệch chọn mẫu của tập cặp; Stage 2 trên bản đó **không có độ phân giải** (0,0% triple trong cửa sổ Δt); KGAT hơn KGAT không KG ở Recall@20 có ý nghĩa sau Holm (p Holm 0,028) nhưng NDCG@20 thì không (0,225); T-KGAT không hơn KGAT, và "T-KGAT thật hơn hoán vị" **không chứng minh thời gian giúp** vì bản hoán vị thấp hơn KGAT.

**Chuỗi kiến thức cần học:**

```
Hệ gợi ý + phản hồi ngầm + BPR (1)
        │
Đánh giá offline + rò rỉ + chia theo thời gian (2) ──► Thống kê: seed, kiểm định cặp, Holm (7)
        │
KG + KGE (TransE, TransR) + CKG (3) ◄── Tích hợp dữ liệu, ngày của fact, đúng theo thời điểm (6)
        │
KGAT: attention + lan truyền + hai loss (4)
        │
Thời gian: T-KGAT Stage 1, Stage 2, đối chứng hoán vị (5)
```

**Điểm khác lớn nhất so với MovieLens (nhớ năm điều):**
1. Không có rating: mọi tương tác là một lần nghe, tất cả là dương (`min_rating: 1`).
2. Timestamp là lúc **nghe**, không phải lúc nhập điểm; Δt nghĩa là "thời gian kể từ lần nghe đầu tiên của cặp (user, item)".
3. Val/test giữ được nhiều tương tác và nhiều user hơn hẳn (2.319 user val, 1.608 user test, so với 66 và 51 của ML-100K) nên kiểm định có sức mạnh thật.
4. KG không có ngày của statement; ngày lấy từ bản chụp Freebase 09/08/2015 qua ngày tồn tại của đích.
5. Chính **tập cặp (user, item)** có thiên lệch chọn mẫu (mục 2.8): không có kiểm tra tự động nào thấy được, phải nói cùng mọi kết quả.

---

## 1. Hệ gợi ý, phản hồi ngầm và BPR

### 1.1 Bài toán
- **Gợi ý top-K:** với mỗi user, trả về K item chưa tương tác mà user có khả năng thích nhất. Đây là bài toán **xếp hạng**, không phải dự đoán số sao.
- **Phản hồi tường minh** (explicit): user cho điểm. **Phản hồi ngầm** (implicit): chỉ biết user có tương tác. Không tương tác **không** có nghĩa là không thích, nên không có nhãn âm thật và không dùng được sai số bình phương (r − ŷ)².
- **Last-FM vốn đã là phản hồi ngầm:** một dòng là một sự kiện nghe nhạc, không có điểm. Nhóm đặt `min_rating: 1`, nghĩa là mọi cặp (user, item) đều là tương tác dương. (MovieLens thì phải đổi rating ≥ 4 thành dương.)
- **Cặp (user, item) và timestamp:** bộ Last-FM mà tác giả KGAT công bố chỉ có cặp, không có thời gian. Nhóm lấy thời gian từ LFM-1b và gán cho mỗi cặp **lần nghe sớm nhất** của cặp đó.

### 1.2 Matrix Factorization (MF)
- Mỗi user u có vector p_u, mỗi item i có vector q_i, cùng độ dài `dim`.
- Điểm: ŷ_ui = p_u · q_i. Tích vô hướng càng lớn thì item càng đứng cao.

### 1.3 BPR (Bayesian Personalized Ranking)
- Giả định: user u thích item đã tương tác i hơn item chưa tương tác j. Mỗi mẫu huấn luyện là bộ ba (u, i, j).
- x̂_uij = ŷ_ui − ŷ_uj; loss L = Σ −ln σ(x̂_uij) + λ‖Θ‖².
- Đẳng thức hay bị hỏi: −ln σ(x) = ln(1 + e^(−x)) = softplus(−x). Code dùng `F.softplus(-(x_ui - x_uj))`.
- Ví dụ số: x̂ = 0 ⇒ loss = ln 2 ≈ 0,693; x̂ = 3 ⇒ loss ≈ 0,049.
- **Lấy mẫu âm:** j lấy ngẫu nhiên trong các item user chưa tương tác **trên train**, không bao giờ dùng val hay test. `NegativeSampler` lấy mẫu trên GPU bằng `torch.Generator`.
- Siêu tham số: `dim`, `lr`, `l2`.
- **Overfit và dừng sớm:** loss train vẫn giảm nhưng Recall@20 trên val ngừng tăng. Repo đo val mỗi 5 epoch, patience 5 cho BPR-MF (10 cho KGAT), giữ epoch tốt nhất trên val.

### 1.4 Baseline
- **MostPop:** cùng một danh sách K item nhiều tương tác nhất trong train cho mọi user.
- **MostPop recent (MostPop gần đây):** chỉ đếm N ngày cuối của train; N chọn trên val trong lưới {1, 3, 5, 7, 14, 30} ngày (quyết định 0007). Cửa sổ neo ở timestamp cuối của train. **Trên `lastfm-u25` val chọn N = 14 ngày.**
- **BPR-MF:** MF học bằng BPR; lưới 3 điểm (dim, l2), 5 seed.
- Vì sao bắt buộc có MostPop recent: là baseline thời gian đơn giản nhất. Mô hình có yếu tố thời gian phải thắng nó, không chỉ thắng MostPop. (Trên Last-FM nó **không** thắng MostPop thường ở test, mục 9.)

### 1.5 Code
`src/tkgat/models/baselines.py` (`MostPop`, `BPRMF`), `src/tkgat/train/bpr.py` (`NegativeSampler`, `fit_bpr`), `configs/model/bprmf.yaml`, `mostpop.yaml`, `mostpop_recent.yaml`.

---

## 2. Đánh giá offline và rò rỉ dữ liệu

### 2.1 Rò rỉ dữ liệu là gì
Rò rỉ là khi thông tin của val/test (hoặc của tương lai) ảnh hưởng tới một quyết định lúc huấn luyện hay tiền xử lý. Hậu quả: điểm test lạc quan.

**Ba lỗi rò rỉ của kết quả cũ (lý do làm lại, quyết định 0001):**
1. Chọn epoch tốt nhất trên test (không có tập val).
2. Lọc k-core trên toàn bộ dữ liệu rồi mới chia.
3. KG dựng từ metadata hiện tại (tri thức tương lai).
Thêm: chỉ 1 seed, CPU và GPU lệch nhau khoảng 18%.

### 2.2 Chia theo mốc thời gian toàn cục (quyết định 0002)
- Hai mốc t1 < t2 là phân vị 0,8 và 0,9 của **mọi** timestamp, áp chung cho mọi user.
- train: ts < t1; val: t1 ≤ ts < t2; test: ts ≥ t2.
- Vì sao không chia theo từng user: tương lai của user A vẫn lọt vào train qua user B.
- Các bước: mọi cặp là dương → 5-core **chỉ trên train** → val/test chỉ giữ user và item đã có trong train (user/item mới bị loại và được đếm).
- Không huấn luyện lại trên train + val trước khi chấm test: chọn trên val, chấm test một lần.

**Số của nhóm (`docs/data-report-lastfm.md`, bảng 1):**

| | `lastfm` (đầy đủ) | `lastfm-u25` (dùng chính thức) |
|---|---|---|
| Cặp (user, item) | 1.542.856 | 370.979 |
| t1 / t2 (UTC) | 28/04/2014 / 06/06/2014 | 29/04/2014 / 07/06/2014 |
| Train (sau 5-core) | 1.220.988 | 236.053 |
| User / item của train | 22.040 / 44.788 | 5.321 / 18.665 |
| Val: positive / user | 142.023 / 10.618 | 23.708 / 2.319 |
| Test: positive / user | 133.896 / 7.149 | 22.419 / 1.608 |
| Positive val / test còn lại sau lọc | 92,1% / 86,8% | 63,9% / 60,4% |
| Positive bị 5-core loại khỏi train | 13.295 | 60.730 (trong 296.783) |

### 2.3 Cái giá của chia theo thời gian: cold-start
- **Last-FM đầy đủ giữ lại 92,1% val và 86,8% test** (MovieLens ML-100K: 12,6% và 13,3%), vì timestamp của Last-FM rải đều theo thời gian chứ không dồn vào vài ngày đầu của mỗi user.
- **Nhưng `lastfm-u25` chỉ giữ 63,9% val và 60,4% test.** Lấy một phần user làm rơi nhiều item khỏi train: 28,8% positive val và 28,1% positive test bị loại vì **cold item** (trên bộ đầy đủ chỉ 3,0% và 4,6%); cold user là 7,3% (val) và 11,5% (test).
- Hệ quả: `lastfm-u25` vẫn có val/test lớn hơn rất nhiều so với MovieLens, nhưng **không còn là val gần như nguyên vẹn của Last-FM đầy đủ**. Phải nêu cùng mọi số `lastfm-u25`. Kết quả chỉ nói cho mẫu 25% user, không nói cho toàn bộ Last-FM.

### 2.4 Mask
- Khi xếp hạng cho user u, loại các item u đã tương tác.
- Mask val: mọi tương tác có ts < t1. Mask test: mọi tương tác có ts < t2.
- Mask không bao giờ chứa tương tác từ mốc của chính phase đó trở đi.

### 2.5 Chỉ số
- **Xếp hạng toàn bộ item** rồi tính chỉ số; không dùng sampled metrics (Krichene & Rendle, 2020).
- Recall@K(u) = |top-K(u) ∩ T_u| / |T_u|.
- DCG@K(u) = Σ_{k=1..K} 1[item hạng k ∈ T_u] / log₂(k+1); NDCG@K = DCG@K / IDCG@K.
- Ví dụ tính tay: xếp hạng [4, 7, 1, 9], đáp án {7, 9}, K = 3 ⇒ Recall@3 = 0,5; NDCG@3 ≈ 0,387.
- **Cỡ số của Last-FM:** Recall@20 trên `lastfm-u25` chỉ khoảng 0,05 (MovieLens u10: khoảng 0,10 đến 0,11), vì có 18.665 item để chọn và mỗi user có ít đáp án trong tập còn lại. Không so tuyệt đối giữa hai dataset.

### 2.6 `check_split`: 6 kiểm tra (thuộc lòng)
1. Thứ tự thời gian (train < t1 ≤ val < t2 ≤ test).
2. Không cặp (user, item) nào nằm ở hai phần.
3. Val/test chỉ có user và item đã biết lúc train.
4. Train thỏa 5-core khi tính riêng trên train.
5. Mask chỉ dùng quá khứ.
6. Đáp án không trùng mask của chính phase đó.

Ba nguồn rò rỉ `check_split` không bắt, và nơi chặn chúng: chọn mô hình bằng test (chỉ `scripts/run.py` chấm test, sau khi chọn trên val), thống kê dùng tương lai (`MostPop` chỉ đọc train), tri thức tương lai trong KG (`check_kg`). Với Last-FM, `check_split` không báo lỗi và `interaction_leakage_problems` rỗng.

### 2.7 Code
`src/tkgat/data/split.py`, `src/tkgat/data/leakage.py`, `src/tkgat/data/encode.py`, `src/tkgat/eval/metrics.py`, `scripts/run.py`, `tests/test_leakage.py`; cho Last-FM: `scripts/02_build_splits.py --data configs/data/lastfm_u25.yaml`.

### 2.8 Thiên lệch chọn mẫu của tập cặp (hạn chế riêng của Last-FM; quyết định 0012)
**`check_split` không thấy được lỗi này vì lỗi nằm trước khi chia.**
- Tập cặp (user, item) của bản Last-FM mà tác giả KGAT công bố **chỉ chứa các cặp còn được nghe lại ở giai đoạn cuối** (có ít nhất một lần nghe từ 01/04/2014). Kiểm trên mẫu 2.475 user: 99,97% cặp có lần nghe **muộn nhất** từ 01/04/2014 trở đi.
- Dấu vết thấy trong dữ liệu: số cặp mới mỗi ngày nhảy từ khoảng 2.800 lên khoảng 11.900 tại 01/04/2014 rồi giảm dần; trên bộ đầy đủ khoảng 23% cặp của train rơi vào tháng trước t1 (trên `lastfm-u25`: 22,0% cạnh train nằm trong 30 ngày trước `t_ref`, số mình tính thêm). Sự kiện thô của LFM-1b không có bước nhảy này.
- Nghĩa là tập cặp được chọn bằng **thông tin sau mốc** (cặp nào còn sống đến cuối), có thể thiên vị các cặp "sống lâu". Tác động lên kết quả **chưa đo** (phương án A trong 0012, tập cặp không giới hạn theo cửa sổ 04/2014, chưa chạy lại trên `lastfm-u25` vì thiếu thời gian, 0013 mục 7).
- Thuộc cùng nhóm: bài báo KGAT ghi Last-FM là dữ liệu tháng 1–6/2015, nhưng LFM-1b của nhóm có timestamp đến khoảng 09/2014. Nhiều khả năng tập cặp của KGAT lấy từ giai đoạn **từ 01/04/2014 đến hết dữ liệu (khoảng năm tháng)**; đây là suy luận, tác giả không ghi rõ.
- Cách nói: "Đây là đặc điểm của bộ dữ liệu công bố; nhóm báo cáo, không giấu; chưa biết nó làm kết quả lệch bao nhiêu".
- Ba cách nhìn cùng một lỗi (dùng ở buổi 14/11): **A** tập cặp chọn bằng thông tin sau 01/04/2014; **B** số cạnh train nhảy lên quanh 01/04/2014 (dấu vết nhìn thấy của A); **C** `check_split` không thấy được A và B. Bản `latest` (mục 12) là một nỗ lực xử lý B; nó **không** xử lý A.

---

## 3. Knowledge Graph và Knowledge Graph Embedding

### 3.1 KG
- Tập triple (h, r, t): thực thể đầu, quan hệ, thực thể đuôi. Ví dụ trên Last-FM: (bản thu, `music.recording.artist`, nghệ sĩ).
- KG giúp gợi ý ở hai phía: item ít lượt nghe nhận thêm thông tin qua thuộc tính chung; user đi qua thuộc tính để tới item mới.
- Quan hệ 1-N: một bản thu có nhiều kỹ sư/nhà sản xuất.

### 3.2 KGE: TransE và TransR
- **TransE:** e_h + e_r ≈ e_t; điểm g(h, r, t) = ‖e_h + e_r − e_t‖² (nhỏ là đúng). Yếu với 1-N.
- **TransR (KGAT dùng):** mỗi quan hệ có ma trận chiếu W_r: g(h, r, t) = ‖W_r e_h + e_r − W_r e_t‖², với e_h, e_t ∈ ℝᵈ, e_r ∈ ℝᵏ, W_r ∈ ℝᵏˣᵈ.
- **Loss KGE:** L = Σ −ln σ(g(h, r, t′) − g(h, r, t)), t′ là tail hỏng. Ví dụ g(đúng) = 1, g(sai) = 4 ⇒ loss ≈ 0,049; bằng nhau ⇒ ln 2.

### 3.3 CKG (Collaborative Knowledge Graph)
- Gộp tương tác và KG vào một đồ thị: mỗi tương tác dương thành triple (u, Interact, i); item căn chỉnh với entity KG; thêm quan hệ ngược.
- Ví dụ đường đi: user A → Interact → bản thu B → artist → nghệ sĩ X ← artist⁻¹ ← bản thu C.
- **Trong repo:** cạnh tương tác chỉ từ train; KG cho train/val dựng tại t1, KG cho test dựng tại t2. **Trên Last-FM KG t1 và t2 giống hệt nhau** (cùng 152.976 triple ở `lastfm-u25`).
- **KG của repo là 1-hop item → thuộc tính.** 1.097 triple (0,7%) trong `lastfm-u25` có head không phải item của train nên bị loại (chủ yếu `music.recording.canonical_version`, `music.single.versions`); quyết định 0013 mục 4.

### 3.4 Ba họ mô hình dùng KG
Embedding-based (CKE), path-based (PER, KPRN), propagation-based (RippleNet, KGCN, KGAT). KGAT = TransR như CKE + lan truyền nhiều bước có attention. CKE là baseline KG tự nhiên, **chưa làm**.

### 3.5 Code
`src/tkgat/kg/ckg.py`, `src/tkgat/kg/lastfm.py` (đọc nguồn Last-FM), `scripts/06_build_kg.py`.

---

## 4. KGAT

### 4.1 Ba khối
1. **Embedding** của user, item, entity, relation.
2. **Các lớp lan truyền có attention** trên CKG.
3. **Dự đoán** bằng tích vô hướng của vector đã nối qua các lớp.

### 4.2 Attention
- Hàng xóm của node h: mọi cạnh (h, r, t) trong CKG, kể cả cạnh ngược.
- Điểm trước chuẩn hóa: π(h, r, t) = (W_r e_t)ᵀ tanh(W_r e_h + e_r).
- **Softmax trên toàn bộ hàng xóm của h:** thứ quyết định là π **so với** các hàng xóm khác của cùng node.
- Ví dụ số: π = (2, 1, 0) ⇒ trọng số ≈ (0,665; 0,245; 0,090).
- Repo tính attention theo từng nhóm quan hệ, trong `no_grad`; A khởi tạo đều 1/|N_h|; attention tính lại sau mỗi epoch.

### 4.3 Lan truyền và aggregator Bi-Interaction
- Hàng xóm gộp: e_{N_h} = Σ π(h, r, t) e_t.
- Bi-Interaction: f = LeakyReLU(W₁(e_h + e_{N_h})) + LeakyReLU(W₂(e_h ⊙ e_{N_h})).
- Sau L lớp, node nhận thông tin từ node cách tối đa L bước. L = 3.
- Vector cuối là **phép nối** mọi lớp; ŷ(u, i) = e*_uᵀ e*_i.

### 4.4 Hai loss và huấn luyện luân phiên
- L = L_KG (TransR) + L_CF (BPR) + λ‖Θ‖².
- Mỗi epoch: các bước CF với Adam riêng, các bước KG với Adam riêng, rồi cập nhật attention.

### 4.5 Cấu hình của nhóm
`configs/model/kgat.yaml`: dim 64, kge_dim 64, ba lớp [64, 32, 16], mess_dropout 0,1, batch CF 1.024, batch KG 2.048, tối đa 300 epoch, đánh giá mỗi 5 epoch, patience 10. Lưới 4 điểm lr ∈ {1e-3, 1e-4} × l2 ∈ {1e-5, 1e-4}, 5 seed (2019–2023), chọn theo trung bình Recall@20 val. `kgat_nokg`: cùng lưới, không có triple KG.

**Điểm được chọn trên `lastfm-u25`:** KGAT lr 1e-3, l2 1e-5 (val 0,0561; ba điểm kia 0,0555, 0,0434, 0,0433, epoch tốt nhất của hai điểm lr = 1e-4 nằm sát trần 300, tức chưa hội tụ). BPR-MF dim 64, l2 1e-3 (val 0,0559; hai điểm kia 0,0503 và 0,0391).

### 4.6 Khác paper và lý do
- Viết lại bằng PyTorch; code TensorFlow 1.x của tác giả chỉ để tham chiếu (`docs/design-kgat.md`, bảng 2.6).
- Chia theo thời gian (paper chia ngẫu nhiên), 5-core (paper 10-core). Số của nhóm không so trực tiếp với số trong paper.
- **Không tái lập từng bit trên GPU:** `torch.sparse.mm` (cuSPARSE) không tất định. Hai lần chạy cùng seed có thể lệch; coi là nhiễu seed. BPR-MF thì tái lập được.

### 4.7 Code
`src/tkgat/models/kgat.py`, `src/tkgat/train/kgat.py`, `src/tkgat/kg/ckg.py`, `docs/design-kgat.md`.

---

## 5. Thời gian trong gợi ý và T-KGAT

### 5.1 Vì sao thời gian quan trọng
Ba cách: sở thích trôi (user đổi gu), độ phổ biến đổi (bài hát hot theo đợt phát hành), tri thức đổi. Mô hình tĩnh coi mọi tương tác như nhau. Bốn họ phương pháp: tham số đổi theo thời gian (timeSVD++), trọng số suy giảm theo độ cũ, mô hình chuỗi (SASRec, TiSASRec), KG có thời gian hiệu lực. T-KGAT nằm giữa "suy giảm" và "attention có khoảng thời gian".

### 5.2 T-KGAT Stage 1: Δt trên cạnh user–item (mô hình chính; quyết định 0005, 0009)
- π_time(h, r, t) = π(h, r, t) − γ · δ_r · Δt_norm, **trước** softmax.
- δ_r = 1 cho Interact và Interact⁻¹, 0 cho mọi quan hệ KG.
- Δt_norm = max(0, t_ref − ts) / dt_den. **Trên `lastfm-u25`:** t_ref = 1398783364 (29/04/2014), dt_den = 290.426.881 giây (9,20 năm, từ 14/02/2005 đến t_ref).
- **Tương đương suy giảm mũ:** softmax(π − γΔt) = trọng số cũ × e^(−γΔt), chuẩn hóa lại. Cạnh cũ bị giảm **so với các hàng xóm khác của cùng node**.
- Ví dụ số: hai cạnh cùng π, Δt_norm = 0 và 1, γ = 3 ⇒ trọng số ≈ (0,953; 0,047).
- **γ = 0 cho đúng KGAT** ⇒ so sánh lồng nhau. γ chọn trên val (lưới {1, 3, 10}), không học bằng gradient. (lr, l2) lấy từ điểm KGAT đã chọn (giao thức hai tầng). **Trên `lastfm-u25` val chọn γ = 10** (val 0,0571; γ = 3: 0,0569; γ = 1: 0,0557), tức mép trên của lưới và chênh lệch rất nhỏ.
- Nếu mọi cạnh của một node có cùng Δt thì tác dụng triệt tiêu ở node đó.

### 5.3 Dữ liệu của Last-FM cho Stage 1 điều kiện tốt hơn MovieLens
*(Các số trong mục này: số cạnh theo tuần và độ trải thời gian có trong `docs/data-report-lastfm-numbers.md` (script 07); phân bố Δt_norm và tỷ lệ cạnh trong 30, 365 ngày cuối do notebook tính từ `data/processed/lastfm-u25/train.csv`.)*

- Δt trên Last-FM nghĩa là thời gian kể từ **lần nghe đầu tiên của cặp**, không phải thời gian nhập điểm.
- Phân bố Δt_norm của 236.053 cạnh train: trung vị 0,100; tứ phân vị 0,0147 và 0,2229; phân vị 90 là 0,4316. 22,0% cạnh nằm trong 30 ngày cuối, 52,1% trong 365 ngày cuối, 4,3% trong 7 ngày cuối.
- **Phía user:** trung vị độ lệch chuẩn thời gian của một user là 249,6 ngày, trung vị độ trải là 724,2 ngày; chỉ 0,53% user có mọi cạnh train trong 1 giờ (ML-100K: 53,0%). Chỉ 5,9% cạnh nằm trong 24 giờ đầu của user (ML-100K: 77,9%).
- **Phía item:** trung vị độ lệch chuẩn thời gian theo item là 552,6 ngày (ML-100K: 45 ngày; ML-1M: 67 ngày).
- Đọc: Δt có độ phân giải thật ở **cả hai phía**, nên Stage 1 trên Last-FM được đo trong điều kiện thuận lợi hơn MovieLens rõ rệt. Điều này làm kết quả không có tác dụng (mục 9) **mạnh hơn** kết luận trên MovieLens, không yếu đi.
- Cảnh báo: Δt tính theo lần nghe đầu của các cặp *đã được chọn* (mục 2.8); 21,7% cạnh train rơi vào giai đoạn từ 01/04/2014, một phần là dấu vết của chính phép chọn cặp.

### 5.4 T-KGAT Stage 2: Δt trên triple của KG (biến thể so sánh; 0009 mục 7–10)
- Mỗi cạnh KG mang ngày từ đó fact đúng. δ_r = 1 cho quan hệ KG, 0 cho Interact (so sánh với Stage 1, không cộng dồn).
- Fact không ngày: Δt = 0. Δt âm cắt về 0. **Δt_norm của cạnh KG bị chặn ở 1.**
- **Trên Last-FM (`lastfm-u25`, KG default tại t1, 152.976 triple; `docs/data-report-lastfm.md` bảng 4):** 24,4% triple nằm trong cửa sổ (`dt_norm < 1`); 55,7% bị chặn ở 1; 19,8% không có ngày. Tính riêng trên các fact **có ngày**: 30,5% nằm trong cửa sổ, 69,5% bị chặn ở 1 (hai tỷ lệ này mình tính từ ba số trên). Trung vị `dt_norm` của fact có ngày là 1,45.
- So với MovieLens (0,7% ở ML-100K và 0,0% ở `ml-1m-u10`), Stage 2 trên Last-FM có độ phân giải **chỉ với phần fact nằm trong cửa sổ**; hơn 2/3 fact có ngày vẫn bị chặn ở 1 vì `dt_den` lấy theo cửa sổ của tương tác (lỗi thiết kế, xem ghi chú đầu tài liệu), nên đây chưa phải phép thử đầy đủ về fact cũ hay mới.
- Ngày của fact ở đây là ngày tồn tại của đích (ngày phát hành, ngày nghệ sĩ bắt đầu hoạt động, ngày sinh), **không phải lúc fact vào KG**.

### 5.5 Đối chứng hoán vị (bắt buộc)
- Stage 1: xáo `train_ts` trên **toàn bộ** cạnh train (không phải trong từng user), mỗi seed một hoán vị, giữ nguyên tập giá trị timestamp.
- Stage 2: xáo ngày của fact trên các triple của KG t1 (kể cả NaN).
- Logic: nếu bản hoán vị tốt bằng bản thật thì phần cải thiện (nếu có) không đến từ thông tin thời gian.
- Kết quả âm là kết quả hợp lệ nếu quy trình đúng (biên bản 07/08).
- Hạn chế đã biết: hoán vị toàn cục chỉ là một dạng đối chứng. Phiên bản hoán vị theo khối thời gian đã được đề xuất với cô nhưng **chưa chạy** (0014, mục Consequences).

### 5.6 Code
`src/tkgat/models/kgat.py` (`time_on`, `delta_r`), `src/tkgat/kg/ckg.py` (`build_ckg`, `permute_train_ts`, `permute_kg_ts`, `load_kg_dates`), `src/tkgat/kg/temporal.py`, configs `tkgat.yaml`, `tkgat_perm.yaml`, `tkgat_kg.yaml`, `tkgat_kg_perm.yaml`.

---

## 6. Tích hợp dữ liệu, ngày của fact và tính đúng theo thời điểm

Phần gắn đề tài với ngành Hệ thống thông tin: ghép nhiều nguồn, đo chất lượng, bảo đảm dữ liệu "đúng tại thời điểm".

### 6.1 Chuỗi tích hợp của Last-FM (khác MovieLens)
Có **ba nguồn** ghép với nhau:
1. **Bản Last-FM của tác giả KGAT:** cặp (user, item) đã remap id (3.034.763 token nghe, 1.542.856 cặp duy nhất), cùng KG `kg_final.txt` (464.567 triple, 9 quan hệ), `entity_list`, `relation_list`, `item_list`.
2. **LFM-1b** (`lfm1b-tracks.inter`, file khoảng 29 GB, hơn một tỷ sự kiện): lấy **lần nghe sớm nhất** của từng cặp, lưu cache `kgat_author_earliest.tsv`.
3. **Bản chụp Freebase ngày 09/08/2015** (dump RDF, lấy từ bản sao cộng đồng vì Google không còn cho tải; đã kiểm SHA-256): cho ngày của fact.

**Không cần bước liên kết thực thể** như MovieLens (KB4Rec, Wikidata): dùng đúng tập user/item của tác giả nên KG `kg_final.txt` khớp trực tiếp, và `entity_list` đã ghi sẵn id Freebase `m.xxxx` khớp với cột `subject` của bản chụp.

### 6.2 Làm sạch và dựng cặp
Mỗi cặp (user, item) giữ **lần nghe sớm nhất** (`kgat_author_earliest.tsv`). Với `lastfm-u25` lấy ngẫu nhiên 25% user (5.892 trong 23.566, seed 2019, quyết định 0008 và 0013).

### 6.3 Độ phủ khác độ chính xác (áp dụng cho Last-FM thế nào)
- Vì không liên kết, **không có độ chính xác liên kết phải đo** như ở MovieLens (0/50 sai); thay vào đó phải nói về **độ phủ ngày**: 78,5% trong 331.097 triple gắn với item của train có ngày ở đích (bản chưa bỏ 1.293 triple không chở được); 21,2% trong 329.804 triple pipeline giữ là không có ngày; trên `lastfm-u25`: 30.323 / 152.979 = 19,8%.
- Ngày theo quan hệ: `music.recording.releases` chỉ 1,9% không có ngày và `music.recording.artist` 1,1%; 4 quan hệ (`common.topic.notable_types`, `music.single.versions`, `music.recording.canonical_version`, `music.recording.song`) 100% không có ngày; `music.recording.engineer` 74,8% và `music.recording.producer` 27,5% không có ngày (số của `lastfm-u25`, in trong notebook phần 2 từ `docs/data-report-lastfm-numbers.md`).
- Nghĩa là drop_undated trên Last-FM chủ yếu cắt quan hệ phân loại và quan hệ phiên bản, giữ gần nguyên quan hệ bản phát hành và nghệ sĩ.

### 6.4 Ngày trong Freebase (quyết định 0006, áp dụng cho Last-FM)
- Ngày của một fact = ngày tồn tại của **đích** (quyết định 0006 D: một fact không thể đúng trước khi đích tồn tại): ngày sớm nhất trong `music.release.release_date`, `music.artist.active_start`, `people.person.date_of_birth`.
- Ba dạng literal: `gYear` là ngày cuối năm; `gYearMonth` là ngày cuối tháng; `date` là đúng ngày, **trừ** ngày kết thúc bằng `-01-01` (đọc là chỉ có năm, nên thành 31/12). Năm trước Công nguyên và giá trị không đọc được coi như không có ngày.
- Head là item của train nên ngày phát hành của chính nó không bao giờ loại fact của nó.
- Hướng sai của quy tắc luôn an toàn (đặt ngày muộn hơn thực tế), không gây rò rỉ.

### 6.5 KG đúng theo thời điểm (quyết định 0004) và vì sao trên Last-FM nó "gần như rỗng"
- Quy tắc như MovieLens: fact có ngày giữ nếu ngày < mốc; fact trỏ tới thực thể ra đời sau mốc bỏ; fact không ngày giữ nếu hai đầu đã tồn tại (biến thể `default`); `drop_undated` bỏ fact không ngày; `static` không lọc (tham chiếu).
- **Kết quả đo:** static − default chỉ là 3 triple ở `lastfm-u25` (15 ở `lastfm` đầy đủ), tức 0,0%; đó là các bản phát hành có ngày cuối năm 2014, sau cả hai mốc. KG t1 = KG t2; số thực thể chỉ có ở t2 là 0. MovieLens bị cắt 1,3% (ML-100K) và 1,5% (ML-1M).
- **Hệ quả:** Last-FM **không đo được** hiện tượng "fact tương lai rơi vào giữa hai mốc" (ví dụ *Titanic* của ML-100K); RQ về rò rỉ qua KG vẫn phải trả lời bằng MovieLens. Đây là kết quả của dữ liệu, không phải lỗi.

### 6.6 Kích thước KG của `lastfm-u25`
KG gốc 464.567 triple → lọc theo item 154.076 → chở được trong 1-hop 152.979 (bỏ 1.097, 0,7%); 44.392 thực thể (25.727 thực thể đích), 9 quan hệ. KG default tại t1: 152.976 triple; `drop_undated` 122.653; `static` 152.979. Đồ thị huấn luyện: 49.713 node, 778.058 cạnh (không KG: 23.986 node, 472.106 cạnh).

### 6.7 Giới hạn phải nói
1. Freebase là bản chụp 09/08/2015, không phải tri thức có năm 2014.
2. KG không có ngày của statement; chỉ có ngày tồn tại của đích.
3. Ngày `-01-01` đọc là chỉ có năm (đặt muộn hơn thực tế, an toàn).
4. 1.097 triple không chở được vào định dạng 1-hop.
5. KG t1 = KG t2 nên không kiểm được đường rò rỉ giữa hai mốc; bộ lọc thời gian gần như không cắt gì.
6. Thiên lệch chọn mẫu của tập cặp (mục 2.8).
7. Khoảng thời gian của train, val, test lệch nhau rất nhiều (train 02/2005–04/2014, val khoảng 39 ngày, test khoảng 86 ngày đến 02/09/2014): một mô hình có thể được lợi từ tính mới theo cách MovieLens không cho phép; đây là đặc điểm của dữ liệu.

### 6.8 Code
`src/tkgat/kg/lastfm.py`, `src/tkgat/kg/temporal.py`, `scripts/06_build_kg.py`, `scripts/07_report_numbers_lastfm.py`, `scripts/generate_lastfm_temporal_kgat.py`, `tests/test_data_report_lastfm.py`.

---

## 7. Kiểm định thống kê

### 7.1 Hai nguồn biến thiên
- **Ngẫu nhiên khi huấn luyện:** 5 seed (2019–2023); báo cáo trung bình ± **độ lệch chuẩn mẫu** (ddof = 1).
- **Mẫu user:** kiểm định cặp theo user.

### 7.2 Kiểm định cặp và Wilcoxon
- Hai mô hình chấm trên cùng user ⇒ xét hiệu d_u = m_A(u) − m_B(u).
- **Wilcoxon signed-rank** (mặc định): xếp hạng |d_u|, cộng hạng của d_u > 0; không cần giả định chuẩn. User có d_u = 0 bị bỏ ⇒ báo cả `n_pairs` và `n_nonzero`.
- Điểm theo user = trung bình qua các seed chung rồi mới so cặp.
- Cảnh báo khi dưới 100 user hoặc dưới 30 chênh lệch khác 0: ngưỡng **tự đặt**. Với `lastfm-u25`, `n_pairs` = 1.608 và `n_nonzero` từ 651 đến 929 nên **không có cảnh báo nào**.

### 7.3 So sánh nhiều lần và Holm
- m kiểm định ở mức 0,05 ⇒ xác suất có ít nhất một dương tính giả là 1 − 0,95^m.
- Holm–Bonferroni: sắp p tăng dần; bác bỏ H_(k) nếu p_(j) ≤ α/(m − j + 1) với mọi j ≤ k.
- Họ Holm trong repo: cùng bộ dữ liệu, cùng metric. Bảng baseline 3 cặp; bảng RQ3 3 phép cố định (T-KGAT vs KGAT, T-KGAT thật vs hoán vị, KGAT vs không KG); Stage 2 họ riêng 2 phép.
- **Chưa có trong họ:** BPR-MF so với KGAT. Chênh 0,0013 Recall@20 (nhỏ hơn độ lệch chuẩn 0,0028 của BPR-MF) nhưng **chưa kiểm định**; không được nói "KGAT không hơn BPR-MF có ý nghĩa" mà phải nói "chưa kiểm định".

### 7.4 Đọc kết quả đúng cách
- **Khác MovieLens:** p lớn với 1.608 user là bằng chứng mạnh hơn nhiều rằng chênh lệch nhỏ, vì kiểm định có sức mạnh. Nhưng vẫn **không chứng minh** hai mô hình bằng nhau; chênh lệch nhỏ hơn độ nhiễu seed thì không phân biệt được.
- Không chọn seed đẹp nhất; báo cáo kết quả âm kèm giải thích.
- Nhiều dataset: Demšar (2006) khuyên Friedman + Nemenyi; nhóm có 2–3 bộ nên báo riêng từng bộ.

### 7.5 Code
`src/tkgat/eval/stats.py`, `src/tkgat/analysis/significance.py`, `scripts/09_significance.py`, `src/tkgat/analysis/rq3.py`, `scripts/10_rq3_table.py`.

---

## 8. Quy trình thực nghiệm, tính toán và quản lý kết quả

- **Một lệnh cho mỗi thí nghiệm:** `scripts/run.py --data <config> --model <config>` chạy mọi điểm lưới × 5 seed, chọn điểm theo trung bình Recall@20 val, rồi chấm test **một lần**. `--val-only` không mở file test. `--resume` dùng lại checkpoint.
- **Registry chỉ thêm dòng** (`results/registry.csv`): thời gian, git commit, dataset, hash split, model, config, seed, phase, epoch, chỉ số, device. Dòng `-dirty` bị loại khỏi bảng chính thức. **`lastfm-u25` có 139 dòng, tất cả ở commit `ce55c0c`, không dòng nào `-dirty`.**
- **Không số nào gõ tay:** báo cáo, bảng đều đọc từ registry.
- **Decision record** (`docs/decisions/NNNN-*.md`): quyết định đã accepted không sửa, chỉ thay thế bằng quyết định mới.
- **Tính toán trên Last-FM (0013, 0014):** KGAT trên `lastfm-u25` mất 20,13 và 20,52 giây mỗi epoch trên laptop (RTX 3050 4 GB, đo 06/11); bản đầy đủ `lastfm` cần khoảng 357 giờ chỉ cho lưới KGAT nên không chạy được. Ước lượng cận trên không dừng sớm cho `lastfm-u25`: 121,7 giờ. **Thực tế (tổng cột `seconds` của các dòng val trong registry): laptop khoảng 59,1 giờ, Kaggle Tesla T4 khoảng 16,2 giờ.** KGAT riêng 23,2 giờ trên laptop.
- **Stage 2 chạy trên Kaggle** (0014): `tkgat_kg` và `tkgat_kg_perm` chạy trên Tesla T4, cùng commit `ce55c0c`, cùng split (hash khớp laptop). Cổng đo là 14 giây mỗi epoch; đo được 13,81 giây. **Cổng ban đầu viết là 12 giây; sau khi thấy số đo, cổng được nâng lên 14 giây theo yêu cầu của Triết**: đây là thay đổi có chủ đích, không phải lần qua cổng đăng ký trước. Ước lượng ban đầu cho 15 lần chạy `tkgat_kg` là khoảng 9,9 giờ, thực tế khoảng 12,3 giờ; phiên 1 chạm trần 12 giờ sau 14/15 lần, phiên 2 (`--resume`) chạy nốt lần còn lại, phiên 3 chạy `tkgat_kg_perm` (5 lần).
- **Hệ quả:** so sánh Stage 2 với KGAT hoặc Stage 1 là so sánh giữa **hai máy** (T4 và RTX 3050); so sánh `tkgat_kg` với `tkgat_kg_perm` là cùng một máy.
- **Thiết bị:** mọi model và trainer nhận `device`; `seed_all(seed)` đầu mỗi lần chạy.
- **Dữ liệu không vào git** (giấy phép).
- **Bản `lastfm-u25-latest` (PR #44):** chưa có dòng nào trong `results/registry.csv` (chỉ thêm dòng, cần quyết định riêng). Số của nó lấy từ registry gộp của năm zip kết quả của Phôn (323 dòng, 132 dòng riêng biệt, commit `873ca7f` của lịch sử riêng, 0 dòng `-dirty`) và được tóm tắt trong `results/rq3_*_lastfm-latest.csv`; nguồn gốc ở `docs/lastfm-latest-provenance.md`. Code xuất của Phôn và quyết định 0015 (nháp, chưa accepted) chưa vào repo.
- **Tag:** `v0.1-review` (commit `0e194e3`) đóng băng dữ liệu, KG, baseline MovieLens. **Chưa có `v0.2-kgat`**; các số Last-FM vẫn là sơ bộ, chưa tái lập độc lập.

**Các quyết định (đọc tên là phải nhớ nội dung):**

| # | Nội dung |
|---|---|
| 0001 | Làm lại từ đầu vì kết quả cũ rò rỉ |
| 0002 | Chia theo mốc thời gian toàn cục, val/test chỉ user/item đã biết |
| 0003 | Liên kết thực thể ML-100K → ML-1M → Freebase |
| 0004 | Luật KG theo thời gian |
| 0005 | Thời gian trong attention |
| 0006 | Xử lý ngày: năm ⇒ 31/12, tháng ⇒ cuối tháng |
| 0007 | Lưới cửa sổ MostPop recent {1, 3, 5, 7, 14, 30} |
| 0008 | Tập con user, xuất định dạng KGAT gốc, giả định timestamp |
| 0009 | T-KGAT: công thức Δt, t_ref, chuẩn hóa, lưới γ, hoán vị, Stage 2 |
| 0010 | Độ nhạy γ chỉ trên val |
| 0011 | Thêm metadata MovieLens vào KG, chạy lại mọi mô hình dùng KG |
| **0012** | **Last-FM là dataset thứ ba: cặp của KGAT + timestamp LFM-1b + ngày Freebase 2015; hạn chế của tập cặp** |
| **0013** | **Lần chạy chính thức trên `lastfm-u25`; cổng đo thời gian; Stage 2 bỏ đầu tiên nếu hết giờ** |
| **0014** | **Stage 2 chạy trên Kaggle; cổng 14 giây; kết quả thực tế** |
| *0015* | *(Nháp của Phôn, **chưa accepted, chưa vào repo**) bản `lastfm-u25-latest`: timestamp là lần nghe muộn nhất* |

---

## 9. Kết quả hiện tại trên `lastfm-u25` (sơ bộ)

Recall@20 trên test (1.608 user), trung bình ± độ lệch chuẩn mẫu qua 5 seed. Nguồn `docs/rq3-table.md`.

| Mô hình | Điểm được chọn trên val | Recall@20 | NDCG@20 |
|---|---|---|---|
| MostPop | không tham số | 0,0238 | 0,0191 |
| MostPop recent | cửa sổ 14 ngày | 0,0211 | 0,0169 |
| BPR-MF | dim 64, l2 1e-3, lr 1e-3 | 0,0520 ± 0,0028 | 0,0435 ± 0,0030 |
| KGAT | lr 1e-3, l2 1e-5 | 0,0534 ± 0,0008 | 0,0444 ± 0,0008 |
| KGAT không KG | lr 1e-3, l2 1e-5 | 0,0518 ± 0,0016 | 0,0426 ± 0,0004 |
| T-KGAT Stage 1 | γ = 10 | 0,0529 ± 0,0025 | 0,0431 ± 0,0016 |
| T-KGAT Stage 1 hoán vị | γ = 10 | 0,0534 ± 0,0024 | 0,0441 ± 0,0017 |
| T-KGAT Stage 2 (Δt trên KG) | γ = 1 | 0,0513 ± 0,0018 | 0,0426 ± 0,0014 |
| Stage 2 hoán vị ngày fact | γ = 1 | 0,0537 ± 0,0020 | 0,0436 ± 0,0016 |

p Holm (Wilcoxon cặp theo user, 1.608 user), Recall@20 / NDCG@20, hiệu = trước − sau:
- T-KGAT vs KGAT: 0,420 / 0,230 (hiệu −0,0005 / −0,0012).
- T-KGAT thật vs hoán vị: 0,803 / 0,506 (hiệu −0,0005 / −0,0009).
- KGAT vs KGAT không KG: 0,803 / 0,506 (hiệu +0,0016 / +0,0017).
- Stage 2 vs KGAT: 0,278 / 0,523 (hiệu −0,0021 / −0,0018).
- Stage 2 thật vs hoán vị ngày fact: 0,278 / 0,442 (hiệu −0,0024 / −0,0010).
- Baseline: BPR-MF > MostPop và BPR-MF > MostPop recent, đều có ý nghĩa mạnh (p Holm nhỏ hơn 10⁻³⁵ ở Recall); MostPop vs MostPop recent: Recall p Holm 0,103, NDCG p Holm 0,016 (MostPop thường **cao hơn**).

**Đọc:**
1. **Baseline:** BPR-MF gần gấp đôi MostPop (0,0520 so với 0,0238) và thắng có ý nghĩa. MostPop recent (cửa sổ 14 ngày chọn trên val: 0,0247 so với 0,0192 của MostPop) **không** thắng MostPop thường ở test (0,0211 so với 0,0238). Chiều ngược với ML-100K. Giải thích chỉ là giả thuyết, chưa kiểm: cửa sổ 14 ngày neo ở cuối train, còn test bắt đầu ở t2 (khoảng 39 ngày sau t1) và kéo dài khoảng 86 ngày; độ "hot" của 14 ngày cuối train có thể đã nguội lúc test.
2. **KG (RQ2):** KGAT hơn KGAT không KG 0,0016 nhưng không có ý nghĩa (p Holm 0,803 / 0,506). Chưa có bằng chứng KG giúp trên Last-FM. KGAT (0,0534) nhỉnh hơn BPR-MF (0,0520) 0,0013, nhưng chênh này nhỏ hơn độ lệch chuẩn qua seed của BPR-MF (0,0028) và **chưa kiểm định**.
3. **Thời gian (RQ3):** T-KGAT Stage 1 không khác KGAT và không khác bản hoán vị; bản hoán vị thậm chí nhỉnh hơn 0,0005. Chưa có bằng chứng thời gian giúp. Stage 2 cũng không: thấp hơn KGAT 0,0021 và thấp hơn bản hoán vị 0,0024, không có ý nghĩa.
4. **Mức độ chắc chắn:** khác MovieLens, mẫu test 1.608 user, `n_nonzero` từ 651 đến 929; kiểm định đủ sức để nói chênh lệch (nếu có) nhỏ, cỡ vài phần nghìn Recall@20, nhỏ hơn độ nhiễu qua seed (độ lệch chuẩn 0,0008 đến 0,0028). Đây vẫn không phải bằng chứng "bằng nhau".
5. **Val và epoch:** val chọn γ = 10 (Stage 1) và γ = 1 (Stage 2) đều ở mép lưới; chênh lệch val giữa các γ chỉ 0,0002 đến 0,0014. Một lần chạy `tkgat_perm` (seed 2019) chạm trần 300 epoch nên chưa chắc đã hội tụ.
6. **Hình:** `docs/figures/rq3_curves_lastfm-u25.png` (Recall@20 val theo epoch; ba đường KGAT, T-KGAT, hoán vị chồng lên nhau, phẳng dần sau khoảng epoch 100); `docs/figures/fig5_bprmf_curve_lastfm-u25.png` (BPR-MF: loss và val).

**Khác biệt về chất lượng bằng chứng so với MovieLens:** Last-FM cho giá trị lớn nhất ở chỗ val/test đủ lớn để kết luận âm có trọng lượng. Điều này hợp với mục tiêu của đề tài (quy trình không rò rỉ + kết luận trung thực).

---

## 10. Từ điển thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Implicit feedback | Phản hồi ngầm: chỉ biết có tương tác (một lần nghe) |
| Top-K, full ranking | Xếp hạng toàn bộ item, lấy K đầu |
| Positive / negative | Tương tác dương (mọi cặp trên Last-FM) / item lấy mẫu làm âm |
| k-core | Giữ user và item có ít nhất k tương tác (lặp đến ổn định), tính trên train |
| Cold-start (cold user, cold item) | User/item mới chưa có trong train; trên `lastfm-u25` cold item là 28% positive val/test |
| Mask | Item đã tương tác bị loại khỏi danh sách xếp hạng |
| Recall@K, NDCG@K | Tỷ lệ đáp án lọt top-K; như trên nhưng thưởng thứ hạng cao |
| Early stopping, patience | Dừng khi val không tăng sau một số lần đánh giá |
| Seed | Hạt giống ngẫu nhiên; 5 seed để đo dao động khi huấn luyện |
| Triple (h, r, t) | Một fact trong KG |
| CKG | Đồ thị gộp tương tác và KG |
| TransE / TransR | KGE tịnh tiến / tịnh tiến sau chiếu theo quan hệ |
| Attention, softmax | Trọng số cho hàng xóm, chuẩn hóa tổng bằng 1 |
| Bi-Interaction | Aggregator tổng + nhân từng phần tử |
| Δt, t_ref, dt_den | Độ cũ của cạnh, mốc tham chiếu (max train), độ dài cửa sổ train (9,20 năm ở `lastfm-u25`) |
| γ, δ_r | Cường độ suy giảm; 1 nếu quan hệ mang thời gian, 0 nếu không |
| Permutation control | Đối chứng hoán vị timestamp (Stage 1) hoặc ngày fact (Stage 2) |
| Point-in-time correctness | Chỉ dùng dữ liệu đã biết tại thời điểm dự đoán |
| LFM-1b | Bộ dữ liệu nghe nhạc Last.fm có timestamp từng sự kiện; nguồn của thời gian |
| Bản Last-FM của KGAT | Tập (user, item) và KG mà tác giả KGAT công bố, không có timestamp |
| Lần nghe sớm nhất | Quy tắc gán timestamp cho một cặp (user, item) |
| Freebase dump 09/08/2015 | Nguồn ngày của fact Last-FM |
| `lastfm-u25` | 25% user ngẫu nhiên (seed 2019) của Last-FM; tập chính thức |
| Thiên lệch chọn mẫu của tập cặp | Tập cặp của KGAT chỉ chứa cặp còn được nghe từ 01/04/2014 (dùng thông tin sau mốc) |
| Kaggle T4 | GPU của phiên Kaggle chạy Stage 2 (khác laptop RTX 3050); với `lastfm-u25-latest` mọi mô hình học sâu đều chạy trên T4 |
| `lastfm-u25-latest` | Cùng user, t1/t2 theo phân vị, 5-core, KG như `lastfm-u25`; timestamp của cặp là lần nghe **muộn nhất**; của Phôn |
| Lần nghe muộn nhất | Quy tắc gán timestamp của bản `latest` |
| Vấn đề A, B, C | A: tập cặp chọn bằng thông tin sau 01/04/2014; B: bước nhảy số cạnh quanh 01/04/2014; C: `check_split` không thấy A và B |
| Registry gộp (zip) | Registry của năm zip kết quả của Phôn; 323 dòng, 132 dòng riêng biệt; không nằm trong `results/registry.csv` |
| Wilson interval | Khoảng tin cậy cho một tỷ lệ |
| Wilcoxon signed-rank | Kiểm định cặp phi tham số |
| Holm–Bonferroni | Hiệu chỉnh nhiều phép so sánh |
| Registry | File kết quả chỉ thêm dòng |
| Decision record | File ghi quyết định giao thức |
| `-dirty` | Dòng registry chạy khi code chưa commit sạch |

---

## 11. 42 câu tự kiểm tra tổng hợp (có đáp án ngắn; câu 31 đến 42 về bản `lastfm-u25-latest`)

1. Vì sao Last-FM không cần đổi rating thành dương? — Mỗi dòng là một lần nghe, vốn là phản hồi ngầm; `min_rating: 1`.
2. Timestamp của một cặp (user, item) lấy thế nào? — Lần nghe sớm nhất của cặp, từ LFM-1b.
3. Δt trên Last-FM nghĩa là gì? — Thời gian kể từ lần nghe đầu tiên của cặp đến `t_ref` (cuối train).
4. Vì sao không dùng (r − ŷ)²? — Không có nhãn âm thật.
5. Item âm của BPR lấy từ đâu? — Item user chưa tương tác trên train.
6. Vì sao MostPop recent là baseline bắt buộc, và nó thắng không? — Là baseline thời gian đơn giản nhất; trên `lastfm-u25` nó không thắng MostPop thường ở test (0,0211 so với 0,0238).
7. Ba lỗi rò rỉ của kết quả cũ? — Chọn epoch trên test; k-core trên toàn bộ; KG từ metadata hiện tại.
8. Vì sao chia theo mốc toàn cục? — Theo từng user thì tương lai của user khác lọt vào train.
9. Vì sao Last-FM đầy đủ giữ 92,1% val còn ML-100K chỉ 12,6%? — Timestamp Last-FM rải đều; MovieLens dồn vào phiên đầu của user nên phần sau t1 chủ yếu là user mới.
10. Vì sao `lastfm-u25` chỉ giữ 63,9% val và 60,4% test? — Lấy 25% user làm rơi nhiều item khỏi train: 28,8% và 28,1% positive bị loại vì cold item.
11. Tại sao chọn `lastfm-u25` mà không chạy `lastfm` đầy đủ? — Lưới KGAT đầy đủ khoảng 357 giờ; ước lượng cận trên cho u25 là 121,7 giờ (0013).
12. Mask test gồm gì? — Mọi tương tác trước t2.
13. Kể 6 kiểm tra của `check_split`. — Mục 2.6.
14. Vì sao không dùng sampled metrics? — Có thể đảo thứ tự mô hình.
15. Tính NDCG@3 cho [4, 7, 1, 9], đáp án {7, 9}. — ≈ 0,387.
16. TransR sửa điểm yếu nào của TransE? — Quan hệ 1-N, nhờ chiếu W_r.
17. CKG gồm những cạnh nào? — Interact (chỉ train), triple KG và các cạnh ngược.
18. Vì sao KGAT không tái lập từng bit trên GPU? — `torch.sparse.mm` không tất định.
19. T-KGAT khác KGAT chỗ nào? — Trừ γ·δ_r·Δt_norm khỏi π trước softmax, chỉ trên cạnh Interact (Stage 1); γ = 0 là KGAT.
20. Hoán vị để làm gì? — Nếu bản hoán vị tốt bằng bản thật thì cải thiện không đến từ thời gian.
21. Ngày của một fact trên Last-FM lấy ở đâu? — Ngày tồn tại của đích, từ Freebase 09/08/2015 (ngày phát hành, bắt đầu hoạt động, ngày sinh).
22. Vì sao bộ lọc thời gian của KG gần như không cắt gì ở Last-FM? — Chỉ 3 triple (15 trên bản đầy đủ) có ngày sau t1; KG t1 = KG t2.
23. Stage 2 trên Last-FM khác MovieLens thế nào? — 24,4% triple trong cửa sổ Δt (MovieLens 0,7% và 0,0%), nên có độ phân giải một phần; 69,5% fact có ngày vẫn bị chặn ở 1 do chuẩn hóa theo cửa sổ tương tác (lỗi thiết kế, Stage 1 không bị).
24. Thiên lệch chọn mẫu của tập cặp là gì? — Tập cặp của KGAT chỉ có cặp còn được nghe từ 01/04/2014, chọn bằng thông tin sau mốc; tác động chưa đo.
25. Vì sao kết quả KGAT/T-KGAT là "sơ bộ"? — Chưa có tag `v0.2-kgat`, chưa tái lập độc lập, PR #42 còn draft; bảng RQ3 ghi rõ "chạy trước khi cô duyệt giao thức t1/t2".
26. Kết quả RQ3 trên `lastfm-u25`? — T-KGAT không khác KGAT (p Holm 0,420 / 0,230) và không khác hoán vị (0,803 / 0,506); Stage 2 cũng không.
27. Vì sao p lớn ở Last-FM đáng tin hơn ở MovieLens? — 1.608 user và 651–929 chênh lệch khác 0, so với 51–95 user.
28. Vì sao vẫn không nói "thời gian không giúp gì"? — p lớn không chứng minh bằng nhau; chỉ nói "chưa có bằng chứng", và hoán vị toàn cục mới là một dạng đối chứng.
29. Vì sao so Stage 2 với KGAT phải kèm ghi chú? — Stage 2 chạy trên Kaggle T4, KGAT trên laptop RTX 3050 (0014).
30. Cổng 14 giây của Kaggle có phải đăng ký trước không? — Không: cổng gốc 12 giây, đo được 13,81 giây, rồi nâng lên 14 giây theo yêu cầu của Triết; phải nói rõ là thay đổi có chủ đích.
31. Bản `lastfm-u25-latest` khác `lastfm-u25` ở đâu? — Chỉ timestamp của cặp: lần nghe muộn nhất thay vì sớm nhất; cùng user, cùng cách đặt t1/t2, 5-core, KG, lưới, seed (mục 12.1).
32. Ba vấn đề A, B, C của `lastfm-u25` là gì? — A: tập cặp chọn bằng thông tin sau 01/04/2014; B: số cạnh train nhảy lên quanh 01/04/2014; C: `check_split` không thấy được A và B (mục 12.1).
33. Bản latest sửa được vấn đề nào, và không sửa được vấn đề nào? — Hết bước nhảy B (mọi timestamp từ 01/04/2014); A còn nguyên vì tập cặp không đổi; C vẫn đúng (`check_split` báo 0 vấn đề) (mục 12.2).
34. Với timestamp là lần nghe muộn nhất, một cặp nằm ở train khi nào? — Khi và chỉ khi trong dữ liệu LFM-1b user không nghe cặp đó từ t1 trở đi (trước khi lọc k-core và cold-start). Cặp còn được nghe sau t1 chỉ có thể ở val hoặc test (mục 12.2; suy từ định nghĩa, chưa đo ảnh hưởng).
35. Cửa sổ train của latest dài bao nhiêu và hệ quả cho Δt? — 83 ngày (từ 01/04/2014 đến 23/06/2014), so với 3.361 ngày của `lastfm-u25`; một đơn vị `dt_norm` là 83 ngày, nên cùng γ không cùng nghĩa theo ngày trên hai bản (mục 12.4).
36. Vì sao Stage 2 trên latest không có độ phân giải? — Fact có ngày mới nhất là 31/12/2013, sớm hơn ngày đầu cửa sổ train 91 ngày; nên 0,0% triple trong cửa sổ, 80,1% bị chặn ở 1, 19,9% không có ngày. Mọi fact có ngày có cùng `dt_norm` = 1 (mục 12.5).
37. Hệ quả cho bản hoán vị ngày fact (Stage 2) trên latest? — Vì mọi fact có ngày có cùng `dt_norm`, hoán vị chỉ xáo xem triple nào là "không có ngày" (suy từ phép chặn ở 1) (mục 12.5).
38. Trên latest, "T-KGAT thật hơn hoán vị" (p Holm 0,033 và 0,0046) có chứng minh thời gian giúp không? — Không: T-KGAT không hơn KGAT (p Holm 0,733 và 0,910) và bản hoán vị thấp hơn KGAT 0,0021; khoảng cách thật so với hoán vị có thể chỉ do xáo timestamp làm mô hình tệ đi (mục 12.6).
39. KGAT so với KGAT không KG trên latest khác `lastfm-u25` thế nào? — Trên latest Recall@20 có ý nghĩa sau Holm (hiệu 0,0040, p Holm 0,028), NDCG@20 không (hiệu 0,0013, p Holm 0,225); trên `lastfm-u25` cả hai không có ý nghĩa. Sơ bộ, trên dữ liệu có thiên lệch chọn mẫu (mục 12.6).
40. Vì sao không so điểm tuyệt đối giữa hai bản? — Test khác nhau (1.341 so với 1.608 user; test 45 so với 86 ngày; cold item 21,9% so với 28,1%) và phép chia khác (train 83 so với 3.361 ngày) (mục 12.3).
41. Vì sao số của latest chưa tái lập được từ repo? — Code xuất và registry của Phôn có commit `873ca7f` của lịch sử riêng, quyết định 0015 là nháp; repo chỉ chứa bảng tóm tắt và nguồn gốc (mục 12.8).
42. Vì sao so Stage 2 với KGAT trên latest không vướng "hai máy"? — Mọi mô hình học sâu của bản latest chạy trên Tesla T4 (tổng 55,1 giờ ở dòng val); MostPop chạy trên CPU (mục 12.8).

---

## 12. Bản `lastfm-u25-latest` (của Phôn): timestamp là lần nghe muộn nhất

*Mục này mới, cho buổi 14/11. Phôn trình bày. Mọi số của bản này đọc từ `docs/data-report-lastfm-latest-numbers.md`, `results/rq3_models_lastfm-latest.csv`, `results/rq3_pairs_lastfm-latest.csv`, `results/lastfm-latest-provenance.json` hoặc được notebook tính từ `data/processed/lastfm-u25-latest/`. Kết quả sơ bộ.*

### 12.1 Bản latest là gì và vì sao có
- **Giữ nguyên:** 25% user (cùng seed 2019), phân vị 0,8 / 0,9 cho t1/t2, 5-core chỉ trên train, quy trình KG, lưới, 5 seed, `check_split`. Cùng 370.979 cặp.
- **Đổi một thứ:** timestamp của cặp (user, item) là **lần nghe muộn nhất** (không phải sớm nhất) trong LFM-1b. Chính sách ghi trong `meta.json` của bản này là `timestamp_policy: latest`.
- **Vì sao:** `lastfm-u25` có ba vấn đề (mục 2.8 và 12.2): **A** tập cặp chọn bằng thông tin sau 01/04/2014; **B** số cạnh train nhảy lên quanh 01/04/2014 (từ khoảng 2.494 lên khoảng 12.416 cạnh mỗi tuần, gấp 5,0 lần; 21,7% cạnh train có timestamp từ 01/04/2014); **C** `check_split` báo 0 vấn đề nên không thấy A và B.
- **Tình trạng:** quyết định 0015 (nháp của Phôn) chưa accepted và chưa vào repo; code xuất và registry của bản này nằm ngoài repo. Phần repo có: bảng tóm tắt và nguồn gốc (PR #44).

### 12.2 Hai quy tắc timestamp: lý thuyết
- **Sớm nhất:** Δt đo từ lần nghe đầu của cặp. Cặp nằm ở train nếu lần nghe đầu < t1. Một cặp được nghe đều đặn suốt nhiều tháng có thể có timestamp từ rất lâu (u25: cửa sổ train gần chín năm, từ 14/02/2005).
- **Muộn nhất:** Δt đo từ lần nghe cuối. Vì tập cặp của KGAT chỉ chứa cặp còn được nghe từ 01/04/2014 (99,97% trên mẫu kiểm của quyết định 0012), mọi timestamp đều từ 01/04/2014 trở đi: cạnh train của bản latest đều từ 01/04/2014, bước nhảy B biến mất, train bắt đầu đúng 01/04/2014.
- **Ví dụ (số giả định để hiểu):** cặp (A, X) được nghe vào 10/03, 05/04 và 20/07. Theo quy tắc sớm nhất, timestamp là 10/03 (thuộc train nếu t1 sau 10/03). Theo quy tắc muộn nhất, timestamp là 20/07 (thuộc val hoặc test nếu t1 và t2 trước đó). Cùng một cặp, hai phần khác nhau của phép chia.
- **Hệ quả logic (suy từ định nghĩa, không phải số đo):** với quy tắc muộn nhất, một cặp (trước khi lọc k-core và cold-start) nằm trong train **khi và chỉ khi** user không nghe cặp đó từ t1 trở đi. Train chỉ chứa các cặp mà việc nghe đã dừng trong dữ liệu có sẵn. Biết cặp nào còn được nghe sau t1 là thông tin sau mốc t1.
- **Kết luận cho A, B, C:** B hết; **A còn nguyên** (tập cặp không đổi); hình thức thiên lệch **đổi chứ không mất** (thêm ở việc cặp rơi vào phần nào của phép chia); C vẫn đúng: `check_split` báo 0 vấn đề trên cả hai bản. Tác động lên kết quả **chưa đo**.

### 12.3 So sánh hai bản (bảng của notebook, mục 6.2)

| | `lastfm-u25` | `lastfm-u25-latest` |
|---|---|---|
| Cặp (user, item) | 370.979 | 370.979 |
| Timestamp của cặp | lần nghe sớm nhất | lần nghe muộn nhất |
| t1 / t2 | 29/04/2014 / 07/06/2014 | 23/06/2014 / 19/07/2014 |
| User / item train | 5.321 / 18.665 | 5.367 / 19.165 |
| Positive train | 236.053 | 236.339 |
| Positive val (user) | 23.708 (2.319) | 25.860 (1.516) |
| Positive test (user) | 22.419 (1.608) | 24.244 (1.341) |
| Val / test mất vì cold item | 28,8% / 28,1% | 24,2% / 21,9% |
| Val / test mất vì cold user | 7,3% / 11,5% | 6,1% / 12,8% |
| Positive bị 5-core loại khỏi train | 60.730 | 60.444 |
| KG default tại t1 (triple) | 152.976 | 156.826 |
| KG: thực thể / % triple không ngày | 44.392 / 19,8% | 45.346 / 19,9% |
| Train: từ ngày – đến ngày | 14/02/2005 – 29/04/2014 | 01/04/2014 – 23/06/2014 |
| Train / val / test (ngày) | 3.361 / 39 / 86 | 83 / 26 / 45 |
| Cửa sổ train theo Δt (năm) | 9,20 | 0,23 |

Không trình bày chỉ tiêu "phần val/test còn lại sau lọc" của bản latest: giá trị 69,7% (val) trùng với một con số khác đã nói là không tái lập (xem tài liệu 13, mục A); notebook chỉ hiển thị phần mất vì cold item và cold user.

### 12.4 Stage 1 trên latest
- Δt_norm = (t_ref − ts) / dt_den với t_ref là timestamp lớn nhất của train, dt_den là độ dài cửa sổ train. **Latest: dt_den = 83 ngày; u25: 3.361 ngày.**
- Trung vị `dt_norm`: 0,73 (u25: 0,10). `dt_norm` < 0,25: 10,2% (u25: 78,8%). Cạnh trong 7 ngày cuối: 3,3% (u25: 4,3%); trong 30 ngày cuối: 16,2% (u25: 22,0%).
- Phía user: 0,93% user có mọi cạnh train trong 1 giờ (u25: 0,53%); trung vị độ lệch chuẩn thời gian theo user 9,2 ngày (u25: 249,6 ngày).
- Số cạnh theo tuần: tuần đầy đủ đầu tiên 39.197 cạnh, tuần đầy đủ cuối cùng 7.717 (giảm dần về cuối cửa sổ; nguyên nhân chưa kiểm).
- **Đọc:** Δt vẫn có biến thiên, nhưng trên cửa sổ 83 ngày; một đơn vị `dt_norm` là 83 ngày (u25: 3.361 ngày), nên cùng γ trong lưới {1, 3, 10} không cùng nghĩa theo ngày trên hai bản.

### 12.5 Stage 2 trên latest: không có độ phân giải
- Ngày của fact không đổi giữa hai bản (ngày tồn tại của đích, Freebase 09/08/2015). Cái đổi là cửa sổ train: `dt_norm = (t_ref − ngày fact) / dt_den` bị chặn ở 1; fact chỉ nằm trong cửa sổ nếu ngày của nó sau ngày đầu của train.
- **Số đo:** 0,0% triple trong cửa sổ, 80,1% bị chặn ở 1, 19,9% không có ngày (u25: 24,4% / 55,7% / 19,8%). Trung vị `dt_norm` của fact có ngày 59,53 (u25: 1,45).
- Fact có ngày mới nhất: 31/12/2013, **sớm hơn ngày đầu cửa sổ train** (01/04/2014) **91 ngày**.
- **Hệ quả (suy từ phép chặn ở 1, không phải số đo):** mọi fact có ngày có cùng `dt_norm` = 1; Stage 2 chỉ phân biệt "có ngày" với "không có ngày", không phân biệt cũ với mới; bản hoán vị ngày fact chỉ xáo xem triple nào là "không có ngày".

### 12.6 Kết quả (sơ bộ), Recall@20 / NDCG@20 trên test, 1.341 user, trung bình ± độ lệch chuẩn mẫu qua 5 seed

| Mô hình | Điểm chọn trên val | Recall@20 | NDCG@20 |
|---|---|---|---|
| MostPop | không tham số | 0,0296 | 0,0268 |
| MostPop recent | *chưa chạy trên bản này* | — | — |
| BPR-MF | dim 64, l2 1e-3, lr 1e-3 | 0,0609 ± 0,0051 | 0,0618 ± 0,0054 |
| KGAT | lr 1e-3, l2 1e-4 | 0,0673 ± 0,0029 | 0,0672 ± 0,0022 |
| KGAT không KG | lr 1e-3, l2 1e-4 | 0,0633 ± 0,0046 | 0,0658 ± 0,0044 |
| T-KGAT Stage 1 | γ = 1 (mép dưới) | 0,0676 ± 0,0037 | 0,0669 ± 0,0031 |
| T-KGAT Stage 1 hoán vị | γ = 1 | 0,0651 ± 0,0049 | 0,0645 ± 0,0034 |
| T-KGAT Stage 2 | γ = 10 (mép trên) | 0,0661 ± 0,0018 | 0,0653 ± 0,0015 |
| Stage 2 hoán vị ngày fact | γ = 10 | 0,0650 ± 0,0033 | 0,0645 ± 0,0017 |

p Holm (Wilcoxon cặp theo user, 1.341 user, `n_nonzero` từ 643 đến 855, không cảnh báo), Recall@20 / NDCG@20, hiệu = trước − sau:
- T-KGAT vs KGAT: p Holm 0,733 / 0,910 (hiệu 0,0003 / −0,0002).
- T-KGAT thật vs hoán vị: 0,033 / 0,0046 (hiệu 0,0024 / 0,0025).
- KGAT vs KGAT không KG: 0,028 / 0,225 (hiệu 0,0040 / 0,0013).
- Stage 2 vs KGAT: 0,270 / 0,161 (hiệu −0,0011 / −0,0019).
- Stage 2 thật vs hoán vị: 0,843 / 0,554 (hiệu 0,0011 / 0,0008).

**Cách đọc đúng (bốn điểm):**
1. **KG:** trên latest, KGAT hơn KGAT không KG ở Recall@20 có ý nghĩa sau Holm (0,028) nhưng NDCG@20 không (0,225). **Khác `lastfm-u25`** (cả hai không có ý nghĩa). Điểm KGAT được chọn cũng khác (l2 1e-4 so với 1e-5). Sơ bộ; chưa đủ để nói KG giúp.
2. **Stage 1:** T-KGAT không hơn KGAT. "T-KGAT thật hơn hoán vị" có ý nghĩa, **nhưng bản hoán vị thấp hơn KGAT 0,0021**, nên điều đó **không chứng minh thời gian giúp** (xáo timestamp có thể chỉ làm mô hình tệ đi).
3. **Stage 2:** không so sánh nào có ý nghĩa; và thang Δt không có độ phân giải (12.5), nên kết quả không trả lời "fact cũ hay mới".
4. **Không so điểm tuyệt đối với `lastfm-u25`:** test khác (số user, độ dài, cold item, phép chia). Chỉ so thứ tự và hiệu giữa các mô hình trong cùng một bản.

### 12.7 Hạn chế của bản latest
1. Không loại được A; hình thức thiên lệch đổi (12.2); `check_split` không thấy.
2. Phép chia lệch về độ dài: train 83 ngày, val 26, test 45.
3. Stage 2 không có độ phân giải.
4. Cold item vẫn lớn (24,2% val, 21,9% test); chỉ 25% user.
5. Không tái lập được chỉ từ repo (12.8).
6. MostPop recent chưa chạy; payload chỉ có biến thể KG `default`.
7. Freebase là bản chụp 09/08/2015; 19,9% triple không có ngày; KGAT không tái lập từng bit trên GPU; điểm chọn của hai giai đoạn ở mép lưới γ.
8. Hoán vị toàn cục chỉ là một dạng đối chứng; kết quả sơ bộ, chưa có tag `v0.2-kgat`.

### 12.8 Nguồn gốc dữ liệu và kết quả của bản latest (PR #44)
- Năm zip kết quả của Phôn, gộp bằng `scripts/12_merge_latest_zips.py`: 323 dòng registry, 132 dòng riêng biệt, 0 dòng `-dirty`, commit `873ca7f` (lịch sử riêng, ngoài repo). Thiết bị: CPU (MostPop) và Tesla T4 (mọi mô hình học sâu; tổng 55,1 giờ ở dòng val).
- Bảng RQ3 và kiểm định do `scripts/10_rq3_table.py` (script của repo, không sửa) ghi từ registry gộp; số của bản `latest` do `scripts/11_numbers_latest_lastfm.py` ghi.
- `results/registry.csv` **không** có dòng nào của `lastfm-u25-latest`. Bản `latest` chưa được duyệt (0015 nháp). Muốn dùng làm kết quả chính thức cần: duyệt 0015, đưa code của Phôn vào repo bằng PR riêng, quyết định cách thêm dòng registry.
- Hướng dẫn tái sinh các bảng: `docs/lastfm-latest-provenance.md`.

### 12.9 Cách nói và không nói về bản latest
| Nên nói | Tránh nói |
|---|---|
| "Latest hết bước nhảy B nhưng không xử lý A; thiên lệch đổi hình thức" | "Latest sửa được thiên lệch của tập cặp" |
| "T-KGAT thật hơn hoán vị, nhưng hoán vị thấp hơn KGAT, nên chưa chứng minh thời gian giúp" | "Thời gian giúp (p Holm 0,033)" |
| "Stage 2 trên latest không có độ phân giải: 0,0% triple trong cửa sổ" | "Stage 2 cho thấy fact mới quan trọng" hay "ngược lại" |
| "KGAT hơn không KG ở Recall@20 có ý nghĩa, NDCG@20 không; sơ bộ" | "KG giúp" |
| "Số của latest là sơ bộ, chưa tái lập được chỉ từ repo" | Trình bày như kết quả chính thức |
| "Không so điểm tuyệt đối giữa u25 và latest" | "Latest tốt hơn vì Recall cao hơn" |
