# 09 — Học để báo cáo với cô ngày 31/10/2026

Tài liệu này đi theo đúng dàn ý đã chốt cho buổi 31/10: **mở đầu → chuẩn bị dữ liệu → huấn luyện mô hình → kết quả và nhận xét → kế hoạch và câu hỏi**. Mỗi phần có: điều cần nói được, số liệu kèm nguồn, hình dùng, và câu cô có thể hỏi kèm đáp án. Lý thuyết nền nằm ở `docs/study/08-tong-hop-ly-thuyet.md` (ghi số mục để tra). Kịch bản nói từng câu nằm ở `docs/study/10-huong-dan-present-31-10.md`.

Quy ước: mọi số lấy từ `results/registry.csv` (qua `scripts/10_rq3_table.py`, `scripts/09_significance.py`), `docs/data-report-numbers.md`, `docs/bprmf-training-table.md` hoặc file `results/runs/*`. KGAT và T-KGAT là **sơ bộ**. "std" là độ lệch chuẩn mẫu qua 5 seed.

---

## Bố cục đã chốt (khoảng 24 phút)

| # | Phần | Thời lượng | Người trình bày | Mục notebook |
|---|---|---|---|---|
| 0 | Mở đầu | 2' | Phôn | 1, 2, 3 |
| 1 | Chuẩn bị dữ liệu (kèm EDA) | 8' | Triết | 4, 5, 6, 7 |
| 2 | Huấn luyện mô hình (kèm learning curve, bảng siêu tham số) | 6' | Phôn (baseline, quy trình) + Triết (KGAT, T-KGAT) | 9, 11 |
| 3 | Kết quả và nhận xét (kèm minh bạch 0011, hạn chế) | 6' | Phôn (baseline) + Triết (KGAT, T-KGAT) | 8, 11 |
| 4 | Kế hoạch và câu hỏi | 2' | Cả hai | 10 |

**Ba thông điệp phải còn lại trong đầu cô sau buổi họp:**
1. Quy trình không rò rỉ, có bằng chứng tự động kiểm tra được.
2. Nhóm hiểu dữ liệu: rating được nhập theo phiên, KG đúng theo thời điểm.
3. Kết quả trung thực: chưa có bằng chứng thời gian giúp ích, và giải thích được vì sao.

---

## Phần 0 — Mở đầu (2 phút)

**Cần nói được:**
- Ngày 17/10 cô góp ý: thiếu tập validation nên có rò rỉ; chưa khám phá dữ liệu đủ kỹ; KG chưa tôn trọng chiều thời gian.
- Nhóm xác định kết quả cũ mắc **ba lỗi rò rỉ**: chọn epoch trên test, k-core trên toàn bộ dữ liệu, KG từ metadata hiện tại ⇒ làm lại từ đầu (quyết định 0001).
- Đề tài được đặt theo ngành Hệ thống thông tin: bài toán tích hợp và quản lý dữ liệu cho hệ gợi ý; T-KGAT là mô hình dùng để đánh giá giá trị của dữ liệu.
- Ba câu hỏi nghiên cứu: RQ1 đặc điểm dữ liệu theo thời gian; RQ2 KG có giúp khi đã loại rò rỉ; RQ3 thời gian trong attention có giúp và có thật đến từ thời gian (08, mục 0).

**Cô có thể hỏi:**
- *"Các em sửa từng góp ý thế nào?"* — Có tập val, chọn mọi thứ trên val; báo cáo dữ liệu và EDA; KG dựng theo mốc t1/t2 với `check_kg`. Mỗi thay đổi có file quyết định trong `docs/decisions/`.

---

## Phần 1 — Chuẩn bị dữ liệu (8 phút)

### 1.1 EDA: dữ liệu trông thế nào (bổ sung theo góp ý "khám phá dữ liệu kỹ hơn")

| | ML-100K | ML-1M |
|---|---|---|
| Rating | 100.000 | 1.000.209 |
| User / item được rating | 943 / 1.682 | 6.040 / 3.706 |
| Mật độ | 6,3% | 4,5% |
| Tỷ lệ rating ≥ 4 | 55,4% | 57,5% |
| Khoảng thời gian | 20/09/1997 – 22/04/1998 (215 ngày) | 25/04/2000 – 28/02/2003 (1.039 ngày) |
| Năm chiếm nhiều rating nhất | 1997 (52,9%) | 2000 (90,5%) |
| Trung vị rating mỗi user | 65 | 96 |

*(943 user của ML-100K là con số của bộ dữ liệu gốc; các số còn lại từ `docs/data-report-numbers.md`.)*

**Phát hiện chính — rating được nhập theo phiên:**
- 77,9% rating của ML-100K (72,4% của ML-1M) nằm trong 24 giờ đầu kể từ rating đầu tiên của user.
- 63,5% user ML-100K chỉ hoạt động một ngày.
- 53,0% user ML-100K có toàn bộ cạnh train trong chưa đầy 1 giờ; trung vị độ lệch chuẩn thời gian theo user chỉ 14 phút. Ngược lại, theo item là 45 ngày (ML-100K), 67 ngày (ML-1M).
- Hình: `docs/figures/fig1_rating_theo_tuan.png`, `fig2_do_trai_thoi_gian.png`.
- **Hệ quả** (nói ngay): (a) chia theo mốc chung làm mất nhiều user ở val/test; (b) Δt theo user gần như bằng 0, nên tác dụng của thời gian (nếu có) chỉ đến từ phía item; (c) đối chứng hoán vị là bắt buộc.

### 1.2 Làm sạch
18 tiêu đề trùng gộp về một id; item 267 "unknown" bị loại; cặp (user, item) trùng giữ lần đầu; checksum file thô để mọi người dùng cùng một bản.

### 1.3 Chia train/val/test theo mốc chung (08, mục 2.2–2.4)
- t1, t2 = phân vị 0,8 và 0,9 của mọi timestamp. ML-100K: 07/03/1998 và 31/03/1998; ML-1M: 02/12/2000 và 29/12/2000.
- Rating ≥ 4 là dương → 5-core chỉ trên train → val/test chỉ giữ user/item đã biết.
- **Cái giá** (hình `fig3_positive_con_lai.png`): ML-100K giữ 12,6% số dương ở val (66 user) và 13,3% ở test (51 user); trong số dương bị loại ở test vì user không có trong train, 99,6% thuộc user mới (rating đầu tiên sau t1). ML-1M giữ 75,7% ở test (952 user) ⇒ kết luận dựa chủ yếu vào ML-1M.
- Cửa sổ: ML-100K val 25 ngày, test 22 ngày; ML-1M val 27 ngày, test 791 ngày.

### 1.4 Liên kết thực thể (08, mục 6.1–6.3)
- Chuỗi: ML-100K → ML-1M (tên + năm) → ML-20M → Freebase (KB4Rec) → Wikidata (P646).
- Độ phủ 97,2% số phim, 99,5% số rating (ML-100K). Độ chính xác: 0/50 sai trong mẫu ngẫu nhiên, Wilson 95% [0%; 7,1%] (40 phim máy xác nhận bằng hai nguồn, 10 phim kiểm tay). Lỗi đã biết: *Clubland* → Cloudland Canyon State Park.
- Notebook: `notebooks/02_entity_linking.ipynb`.

### 1.5 KG theo thời điểm (08, mục 6.4–6.6)
- Hai nguồn: Wikidata (119.748 fact, 18 quan hệ; bản chụp 17/10/2026) + metadata MovieLens (thể loại, năm, thập niên; quyết định 0011).
- Fact có ngày giữ nếu trước mốc (năm ⇒ 31/12); thực thể ra đời sau mốc bị bỏ; fact không ngày giữ nếu hai đầu đã tồn tại. KG train/val dựng tại t1, test tại t2.
- Kích thước KG default tại t1: ml-100k 40.210 triple, ml-1m-u10 71.553.
- Ví dụ: 11 giải Oscar của *Titanic* (23/03/1998) nằm giữa t1 và t2 ⇒ chỉ có trong KG test; *Titanic* mất 28 fact ở t1.
- Bộ lọc chỉ bỏ 1,3% fact (ML-100K; 1,5% ở ML-1M) nhưng chúng tập trung ở phim phổ biến: item bị mất ít nhất một fact mang 53,1% số dương ở val và 48,1% ở test (ML-100K). Hình `fig4_fact_bi_loai.png`.

### 1.6 Bằng chứng không rò rỉ (08, mục 2.6)
- `check_split`: 6 kiểm tra (thứ tự thời gian; không cặp ở hai phần; user/item đã biết; 5-core trên train; mask chỉ dùng quá khứ; đáp án không trùng mask). Script 02 từ chối lưu split vi phạm.
- `check_kg`: script 06 từ chối lưu KG có fact sau mốc.
- Test chỉ được chấm trong `scripts/run.py`, sau khi chọn trên val. CI chạy `ruff` và `pytest` mỗi lần đẩy code.

**Cô có thể hỏi:**
- *"Vì sao test ML-100K chỉ 51 user?"* — Rating nhập theo phiên; trong số dương bị loại ở test vì user không có trong train, 99,6% thuộc user mới. Vì vậy chạy song song ML-1M.
- *"97% có phải liên kết đúng 97%?"* — Không, đó là độ phủ. Độ chính xác đo bằng mẫu 50 phim: 0 sai, cận trên khoảng 7%.
- *"Wikidata là 2026, sao gọi là đúng theo thời điểm?"* — Lọc được theo ngày sự kiện, không lọc được ngày fact được nhập vào Wikidata; đây là hạn chế đã ghi.
- *"Ngày chỉ có năm xử lý sao?"* — Coi là 31/12 để không đưa fact tương lai vào (quyết định 0006).

---

## Phần 2 — Huấn luyện mô hình (6 phút)

### 2.1 Các mô hình
| Mô hình | Ý tưởng một câu | 08, mục |
|---|---|---|
| MostPop | Phim nhiều tương tác nhất trong train, giống nhau cho mọi user | 1.4 |
| MostPop recent | Như trên nhưng chỉ đếm N ngày cuối của train (N chọn trên val) | 1.4 |
| BPR-MF | Matrix factorization, học xếp hạng bằng BPR | 1.2–1.3 |
| KGAT | Lan truyền trên đồ thị gộp tương tác + KG bằng attention, hai loss (BPR + TransR) | 4 |
| KGAT không KG | KGAT bỏ triple KG (đo giá trị của KG, RQ2) | 4.5 |
| T-KGAT Stage 1 | KGAT, trừ γ·Δt khỏi điểm attention của cạnh user–item trước softmax | 5.2 |
| T-KGAT hoán vị | T-KGAT với timestamp train bị xáo (đo phần đến từ thời gian, RQ3) | 5.5 |
| T-KGAT Stage 2 | Δt đặt trên triple KG thay vì cạnh user–item | 5.4 |

### 2.2 Quy trình huấn luyện chung (bằng chứng có tập validation)
1. Mỗi điểm lưới × 5 seed (2019–2023).
2. Huấn luyện trên train; đánh giá Recall@20 trên val mỗi 5 epoch; dừng sớm khi val không tăng (patience 5 với BPR-MF, 10 với KGAT); giữ epoch tốt nhất trên val.
3. Chọn điểm lưới có trung bình Recall@20 val qua 5 seed cao nhất.
4. Chấm test **một lần** cho điểm đã chọn; ghi mọi dòng val và test vào `results/registry.csv` (chỉ thêm, có git commit).
5. T-KGAT: giữ (lr, l2) KGAT đã chọn, chỉ dò γ ∈ {1, 3, 10}.

**Hình learning curve** (Recall@20 val theo epoch): `docs/figures/fig5_bprmf_curve_ml-100k.png`, `fig5_bprmf_curve_ml-1m.png` (BPR-MF) và `docs/figures/rq3_curves_ml-100k.png`, `rq3_curves_ml-1m-u10.png` (KGAT, T-KGAT, hoán vị). Ý cần nói: epoch tốt nhất được chọn theo val, test không tham gia.

### 2.3 Bảng siêu tham số (bổ sung theo góp ý 03/07)

**BPR-MF** (`docs/bprmf-training-table.md`):

| Tham số | ML-100K | ML-1M |
|---|---|---|
| Batch size | 2.048 | 2.048 |
| Epoch tối đa / đánh giá mỗi / patience | 300 / 5 / 5 | 300 / 5 / 5 |
| Điểm được chọn (dim, lr, l2) | 32, 1e-3, 1e-4 | 64, 1e-3, 1e-4 |
| Epoch đã chạy (qua seed) | 60–135 | 40–70 |
| Epoch tốt nhất (qua seed) | 35–110 | 15–45 |

**KGAT / T-KGAT** (`configs/model/kgat.yaml`, `results/runs/*/kgat_summary.json`, history):

| Tham số | Giá trị |
|---|---|
| dim / kge_dim / các lớp | 64 / 64 / [64, 32, 16] |
| mess_dropout, l2_kg | 0,1, 1e-5 |
| Batch CF / batch KG | 1.024 / 2.048 |
| Epoch tối đa / đánh giá mỗi / patience | 300 / 5 / 10 |
| Lưới (lr, l2) | {1e-3, 1e-4} × {1e-5, 1e-4} (4 điểm) |
| Điểm KGAT được chọn | ml-100k: lr 1e-3, l2 1e-4; ml-1m-u10: lr 1e-3, l2 1e-5 |
| Epoch đã chạy / tốt nhất (KGAT, điểm chọn) | ml-100k: 130–200 / 40–155; ml-1m-u10: 70–165 / 20–75 |
| Lưới γ (T-KGAT) | {1, 3, 10}; chọn γ = 3 (ml-100k), γ = 1 (ml-1m-u10) |
| Thiết bị | GPU RTX 3050 Laptop 4 GB; bộ nhớ đỉnh khoảng 200–500 MB |

### 2.4 Tái lập
- Tag `v0.1-review` (commit `0e194e3`) đóng băng dữ liệu, KG, baseline, báo cáo; đúng commit của lần chạy KGAT/T-KGAT mới nhất.
- BPR-MF tái lập được trên GPU; KGAT không tái lập từng bit (`torch.sparse.mm`) ⇒ Phôn chạy lại từ clone sạch, tiêu chí khớp là trung bình 5 seed nằm trong một độ lệch chuẩn.

**Cô có thể hỏi:**
- *"Tập val dùng vào việc gì?"* — Dừng sớm, chọn điểm lưới, chọn γ, chọn cửa sổ MostPop recent. Test chỉ chấm một lần.
- *"Sao không train lại trên train + val?"* — Quyết định từ 17/10: đơn giản, dễ kiểm chứng, ít chỗ sai.
- *"T-KGAT khác KGAT chỗ nào?"* — Trừ γ·Δt khỏi điểm attention trước softmax, chỉ trên cạnh user–item; γ = 0 là đúng KGAT; tương đương nhân trọng số với e^(−γΔt).
- *"Vì sao KGAT chạy hai lần cùng seed lại khác?"* — `torch.sparse.mm` trên GPU không tất định; coi là nhiễu seed, báo cáo trung bình ± std.

---

## Phần 3 — Kết quả và nhận xét (6 phút)

### 3.1 Bảng chính (Recall@20 test, trung bình ± std qua 5 seed)

| | ml-100k (51 user) | ml-1m-u10 (95 user) | ML-1M đầy đủ (952 user) |
|---|---|---|---|
| MostPop | 0,1555 | 0,0846 | 0,0902 |
| MostPop recent | **0,3372** | 0,1007 | 0,1060 |
| BPR-MF | 0,1471 ± 0,0295 | 0,1000 ± 0,0069 | 0,1194 ± 0,0031 |
| KGAT | 0,2096 ± 0,0284 | 0,1086 ± 0,0054 | chưa chạy |
| KGAT không KG | 0,2017 ± 0,0232 | 0,1042 ± 0,0101 | chưa chạy |
| T-KGAT | 0,2151 ± 0,0174 | 0,1113 ± 0,0046 | chưa chạy |
| T-KGAT hoán vị | 0,1950 ± 0,0156 | 0,1119 ± 0,0049 | chưa chạy |
| T-KGAT Stage 2 | 0,1933 ± 0,0216 (γ = 1) | 0,1145 ± 0,0148 (γ = 1) | chưa chạy |
| Stage 2 hoán vị ngày fact | 0,1971 ± 0,0238 | 0,1164 ± 0,0099 | chưa chạy |

Nguồn: `docs/rq3-table.md` (ml-100k, ml-1m-u10), `results/significance.csv` và `docs/data-report.md` (ML-1M).

### 3.2 Ba nhận xét (mỗi nhận xét: số → ý nghĩa → giải thích)

**Nhận xét 1 — Baseline.** Trên ML-1M (952 user) BPR-MF thắng MostPop recent có ý nghĩa (0,1194 so với 0,1060, p Holm ≈ 1,1·10⁻⁵). Trên ML-100K, MostPop recent thắng mọi mô hình (0,3372; thắng BPR-MF với p Holm 0,029). Giải thích: test ML-100K chỉ 22 ngày và 51 user, phim hot gần mốc cắt chiếm ưu thế. Mô hình thời gian phải vượt được MostPop recent.

**Nhận xét 2 — KG (RQ2).** KGAT nhỉnh hơn KGAT không KG nhưng chỉ sát ngưỡng: ml-1m-u10 NDCG p Holm 0,049, Recall p Holm 0,14; ml-100k không khác biệt. Với KG cũ (chỉ Wikidata) bằng chứng ở Recall mạnh hơn (p Holm 0,020). Đọc: KG có thể giúp, chưa chắc chắn.

**Nhận xét 3 — Thời gian (RQ3).** T-KGAT so với bản hoán vị: không khác biệt trên cả hai bộ (p Holm ≥ 0,45). T-KGAT hơn KGAT ở NDCG trên ml-1m-u10 (p Holm 0,049) nhưng bản hoán vị cũng đạt mức đó ⇒ phần hơn không đến từ thời gian. Độ nhạy γ chỉ trên val cho hai chiều ngược nhau (ml-100k nghiêng về γ = 0; ml-1m-u10 về γ = 1). Giải thích bằng dữ liệu: rating nhập theo phiên nên Δt theo user ≈ 0; tín hiệu thời gian yếu.

**Stage 2** (Δt trên triple KG, γ = 1 được chọn trên cả hai bộ, là mép dưới lưới): ml-100k 0,1933 ± 0,0216, thấp hơn KGAT nhưng không có ý nghĩa (Recall p Holm 0,162); ml-1m-u10 0,1145 ± 0,0148, cao hơn KGAT nhưng không có ý nghĩa (Recall p Holm 0,189; NDCG 0,065). So với bản hoán vị ngày fact: không khác biệt trên cả hai bộ (p Holm từ 0,155 trở lên), và bản hoán vị còn nhỉnh hơn ở Recall. Vì Δt_norm của cạnh KG bị chặn ở 1 (chỉ 0,7% / 0,0% fact có ngày nằm dưới mức này), Stage 2 thực chất giảm trọng số fact có ngày so với fact không ngày, không đo được "fact cũ hay mới". Đọc: không có bằng chứng; cần thiết kế lại thang Δt cho KG. Phải nói đúng như vậy.

### 3.3 Minh bạch về quyết định 0011 (bắt buộc nói)
"KG được thêm metadata MovieLens (PR #34) sau khi điểm test của KG cũ đã được xem. Quyết định 0011 quy định: KG mới là chính thức, chạy lại mọi mô hình dùng KG với cùng lưới và seed, lấy kết quả mới dù tốt hay xấu; kết quả KG cũ để ở phụ lục của `docs/rq3-table.md`. Nhóm thừa nhận đây là lần nhìn test thứ hai của KGAT và T-KGAT."

### 3.4 Hạn chế (một slide)
1. Wikidata là bản chụp 2026 (lọc theo ngày sự kiện, không theo ngày nhập).
2. Timestamp được giả định là lúc nhập rating (quyết định 0008).
3. Test dưới 100 user ở ml-100k và ml-1m-u10: p lớn không chứng minh "không khác biệt".
4. KGAT không tái lập từng bit trên GPU.
5. ml-1m-u10 là 10% user của ML-1M, không phải ML-1M đầy đủ (KGAT trên ML-1M đầy đủ mất khoảng 29 giờ mỗi cấu hình).
6. KGAT/T-KGAT chạy trước khi cô duyệt quy trình, và KG đổi sau khi đã xem test (0011).

**Cô có thể hỏi:**
- *"MostPop recent thắng, vậy mô hình có ích gì?"* — Trên ML-1M đủ user, BPR-MF thắng có ý nghĩa; ML-100K test quá ngắn và ít user. Mục tiêu là vượt MostPop recent trên ML-1M.
- *"Kết quả âm thì khóa luận đóng góp gì?"* — Một quy trình đánh giá không rò rỉ cho KG có thời gian, bằng chứng rằng tín hiệu thời gian của MovieLens yếu vì cách dữ liệu được nhập, và đối chứng hoán vị cho thấy cải thiện nhỏ không đến từ thời gian. Biên bản 07/08 đã ghi kết quả âm là hợp lệ.
- *"Sao không chọn γ nhỏ hơn?"* — Lưới đã đăng ký trước khi chạy; γ ∈ {0,1; 0,3} chỉ xem trên val (quyết định 0010), không chấm test.

---

## Phần 4 — Kế hoạch và câu hỏi (2 phút)

**Kế hoạch:** giai đoạn 4 (02–29/11): biến thể KG `drop_undated`, `static`, KG 2-hop, baseline CKE, một lần chạy theo quy trình gốc của paper (chia ngẫu nhiên) làm mốc; giai đoạn 5 (30/11–27/12): thời gian trên KG và ba hướng cô gợi ý 18/09 (rating như relation hoặc trọng số, timestamp như relation, LLM làm sạch metadata); giai đoạn 6 (28/12–17/01): viết khóa luận.

**Ba câu hỏi xin ý kiến cô:**
1. Kết quả KGAT/T-KGAT đã chạy sớm: giữ làm chính thức, hay chạy lại sau khi cô duyệt quy trình?
2. Chạy ML-1M đầy đủ, hay dùng ml-1m-u10 và ghi là giới hạn?
3. Ưu tiên tiếp theo: thiết kế lại Stage 2 (chuẩn hóa Δt của KG theo năm), hay ba hướng cô gợi ý ngày 18/09?

---

## Phụ lục A — Thẻ số một trang (in ra mang theo)

| Chủ đề | Số |
|---|---|
| Rating trong 24 giờ đầu | 77,9% (100K), 72,4% (1M) |
| User có mọi cạnh train trong 1 giờ | 53,0% (100K) |
| Độ lệch chuẩn thời gian trung vị theo user / item | 14 phút / 45 ngày (100K) |
| t1, t2 | 07/03/1998, 31/03/1998 (100K); 02/12/2000, 29/12/2000 (1M) |
| User val / test | 66 / 51 (100K); 480 / 952 (1M); test 95 (ml-1m-u10) |
| Số dương giữ lại ở test | 13,3% (100K), 75,7% (1M) |
| Độ phủ liên kết / độ chính xác | 97,2% / 0/50 sai, Wilson [0%; 7,1%] |
| Wikidata | 119.748 fact, 18 quan hệ, bản chụp 17/10/2026 |
| KG default t1 | 40.210 (ml-100k), 71.553 (ml-1m-u10) |
| Titanic | 11 Oscar ngày 23/03/1998, mất 28 fact ở t1 |
| ML-1M test Recall@20 | MostPop 0,0902; recent 0,1060; BPR-MF 0,1194 ± 0,0031 |
| ml-100k KGAT / T-KGAT / hoán vị | 0,2096 / 0,2151 / 0,1950 |
| ml-1m-u10 KGAT / T-KGAT / hoán vị | 0,1086 / 0,1113 / 0,1119 |
| Stage 2: fact có ngày với Δt_norm < 1 | 0,7% (ml-100k), 0,0% (ml-1m-u10) |
| Stage 2 / hoán vị ngày fact | 0,1933 / 0,1971 (ml-100k); 0,1145 / 0,1164 (ml-1m-u10) |

## Phụ lục B — Lịch ôn đến sáng 31/10

| Khi nào | Việc |
|---|---|
| Sáng 30/10 | Đọc tài liệu 08 mục 2, 4, 5 (rò rỉ, KGAT, thời gian); đọc kỹ đoạn Stage 2 ở phần 3 (đã có số) |
| Trưa 30/10 | Thuộc Phụ lục A; tự trả lời 30 câu ở cuối tài liệu 08 |
| Chiều 30/10 | Chạy notebook từ đầu đến cuối; xem lại các hình |
| Tối 30/10 | Tập dượt 2 lượt theo tài liệu 10, lượt 2 đổi vai |
| Sáng 31/10 | Đọc lại 3 thông điệp, 3 nhận xét, 6 hạn chế, 3 câu hỏi |
