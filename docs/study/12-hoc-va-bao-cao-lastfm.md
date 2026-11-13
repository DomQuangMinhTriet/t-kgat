# 12 — Học để báo cáo kết quả Last-FM (`lastfm-u25` và `lastfm-u25-latest`) với cô, 14/11/2026 (13:00)

Tài liệu này đi theo cùng khung với `docs/study/09-hoc-va-bao-cao-31-10.md`: **mở đầu → dữ liệu → huấn luyện → kết quả → (mới) ba vấn đề và bản `latest` → kế hoạch và câu hỏi**. Mỗi phần có: điều cần nói được, số liệu kèm nguồn, hình dùng, và câu cô có thể hỏi kèm đáp án. Lý thuyết nền ở `11-tong-hop-ly-thuyet-lastfm.md` (ghi số mục để tra). Kịch bản nói từng câu ở `13-huong-dan-present-lastfm.md`. Notebook trình bày: `notebooks/00_present_lastfm.ipynb`.

**Phạm vi:** phần 0 đến 3 chỉ `lastfm-u25` (Triết, và Phôn ở baseline và quy trình); phần 4 là bản mới `lastfm-u25-latest` (Phôn): timestamp của cặp là lần nghe **muộn nhất**; phần 5 là kế hoạch và ba câu hỏi. **Không trộn số của hai bản**: mọi bảng ghi rõ tên run.

**Quy ước:** số của `lastfm-u25` lấy từ `results/registry.csv` (qua `scripts/10_rq3_table.py`, `scripts/09_significance.py`), `docs/rq3-table.md`, `docs/data-report-lastfm.md`, `docs/data-report-lastfm-numbers.md`, `docs/bprmf-training-table.md`, các quyết định 0012–0014. Số của `lastfm-u25-latest` lấy từ `docs/data-report-lastfm-latest-numbers.md`, `docs/rq3-table-lastfm-latest.md`, `results/rq3_models_lastfm-latest.csv`, `results/rq3_pairs_lastfm-latest.csv`, `results/lastfm-latest-provenance.json` (PR #43), hoặc được notebook tính từ `data/processed/lastfm-u25-latest/`. "std" là độ lệch chuẩn mẫu qua 5 seed. Kết quả KGAT và T-KGAT của cả hai bản là **sơ bộ**. Số có dấu **(tính thêm)** do notebook tính từ `train.csv`, chưa nằm trong báo cáo dữ liệu của repo.

**Tình trạng (đã kiểm ngày 13/11):**
- `lastfm-u25`: 9 biến thể mô hình × 5 seed đã chạy xong cả val và test (MostPop và MostPop recent là một lần chạy tất định); 139 dòng registry, cùng commit `ce55c0c`, không dòng `-dirty`; PR #42 (nháp). Chưa có: tag `v0.2-kgat`, tái lập độc lập từ clone sạch, các phân tích bổ sung đã đề xuất.
- `lastfm-u25-latest`: 8 mô hình đã chạy (MostPop recent **chưa chạy**); registry gộp của năm zip của Phôn có 323 dòng, 132 dòng riêng biệt, commit `873ca7f` (lịch sử riêng), 0 dòng `-dirty`; **không có dòng nào trong `results/registry.csv`**; quyết định 0015 là nháp của Phôn, chưa accepted, chưa vào repo; code xuất của Phôn chưa vào repo; PR #43 xếp chồng lên #42 và chỉ chứa bảng tóm tắt, script tái sinh và nguồn gốc.

---

## Bố cục đề xuất (khoảng 24 phút; chưa chốt với cô, chia người nói là đề xuất)

| # | Phần | Thời lượng | Người trình bày (đề xuất) | Mục notebook |
|---|---|---|---|---|
| 0 | Mở đầu: vì sao Last-FM, protocol không đổi | 1,5' | Phôn | 1 |
| 1 | Dữ liệu `lastfm-u25`: nguồn, chia, KG, hạn chế của tập cặp | 5,5' | Triết | 2 |
| 2 | Huấn luyện: quy trình, siêu tham số, tính toán, Kaggle | 4' | Phôn (baseline, quy trình) + Triết (KGAT, T-KGAT) | 3 |
| 3 | Kết quả `lastfm-u25`, minh bạch, hạn chế | 4,5' | Phôn (baseline) + Triết (KGAT, T-KGAT) | 4, 5 |
| 4 | Ba vấn đề của `lastfm-u25` và bản `lastfm-u25-latest` | 6' | Phôn | 6 |
| 5 | Kế hoạch và ba câu hỏi | 2,5' | Cả hai | 7 |

**Bốn thông điệp phải còn lại trong đầu cô:**
1. Protocol không đổi so với MovieLens và vẫn không rò rỉ (có kiểm tra tự động, test chấm một lần).
2. Last-FM cho val/test đủ lớn (2.319 và 1.608 user ở `lastfm-u25`) nên kết luận có trọng lượng hơn MovieLens: trên `lastfm-u25` **chưa có bằng chứng KG hay thời gian giúp**.
3. `lastfm-u25` có ba vấn đề (A, B, C) mà nhóm tự phát hiện và báo; bản `latest` của Phôn xử lý B nhưng **không loại được A**, thiên lệch chỉ **đổi hình thức**.
4. Trên `latest`: Stage 2 không có độ phân giải; "T-KGAT thật hơn hoán vị" không chứng minh thời gian giúp; kết quả sơ bộ và chưa tái lập được chỉ từ repo. Nhóm xin cô chọn bản dữ liệu chính thức.

---

## Phần 0 — Mở đầu (1,5 phút)

**Cần nói được:**
- Ngày 01/11 cô nhận xét kết quả MovieLens cho thấy vấn đề ở dữ liệu đầu vào và đề nghị chuyển sang Last-FM, lặp lại thí nghiệm của MovieLens (quyết định 0012).
- Nhóm làm đúng theo đó: cùng mã chia, cùng mốc 0,8 / 0,9, cùng k-core trên train, cùng `check_split`, cùng luật KG, cùng lưới, cùng 5 seed (11, mục 0).
- **Minh bạch ngay:** quyết định chuyển sang Last-FM được đưa ra sau khi điểm test MovieLens đã được xem; nhóm làm theo hướng cô chỉ, không chọn để cải thiện điểm số.
- Ba câu hỏi nghiên cứu như trước: RQ1 đặc điểm dữ liệu theo thời gian; RQ2 KG có giúp không; RQ3 thời gian trong attention có giúp và có thật đến từ thời gian không.

**Cô có thể hỏi:**
- *"Vì sao lấy 25% user?"* — Lưới KGAT đầy đủ trên Last-FM cần khoảng 357 giờ trên laptop; 25% user đo được khoảng 20 giây mỗi epoch, ước lượng cận trên 121,7 giờ (0013). Cái giá là mất 28% positive val/test vì cold item.

---

## Phần 1 — Dữ liệu `lastfm-u25` (5,5 phút)

### 1.1 Nguồn (11, mục 6.1)
Ba nguồn ghép lại:
1. **Bản Last-FM của tác giả KGAT:** 3.034.763 token nghe → 1.542.856 cặp (user, item) duy nhất; KG 464.567 triple, 9 quan hệ. Không có timestamp, không có validation.
2. **LFM-1b:** nguồn timestamp. Mỗi cặp giữ **lần nghe sớm nhất**.
3. **Freebase dump 09/08/2015:** nguồn ngày của fact, qua ngày tồn tại của đích.

Không có bước liên kết thực thể như MovieLens vì dùng đúng tập user/item của tác giả; KG khớp trực tiếp qua id Freebase.

### 1.2 Số chính (`docs/data-report-lastfm.md` bảng 1; nguồn script `07_report_numbers_lastfm.py`)

| | `lastfm` (đầy đủ) | `lastfm-u25` (chính thức) |
|---|---|---|
| Cặp (user, item) | 1.542.856 | 370.979 |
| t1 / t2 | 28/04/2014 / 06/06/2014 | 29/04/2014 / 07/06/2014 |
| Train (sau 5-core) | 1.220.988 | 236.053 |
| User / item của train | 22.040 / 44.788 | 5.321 / 18.665 |
| Val: positive / user | 142.023 / 10.618 | 23.708 / 2.319 |
| Test: positive / user | 133.896 / 7.149 | 22.419 / 1.608 |
| Positive val / test giữ lại | 92,1% / 86,8% | 63,9% / 60,4% |
| Bị 5-core loại khỏi train | 13.295 | 60.730 (trong 296.783) |

So với MovieLens: ML-100K giữ 12,6% val và 13,3% test (66 và 51 user); `ml-1m-u10` test 95 user. Last-FM đầy đủ giữ 92,1% / 86,8%.

### 1.3 Cái giá của việc lấy 25% user (nói thẳng)
- 28,8% positive val và 28,1% positive test bị loại vì **cold item** (bộ đầy đủ chỉ 3,0% và 4,6%); cold user 7,3% (val) và 11,5% (test) (`meta.json`: 10.675, 10.424 cold-item interactions; 2.715, 4.255 cold-user interactions trên 37.098).
- Vẫn lớn hơn MovieLens rất nhiều, nhưng **không còn là val gần như nguyên vẹn** của `lastfm` đầy đủ. Kết quả chỉ nói cho mẫu 25% user (seed 2019), không nói cho toàn bộ Last-FM.
- `lastfm-u10` chỉ là tập phát triển; không báo số chính thức trên nó.

### 1.4 Δt trong dữ liệu Last-FM (điều kiện cho Stage 1) **(tính thêm)**
Phát hiện ngược với MovieLens, trả lời trực tiếp cho điều cô sẽ hỏi "thời gian có độ phân giải không?":

| | ML-100K | `lastfm-u25` (tính thêm) |
|---|---|---|
| Cạnh nằm trong 24 giờ đầu của user | 77,9% | 5,9% |
| User có mọi cạnh train trong 1 giờ | 53,0% | 0,53% |
| Trung vị độ lệch chuẩn thời gian theo user | 14 phút | 249,6 ngày |
| Trung vị độ lệch chuẩn thời gian theo item | 45 ngày | 552,6 ngày |

- Δt_norm của 236.053 cạnh train: trung vị 0,100; tứ phân vị 0,0147 và 0,2229; phân vị 90 là 0,4316; 22,0% cạnh trong 30 ngày cuối; 52,1% trong 365 ngày cuối. Cửa sổ train dài 9,20 năm (14/02/2005 – 29/04/2014).
- **Hệ quả:** Δt thật sự có biến thiên ở cả hai phía user và item. Nghĩa là Stage 1 được đo trong **điều kiện thuận lợi hơn** MovieLens, nên kết quả "không có tác dụng" (phần 3) mạnh hơn kết luận trên MovieLens, không yếu hơn.
- Cảnh báo: Δt tính từ lần nghe đầu của cặp; 21,7% cạnh train rơi vào giai đoạn từ 01/04/2014 (một phần là dấu vết của phép chọn cặp, mục 1.6).
- `docs/data-report-lastfm.md` mục 7.3 ghi phép đo này là "chưa làm"; đây là lần đầu có số.

### 1.5 KG và ngày của fact (11, mục 6.3–6.6)
- Kích thước `lastfm-u25`: 464.567 → 154.076 (lọc theo item) → 152.979 (chở được 1-hop; bỏ 1.097, 0,7%, head không phải item train); 44.392 thực thể, 9 quan hệ. KG default tại t1: 152.976 triple; `drop_undated` 122.653; `static` 152.979.
- **Ngày:** fact được định ngày qua ngày tồn tại của đích (sớm nhất của `release_date`, `active_start`, `date_of_birth`). Không có ngày của statement. 30.323 triple (19,8%) không có ngày; 4 trong 9 quan hệ hoàn toàn không có ngày (`notable_types`, `single.versions`, `canonical_version`, `recording.song`); `releases` 1,9% và `artist` 1,1% không có ngày (`producer` 27,5%, `engineer` 74,8%).
- **Bộ lọc thời gian gần như không cắt gì:** static − default = 3 triple (15 ở bản đầy đủ), 0,0%; KG t1 = KG t2; 0 thực thể chỉ có ở t2. MovieLens: bộ lọc cắt 1,3% (ML-100K) và 1,5% (ML-1M). **Vì vậy Last-FM không đo được hiện tượng "fact tương lai giữa hai mốc"**; câu hỏi rò rỉ qua KG vẫn trả lời bằng MovieLens (ví dụ *Titanic*).
- **Thang Δt của Stage 2** (bảng 4 của báo cáo dữ liệu): 24,4% triple nằm trong cửa sổ (`dt_norm < 1`); 55,7% bị chặn ở 1; 19,8% không có ngày; trung vị `dt_norm` của fact có ngày 1,45. Tính riêng fact có ngày (tính thêm): 30,5% trong cửa sổ, 69,5% bị chặn. MovieLens: 0,7% (ML-100K) và 0,0% (`ml-1m-u10`).
- **Hệ quả:** Stage 2 trên Last-FM có độ phân giải để so fact cũ với fact mới, khác MovieLens; nhưng hơn 2/3 fact có ngày vẫn bị chặn ở 1.

### 1.6 Bằng chứng không rò rỉ và hạn chế riêng của tập cặp
- `check_split`: 6 kiểm tra (thứ tự thời gian; không cặp ở hai phần; user/item đã biết; 5-core trên train; mask chỉ dùng quá khứ; đáp án không trùng mask); `interaction_leakage_problems: []`. `check_kg` từ chối lưu KG có fact sau mốc. Test chỉ được chấm trong `scripts/run.py` sau khi chọn trên val.
- **Hạn chế mà kiểm tra tự động không thấy được (quyết định 0012):** tập cặp của KGAT chỉ chứa các cặp có ít nhất một lần nghe từ 01/04/2014 (99,97% trên mẫu 2.475 user). Dấu vết: số cặp mới mỗi ngày nhảy từ khoảng 2.800 lên khoảng 11.900 tại 01/04/2014; trên bộ đầy đủ khoảng 23% cặp của train rơi vào tháng trước t1 (`lastfm-u25`: 22,0% cạnh trong 30 ngày cuối, tính thêm). Tức tập cặp được chọn bằng thông tin **sau mốc**; có thể thiên vị các cặp "sống lâu". Tác động lên kết quả **chưa đo**: dựng lại tập cặp không giới hạn bởi cửa sổ 04/2014 (gọi là "phương án A" ở 0013 mục 7; trong buổi 14/11 gọi là **phương án C** để khỏi trùng với A, B, C của ba phương án dữ liệu ở phần 5) chưa chạy lại trên `lastfm-u25` vì thiếu thời gian.
- Bài báo KGAT ghi Last-FM là dữ liệu 01–06/2015 nhưng LFM-1b của nhóm có timestamp đến khoảng 09/2014 (test đến 02/09/2014). Nhiều khả năng tập cặp lấy từ giai đoạn từ 01/04/2014 đến hết dữ liệu (khoảng năm tháng). Suy luận, tác giả không ghi rõ.

**Cô có thể hỏi:**
- *"Vì sao chắc dữ liệu Last-FM không rò rỉ?"* — Nêu 6 kiểm tra; nhưng nói thêm: chính tập cặp có thiên lệch chọn mẫu mà `check_split` không bắt được; đã ghi ở 0012; chưa biết ảnh hưởng.
- *"Timestamp là gì?"* — Lần nghe sớm nhất của cặp. Δt = thời gian từ lần nghe đầu đến cuối train.
- *"Freebase 2015 mà dữ liệu 2014, có sao không?"* — Lọc được theo ngày tồn tại của đích, không lọc được ngày fact vào KG; hạn chế đã ghi; chỉ 3 triple có ngày sau t1 nên tác động lên lọc thời gian nhỏ.
- *"Sao KG t1 và t2 giống nhau, không phải lỗi?"* — Không: không fact nào mang ngày rơi giữa 28/04 và 06/06/2014; 0 thực thể chỉ có ở t2. Là đặc điểm dữ liệu.
- *"Vì sao val/test của u25 nhỏ hơn bản đầy đủ?"* — Cold item 28%; xem 1.3.

---

## Phần 2 — Huấn luyện mô hình (4 phút)

### 2.1 Các mô hình (11, mục 1.4, 4, 5)
| Mô hình | Ý tưởng một câu |
|---|---|
| MostPop | Item nhiều tương tác nhất trong train, giống nhau cho mọi user |
| MostPop recent | Như trên nhưng chỉ đếm N ngày cuối của train (N chọn trên val; ở đây 14) |
| BPR-MF | Matrix factorization, học xếp hạng bằng BPR |
| KGAT | Lan truyền trên đồ thị gộp tương tác + KG bằng attention; hai loss (BPR + TransR) |
| KGAT không KG | KGAT bỏ triple KG (đo giá trị của KG, RQ2) |
| T-KGAT Stage 1 | KGAT, trừ γ·Δt khỏi điểm attention của cạnh user–item trước softmax |
| T-KGAT hoán vị | Stage 1 với timestamp train bị xáo (RQ3) |
| T-KGAT Stage 2 | Δt đặt trên triple KG thay vì cạnh user–item |
| Stage 2 hoán vị | Stage 2 với ngày fact bị xáo |

### 2.2 Quy trình chung (bằng chứng có tập validation)
1. Mỗi điểm lưới × 5 seed (2019–2023).
2. Huấn luyện trên train; Recall@20 val mỗi 5 epoch; dừng sớm (patience 5 với BPR-MF, 10 với KGAT); giữ epoch tốt nhất trên val; tối đa 300 epoch.
3. Chọn điểm lưới có trung bình Recall@20 val cao nhất qua 5 seed.
4. Chấm test **một lần**; mọi dòng val và test ghi vào `results/registry.csv` (chỉ thêm, có git commit).
5. T-KGAT: giữ (lr, l2) của KGAT, chỉ dò γ ∈ {1, 3, 10}.

### 2.3 Bảng siêu tham số và lưới (từ registry; trung bình val Recall@20 qua 5 seed)

**BPR-MF** (`docs/bprmf-training-table.md`): batch 2.048; tối đa 300 epoch; đánh giá mỗi 5; patience 5; lr 1e-3.

| dim | l2 | val Recall@20 | Epoch tốt nhất (qua seed) |
|---|---|---|---|
| 32 | 1e-4 | 0,0391 | 45–115 |
| 64 | 1e-4 | 0,0503 | 135–200 |
| **64** | **1e-3** | **0,0559** | 120–210 |

Điểm chọn nằm ở **mép lưới** (dim lớn nhất, l2 lớn nhất): chưa biết l2 lớn hơn có tốt hơn không. Epoch đã chạy 145–235.

**KGAT** (`configs/model/kgat.yaml`): dim 64, kge_dim 64, lớp [64, 32, 16], mess_dropout 0,1, l2_kg 1e-5, batch CF 1.024 và KG 2.048, tối đa 300 epoch, đánh giá mỗi 5, patience 10.

| lr | l2 | val Recall@20 | Epoch tốt nhất |
|---|---|---|---|
| 1e-4 | 1e-4 | 0,0433 | 275–300 |
| 1e-3 | 1e-4 | 0,0555 | 80–300 |
| 1e-4 | 1e-5 | 0,0434 | 275–300 |
| **1e-3** | **1e-5** | **0,0561** | 125–240 |

Hai điểm lr = 1e-4 có epoch tốt nhất sát trần 300 nên chưa hội tụ trong 300 epoch (không được chọn). `kgat_nokg`: điểm chọn cùng (lr 1e-3, l2 1e-5), val 0,0553 (điểm l2 1e-4 là 0,0552, gần như bằng nhau).

**T-KGAT** (giữ lr, l2 của KGAT; val Recall@20 theo γ):

| | γ = 1 | γ = 3 | γ = 10 |
|---|---|---|---|
| Stage 1 | 0,0557 | 0,0569 | **0,0571** |
| Stage 2 | **0,0576** | 0,0570 | 0,0566 |

Stage 1 chọn γ = 10 (mép trên), Stage 2 chọn γ = 1 (mép dưới); khác biệt giữa các γ chỉ 0,0002–0,0014, cùng cỡ nhiễu seed. Epoch tốt nhất ở điểm chọn (**tính thêm** từ registry): KGAT 125–240; KGAT không KG 100–275; Stage 1 135–200; Stage 1 hoán vị 80–300 (một seed chạm trần 300); Stage 2 95–210; Stage 2 hoán vị 85–240.

### 2.4 Tính toán và Kaggle (bắt buộc nói; quyết định 0013, 0014)
- Laptop RTX 3050 4 GB: KGAT trên `lastfm-u25` 20,13 và 20,52 giây mỗi epoch (đo 06/11); T-KGAT bằng KGAT (0,05–0,07 giây thêm mỗi epoch).
- Tổng thời gian thực (cột `seconds` của dòng val, **tính thêm**): laptop khoảng 59,1 giờ; Kaggle T4 khoảng 16,2 giờ. KGAT riêng 23,2 giờ trên laptop.
- **Stage 2 chạy trên Kaggle** (Tesla T4, commit `ce55c0c`, cùng hash split): cổng đo là 14 giây mỗi epoch; đo được 13,81 giây. **Cổng ban đầu viết 12 giây; sau khi thấy số đo cổng được nâng lên 14 giây theo yêu cầu của Triết**: thay đổi có chủ đích, không phải lần qua cổng đăng ký trước. Ước lượng ban đầu cho `tkgat_kg` khoảng 9,9 giờ, thực tế khoảng 12,3 giờ; phiên 1 chạm trần 12 giờ sau 14/15 lần; phiên 2 (`--resume`) chạy nốt lần còn lại; phiên 3 chạy `tkgat_kg_perm`.
- **Hệ quả cho cách đọc:** so Stage 2 với KGAT hoặc Stage 1 là so giữa hai máy (T4 và RTX 3050), phải kèm ghi chú; so `tkgat_kg` với `tkgat_kg_perm` là cùng máy.
- Ngày 02/11 cô gợi ý dùng Kaggle trong nhóm chat; các chi tiết cụ thể (Stage 2 trên T4, ước lượng thấp, nâng cổng 12 lên 14 giây) **chưa báo cô** và được báo trong buổi này (0014, mục 6).

### 2.5 Tái lập
- Chưa có tag `v0.2-kgat`; chưa chạy lại độc lập từ clone sạch trên máy khác. BPR-MF tái lập được trên GPU; KGAT không tái lập từng bit (`torch.sparse.mm`), tiêu chí khớp là trung bình 5 seed trong một độ lệch chuẩn.
- Hình learning curve: `docs/figures/fig5_bprmf_curve_lastfm-u25.png` (loss và val Recall@20 của BPR-MF, vạch chấm là epoch tốt nhất từng seed) và `docs/figures/rq3_curves_lastfm-u25.png` (val Recall@20 của KGAT, T-KGAT, hoán vị; ba đường chồng lên nhau và phẳng dần sau khoảng epoch 100).

**Cô có thể hỏi:**
- *"Tập val dùng vào việc gì?"* — Dừng sớm, chọn điểm lưới, chọn γ, chọn cửa sổ MostPop recent. Test chấm một lần.
- *"Sao chọn lưới đó, và điểm chọn ở mép thì sao?"* — Lưới đăng ký từ MovieLens để giữ protocol; điểm ở mép nghĩa là có thể còn tốt hơn ngoài lưới; chưa mở rộng; ghi là hạn chế.
- *"T-KGAT khác KGAT chỗ nào?"* — Trừ γ·Δt khỏi điểm attention trước softmax, chỉ trên cạnh user–item; γ = 0 là KGAT.
- *"Vì sao Stage 2 chạy trên máy khác?"* — Laptop quá chậm; Kaggle GPU; cổng đo 14 giây được nâng từ 12 sau khi thấy số đo; số được ghi kèm cột `device`.
- *"Sao không chạy lại trên train + val?"* — Quyết định từ đầu để đơn giản và dễ kiểm chứng.

---

## Phần 3 — Kết quả và nhận xét của `lastfm-u25`, minh bạch và hạn chế (4,5 phút)

### 3.1 Bảng chính (Recall@20 / NDCG@20 test, 1.608 user, trung bình ± std qua 5 seed; `docs/rq3-table.md`)

| Mô hình | Recall@20 | NDCG@20 |
|---|---|---|
| MostPop | 0,0238 | 0,0191 |
| MostPop recent (14 ngày) | 0,0211 | 0,0169 |
| BPR-MF | 0,0520 ± 0,0028 | 0,0435 ± 0,0030 |
| KGAT | 0,0534 ± 0,0008 | 0,0444 ± 0,0008 |
| KGAT không KG | 0,0518 ± 0,0016 | 0,0426 ± 0,0004 |
| T-KGAT Stage 1 (γ = 10) | 0,0529 ± 0,0025 | 0,0431 ± 0,0016 |
| T-KGAT hoán vị | 0,0534 ± 0,0024 | 0,0441 ± 0,0017 |
| T-KGAT Stage 2 (γ = 1) | 0,0513 ± 0,0018 | 0,0426 ± 0,0014 |
| Stage 2 hoán vị ngày fact | 0,0537 ± 0,0020 | 0,0436 ± 0,0016 |

**Kiểm định cặp** (Wilcoxon theo user, 1.608 user, 651–929 chênh lệch khác 0, không cảnh báo; p Holm Recall / NDCG; hiệu = trước − sau):

| So sánh | p Holm Recall | p Holm NDCG | Hiệu Recall / NDCG |
|---|---|---|---|
| T-KGAT vs KGAT | 0,420 | 0,230 | −0,0005 / −0,0012 |
| T-KGAT thật vs hoán vị | 0,803 | 0,506 | −0,0005 / −0,0009 |
| KGAT vs KGAT không KG | 0,803 | 0,506 | +0,0016 / +0,0017 |
| Stage 2 vs KGAT | 0,278 | 0,523 | −0,0021 / −0,0018 |
| Stage 2 thật vs hoán vị | 0,278 | 0,442 | −0,0024 / −0,0010 |

Baseline: BPR-MF > MostPop và BPR-MF > MostPop recent, đều rất có ý nghĩa (Recall p Holm nhỏ hơn 10⁻³⁵); MostPop vs MostPop recent: Recall p Holm 0,103, NDCG 0,016 (MostPop thường **cao hơn**).

### 3.2 Bốn nhận xét (mỗi nhận xét: số → ý nghĩa → giải thích)

**Nhận xét 1 — Baseline.** BPR-MF gần gấp đôi MostPop (0,0520 so với 0,0238) và thắng rất có ý nghĩa. MostPop recent (14 ngày chọn trên val: 0,0247 so với 0,0192 của MostPop) lại **không** thắng MostPop thường ở test (0,0211 so với 0,0238), chiều ngược với ML-100K. Giải thích chỉ là giả thuyết chưa kiểm: cửa sổ 14 ngày neo ở cuối train, còn test bắt đầu ở t2 (khoảng 39 ngày sau t1) và kéo dài khoảng 86 ngày, nên độ "hot" gần cuối train có thể đã nguội. Mô hình thời gian vẫn phải vượt được cả hai.

**Nhận xét 2 — KG (RQ2).** KGAT hơn KGAT không KG 0,0016 Recall@20 và 0,0017 NDCG@20, **không có ý nghĩa** (p Holm 0,803 và 0,506). KGAT (0,0534) cũng chỉ nhỉnh hơn BPR-MF (0,0520) 0,0013, nhỏ hơn độ lệch chuẩn qua seed của BPR-MF (0,0028); cặp này **chưa nằm trong họ kiểm định** nên chưa có p. Đọc: chưa có bằng chứng KG giúp trên Last-FM. Phải nói kèm: 19,8% fact không có ngày, KG Freebase 2015, KG t1 = t2.

**Nhận xét 3 — Thời gian, Stage 1 (RQ3).** T-KGAT không khác KGAT (p Holm 0,420 / 0,230) và không khác bản hoán vị (0,803 / 0,506); bản hoán vị còn nhỉnh hơn 0,0005. Dữ liệu Last-FM có Δt biến thiên thật (mục 1.4), nên không thể giải thích âm tính này bằng "Δt gần như bằng 0" như ở MovieLens. Đọc: chưa có bằng chứng thời gian trên cạnh user–item giúp.

**Nhận xét 4 — Thời gian, Stage 2.** Stage 2 thấp hơn KGAT 0,0021 và thấp hơn bản hoán vị ngày fact 0,0024, đều không có ý nghĩa (p Holm 0,278). Khác MovieLens, thang Δt của KG ở đây có độ phân giải (24,4% triple trong cửa sổ), nên đây là phép đo thật cho "fact cũ hay mới", dù hơn 2/3 fact có ngày bị chặn ở 1. Lưu ý Stage 2 chạy trên Kaggle T4, KGAT trên laptop (so sánh hai máy, 0014).

**Mức độ chắc chắn:** với 1.608 user kiểm định đủ sức để nói chênh lệch nếu có thì nhỏ, cỡ vài phần nghìn Recall@20, nhỏ hơn độ nhiễu qua seed (std 0,0008–0,0028). Đây vẫn không phải bằng chứng "bằng nhau".

### 3.3 Minh bạch (bắt buộc nói)
1. Quyết định chuyển sang Last-FM được đưa ra sau khi điểm test MovieLens đã được xem (0012).
2. Kết quả KGAT/T-KGAT là sơ bộ: bảng RQ3 ghi "chạy trước khi cô duyệt giao thức t1/t2"; chưa tag `v0.2-kgat`; PR #42 còn draft.
3. Stage 2 chạy trên máy khác, cổng được nâng 12 → 14 giây sau khi thấy số đo, ước lượng giờ chạy thấp hơn thực tế (0014).
4. Các phân tích bổ sung đã đề xuất với cô (hoán vị theo khối thời gian, nhóm từ 01/04/2014, biến thể `drop_undated`, γ ∈ {30, 100} chỉ trên val, MostPop và BPR-MF trên Last-FM đầy đủ) **chưa chạy**; mỗi cái cần quyết định riêng và sẽ được ghi là thêm sau khi đã thấy điểm test (0014).

### 3.4 Hạn chế (một slide)
1. Tập cặp (user, item) của KGAT được chọn bằng thông tin sau mốc 01/04/2014; ảnh hưởng chưa đo; dựng lại tập cặp (phương án C của buổi 14/11, "phương án A" của 0013) chưa chạy (0012, 0013).
2. Chỉ 25% user (seed 2019); mất 28,8% val và 28,1% test vì cold item; không nói cho toàn Last-FM.
3. Timestamp là lần nghe sớm nhất của cặp; khoảng thời gian train (02/2005–04/2014), val (khoảng 39 ngày) và test (khoảng 86 ngày) rất lệch nhau.
4. Freebase là bản chụp 09/08/2015; KG không có ngày của statement; 19,8% fact không có ngày; KG t1 = KG t2 nên không kiểm được rò rỉ giữa hai mốc.
5. KGAT không tái lập từng bit trên GPU; chưa tái lập độc lập; không có tag `v0.2-kgat`.
6. Điểm được chọn ở mép lưới (BPR-MF l2 = 1e-3; KGAT lr = 1e-3; γ = 10 và γ = 1); một lần chạy hoán vị chạm trần 300 epoch.
7. Hoán vị toàn cục là một dạng đối chứng; biến thể theo khối thời gian chưa chạy.
8. Stage 2 so với KGAT và Stage 1 là so giữa hai máy; BPR-MF so với KGAT chưa kiểm định.
9. Số của nhóm không so trực tiếp với số trong paper KGAT (chia theo thời gian, 5-core, KG Freebase thêm ngày).

**Cô có thể hỏi:**
- *"Vậy mô hình thời gian vô dụng à?"* — Chưa có bằng chứng nó giúp trên hai bộ dữ liệu với hai cấu hình; trên Last-FM có Δt thật và test lớn, nên kết quả âm có trọng lượng hơn MovieLens; nhưng p lớn không chứng minh bằng nhau và vẫn còn các đối chứng chưa chạy.
- *"Kết quả âm thì đóng góp gì?"* — Quy trình đánh giá không rò rỉ cho KG có thời gian; đo được khi nào thời gian có độ phân giải (Last-FM) và khi nào không (MovieLens); đối chứng hoán vị chỉ ra cải thiện nhỏ không đến từ thời gian. Biên bản 07/08 ghi kết quả âm hợp lệ.
- *"Sao MostPop recent không thắng MostPop?"* — Quan sát có thật (0,0211 so với 0,0238; NDCG p Holm 0,016); giải thích còn là giả thuyết về cửa sổ 14 ngày so với khoảng cách tới test.
- *"Nhóm có lọc sai fact tương lai không?"* — Chỉ 3 triple có ngày sau t1 và bị lọc; KG t1 = t2; nên Last-FM không đo được rò rỉ qua KG.
- *"Vì sao γ lại ở mép lưới?"* — Lưới đăng ký từ MovieLens; chênh lệch val giữa các γ 0,0002–0,0014 ≈ nhiễu; chưa mở rộng.
- *"Số của em khác paper KGAT nhiều?"* — Khác cách chia (thời gian, không ngẫu nhiên), 5-core, 25% user, nên không so trực tiếp; Recall@20 khoảng 0,05 là cỡ số của thiết lập này.

---

## Phần 4 — Ba vấn đề của `lastfm-u25` và bản `lastfm-u25-latest` (6 phút) · Phôn

Phân bổ thời gian đề xuất: 4.1 (1'), 4.2 và 4.3 (1'), 4.4 (1'), 4.5 (1'), 4.6 (1,5'), 4.7 (0,5').

### 4.1 Ba vấn đề của `lastfm-u25` (notebook 6.1; 11, mục 2.8 và 12.1)
- **A. Tập cặp chọn bằng thông tin sau 01/04/2014** (quyết định 0012): tập cặp mà tác giả KGAT công bố chỉ chứa cặp còn được nghe từ 01/04/2014. Có thể thiên vị cặp "sống lâu". Tác động **chưa đo**.
- **B. Bước nhảy số cạnh quanh 01/04/2014** (dấu vết nhìn thấy của A; `train.csv` của `lastfm-u25`): trung bình 2.494 cạnh mỗi tuần trong 4 tuần trước 31/03/2014 và 12.416 trong 4 tuần từ đó (gấp 5,0 lần); 21,7% cạnh train có timestamp từ 01/04/2014 trở đi, dù cửa sổ train dài 3.361 ngày.
- **C. `check_split` không thấy được A và B**: nó báo 0 vấn đề trên `lastfm-u25`, vì lỗi nằm trước khi chia.

### 4.2 Bản latest là gì (notebook 6.0; 11, mục 12.1 và 12.2)
- Cùng user (25%, seed 2019), cùng t1/t2 theo phân vị, 5-core, KG, lưới, 5 seed; cùng 370.979 cặp. **Chỉ đổi:** timestamp của cặp là lần nghe **muộn nhất**.
- Hết bước nhảy B: cạnh train của latest đều từ 01/04/2014; train bắt đầu đúng 01/04/2014.
- **A còn nguyên** (tập cặp không đổi). **Hình thức thiên lệch đổi chứ không mất** *(suy từ định nghĩa, không phải số đo)*: một cặp nằm trong train khi và chỉ khi user không nghe cặp đó từ t1 trở đi (trong dữ liệu LFM-1b); cặp còn được nghe sau t1 chỉ có thể ở val hoặc test. `check_split` báo 0 vấn đề trên latest, và con số 0 này không chứng minh tập cặp không thiên lệch.
- Nguồn: quyết định 0015 là nháp của Phôn (chưa accepted, chưa vào repo); không dòng nào trong `results/registry.csv`; registry gộp 323 dòng, 132 dòng riêng biệt, commit `873ca7f`.

### 4.3 So sánh hai bản (notebook 6.2; 11, mục 12.3)

| | `lastfm-u25` | `lastfm-u25-latest` |
|---|---|---|
| Timestamp của cặp | lần nghe sớm nhất | lần nghe muộn nhất |
| t1 / t2 | 29/04/2014 / 07/06/2014 | 23/06/2014 / 19/07/2014 |
| User / item train | 5.321 / 18.665 | 5.367 / 19.165 |
| Positive train | 236.053 | 236.339 |
| Positive val (user) | 23.708 (2.319) | 25.860 (1.516) |
| Positive test (user) | 22.419 (1.608) | 24.244 (1.341) |
| Val / test mất vì cold item | 28,8% / 28,1% | 24,2% / 21,9% |
| Val / test mất vì cold user | 7,3% / 11,5% | 6,1% / 12,8% |
| KG default tại t1 (triple) / thực thể / % không ngày | 152.976 / 44.392 / 19,8% | 156.826 / 45.346 / 19,9% |
| Train: từ ngày – đến ngày | 14/02/2005 – 29/04/2014 | 01/04/2014 – 23/06/2014 |
| Train / val / test (ngày) | 3.361 / 39 / 86 | 83 / 26 / 45 |

Điểm cần nói: cùng 370.979 cặp, nhưng t1/t2 khác hẳn; cửa sổ train của latest chỉ 83 ngày; val và test ngắn (26 và 45 ngày); số user test 1.341 so với 1.608. Payload dữ liệu của Phôn chỉ có biến thể KG `default`.

### 4.4 Stage 1 trên latest (notebook 6.3; 11, mục 12.4)
- Số cạnh theo tuần: tuần đầy đủ đầu tiên 39.197 cạnh, tuần đầy đủ cuối cùng 7.717 (giảm dần về cuối cửa sổ; nguyên nhân chưa kiểm).
- Trung vị `dt_norm` 0,73 (u25: 0,10); `dt_norm` < 0,25: 10,2% (u25: 78,8%) *(tính thêm cho u25)*; cạnh trong 7 ngày cuối 3,3% (u25: 4,3%); trong 30 ngày cuối 16,2% (u25: 22,0%).
- 0,93% user có mọi cạnh trong 1 giờ (u25: 0,53%); trung vị độ lệch chuẩn thời gian theo user 9,2 ngày (u25: 249,6 ngày).
- **Một đơn vị `dt_norm` là 83 ngày** (u25: 3.361 ngày): cùng γ trong lưới {1, 3, 10} không cùng nghĩa theo ngày trên hai bản.

### 4.5 Stage 2 trên latest (notebook 6.4; 11, mục 12.5)
- 0,0% triple trong cửa sổ, 80,1% bị chặn ở 1, 19,9% không có ngày (u25: 24,4% / 55,7% / 19,8%). Trung vị `dt_norm` của fact có ngày: 59,53 (u25: 1,45).
- Fact có ngày mới nhất: 31/12/2013, sớm hơn ngày đầu cửa sổ train (01/04/2014) 91 ngày.
- **Hệ quả (suy từ phép chặn ở 1):** mọi fact có ngày có cùng `dt_norm` = 1; Stage 2 chỉ phân biệt "có ngày" với "không có ngày"; bản hoán vị ngày fact chỉ xáo xem triple nào là "không có ngày". Stage 2 trên latest **không** trả lời "fact cũ hay mới".

### 4.6 Kết quả trên latest (notebook 6.5; 11, mục 12.6)

Recall@20 / NDCG@20 trên test, 1.341 user, trung bình ± std mẫu qua 5 seed (`docs/rq3-table-lastfm-latest.md`):

| Mô hình | Recall@20 | NDCG@20 |
|---|---|---|
| MostPop | 0,0296 | 0,0268 |
| MostPop recent | chưa chạy | chưa chạy |
| BPR-MF | 0,0609 ± 0,0051 | 0,0618 ± 0,0054 |
| KGAT | 0,0673 ± 0,0029 | 0,0672 ± 0,0022 |
| KGAT không KG | 0,0633 ± 0,0046 | 0,0658 ± 0,0044 |
| T-KGAT Stage 1 (γ = 1) | 0,0676 ± 0,0037 | 0,0669 ± 0,0031 |
| T-KGAT hoán vị | 0,0651 ± 0,0049 | 0,0645 ± 0,0034 |
| T-KGAT Stage 2 (γ = 10) | 0,0661 ± 0,0018 | 0,0653 ± 0,0015 |
| Stage 2 hoán vị ngày fact | 0,0650 ± 0,0033 | 0,0645 ± 0,0017 |

| So sánh (hiệu = trước − sau) | Hiệu Recall / NDCG | p Holm Recall / NDCG |
|---|---|---|
| T-KGAT vs KGAT | 0,0003 / −0,0002 | 0,733 / 0,910 |
| T-KGAT thật vs hoán vị | 0,0024 / 0,0025 | 0,033 / 0,0046 |
| KGAT vs KGAT không KG | 0,0040 / 0,0013 | 0,028 / 0,225 |
| Stage 2 vs KGAT | −0,0011 / −0,0019 | 0,270 / 0,161 |
| Stage 2 thật vs hoán vị | 0,0011 / 0,0008 | 0,843 / 0,554 |

1.341 user, `n_nonzero` từ 643 đến 855, không cảnh báo cỡ mẫu. Mọi mô hình học sâu chạy trên Tesla T4 (một loại máy).

**Bốn nhận xét (nói đúng từng chữ):**
1. **KG:** KGAT hơn KGAT không KG ở Recall@20 **có ý nghĩa** sau Holm (p Holm 0,028), NDCG@20 **không** (0,225). **Khác `lastfm-u25`** (cả hai không có ý nghĩa). Sơ bộ, trên dữ liệu có thiên lệch chọn mẫu; chưa đủ để nói KG giúp.
2. **Stage 1:** T-KGAT không hơn KGAT (p Holm 0,733 / 0,910). T-KGAT thật hơn hoán vị có ý nghĩa (0,033 / 0,0046), **nhưng bản hoán vị thấp hơn KGAT 0,0021**, nên điều đó **không chứng minh thời gian giúp**: khoảng cách có thể chỉ phản ánh việc xáo timestamp làm mô hình tệ đi.
3. **Stage 2:** không so sánh nào có ý nghĩa; thang Δt không có độ phân giải (4.5), nên kết quả không nói gì về "fact cũ hay mới".
4. **Không so điểm tuyệt đối với `lastfm-u25`:** test khác (1.341 so với 1.608 user; test 45 so với 86 ngày; cold item khác; phép chia khác). Latest có Recall cao hơn không phải bằng chứng latest tốt hơn.

### 4.7 Hạn chế của latest (notebook 6.6; 11, mục 12.7)
1. Không loại được A; hình thức thiên lệch đổi; `check_split` không thấy.
2. Phép chia lệch: train 83 ngày, val 26, test 45.
3. Stage 2 không có độ phân giải.
4. Cold item 24,2% val và 21,9% test; chỉ 25% user.
5. Không tái lập được chỉ từ repo (code và registry của Phôn ngoài repo; 0015 nháp).
6. MostPop recent chưa chạy; payload chỉ có KG `default`.
7. Freebase 09/08/2015; 19,9% triple không có ngày; KGAT không tái lập từng bit; điểm chọn γ ở mép lưới (Stage 1 mép dưới, Stage 2 mép trên).
8. Hoán vị toàn cục chỉ là một dạng đối chứng; sơ bộ, chưa có tag.

**Cô có thể hỏi:**
- *"Bản latest khác bản trước chỗ nào, vì sao đổi?"* — Chỉ timestamp: lần nghe muộn nhất thay vì sớm nhất; để hết bước nhảy số cạnh quanh 01/04/2014 (B). Nhóm thấy nó không xử lý được A.
- *"Latest hết thiên lệch chưa?"* — Chưa. Tập cặp không đổi (A còn nguyên); thiên lệch đổi hình thức (cặp nằm ở train khi và chỉ khi không còn được nghe từ t1). Tác động chưa đo; `check_split` báo 0 vấn đề nhưng không thấy được lỗi này.
- *"Sao Recall của latest cao hơn?"* — Không so điểm tuyệt đối: test khác, phép chia khác (4.3). Chưa tách được nguyên nhân; nhóm không dùng nó làm bằng chứng latest tốt hơn.
- *"KG có giúp trên latest không?"* — Recall@20 có ý nghĩa (p Holm 0,028), NDCG@20 không (0,225); khác `lastfm-u25`; sơ bộ; chưa đủ để kết luận.
- *"T-KGAT hơn hoán vị có ý nghĩa, vậy thời gian giúp chứ?"* — Chưa: T-KGAT không hơn KGAT, và hoán vị thấp hơn KGAT 0,0021.
- *"Stage 2 trên latest cho biết gì?"* — Không gì về fact cũ hay mới: 0,0% triple trong cửa sổ; chỉ so "có ngày" với "không có ngày".
- *"Sao số của latest không có trong registry?"* — Registry chỉ thêm dòng và có commit của repo; số của Phôn ở commit `873ca7f` của lịch sử riêng; thêm dòng cần quyết định riêng. PR #43 chỉ có bảng tóm tắt và nguồn gốc (`docs/lastfm-latest-provenance.md`).

---

## Phần 5 — Kế hoạch và ba câu hỏi (2,5 phút) · Cả hai
(Notebook mục 7.)

**Kế hoạch (đề xuất; cần cô chọn):** (1) chốt bản dữ liệu chính thức; (2) nếu cô chọn bản của Phôn: đưa code xuất dữ liệu và quyết định 0015 vào repo bằng PR riêng để duyệt, rồi thêm dòng registry theo quyết định riêng; (3) hoàn tất tái lập độc lập và tag `v0.2-kgat`; (4) chọn một số phân tích bổ sung theo ý cô; (5) viết khóa luận.

**Ba câu hỏi xin ý kiến cô:**
1. **Chọn bản dữ liệu chính thức.** Phương án **A**: giữ `lastfm-u25` (đã chạy đủ, có trong registry; giữ vấn đề A và B kèm lời báo). Phương án **B**: dùng `lastfm-u25-latest` (hết B; A còn nguyên và thiên lệch đổi hình thức; Stage 2 không có độ phân giải; cần duyệt 0015 và đưa code, registry của Phôn vào repo). Phương án **C**: dựng lại tập cặp từ sự kiện thô của LFM-1b, không giới hạn bởi cửa sổ tháng 4/2014 (trước đây gọi là "phương án A" ở quyết định 0013 mục 7; xử lý gốc của A nhưng cần quyết định mới và thêm thời gian tính toán).
2. **Stage 2 và các phân tích bổ sung.** Trên bản nào Stage 2 còn đáng chạy (u25 có độ phân giải, latest không)? Trong các phân tích đã đề xuất (hoán vị theo khối thời gian, nhóm từ 01/04/2014, `drop_undated`, γ ∈ {30, 100} chỉ trên val, MostPop và BPR-MF trên Last-FM đầy đủ), cô ưu tiên cái nào? Mỗi cái có quyết định riêng và được báo là thêm sau khi đã thấy test.
3. **Kết quả sơ bộ.** Giữ kết quả của bản được chọn làm chính thức sau khi tái lập độc lập và gắn tag, hay cô muốn chỉnh giao thức rồi chạy lại? (Kết quả `latest` hiện chỉ là bảng tóm tắt từ registry của Phôn.)

---

## Phụ lục A — Thẻ số một trang (in ra mang theo)

**`lastfm-u25`** (test 1.608 user)

| Chủ đề | Số |
|---|---|
| Dữ liệu | 1.542.856 cặp (đầy đủ); `lastfm-u25` 370.979 cặp, 5.892 trong 23.566 user |
| t1, t2 | 29/04/2014, 07/06/2014 |
| Train / val / test positive | 236.053 / 23.708 / 22.419 |
| User train / val / test; item train | 5.321 / 2.319 / 1.608; 18.665 |
| Cold item val / test; cold user val / test | 28,8% / 28,1%; 7,3% / 11,5% (đầy đủ cold item: 3,0% / 4,6%) |
| Bước nhảy B | 2.494 → 12.416 cạnh mỗi tuần (gấp 5,0 lần); 21,7% cạnh train từ 01/04/2014 |
| KG | 464.567 → 152.979 (bỏ 1.097, 0,7%); 44.392 thực thể; 9 quan hệ; fact không ngày 30.323 (19,8%); bộ lọc thời gian 3 triple; KG t1 = t2 |
| Thang Δt Stage 2 | 24,4% trong cửa sổ; 55,7% chặn ở 1; 19,8% không ngày |
| Stage 1: Δt_norm trung vị; 30 ngày cuối *(tính thêm)* | 0,100; 22,0% |
| User có mọi cạnh trong 1 giờ *(tính thêm)* | 0,53% (ML-100K: 53,0%) |
| dt_den | 290.426.881 giây = 9,20 năm |
| Điểm đã chọn | BPR-MF dim 64, l2 1e-3; KGAT lr 1e-3, l2 1e-5; γ = 10 (Stage 1), γ = 1 (Stage 2); MostPop recent 14 ngày |
| MostPop / recent | 0,0238 / 0,0211 |
| BPR-MF | 0,0520 ± 0,0028 |
| KGAT / không KG | 0,0534 ± 0,0008 / 0,0518 ± 0,0016 |
| T-KGAT / hoán vị | 0,0529 ± 0,0025 / 0,0534 ± 0,0024 |
| Stage 2 / hoán vị ngày fact | 0,0513 ± 0,0018 / 0,0537 ± 0,0020 |
| p Holm Recall: T-KGAT vs KGAT; vs hoán vị; KGAT vs không KG | 0,420; 0,803; 0,803 |
| p Holm Recall: Stage 2 vs KGAT; vs hoán vị | 0,278; 0,278 |
| Registry | 139 dòng, commit `ce55c0c`, không dòng `-dirty` |
| Thời gian chạy *(tính thêm)* | Laptop khoảng 59,1 giờ; Kaggle T4 khoảng 16,2 giờ |
| Kaggle | cổng 14 giây (gốc 12); đo 13,81 giây; ước lượng 9,9 giờ, thực 12,3 giờ |

**`lastfm-u25-latest`** (test 1.341 user)

| Chủ đề | Số |
|---|---|
| Dữ liệu | 370.979 cặp; timestamp = lần nghe muộn nhất |
| t1, t2 | 23/06/2014, 19/07/2014 |
| Train / val / test positive | 236.339 / 25.860 / 24.244 |
| User train / val / test; item train | 5.367 / 1.516 / 1.341; 19.165 |
| Cold item val / test; cold user val / test | 24,2% / 21,9%; 6,1% / 12,8% |
| Train / val / test (ngày) | 83 / 26 / 45 (u25: 3.361 / 39 / 86) |
| KG | 156.826 triple tại t1; 45.346 thực thể; 19,9% không ngày; chỉ có biến thể `default` |
| Stage 1 | trung vị `dt_norm` 0,73; 7 ngày cuối 3,3%; 30 ngày cuối 16,2%; user mọi cạnh trong 1 giờ 0,93%; độ lệch chuẩn theo user 9,2 ngày; cạnh theo tuần 39.197 → 7.717 |
| Stage 2 | 0,0% trong cửa sổ; 80,1% chặn ở 1; 19,9% không ngày; fact mới nhất 31/12/2013, sớm hơn đầu cửa sổ 91 ngày |
| Điểm đã chọn | BPR-MF dim 64, l2 1e-3; KGAT lr 1e-3, l2 1e-4; γ = 1 (Stage 1), γ = 10 (Stage 2) |
| MostPop | 0,0296 (MostPop recent chưa chạy) |
| BPR-MF | 0,0609 ± 0,0051 |
| KGAT / không KG | 0,0673 ± 0,0029 / 0,0633 ± 0,0046 |
| T-KGAT / hoán vị | 0,0676 ± 0,0037 / 0,0651 ± 0,0049 |
| Stage 2 / hoán vị ngày fact | 0,0661 ± 0,0018 / 0,0650 ± 0,0033 |
| p Holm Recall / NDCG: T-KGAT vs KGAT | 0,733 / 0,910 |
| p Holm Recall / NDCG: T-KGAT vs hoán vị | 0,033 / 0,0046 |
| p Holm Recall / NDCG: KGAT vs không KG | 0,028 / 0,225 |
| p Holm Recall / NDCG: Stage 2 vs KGAT; vs hoán vị | 0,270 / 0,161; 0,843 / 0,554 |
| Hoán vị so với KGAT (Recall) | −0,0021 |
| Registry gộp | 323 dòng, 132 riêng biệt, commit `873ca7f`, 0 `-dirty`; không có trong `results/registry.csv` |
| Thiết bị | Tesla T4 cho mọi mô hình học sâu (tổng 55,1 giờ, dòng val); MostPop trên CPU |

## Phụ lục B — Lịch ôn đến buổi báo cáo (rehearsal 13/11, báo cáo 14/11 lúc 13:00)

| Khi nào | Việc |
|---|---|
| Chiều 13/11 | Đọc tài liệu 11 mục 0, 2.8, 5.3–5.4, 6 và **mục 12**; đọc kỹ phần 1.4, 2.4 và **phần 4** của tài liệu này |
| Tối 13/11 | Thuộc Phụ lục A (cả hai khối); tự trả lời 42 câu cuối tài liệu 11; tập dượt theo tài liệu 13, bấm giờ; lượt 2 đổi vai |
| Sáng 14/11 | Đọc lại bốn thông điệp, bốn nhận xét của `lastfm-u25` và bốn nhận xét của `latest`, các hạn chế, ba câu hỏi |
