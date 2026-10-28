# Thiết kế KGAT (PyTorch) cho t-kgat

- Ngày: 2026-10-27. Trạng thái: **Triết đã duyệt Q1–Q14** (mục "Đã chốt" bên dưới); Phôn duyệt từng PR theo CONTRIBUTING.
- Bản đề xuất gốc được chép nguyên từ `E:\KLTN HCMUS\design-kgat.md` (ngoài repo) sang file này, kèm ba chỉnh sửa đã ghi ở
  bảng 2.6 và mục 5.3. Các mục 0–7 và phụ lục vẫn là văn bản đề xuất; khi mâu thuẫn với "Đã chốt" thì "Đã chốt" thắng.
- Thuộc mục nào của `docs/plan.md` §12: **không có dòng nào trong §12 (#1–#24)**. §12 chỉ phủ 18–31/10 và ghi rõ
  "KGAT chỉ bắt đầu từ 02/11". Việc này là **chuẩn bị cho giai đoạn 4** (bảng phân công §10: "KGAT (viết lại, dừng sớm
  theo val)", Phôn làm chính; Triết chuẩn bị KG). Vì vậy đây chỉ là tài liệu thiết kế; code sẽ vào nhánh riêng sau 31/10.
- Dự kiến file B2 sẽ thêm/sửa (chưa làm gì): `src/tkgat/kg/ckg.py` (dựng CKG), `src/tkgat/models/kgat.py`,
  `src/tkgat/train/kgat.py`, `configs/model/kgat*.yaml`, `scripts/run.py` (thêm nhánh `kgat`), `tests/test_ckg.py`,
  `tests/test_kgat.py`. Decision 0009 (Δt của T-KGAT) sẽ soạn sau; Q1 và Q2 không cần decision mới (xem "Đã chốt").

## Đã chốt (Q1–Q14, mục 7)

Triết duyệt ngày 2026-10-27. Không có quyết định nào ở đây đổi luật 1–6 của CLAUDE.md, nên PR đầu tiên không kèm decision record.

| # | Chốt | Ghi chú |
|---|---|---|
| Q1 | (a) Lúc test chỉ dùng cạnh `Interact` của train; chỉ KG đổi từ t1 sang t2. | Hệ quả của decision 0002 mục 7 ("không huấn luyện lại trên train+val"): đồ thị lúc test không có cạnh nào mà mô hình chưa thấy khi huấn luyện. Không phải quyết định giao thức mới. |
| Q2 | (a) Lúc test bỏ triple có tail chưa xuất hiện trong `kg_t1`, ghi số triple bỏ. | Quy tắc bảo thủ: embedding của entity chưa từng huấn luyện không được vào điểm test. Tập tail "đã gặp" chỉ tính từ `kg_t1`. Cài trong `kg/ckg.py` (docstring nêu cả Q1 và Q2). |
| Q3 | Viết đủ 4 config (`kgat`, `kgat_drop_undated`, `kgat_static`, `kgat_nokg`); chỉ chạy `kgat` và `kgat_nokg`. | `kgat_static` gắn nhãn tham chiếu rò rỉ (`time_consistent: false`), không báo như kết quả nhất quán thời gian. |
| Q4 | Attention không nhận gradient từ BPR (như TF1). | `gamma` và `delta_r` là siêu tham số, không học được qua BPR. |
| Q5 | `A` ở epoch 1 đồng đều `1/|N_h|`. | |
| Q6 | CF lấy mẫu theo cạnh bằng `randperm` như `fit_bpr`; KG theo cạnh, tail giả lấy từ nút hoạt động. | |
| Q7 | Hai Adam xen kẽ, cùng `lr`. | |
| Q8 | Khởi tạo Xavier, không pretrain. | |
| Q9 | Regulariser theo TF1 (`l2` và `l2_kg` tách). | Không so sánh được với `l2` của BPR-MF. |
| Q10 | `batch_size` 1024, patience 10 lần đánh giá. | Ablation số lớp để sau 30/10. |
| Q11 | Xong: PR #20 đã merge, `ml1m_u10.yaml` có trên `main`. | |
| Q12 | Giữ giao diện: `edge_ts` song song với cạnh, `delta_r` theo quan hệ, mốc Δt = timestamp train lớn nhất cho cả val và test, `gamma = 0` trùng KGAT. | Chưa cài Δt. |
| Q13 | Giữ nguyên "KG 1-hop", ghi trong báo cáo. | |
| Q14 | `kg/ckg.py` (Triết), `models/kgat.py` và `train/kgat.py` (Phôn); Phôn duyệt từng PR. | |

## 0. Nguồn đã đọc, phạm vi, điều chưa chắc

Đã đọc: `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`, decisions 0001–0007, và **0008 (đọc từ nhánh `feat/22-subset-and-kgat-export`
bằng `git show`, không checkout)**, `plan.md` §10–§12, `encode.py`, `kg/temporal.py`, `train/bpr.py`, `eval/metrics.py`,
`models/baselines.py`, `device.py`, `pipeline/build.py`, `scripts/06_build_kg.py`, `scripts/run.py`, `configs/`,
`data/processed/*/meta.json` và `kg/*/kg_meta.json`; code TF1 gốc `Model/KGAT.py`, `Model/Main.py`,
`Model/utility/{load_data,loader_kgat,parser}.py`; `KGAT_Giai_thich_code.md`.

Hạn chế cần biết:
1. **Mình không có PDF của paper.** "Paper" trong tài liệu này = `KGAT_Ban_dich_tieng_Viet.md` (bản dịch có giải thích).
   Các dòng đánh dấu (*) ở bảng 2.6 cần đối chiếu lại với arXiv 1905.07854 trước khi viết vào luận văn.
2. Mọi con số về kích thước (mục 3) mình **tính ngày 2026-10-27 từ `meta.json` và `kg_meta.json`** (KG variant `default`);
   không phải số của `results/registry.csv`, và là **ước lượng**, chưa đo bằng `torch.cuda.max_memory_allocated()`.
3. `ml-1m-u10` (và `u05`) chỉ có config trong nhánh `feat/22-subset-and-kgat-export` (decision 0008 còn "proposed"); dữ liệu
   đã sinh sẵn trong `data/processed/ml-1m-u10/`. Xem Q11.
4. Chưa có quy tắc nào trong CLAUDE.md bị yêu cầu của bạn đụng tới. Hai điểm chạm vào giao thức (cạnh nào dùng ở test,
   thực thể chỉ có ở KG t2) mình **không tự quyết**: ở mục 7 (Q1, Q2) và cần decision record theo luật 6.

---

## 1. Đồ thị CKG (collaborative knowledge graph)

### 1.1 Đầu vào (chỉ những thứ này)
- `EncodedSplit` (train): `n_users` = U, `n_items` = I, `train` (n,2) `[u, i]` chỉ gồm **positive** (rating ≥ 4, trước t1,
  5-core), `train_ts` cùng thứ tự hàng. Cặp (u,i) đã khử trùng lặp nên mỗi cạnh có đúng một timestamp.
- KG của script 06, variant `default`: `<processed_dir>/kg/default/kg_t1.tsv` và `kg_t2.tsv` (cột `h, r, t`, số nguyên,
  **không có quan hệ ngược**), `entities.csv` (E dòng), `relations.csv` (R = 18 dòng), `kg_meta.json`.
  Mọi head là item (`h < I`), mọi tail có `t ≥ I`: KG này là **1-hop** (item → thuộc tính), không có cạnh thuộc tính → thuộc tính.
  Variant khác (`drop_undated`, `static`) dùng cùng cách, chỉ đổi thư mục.

### 1.2 Id nút
Một không gian nút duy nhất, giống TF1 (`load_data`/`loader_kgat`):

| Nút | Id | Ghi chú |
|---|---|---|
| user `u` | `u` (0..U−1) | user của train; val/test chỉ chứa user đã có trong train (quyết định 0002) |
| entity `e` | `U + e` (U..U+E−1) | item `i` là entity `i` (script 06 đặt item trước, đúng thứ tự `EncodedSplit.item_ids`), nên nút item = `U + i` |

N = U + E nút. E lấy từ `entities.csv`, gồm cả entity **không có cạnh nào** ở t1 (xem 1.6).

### 1.3 Id quan hệ
R lấy **từ `relations.csv` (18), không đếm từ các triple quan sát được**. Lý do: TF1 đặt `n_relations = len(adj_r_list)`
nên nếu một quan hệ không còn triple nào (một variant hoặc mốc cắt) thì số quan hệ co lại và id ngược vượt chỉ số bảng
embedding. Ở đây các variant dùng chung từ vựng quan hệ nên phải cố định.

| Id | Ý nghĩa |
|---|---|
| 0 | `Interact` (user → item) |
| 1 … R | quan hệ KG `r` → id `r + 1` |
| R + 1 | `Interact⁻¹` (item → user) |
| R + 2 … 2R + 1 | nghịch đảo của quan hệ KG `r` → id `r + R + 2` |

Tổng `n_rel = 2R + 2 = 38`. Công thức id trùng TF1 (`0`, `r+1`, `n_rel+1`, `r+2+n_rel`), nên log/so sánh với code gốc dễ.

### 1.4 Cạnh
- Với mỗi cặp train `(u, i)`: `(u, 0, U+i)` và `(U+i, R+1, u)`.
- Với mỗi triple `(h, r, t)` của `kg_t1.tsv`: `(U+h, r+1, U+t)` và `(U+t, r+R+2, U+h)`.
- Tổng `E_ckg = 2·(P + T)` (P = số positive train, T = số triple). Không có self-loop, không có cạnh user–user, không có
  cạnh từ val/test.
- Sắp theo `(head, tail, relation)` để mọi cạnh của một head nằm liền nhau (softmax theo head dùng đoạn liên tiếp).
- Hai cạnh cùng `(h, t)` khác quan hệ (ví dụ một người vừa là đạo diễn vừa là biên kịch) **giữ là hai cạnh**, mỗi cạnh có
  điểm attention riêng; khi nhân A với embedding thì giá trị cộng lại (đúng công thức 3 của paper: tổng theo triple).
- Cạnh `Interact` / `Interact⁻¹` mang thêm `edge_ts` (timestamp train của cặp); cạnh KG có `edge_ts = NaN`. Mảng này chỉ
  để dành cho T-KGAT (mục 6); KGAT thuần không đọc.

### 1.5 KG nào cho pha nào
| Pha | Đồ thị | Ai dựng |
|---|---|---|
| huấn luyện | `CKG(train, kg_t1)` | `fit_kgat` |
| chọn epoch / chọn grid (val) | cùng đồ thị trên, attention tính lại | `fit_kgat` |
| chấm test (một lần) | `CKG(train, kg_t2)`, **tham số đã đóng băng**, attention tính lại trên đồ thị mới | chỉ `scripts/run.py`, sau khi chọn |

`fit_kgat` **không nhận** `kg_t2`, `data.test`, `data.mask_test` (chặn bằng chữ ký hàm + test, mục 6).
Cạnh tương tác ở test vẫn là cạnh train (không thêm cạnh val; xem Q1).

### 1.6 Id entity chia sẻ giữa t1 và t2
Script 06 (`assign_ids`) dựng **một từ vựng duy nhất** cho 3 variant và cả hai mốc: entity `0..I−1` là item theo thứ tự
`EncodedSplit.item_ids`, tail xếp theo id Wikidata, lấy từ "universe" = tail của KG `static`. Hệ quả:
- Cùng một số nguyên là cùng một thực thể trong `kg_t1.tsv` và `kg_t2.tsv`; `kg_t1 ⊂ kg_t2` (script 06 đã kiểm). Không cần
  ánh xạ lại khi chuyển sang test; bảng embedding giữ nguyên kích thước E.
- **Entity có trong từ vựng nhưng không dùng ở t1** (tail chỉ xuất hiện trong fact bị loại ở t1): ML-100K 206, ML-1M-u10 296,
  ML-1M 415 (tính từ `kg_meta.json`: `n_entities − n_items − n_tail_entities_used.t1`). Ở đồ thị t1 chúng **cô lập** (bậc 0):
  không có hàng trong A, không nhận gradient nếu negative KG chỉ lấy từ nút đang hoạt động (6, L5). Chúng chỉ là một id, không
  mang thông tin về item nào.
- **Entity chỉ xuất hiện ở KG t2** (`n_entities_only_in_t2`): ML-100K 1 (KG t1 36,560 → t2 36,580 triple, thêm 20); ML-1M và
  ML-1M-u10 0 (KG t2 = KG t1); u05 113. Embedding của chúng chưa từng được huấn luyện. Đề xuất (Q2): **ở test bỏ các triple có tail
  không nằm trong tập tail đã dùng ở t1** (tập này tính chỉ từ `kg_t1`), ghi số triple bỏ vào `kgat_summary.json`.
- Vì KG t2 gần như trùng KG t1 (ML-100K +20 triple = 0,05%; ML-1M +0), **việc đổi sang KG t2 ở test hầu như không đổi kết quả**.
  Khi báo cáo phải nói rõ điều này để không ai đọc nó như "hiệu ứng của KG theo thời gian ở test". Phần lớn tác dụng của
  tính nhất quán thời gian nằm ở bộ lọc t1 (ML-100K bỏ 543 fact).
- Item không có link/không có fact (ML-100K: 5 item không link) chỉ có cạnh `Interact⁻¹`; không sao vì mọi item có ≥ 5 cạnh train.

---

## 2. Công thức

Quy ước: `d = 64` (embedding), `k = 64` (không gian quan hệ), `e_x ∈ R^d`, `e_r ∈ R^k`, `W_r ∈ R^{k×d}`
(code lưu `d×k` và nhân vector hàng; cùng một phép). `⊙` là tích Hadamard, `‖·‖` là chuẩn L2.

### 2.1 TransR và loss KG (công thức 1–2)
```
e_x^r        = W_r e_x
g(h,r,t)     = ‖ e_h^r + e_r − e_t^r ‖²                       (nhỏ = triple đáng tin)
L_KG         = mean_{(h,r,t,t')}  softplus( g(h,r,t) − g(h,r,t') )
             = mean  −ln σ( g(h,r,t') − g(h,r,t) )              (t' = tail giả; "g(sai) − g(đúng)" trong −ln σ)
reg_KG       = (1/2)·mean( ‖e_h^r‖² + ‖e_r‖² + ‖e_t^r‖² + ‖e_t'^r‖² )
loss_KG      = L_KG + l2_kg · reg_KG
```
- Tập triple huấn luyện = **mọi cạnh CKG** (cả `Interact`, cả cạnh ngược), như TF1. Cần vì attention `π(u, Interact, i)` dùng
  `e_u` trong cùng không gian TransR.
- Không ràng buộc chuẩn (TF1 đã comment bỏ `l2_normalize` ở h/r/t).
- `t'`: nút ngẫu nhiên từ tập **nút có ≥ 1 cạnh trong CKG hiện hành**, loại nếu `(h, r, t') ∈ CKG`
  (kiểm bằng khoá `(h·n_rel + r)·N + t'` đã sắp + `searchsorted`, giống `NegativeSampler`).

### 2.2 Attention theo từng head (công thức 4–5)
```
π(h,r,t)  = (W_r e_t)ᵀ · tanh( W_r e_h + e_r )
π̃(h,r,t) = exp π(h,r,t) / Σ_{(h,r',t') ∈ N_h} exp π(h,r',t')      N_h = MỌI cạnh có head h
```
`N_h` gồm cạnh của mọi quan hệ (cả ngược, cả `Interact`): hàng của một user chỉ chứa `Interact`; hàng của một item chứa
`Interact⁻¹` (user đã xem) và các quan hệ KG; hàng của một tail chứa các cạnh ngược về item. Softmax **ổn định**
(trừ max theo head); hàng rỗng (nút cô lập) cho `A` hàng 0, không lỗi.

Gắn thời gian (quyết định 0005, chưa làm ở bản KGAT): `π(h,r,t) ← π(h,r,t) − γ·δ_r·Δt(h,t)` **trước** softmax, `γ = 0` trả về KGAT.

Ma trận `A ∈ R^{N×N}` thưa: `A[h, t] += π̃(h,r,t)`. **Tính lại mỗi epoch, trong `no_grad`, từ embedding hiện tại**; trong các
bước CF nó là hằng số (như TF1: `A_values` là placeholder). Nghĩa là attention chỉ được học qua `L_KG`, không qua BPR (Q4).

### 2.3 Lan truyền và aggregator Bi-Interaction (công thức 3, 8–11)
Với `e^{(0)} = [user_embed ; entity_embed]` (N×d) và các lớp `d_0 = 64 → 64 → 32 → 16`:
```
e_N^{(l-1)} = A · e^{(l-1)}                                          (công thức 10, torch.sparse.mm)
e^{(l)}     = Dropout_p( LeakyReLU( W1^l (e^{(l-1)} + e_N) + b1^l )
                       + LeakyReLU( W2^l (e^{(l-1)} ⊙ e_N) + b2^l ) )   (công thức 8–9)
ê^{(l)}     = e^{(l)} / ‖e^{(l)}‖₂           (l ≥ 1; chỉ bản chuẩn hoá đi vào ghép, lớp sau dùng e^{(l)} chưa chuẩn hoá)
e*          = e^{(0)} ‖ ê^{(1)} ‖ ê^{(2)} ‖ ê^{(3)}        (N × 176; e^{(0)} KHÔNG chuẩn hoá)
ŷ(u,i)     = e*_u · e*_{U+i}                                          (công thức 12)
```
LeakyReLU **độ dốc 0,2** (mặc định `tf.nn.leaky_relu`; `torch.nn.functional.leaky_relu` mặc định 0,01 nên phải đặt tay).
`p = mess_dropout` là **tỉ lệ bỏ** (TF1 truyền `1 − mess_dropout` làm `keep_prob`; chú thích "keep probability" trong `parser.py` gây hiểu nhầm).
Node dropout: `parser.py` có tham số nhưng `KGAT.py` đã comment bỏ nên không có tác dụng; không cài.
Chế độ eval: `p = 0`.

### 2.4 BPR và regulariser (công thức 13)
```
L_CF   = mean softplus( −(ŷ(u,i) − ŷ(u,j)) )
reg_CF = (1/2)·mean( ‖e*_u‖² + ‖e*_i‖² + ‖e*_j‖² )     # trên vector đã lan truyền (176 chiều), như TF1
loss_CF = L_CF + l2 · reg_CF
```
`(u,i)` lấy mọi cạnh train theo `randperm` (một epoch = P cạnh, như `fit_bpr`), `j` từ `NegativeSampler` (chỉ loại positive train của chính user).
Hệ số `l2` ở đây **không so sánh được** với `l2` của BPR-MF (`BPRMF` không có hệ số ½ và regularise tham số thô, không phải vector đã lan truyền).

### 2.5 Vòng huấn luyện xen kẽ (một epoch)
```
(1) pha CF : ⌈P / B⌉ bước; mỗi bước propagate toàn đồ thị với A hiện tại (dropout bật) → loss_CF → opt_cf
(2) pha KG : ⌈E_ckg / B_kg⌉ bước; lấy B_kg cạnh ngẫu nhiên + tail giả → loss_KG → opt_kg
(3) cập nhật A (no_grad, theo mảnh) từ embedding sau pha KG
(4) mỗi eval_every epoch: đánh giá val (mục 4)
```
Hai optimizer **Adam riêng**, cùng `lr`, như TF1:
- `opt_cf` giữ `user_embed, entity_embed, W1^l, b1^l, W2^l, b2^l`;
- `opt_kg` giữ `user_embed, entity_embed, relation_embed, trans_W` (hai bảng embedding dùng chung, nhưng mỗi optimizer có
  moment riêng).

### 2.6 Khác biệt: paper, code TF1, bản của mình
(*) = lấy từ bản dịch, cần đối chiếu PDF.

| Điểm | Paper (*) | Code TF1 | Bản này |
|---|---|---|---|
| Hàm mục tiêu | `L_KG + L_CF + λ‖Θ‖²` (công thức 14); tối ưu **xen kẽ** | hai loss tách, hai Adam; L2 đặt trên **biểu diễn của batch** (CF: `e*` của u,i,j; KG: h,r,t,t' đã chiếu), hai hệ số `regs[0]`, `regs[1]`; `regs[2]` (mặc định 1e-2) **không dùng** trong `KGAT.py` | theo TF1: `l2`, `l2_kg` |
| Gradient qua attention | attention được tính lại ở mỗi lượt lan truyền nên có thể chịu gradient | `A` là hằng số trong bước CF; chỉ tính lại sau mỗi epoch | theo code TF1, đây là lựa chọn có chủ đích (Q4) |
| `A` ở epoch 1 | — | `sum(lap_list)`: **mỗi quan hệ chuẩn hoá hàng riêng rồi cộng**, nên tổng hàng = số loại quan hệ của nút (≠ 1) | đồng đều `1/|N_h|` (Q5) |
| Bias ở aggregator | không | có `b1, b2` | có |
| Chuẩn hoá L2 từng lớp | không nêu | có, cho lớp ≥ 1; `e^{(0)}` không | theo TF1 |
| Dropout | không nêu | message dropout; node dropout chỉ là tham số thừa | message dropout |
| Độ dốc LeakyReLU | — | 0,2 | 0,2 |
| Ràng buộc chuẩn TransR | thường có | đã bỏ | bỏ |
| Lấy mẫu CF | `(u,i,j)` ngẫu nhiên | chọn **user** đều (không hoàn lại nếu `B ≤ U`), mỗi user một positive; `n_train // B + 1` bước | cạnh đều bằng `randperm` như `fit_bpr` (Q6) |
| Lấy mẫu KG | thay tail bằng entity ngẫu nhiên | chọn **head** đều trong các nút có cạnh, rồi một triple; tail giả là **bất kỳ nút nào** trong `[0, U+E)` (kể cả user) miễn `(t,r) ∉ N_h` | cạnh đều; tail giả chỉ lấy từ nút hoạt động (Q6) |
| Số bước KG | — | `n_A_batch = len(all_h_list)//B_kg + 1` | `⌈E_ckg / B_kg⌉` (cùng ý) |
| Khởi tạo | Xavier; có tuỳ chọn pretrain MF | Xavier; `--pretrain -1` nạp embedding BPR-MF | Xavier, không pretrain (Q8) |
| Chọn epoch / dừng sớm | dừng sớm với patience 50 epoch trên Recall@20 của val | **theo recall@20 trên TEST**, mỗi 10 epoch, kiên nhẫn 10 lần đánh giá (= 100 epoch), `Main.py` chọn "Best Iter" trên test | **theo val** (luật 1 của repo); patience 10 lần đánh giá, đánh giá mỗi 5 epoch (`eval_every: 5`), tương đương 50 epoch, bằng patience của paper |
| Chia mảnh `A` (`n_fold = 100`) | — | để tránh OOM trong TF1 | không cần |

Dòng áp dụng cho Main.py rất quan trọng: đây chính là kiểu rò rỉ "best epoch chọn trên test" của kết quả cũ (decision 0001).
**Không port nguyên `Main.py`**; chỉ dùng công thức của `KGAT.py`.

---

## 3. Bộ nhớ (ma trận kề thưa, float32, GPU 4 GB)

### 3.1 Thiết kế
- Cấu trúc đồ thị (`head, relation, tail` int64, `edge_ts` float32) và `A` (giá trị float32) nằm trên thiết bị; dựng một lần mỗi đồ thị.
- `A` là `torch.sparse_coo_tensor` đã `coalesce()` (cộng các cạnh trùng `(h,t)`), float32, **không cần grad**; `torch.sparse.mm(A, e)`
  chỉ lan truyền gradient về `e`. Chỉ số int64 (yêu cầu của torch) → ≈ 20 byte/cạnh.
- **Tuyệt đối không gather `W_r` theo từng cạnh khi tính attention toàn đồ thị**: `E_ckg × d × k × 4 B` (bảng dưới: 2,4 / 3,3 / 17,3 GiB).
  Thay bằng **nhóm theo quan hệ**: sắp cạnh theo `r` một lần (hoán vị cố định), với mỗi quan hệ chiếu `(E_r × d) @ (d × k)`; xử lý theo mảnh
  ≈ 65.536 cạnh dưới `no_grad` (≈ 64 MB). Vòng lặp chỉ qua 38 quan hệ, không qua từng mẫu.
  Ở bước huấn luyện KG (B_kg = 2048) gather `W_r` theo mẫu chỉ tốn `2048·64·64·4 B = 32 MB`, chấp nhận được; đặt `B_kg ≤ 8192`.
- Softmax theo head bằng đoạn liên tiếp (`torch.segment_reduce` hoặc `scatter`); cần kiểm tính tất định (xem rủi ro R1).
- Đánh giá: tính `e*` một lần (`no_grad`, eval), `score(users) = e*[users] @ e*[U:U+I].T` → tensor (B, I) trên thiết bị
  (`evaluate` đổi sang float64 chỉ trên (B, I): 512×3009×8 B ≈ 12 MB).

### 3.2 Ước lượng (bộ chạy ngày 2026-10-27; d = k = 64, lớp [64,32,16], `n_rel` = 38)

| | ML-100K | ML-1M-u10 | ML-1M |
|---|---:|---:|---:|
| U / I / E (entity) | 745 / 932 / 15.288 | 537 / 1.639 / 23.117 | 5.350 / 3.009 / 32.321 |
| N = U + E | 16.033 | 23.654 | 37.671 |
| Positive train P | 43.018 | 42.554 | 461.858 |
| Triple KG t1 (default) T | 36.560 | 65.344 | 103.443 |
| Cạnh CKG `2(P+T)` | 159.156 | 215.796 | 1.130.602 |
| Entity cô lập ở t1 | 206 | 296 | 415 |
| Tham số (1,2M / 1,7M / 2,6M) | 4,6 MB | 6,4 MB | 9,9 MB |
| Tham số + grad + 2 moment Adam | 18 MB | 26 MB | 39 MB |
| `A` (COO) | 3,0 MB | 4,1 MB | 21,6 MB |
| `e*` (N×176) | 11 MB | 16 MB | 25 MB |
| Activation lưu cho backward (cận trên thô) | ≤ 157 MB | ≤ 231 MB | ≤ 368 MB |
| Gather `W_r` theo cạnh (**không dùng**) | 2,4 GiB | 3,3 GiB | 17,3 GiB |
| Bước CF / epoch (B = 2048) · bước KG / epoch | 22 · 78 | 21 · 106 | 226 · 553 |
| User val / test | 66 / 51 | 41 / 95 | 480 / 952 |

Kết luận: với thiết kế trên, **cả ML-1M đầy đủ** vẫn dưới ~0,5 GB (RTX 3050 còn ~3 GB trống; lúc kiểm `nvidia-smi` đang dùng 897 MiB).
Nút cổ chai không phải `A` mà là việc gather ma trận theo cạnh. Đây là ước lượng; **trước khi chạy dài phải đo**
`torch.cuda.max_memory_allocated()` trên ML-1M-u10 rồi ML-1M (CLAUDE.md, mục GPU rules) và ghi vào meta của file history.
Thời gian mỗi epoch: **chưa đo**, sẽ đo ở B2 trên ML-1M-u10 trước.

### 3.3 Hệ quả cho `run.py`
`run.py` hiện giữ `fitted[g][seed] = model` cho **mọi** điểm lưới × seed đến lúc chấm test. Với KGAT chỉ giữ `state_dict` trên CPU (~10 MB/mô hình
ở ML-1M) và dựng lại mô hình của điểm được chọn khi chấm test. Tránh phụ thuộc vào việc GPU còn đủ chỗ.

### 3.4 Rủi ro
- **R1** `torch.sparse.mm` (backward) và `scatter/index_add` trên CUDA có thể không tất định; `seed_all` chỉ `warn_only`. Cần test: cùng seed, chạy hai lần → `state_dict` bằng nhau
  (BPR-MF đã đạt, CLAUDE.md ghi kết quả). Nếu không, đổi sang CSR hoặc tự viết phép tổng theo đoạn (segment sum).
- **R2** Khi `B` nhỏ (ML-100K có 22–43 bước CF/epoch) cần nhiều epoch; `max_epochs` và `patience` đặt ở `fixed` (mục 5).

---

## 4. Dừng sớm theo val; ground truth và mask từ `EncodedSplit`

- Chỉ đọc `data.train`, `data.train_ts`, `data.val`, `data.mask_val` (cùng đối tượng `fit_bpr` đang dùng). Trainer **không** truy cập
  `data.test`, `data.mask_test`, `kg_t2` (docstring của `train/bpr.py` đã nêu quy tắc này; làm tương tự và thêm test).
- Mỗi `eval_every` epoch, **sau** khi cập nhật A: `model.eval()` → `propagate` một lần (không dropout) → `evaluate(model.score, data.val, data.mask_val)`
  → chọn theo `select_metric` (`recall@20`). `mask_val` chứa mọi tương tác (mọi rating) trước t1; `evaluate` che chúng.
- Giữ trạng thái tốt nhất (`state_dict` sao chép sang CPU), `bad` đếm số lần đánh giá không cải thiện, dừng khi `bad ≥ patience`. Khi nạp lại trạng thái tốt:
  **tính lại A từ tham số đó** (A là hàm tất định của tham số + đồ thị, nên bằng đúng A lúc đạt best epoch).
- `history` mỗi epoch: `epoch, loss_cf, loss_kg` và, ở epoch đánh giá, các metric val; **không có cột nào từ test**. Lưu cùng `config`, `kg_variant`,
  `peak_memory` vào `<tag>_<g>_seed<seed>_history.csv` qua `registry.save_history` như BPR-MF.
- Val chỉ 41–66 user (ML-100K 66, u10 41): epoch tốt nhất dao động. Quy tắc giữ nguyên (chọn **điểm lưới** theo trung bình val 5 seed); báo cáo kèm cảnh báo, ưu tiên ML-1M cho kết luận.
- Test (một lần, trong `run.py`, sau khi chọn điểm lưới): dựng `CKG(train, kg_t2)` (Q2), nạp `state_dict` của từng seed, tính A, `evaluate(model.score, data.test, data.mask_test)`.

---

## 5. Tích hợp `scripts/run.py`

### 5.1 Config (bản nháp `configs/model/kgat.yaml`)
```yaml
# KGAT on the time-consistent KG (variant default). Every grid point is trained with every seed; the point with
# the best mean validation Recall@20 is selected, and only that point is scored on test (with the KG at t2).
name: kgat
seeds: [2019, 2020, 2021, 2022, 2023]
kg: {variant: default}            # default | drop_undated | static | none
grid:
  - {lr: 0.001,  l2: 0.00001, l2_kg: 0.00001, mess_dropout: 0.1}
  - {lr: 0.001,  l2: 0.0001,  l2_kg: 0.00001, mess_dropout: 0.1}
  - {lr: 0.0001, l2: 0.00001, l2_kg: 0.00001, mess_dropout: 0.1}
  - {lr: 0.0001, l2: 0.0001,  l2_kg: 0.00001, mess_dropout: 0.1}
fixed: {dim: 64, kge_dim: 64, layers: [64, 32, 16], batch_size: 1024, batch_size_kg: 2048,
        max_epochs: 300, eval_every: 5, patience: 10, select_metric: recall@20}
```
- **Mỗi biến thể KG là một file riêng** (`kgat.yaml`, `kgat_drop_undated.yaml`, `kgat_static.yaml`, `kgat_nokg.yaml`): khoá registry là
  `(dataset, split_sha256, model, config, seed, phase)` với `model` = tên file, `config` = chỉ **điểm lưới**. `fixed` và `kg` **không** nằm trong khoá,
  nên đổi chúng mà giữ tên file sẽ ghi đè về mặt ngữ nghĩa. Quy tắc: đổi `fixed` hoặc `kg` → đổi tên file.
- `kg: none` = CKG chỉ có `Interact` và `Interact⁻¹` ("KGAT không KG", mốc so sánh cho RQ2). `static` là tham chiếu, luôn gắn nhãn
  `time_consistent: false` (decision 0006 E), không bao giờ báo như kết quả nhất quán thời gian.
- Kiểm quy tắc của decision 0007: nếu điểm được chọn nằm ở biên lưới (`lr`, `l2` nhỏ nhất/lớn nhất) thì nói rõ trong báo cáo.

### 5.2 Thay đổi trong `run.py` (nhỏ)
1. Thêm nhánh `elif mcfg["name"] == "kgat"`: nạp KG t1 **một lần ngoài vòng lưới** (`load_kg(dcfg, variant, "t1")`), `fit_kgat(data, ckg_t1, KGATConfig(**point, **fixed, device=...), seed)`,
   `save_history` như BPR-MF, giữ `state_dict` CPU.
2. Khi chấm test: với mô hình KGAT, trước `evaluate` gọi `ckg_t2 = load_kg(..., "t2")` rồi `model.set_graph(ckg_t2)`; mô hình khác giữ nguyên `model.score`.
3. Không thêm cột vào `registry.csv` (`registry.append` từ chối file khác header). Thông tin KG (variant, số triple t1/t2, số triple t2 bị bỏ theo Q2,
   `time_consistent`, "Wikidata snapshot 2026-10-17") ghi vào `results/runs/<run_name>/kgat_summary.json` bên cạnh `selected` và `test_mean_std`.
4. `seed_all(seed)` đầu mỗi lần chạy; generator trên thiết bị cho cả hai sampler; 5 seed 2019–2023; `seed_std` ddof=1 như cũ; file `kgat_seed<seed>.npz` cho Wilcoxon theo user.
5. `load_kg` kiểm: `kg_meta["inputs"]["split_sha256"] == meta["sha256"]` (KG dựng từ đúng split này), số item bằng `EncodedSplit.n_items`,
   `entities.csv` có `key` của entity `i < I` đúng bằng `item_ids[i]`. Sai thì dừng, không chạy.
6. Chạy thử trên ML-1M-u10 với `--registry` trỏ file nháp (ví dụ `results/dev_registry.csv`) để không trộn số khám phá vào registry chính.

Lệnh dự kiến (sau B2): `python scripts/run.py --data configs/data/ml100k.yaml --model configs/model/kgat.yaml` (cũng cho `ml1m_u10.yaml`, `ml1m.yaml`).

### 5.3 Kiểm thử dự kiến (CPU; GPU kiểm đánh dấu `skipif`)
`test_ckg`: id nút/quan hệ, cặp nghịch đảo, `E_ckg = 2(P+T)`, tập cạnh `Interact` **bằng đúng** cặp train (không có cặp nào của val/test), R cố định kể cả khi một quan hệ không có triple ·
`test_ckg_guard`: KG cũ (sha khác, thứ tự item khác) → lỗi · `test_attention`: tổng hàng = 1, đều khi điểm bằng nhau, khớp softmax dense trên đồ thị nhỏ ·
`test_propagate`: khớp bản dense 2 lớp · `test_transr`: loss đúng với tính tay, tập tham số của `opt_cf`/`opt_kg` đúng · `test_kg_negatives`: không bao giờ là cạnh thật, không bao giờ là nút cô lập ·
`test_test_blind`: thay `data.test` bằng rác và không đưa `kg_t2` → lịch sử val và `state_dict` y hệt · `test_t2_rule`: bỏ đúng các triple có tail chưa dùng ở t1 ·
`test_determinism`: cùng seed hai lần → bằng nhau.
Hai test khoá giao diện cho T-KGAT (chưa viết, đến PR có Δt):
`test_gamma_zero_is_kgat`: với `γ = 0`, mô hình thời gian và KGAT thuần (cùng seed, cùng đồ thị) cho **cùng tensor** `A` và `e*` (so từng phần tử, không chỉ so Recall) ·
`test_zero_dt_is_kgat`: đặt Δt = 0 trên mọi cạnh (với `γ > 0`) thì mô hình thời gian vẫn trùng KGAT.

---

## 6. Các điểm có thể rò rỉ và cách chặn

| # | Rò rỉ có thể | Cách chặn | Kiểm bằng |
|---|---|---|---|
| L1 | Dùng sai KG: KG t2 (có fact trong `[t1, t2)`) khi huấn luyện/chọn, hoặc KG `static` (có fact tương lai) mà báo như nhất quán | `fit_kgat` chỉ nhận `ckg_t1`; `load_kg(cutoff)` là hàm duy nhất đọc `kg_t*.tsv`; chỉ `run.py` gọi `t2`; variant `static` ghi `time_consistent: false` | `test_test_blind`, summary JSON |
| L2 | Cạnh `Interact` lấy từ val/test (hoặc từ rating < 4, hoặc cặp trùng) | CKG chỉ dựng từ `EncodedSplit.train` (positive, trước t1) | `test_ckg` (tập cạnh = cặp train) |
| L3 | KG dựng từ split cũ/khác (id entity lệch item) | `load_kg` kiểm sha của split trong `kg_meta.json` và khoá item của `entities.csv` | `test_ckg_guard` |
| L4 | Negative BPR lấy từ val/test | dùng nguyên `NegativeSampler` (chỉ positive train của chính user) | test sẵn có của sampler + dùng lại |
| L5 | Negative KG chọn entity "tương lai" (tail chỉ có ở fact bị loại ở t1) làm tail giả → embedding bị đẩy về một phía rồi xuất hiện ở KG t2 | tail giả chỉ lấy từ nút có ≥ 1 cạnh trong CKG hiện hành; entity cô lập không bao giờ nhận gradient | `test_kg_negatives` |
| L6 | Entity chỉ có ở KG t2 mang embedding ngẫu nhiên vào điểm test | bỏ triple có tail chưa dùng ở t1 (Q2), ghi số lượng | `test_t2_rule` |
| L7 | Δt: dùng timestamp val/test, hoặc chuẩn hoá bằng thống kê toàn bộ dữ liệu | `edge_ts` chỉ lấy từ `train_ts`; mốc tham chiếu = timestamp train lớn nhất (như MostPop recent, decision 0007), **cùng một mốc** cho val và test (cạnh test cũng là cạnh train); hệ số chuẩn hoá tính chỉ trên `train_ts`; đối chứng hoán vị chỉ xáo `train_ts` giữa cạnh train | test T-KGAT (sau) |
| L8 | Chọn epoch/tham số bằng test (như `Main.py` gốc) | dừng sớm theo `data.val`; điểm lưới chọn theo trung bình val 5 seed; test chỉ trong `run.py` sau khi chọn; `history` không có cột test | `test_test_blind`, đọc `train/kgat.py` |
| L9 | Khởi tạo/pretrain dùng dữ liệu ngoài train | không pretrain ở v1 (Q8); nếu sau này dùng thì chỉ BPR-MF huấn luyện trên train, cùng seed | — |
| L10 | Mô hình hoặc A được chọn theo một đồ thị rồi đánh giá trên đồ thị khác mà không tính lại A | `set_graph` luôn tính lại A từ tham số hiện tại; khi nạp best state cũng tính lại | `test_attention`, `test_determinism` |
| L11 | Kết quả KG bị hiểu như tri thức thời 1997–2003 | mọi báo cáo kết quả KG ghi "Wikidata là ảnh chụp 2026-10-17" (luật 3) | summary JSON + báo cáo |

Cảnh báo ngoài danh sách: KG t2 gần như bằng KG t1 ở ML-1M (+0 triple) và ML-100K (+20), nên một kết quả "KG t2 giúp ở test" là **đáng nghi** (lỗi nối đồ thị, không phải hiệu ứng thật). Nếu gặp, kiểm L1/L2/L3 trước khi báo.

---

## 7. Câu hỏi cần bạn quyết (kèm đề xuất)

**A. Chạm vào giao thức (cần decision record 0009 nếu bạn đồng ý; theo luật 6 mình chỉ soạn sau khi bạn chốt)**

- **Q1. Ở test dùng cạnh tương tác nào?** (a) chỉ cạnh train (đề xuất): đồ thị test = đồ thị train, chỉ KG đổi t1 → t2; nhất quán với quyết định 0002 ("không huấn luyện lại trên train+val").
  (b) thêm cạnh val (hợp lệ về thời gian vì < t2) nhưng đổi giao thức và làm lệch phân bố bậc so với lúc huấn luyện.
- **Q2. Thực thể chỉ có ở KG t2 (ML-100K 1, u05 113):** (a) bỏ triple có tail chưa dùng ở t1, ghi số bỏ (đề xuất; bảo thủ, không rò); (b) giữ với embedding ngẫu nhiên chưa huấn luyện; (c) khởi tạo bằng 0/trung bình. Với ML-1M và u10 không có khác biệt.
- **Q3. Phạm vi v1 của KGAT:** chạy cả `kgat` (default), `kgat_drop_undated`, `kgat_static` (tham chiếu rò rỉ, gắn nhãn), `kgat_nokg` (RQ2)? Đề xuất **có cả 4 file config, nhưng chỉ chạy `kgat` và `kgat_nokg` trước**; hai cái kia sau khi `kgat` ổn.

**B. Kỹ thuật (đề xuất sẽ dùng nếu bạn không phản đối)**

- **Q4. Attention có gradient từ BPR không?** Đề xuất **không** (như TF1: tính lại mỗi epoch, `no_grad`): rẻ bộ nhớ, đúng code tham chiếu. Hệ quả cho T-KGAT: `γ` và `δ_r` là **siêu tham số** (lưới), không học được qua BPR.
- **Q5. `A` ở epoch 1:** đồng đều `1/|N_h|` (đề xuất) thay vì tổng các Laplace chuẩn hoá từng quan hệ của TF1 (tổng hàng ≠ 1; chỉ ảnh hưởng epoch 1).
- **Q6. Lấy mẫu:** CF theo cạnh bằng `randperm` như `fit_bpr` (đề xuất; để khác biệt BPR-MF ↔ KGAT chỉ do mô hình, không do cách lấy mẫu) thay vì user đều của TF1. KG theo cạnh đều; tail giả từ nút hoạt động.
- **Q7. Tối ưu:** hai Adam xen kẽ, cùng `lr` (đề xuất, như TF1), thay vì cộng hai loss (công thức 14 viết thẳng).
- **Q8. Khởi tạo:** Xavier, **không** pretrain bằng BPR-MF ở v1 (có thể thêm sau như một cờ).
- **Q9. Regulariser:** theo TF1 (trên vector đã lan truyền; `l2`, `l2_kg` tách). Không so sánh được với `l2` của BPR-MF, ghi chú trong báo cáo.
- **Q10. Lưới và dừng sớm:** 4 điểm như bản nháp; `batch_size` 1024 (mặc định TF1; ML-100K chỉ có 43 bước/epoch nên 2048 là quá thô), `patience` 10 lần đánh giá vì val ồn. Bạn muốn thêm bước `layers` {1, 2, 3} (bảng 3 của paper) làm ablation sau không?
- **Q11. Dev và phụ thuộc:** phát triển trên ML-100K và ML-1M-u10. Config `ml1m_u10.yaml` chỉ có ở `feat/22-subset-and-kgat-export` (0008 còn "proposed"). Đề xuất: **chỉ tạo `feat/kgat-model` sau khi PR của `feat/22` đã merge** (hoặc rebase lên đó), nếu không Phôn không chạy lại được ML-1M-u10. Số chạy dev đi vào registry nháp.
- **Q12. Giao diện cho T-KGAT (chỉ chốt giao diện, chưa làm thời gian):** mảng `edge_ts` song song với cạnh, `δ_r` theo quan hệ (chỉ `Interact`/`Interact⁻¹` khác 0), mốc tham chiếu = timestamp train lớn nhất cho cả val/test, `γ = 0` bằng đúng KGAT. Biến thể thời gian trên triple KG cần ngày hiệu lực từng fact, mà `kg_t*.tsv` hiện chỉ lưu `h, r, t` (ngày nằm ở `dropped.csv` và `temporal.annotate`); sẽ phải mở rộng đầu ra script 06 ở giai đoạn đó.
- **Q13. "KG 2-hop" của decision 0005:** KG hiện có 1-hop (head luôn là item), nên "2-hop" không có trong dữ liệu; hai lớp lan truyền đã nối item → thuộc tính → item. Đề xuất giữ nguyên, ghi rõ trong báo cáo thay vì mở rộng KG.
- **Q14. Bố cục mã và phân công:** `kg/ckg.py` (đọc KG + dựng CKG; thư mục của Triết), `models/kgat.py` và `train/kgat.py` (thư mục của Phôn). Mỗi PR do người kia duyệt (CONTRIBUTING).

---

## Phụ lục A. Chỗ cần cẩn thận trong `KGAT_Giai_thich_code.md` (đã đối chiếu với `KGAT.py`, `Main.py`, `loader_kgat.py`)

1. Mục 2.8 và phần "Vòng lặp xen kẽ": "Phase II cập nhật `relation_embed`, `trans_W`" thiếu. `opt2` còn cập nhật `user_embed` và `entity_embed` (chúng nằm trong `loss2`); `opt` (pha I) không đụng `relation_embed`/`trans_W`. Hai Adam có moment riêng.
2. "A_in ban đầu 1/degree": thực tế là tổng các Laplace đã chuẩn hoá hàng **từng quan hệ**, nên tổng hàng bằng số loại quan hệ của nút.
3. Phần 4 ("Luồng training") và phần 3 (metric) bỏ qua việc `Main.py` **đánh giá và dừng sớm trên test**. Không làm theo.
4. Skeleton PyTorch (Phần 6): (a) `compute_attention` thiếu softmax theo head; (b) `F.leaky_relu` mặc định 0,01, TF1 dùng 0,2; (c) `forward` và `kg_loss` thiếu hạng L2; (d) `trans_W[r_ids]` gather theo cạnh → 17,3 GiB nếu dùng cho toàn đồ thị ML-1M (mục 3); (e) thiếu cập nhật A, negative sampling KG, và dịch id `+ n_users` cho đồ thị KG.
5. `node_dropout` xuất hiện trong feed_dict nhưng không có tác dụng trong `KGAT.py` (dòng dropout node đã comment).

## Phụ lục B. Thứ tự làm đề xuất cho B2 (mỗi bước một PR nhỏ, `ruff` + `pytest` xanh trước mỗi commit)
1. `kg/ckg.py` + `tests/test_ckg.py` (dựng đồ thị, kiểm sha/khoá item). Không có mô hình.
2. `models/kgat.py`: TransR, attention theo đoạn, propagate, aggregator + test dense tham chiếu.
3. `train/kgat.py`: hai sampler, vòng xen kẽ, dừng sớm theo val, `test_test_blind`, `test_determinism`.
4. `scripts/run.py` + `configs/model/kgat*.yaml`, chạy thử ML-1M-u10 (registry nháp, 1 seed), đo bộ nhớ/thời gian.
5. Chạy chính thức ML-100K (Phôn) và ML-1M (Triết) sau khi 0009 được duyệt; đối chiếu độc lập (Phôn chạy lại).
