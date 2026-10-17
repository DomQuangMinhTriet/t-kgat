# Chủ đề 2 — Đánh giá offline và rò rỉ dữ liệu

> Người trình bày: **Phôn** · Ngày: CN 18/10 · Đọc kèm code: `src/tkgat/data/split.py`, `src/tkgat/data/leakage.py`, `src/tkgat/eval/metrics.py`, `tests/test_leakage.py`

Đây là chủ đề quan trọng nhất: cả 5 góp ý của cô ngày 17/10 đều xoay quanh nó.

## 1. Đánh giá offline là gì
Ta không thể thử mô hình trên người dùng thật, nên **mô phỏng tương lai bằng dữ liệu quá khứ**: giấu một phần tương tác (test), huấn luyện trên phần còn lại (train), rồi xem mô hình có "đoán lại" được phần bị giấu không. Mô phỏng này chỉ đúng nếu **mô hình không được nhìn thấy gì mà ở thời điểm dự đoán thật nó chưa thể biết**. Vi phạm điều đó gọi là **rò rỉ dữ liệu (data leakage)**.

## 2. Ba tập dữ liệu và vai trò

| Tập | Dùng để | Không được dùng để |
|---|---|---|
| Train | Học tham số mô hình | — |
| Validation (val) | Chọn epoch dừng, chọn siêu tham số (dim, l2, γ…), chọn biến thể | Báo cáo kết quả cuối |
| Test | Báo cáo kết quả cuối, **chấm đúng một lần** cho cấu hình đã chọn | Bất kỳ quyết định nào |

Lỗi của giai đoạn trước: chọn `best_epoch` theo Recall trên test. Như vậy test đã tham gia vào việc chọn mô hình, nên con số báo cáo bị thổi phồng. Đây là lý do cô nói "mất tập val rồi, phải có val mới đúng, nếu không là bị data leakage".

## 3. Các cách chia dữ liệu

| Cách chia | Mô tả | Rò rỉ thời gian? |
|---|---|---|
| Ngẫu nhiên (random) | Lấy ngẫu nhiên 80% tương tác làm train (paper KGAT dùng cách này) | **Có**: tương lai của user lọt vào train |
| Leave-one-out | Tương tác cuối của mỗi user làm test | **Có** một phần: tương tác "tương lai" của user khác vẫn ở train |
| Theo thời gian từng user | X% tương tác cuối của mỗi user làm test | **Có** một phần (cùng lý do) |
| **Theo mốc thời gian chung (global temporal)** | Mọi tương tác trước $t_1$ là train, $[t_1, t_2)$ là val, sau $t_2$ là test | **Không** |

Repo dùng mốc chung: $t_1$, $t_2$ là phân vị 80% và 90% của thời gian (`q_val`, `q_test`).

Cái giá phải trả: user chỉ xuất hiện sau $t_1$ là user mới, mô hình không biết gì về họ nên bị loại khỏi val/test. Với MovieLens-100K, 78% rating được nhập ngay trong ngày đầu user đăng ký, nên test chỉ còn 51 user. Đó là lý do nhóm chạy song song MovieLens-1M (952 user test).

## 4. Các nguồn rò rỉ thường gặp (và cách repo chặn)

| # | Nguồn rò rỉ | Ví dụ | Cách chặn trong repo |
|---|---|---|---|
| 1 | Chọn mô hình bằng test | Chọn epoch tốt nhất theo test | `fit_bpr` không nhận test; test chấm sau khi chọn (`scripts/run.py`) |
| 2 | Tiền xử lý trên toàn bộ dữ liệu | Lọc k-core trước khi chia: quyết định giữ user dựa cả vào tương tác tương lai | `kcore()` chỉ chạy trên train dương; test `test_kcore_never_sees_future` |
| 3 | Thống kê dùng dữ liệu tương lai | Độ phổ biến item tính trên toàn bộ dữ liệu | `MostPop` chỉ đọc `data.train` |
| 4 | Trùng lặp | Một cặp (user, item) xuất hiện cả ở train và test (Foursquare: 4,09 check-in/cặp) | `dedupe_pairs`; kiểm tra số 2 trong `check_split` |
| 5 | Thực thể chưa từng thấy | Item chỉ xuất hiện ở test | Loại và đếm (`*_dropped_cold_*`) |
| 6 | Tri thức tương lai trong KG | Dùng "Titanic đoạt Oscar 23/03/1998" cho rating 07/03/1998 | KG theo mốc (quyết định 0004) |
| 7 | Mặt nạ (mask) sai | Loại khỏi danh sách xếp hạng những item user tương tác *sau* mốc | `seen_before_*` chỉ chứa tương tác trước mốc; kiểm tra số 5 |

`check_split` chạy tự động mỗi lần dựng split. Nếu có bất kỳ vi phạm nào, `scripts/02_build_splits.py` **từ chối lưu**.

## 5. Metric xếp hạng
Với mỗi user trong test: xếp hạng **toàn bộ** item (trừ item user đã tương tác trước mốc), lấy top-K, so với tập item đúng $T_u$.

$$\text{Recall@}K(u) = \frac{|\text{top-}K(u) \cap T_u|}{|T_u|}$$

$$\text{DCG@}K(u) = \sum_{k=1}^{K} \frac{\mathbb{1}[\text{item hạng } k \in T_u]}{\log_2(k+1)}, \qquad \text{NDCG@}K(u) = \frac{\text{DCG@}K(u)}{\text{IDCG@}K(u)}$$

- IDCG là DCG của xếp hạng lý tưởng (mọi item đúng đứng đầu), nên NDCG nằm trong [0, 1].
- Kết quả báo cáo là trung bình trên user.
- **Ví dụ** (có trong `tests/test_metrics.py`): xếp hạng [5, 1, 9, 2], đúng {1, 2, 7}, K = 4 → Recall = 2/3; DCG = 1/log₂3 + 1/log₂5; IDCG = 1 + 1/log₂3 + 1/log₂4.
- **Xếp hạng toàn bộ hay lấy mẫu:** nhiều paper chỉ xếp hạng item đúng với 100 item âm ngẫu nhiên. Cách đó có thể làm đảo thứ tự giữa các mô hình (Krichene & Rendle, 2020). Repo xếp hạng toàn bộ, giống paper KGAT.

## 6. Baseline và độ tin cậy
- Luôn có MostPop. Mô hình không thắng MostPop thì không có gì để nói.
- Chạy nhiều seed (5), báo cáo trung bình ± độ lệch chuẩn, kèm kiểm định cặp theo user (chủ đề 7).
- Ghi mọi kết quả vào `results/registry.csv` kèm commit, cấu hình và seed, để con số nào cũng truy được nguồn gốc.

## 7. Liên hệ với repo
| Khái niệm | Vị trí |
|---|---|
| Chia theo mốc, dedupe, k-core trên train | `src/tkgat/data/split.py` |
| 6 kiểm tra rò rỉ | `src/tkgat/data/leakage.py` → `check_split` |
| Test cố tình chèn dữ liệu tương lai | `tests/test_leakage.py` |
| Recall/NDCG xếp hạng toàn bộ | `src/tkgat/eval/metrics.py` |
| Chọn trên val, chấm test một lần | `scripts/run.py` |
| Quyết định chính thức | `docs/decisions/0002-data-split.md` |

## 8. Tự kiểm tra
1. Định nghĩa rò rỉ dữ liệu trong một câu.
2. Vì sao chọn epoch theo test là rò rỉ, dù không train trên test?
3. Lọc k-core trước khi chia dữ liệu rò rỉ thông tin như thế nào? Cho ví dụ.
4. So sánh chia ngẫu nhiên và chia theo mốc chung. Vì sao paper KGAT chia ngẫu nhiên vẫn được đăng?
5. Vì sao test của 100K chỉ còn 51 user? Có cách nào tăng lên mà không rò rỉ?
6. Tính tay Recall@3 và NDCG@3 cho xếp hạng [4, 7, 1, 9], đúng {7, 9}.
7. Vì sao phải loại khỏi xếp hạng những item user đã tương tác trước mốc?
8. Nêu 7 nguồn rò rỉ và kiểm tra tương ứng trong `check_split`.
9. Vì sao xếp hạng trên mẫu 100 item âm có thể cho kết luận sai?
10. Một fact trong KG có thể gây rò rỉ như thế nào? Cho ví dụ với Titanic.

## 9. Tài liệu đọc
- Ji, Sun, Zhang, Li. *A Critical Study on Data Leakage in Recommender System Offline Evaluation.* ACM TOIS 2023 — **đọc kỹ**, sát nhất với góp ý của cô.
- Meng, McCreadie, Macdonald, Ounis. *Exploring Data Splitting Strategies for the Evaluation of Recommendation Models.* RecSys 2020.
- Krichene, Rendle. *On Sampled Metrics for Item Recommendation.* KDD 2020.
- Sun et al. *Are We Evaluating Rigorously? Benchmarking Recommendation for Reproducible Evaluation and Fair Comparison.* RecSys 2020.
- Ferrari Dacrema, Cremonesi, Jannach. *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.* RecSys 2019 — đọc để hiểu vì sao baseline đơn giản quan trọng.
