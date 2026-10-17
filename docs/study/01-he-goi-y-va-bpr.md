# Chủ đề 1 — Hệ gợi ý, phản hồi ngầm và BPR

> Người trình bày: **Triết** · Ngày: T2 19/10 · Đọc kèm code: `src/tkgat/models/baselines.py`, `src/tkgat/train/bpr.py`

## 1. Bài toán gợi ý top-K
Cho tập user $\mathcal{U}$, tập item $\mathcal{I}$ và lịch sử tương tác, ta cần với mỗi user $u$ trả về **K item** mà $u$ có khả năng tương tác nhất trong tương lai. Đây là bài toán **xếp hạng (ranking)**, không phải dự đoán số sao. Mô hình chỉ cần cho điểm $\hat{y}_{ui}$ sao cho item user thích đứng trên item user không thích.

Ba họ phương pháp chính:

| Họ | Ý tưởng | Ví dụ |
|---|---|---|
| Lọc cộng tác (Collaborative Filtering, CF) | User giống nhau thích item giống nhau, chỉ dùng ma trận tương tác | ItemKNN, Matrix Factorization, BPR-MF |
| Dựa trên nội dung (content-based) | Gợi ý item có thuộc tính giống item user đã thích | TF-IDF trên mô tả phim |
| Lai / dựa trên tri thức (hybrid, KG-based) | Kết hợp tương tác với thuộc tính, quan hệ giữa các item | CKE, RippleNet, **KGAT** |

## 2. Phản hồi tường minh và phản hồi ngầm
- **Tường minh (explicit):** user cho điểm 1–5 sao. MovieLens có dữ liệu này.
- **Ngầm (implicit):** chỉ biết user *có* tương tác (xem, click, mua). Không có tương tác **không có nghĩa là không thích**: có thể user chưa từng thấy item đó.

Trong repo, nhóm chuyển MovieLens sang phản hồi ngầm: **rating ≥ 4 là tương tác dương** (`min_rating: 4` trong `configs/data/*.yaml`). Lý do: KGAT và các baseline đều được thiết kế cho phản hồi ngầm, và đánh giá top-K cần biết item nào là "đúng".

Hệ quả quan trọng: tập "âm" không có thật, nên ta phải **lấy mẫu âm (negative sampling)**, tức chọn ngẫu nhiên các item user chưa tương tác và *giả định* user thích chúng ít hơn.

## 3. Matrix Factorization (MF)
Mỗi user và item được biểu diễn bằng một vector $d$ chiều (embedding):
$$\hat{y}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$$
- $\mathbf{p}_u \in \mathbb{R}^d$: sở thích ẩn của user; $\mathbf{q}_i \in \mathbb{R}^d$: đặc trưng ẩn của item.
- Trực giác: nếu chiều thứ 3 mang ý nghĩa "mức độ hành động", user thích phim hành động có $p_{u,3}$ lớn, phim hành động có $q_{i,3}$ lớn, nên tích vô hướng lớn.
- Với phản hồi tường minh, MF học bằng cách giảm sai số $(r_{ui} - \hat{y}_{ui})^2$. Với phản hồi ngầm, cách học đó không hợp vì không có nhãn âm thật, nên ta dùng BPR.

## 4. BPR — Bayesian Personalized Ranking
**Giả định:** user $u$ thích item đã tương tác $i$ hơn item chưa tương tác $j$. Mỗi bộ ba $(u, i, j)$ là một mẫu huấn luyện.

$$\hat{x}_{uij} = \hat{y}_{ui} - \hat{y}_{uj}$$
$$\mathcal{L}_{\text{BPR}} = \sum_{(u,i,j)} -\ln \sigma(\hat{x}_{uij}) + \lambda \lVert \Theta \rVert_2^2$$

- $\sigma$ là hàm sigmoid. Khi $\hat{y}_{ui} \gg \hat{y}_{uj}$ thì $\sigma \to 1$ và loss $\to 0$.
- $-\ln\sigma(x) = \text{softplus}(-x)$. Trong code: `F.softplus(-(x_ui - x_uj))`.
- $\lambda\lVert\Theta\rVert^2$ là regularization L2, chống overfit. Trong code là `cfg.l2 * reg`.
- BPR tối ưu trực tiếp **thứ tự** (AUC), đúng với mục tiêu xếp hạng.

**Lấy mẫu âm trong repo** (`_sample_negatives`): chọn $j$ ngẫu nhiên đều trong toàn bộ item, chọn lại nếu $j$ nằm trong tập dương của $u$ **trên train**. Không bao giờ dùng val/test để lấy mẫu.

## 5. Vòng huấn luyện và các tham số phải báo cáo (góp ý 03/07)
| Tham số | Ý nghĩa | Giá trị trong `configs/model/bprmf.yaml` |
|---|---|---|
| `dim` | Số chiều embedding | 32 hoặc 64 (chọn theo val) |
| `lr` | Learning rate của Adam | 0.001 |
| `l2` | Hệ số regularization | 1e-4 hoặc 1e-3 (chọn theo val) |
| `batch_size` | Số bộ ba $(u,i,j)$ mỗi lần cập nhật | 2048 |
| `max_epochs` | Số epoch tối đa | 300 |
| `eval_every`, `patience` | Đánh giá val mỗi 5 epoch; dừng nếu 5 lần liên tiếp không cải thiện | 5, 5 |

- **Epoch:** một lượt đi qua toàn bộ tương tác train.
- **Learning curve:** vẽ loss train và Recall@20 trên val theo epoch. Loss giảm mà Recall val không tăng nữa là dấu hiệu overfit, và đó là lúc dừng sớm.
- **Underfit:** cả loss train lẫn metric val đều kém. **Overfit:** loss train tiếp tục giảm, metric val giảm.

## 6. Baseline phổ biến (MostPop)
Gợi ý cho mọi user cùng một danh sách: K item có nhiều tương tác dương nhất trong train. Trong repo: `MostPop` (toàn bộ train) và `MostPop(window_days=...)` (chỉ N ngày cuối trước $t_1$).

Vì sao bắt buộc phải có: trên MovieLens, MostPop thường rất mạnh. Kết quả chạy thử ngày 17/10 cho Recall@20 bằng BPR-MF (≈ 0,156). Một mô hình phức tạp mà không thắng MostPop thì kết luận "mô hình tốt" là sai.

## 7. Liên hệ với repo
| Khái niệm | Vị trí |
|---|---|
| Ngưỡng rating ≥ 4 | `src/tkgat/data/split.py` → `SplitConfig.min_rating` |
| MF + BPR | `src/tkgat/models/baselines.py` → `BPRMF.forward` |
| Lấy mẫu âm, dừng sớm theo val | `src/tkgat/train/bpr.py` → `_sample_negatives`, `fit_bpr` |
| MostPop | `src/tkgat/models/baselines.py` → `MostPop` |
| Chạy thử | `python scripts/run.py --data configs/data/ml100k.yaml --model configs/model/bprmf.yaml` |

## 8. Lỗi thường gặp
- Lấy mẫu âm từ toàn bộ dữ liệu (kể cả test) → item test bị coi là âm khi train, hoặc ngược lại làm lộ thông tin test.
- So sánh với baseline yếu (chỉ random) và bỏ qua MostPop.
- Báo cáo kết quả 1 seed.

## 9. Tự kiểm tra
1. Vì sao không thể dùng sai số bình phương trên rating cho bài toán phản hồi ngầm?
2. Viết công thức loss BPR và giải thích từng thành phần.
3. $-\ln\sigma(x)$ và `softplus(-x)` có bằng nhau không? Chứng minh.
4. Vì sao phải lấy mẫu âm, và lấy mẫu từ tập nào là đúng?
5. Nếu tăng `dim` từ 32 lên 256 với dữ liệu 100K, điều gì có thể xảy ra?
6. Learning curve thế nào thì là overfit? Ta xử lý ra sao trong repo?
7. Vì sao MostPop là baseline bắt buộc trên MovieLens?
8. Ngưỡng rating ≥ 4 ảnh hưởng thế nào đến số user ở val/test?

## 10. Tài liệu đọc
- Rendle, Freudenthaler, Gantner, Schmidt-Thieme. *BPR: Bayesian Personalized Ranking from Implicit Feedback.* UAI 2009 — đọc mục 1–4.
- Koren, Bell, Volinsky. *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer 2009 — bài tổng quan ngắn, dễ đọc.
- Hu, Koren, Volinsky. *Collaborative Filtering for Implicit Feedback Datasets.* ICDM 2008 — đọc phần định nghĩa phản hồi ngầm.
