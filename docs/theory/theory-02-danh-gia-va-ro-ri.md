# Chủ đề 2: Đánh giá offline và rò rỉ dữ liệu

Người trình bày: Phôn · Người hỏi vặn: Triết · Ngày teach-back: ../10/2026

Bản nháp do Triết chuẩn bị sau khi học chủ đề này bằng hỏi đáp; Phôn sửa lại bằng lời của mình và điền phần teach-back.

## Ý chính
- Rò rỉ dữ liệu là khi thông tin của val/test (hoặc của tương lai) ảnh hưởng tới một quyết định lúc huấn luyện hay tiền xử lý. Hậu quả: điểm test lạc quan, không còn là ước lượng cho dữ liệu chưa thấy.
- Chia theo thời điểm toàn cục: hai mốc t1 < t2 là phân vị 80% và 90% của mọi timestamp, áp cho tất cả user. train: ts < t1; val: t1 <= ts < t2; test: ts >= t2. Một mốc chung nên tương lai của user này không lọt vào quá khứ của user khác.
- Tương tác dương là rating >= 4. K-core (5-core) chỉ tính trên train dương. Val/test chỉ giữ user và item đã có trong train; user hoặc item mới bị loại và được đếm, không đưa lại.
- Cái giá: user mới sau t1 là cold-start, mô hình không có lịch sử để dự đoán nên bị loại. ML-100K chỉ còn 66 user ở val (12,6% positive) và 51 user ở test (13,3%). Nguyên nhân là user chấm hàng loạt rồi rời đi: 77,9% rating nằm trong 24 giờ đầu của user, 63,5% user chỉ hoạt động một ngày, nên phần lớn rating sau t1 đến từ user mới. Đó là lý do chạy song song ML-1M (test 952 user, giữ 75,7% positive).
- Mask: khi xếp hạng cho user u, loại các item u đã tương tác, vì gợi ý lại item đã xem không có ý nghĩa và chúng chiếm chỗ trong top-K. Mask val gồm mọi tương tác (kể cả rating thấp) có ts < t1; mask test gồm mọi tương tác có ts < t2 (train cộng val). Không bao giờ chứa tương tác từ mốc của chính phase đó trở đi, vì đó là đáp án.
- Xếp hạng toàn bộ item rồi tính Recall/NDCG@20. Không dùng sampled metrics (1 item đúng + 100 item âm) vì chúng có thể đảo thứ tự giữa các mô hình (Krichene & Rendle, 2020).

## Công thức
Recall@K(u) = |top-K(u) ∩ T_u| / |T_u|

DCG@K(u) = Σ_{k=1..K} 1[item hạng k ∈ T_u] / log₂(k+1); NDCG@K = DCG@K / IDCG@K, với IDCG là DCG của xếp hạng lý tưởng.

Ví dụ tính tay: xếp hạng [4, 7, 1, 9], đáp án {7, 9}, K = 3. Top-3 = [4, 7, 1], chỉ 7 đúng (hạng 2). Recall@3 = 1/2 = 0,5. DCG@3 = 1/log₂3 ≈ 0,6309. IDCG@3 = 1/log₂2 + 1/log₂3 ≈ 1,6309. NDCG@3 ≈ 0,387.

## Nguồn rò rỉ và nơi chặn
| Nguồn | Nơi chặn |
|---|---|
| Chọn mô hình bằng test | `fit_bpr` chỉ nhận train và val; `scripts/run.py` chọn trên val, chấm test một lần; test `test_fit_bpr_never_reads_test` |
| K-core trên toàn bộ dữ liệu | `kcore()` chỉ chạy trên train dương; `check_split` kiểm tra 4; test `test_kcore_never_sees_future` |
| Thống kê dùng dữ liệu tương lai | `MostPop` chỉ đọc `data.train` |
| Cặp (user, item) trùng giữa các phần | `dedupe_pairs`; `check_split` kiểm tra 2 |
| User/item chưa từng thấy | Loại và đếm; `check_split` kiểm tra 3 |
| Tri thức tương lai trong KG | KG dựng theo mốc (quyết định 0004); `check_kg` trong `src/tkgat/kg/temporal.py`, script 06 từ chối lưu nếu vi phạm |
| Mask sai | `seen_before_*`; `check_split` kiểm tra 5 và 6 |

`check_split` có 6 kiểm tra: (1) thứ tự thời gian, (2) không cặp nào ở hai phần, (3) val/test chỉ có user và item đã biết, (4) train thỏa 5-core, (5) mask chỉ dùng quá khứ, (6) đáp án không trùng mask của chính phase. Ba nguồn không do hàm này bắt: chọn mô hình bằng test, thống kê tương lai, tri thức tương lai trong KG. `scripts/02_build_splits.py` từ chối lưu split nếu `check_split` báo vi phạm.

## Nằm ở đâu trong code
- Split và mốc t1, t2: `src/tkgat/data/split.py`
- Kiểm tra rò rỉ: `src/tkgat/data/leakage.py` (`check_split`)
- Mask khi chấm: `mask_val`, `mask_test` trong `src/tkgat/data/encode.py`; áp trong `src/tkgat/eval/metrics.py`
- Chọn trên val, chấm test một lần: `scripts/run.py`

## 3 câu tự kiểm tra
1. Vì sao chọn epoch trên test là rò rỉ? Vì test không còn là dữ liệu chưa thấy; điểm bị thổi phồng.
2. Vì sao ML-100K chỉ còn 51 user ở test? User chấm hàng loạt rồi đi, nên rating sau t2 chủ yếu thuộc user mới, bị loại vì cold-start.
3. Mask test gồm những gì, và vì sao lấy cả rating thấp? Mọi tương tác trước t2, vì user đã xem item đó dù không thích.

## Teach-back với Triết (điền sau buổi giảng)
Ngày: ..; 3 câu Triết hỏi vặn: ..; trả lời thế nào: ..
