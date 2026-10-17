# Lộ trình học tuần 1 (18–24/10/2026)

Mục tiêu: hết tuần 1, **cả Triết và Phôn** đều giải thích được bằng lời của mình toàn bộ 7 chủ đề dưới đây, và chỉ ra được mỗi khái niệm nằm ở đâu trong code của repo.

## Cách học mỗi ngày (khoảng 3 giờ)

| Bước | Thời lượng | Ai | Làm gì |
|---|---|---|---|
| 1. Đọc trước | 60–90 phút | Cả hai | Đọc tài liệu của ngày đó (file trong thư mục này + phần paper được chỉ định) |
| 2. Teach-back | 45 phút | Người trình bày | Giảng lại không nhìn tài liệu, viết công thức lên bảng/giấy |
| 3. Hỏi vặn | 20 phút | Người nghe | Hỏi 3 câu đã chuẩn bị + các câu "Tự kiểm tra" cuối mỗi file |
| 4. Đọc code | 30 phút | Cả hai | Mở file code được chỉ ra trong mục "Liên hệ với repo", chạy thử |
| 5. Ghi chú | 20 phút | Người trình bày | Viết `docs/theory/0X-*.md` (ghi chú của nhóm, ≤ 2 trang) |

## Lịch

| Ngày | Chủ đề | Tài liệu | Người trình bày |
|---|---|---|---|
| CN 18/10 | 2 — Đánh giá offline và rò rỉ dữ liệu | [02-danh-gia-va-ro-ri.md](02-danh-gia-va-ro-ri.md) | Phôn |
| T2 19/10 | 1 — Hệ gợi ý, phản hồi ngầm, BPR | [01-he-goi-y-va-bpr.md](01-he-goi-y-va-bpr.md) | Triết |
| T3 20/10 | 3 — Knowledge Graph và KGE | [03-kg-va-kge.md](03-kg-va-kge.md) | Phôn |
| T4 21/10 | 4 — KGAT chi tiết | [04-kgat.md](04-kgat.md) | Triết |
| T5 22/10 | 5 — Gợi ý có yếu tố thời gian | [05-thoi-gian.md](05-thoi-gian.md) | Triết |
| T6 23/10 | 6 — Tích hợp dữ liệu, liên kết thực thể, đúng theo thời điểm | [06-tich-hop-du-lieu.md](06-tich-hop-du-lieu.md) | Phôn |
| T7 24/10 | 7 — Kiểm định thống kê + kiểm tra chéo cả 7 chủ đề | [07-thong-ke.md](07-thong-ke.md) | Cả hai |

Chủ đề 2 học trước chủ đề 1 vì nó trực tiếp gây ra kết quả tệ ngày 17/10.

## Buổi kiểm tra chéo 24/10
- Mỗi người bốc ngẫu nhiên 10 câu trong các mục "Tự kiểm tra" của cả 7 file và trả lời không nhìn tài liệu.
- Câu nào trả lời sai hoặc ấp úng → ghi vào `docs/theory/can-hoc-lai.md`, học lại trước 25/10.
- Đạt yêu cầu khi mỗi người trả lời đúng ít nhất 8/10 câu.

## Ký hiệu dùng chung (thống nhất giữa hai người)

| Ký hiệu | Nghĩa |
|---|---|
| $u, i, j$ | user, item dương (đã tương tác), item âm (lấy mẫu) |
| $\mathcal{U}, \mathcal{I}$ | tập user, tập item |
| $\mathcal{G} = \{(h, r, t)\}$ | knowledge graph, mỗi phần tử là một triple (head, relation, tail) |
| $\mathbf{e}_h, \mathbf{e}_r$ | embedding của entity $h$, của relation $r$ |
| $\mathcal{N}_h$ | các triple có head là $h$ (hàng xóm của $h$) |
| $\pi(h, r, t)$ | trọng số attention của triple $(h, r, t)$ |
| $t_1, t_2$ | mốc cắt train/val và val/test |
| $\Delta t$ | khoảng thời gian (từ tương tác/fact đến thời điểm dự đoán) |
| $\gamma$ | hệ số suy giảm theo thời gian trong T-KGAT |
| $\delta_r$ | bằng 1 nếu $r$ là quan hệ Interact (user–item), bằng 0 nếu là quan hệ KG |
