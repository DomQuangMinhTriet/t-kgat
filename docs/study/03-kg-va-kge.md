# Chủ đề 3 — Knowledge Graph và Knowledge Graph Embedding

> Người trình bày: **Phôn** · Ngày: T3 20/10 · Đọc kèm code: `src/tkgat/kg/linking.py`, `scripts/03_link_entities.py`, `scripts/04_fetch_wikidata.py`

## 1. Knowledge Graph (KG) là gì
KG là tập các **triple** $(h, r, t)$: thực thể đầu (head), quan hệ (relation), thực thể đuôi (tail).

Ví dụ với phim *Titanic* (MovieLens item 313):

| head | relation | tail |
|---|---|---|
| Titanic | director | James Cameron |
| Titanic | cast_member | Leonardo DiCaprio |
| Titanic | genre | romance film |
| Titanic | award_received | Academy Award for Best Picture |
| James Cameron | (là đạo diễn của) | Aliens, Terminator 2, ... |

Vì sao KG giúp gợi ý: hai phim **không có user chung** vẫn có thể "gần nhau" qua đạo diễn, diễn viên hay thể loại chung. Thông tin đó đặc biệt có ích cho item ít tương tác (cold-start / long-tail).

**Nguồn KG trong đề tài:**

| Nguồn | Vai trò | Ghi chú |
|---|---|---|
| Freebase (qua KB4Rec) | Bảng ánh xạ MovieLens id → Freebase id | Freebase đã đóng năm 2016. Paper KGAT dùng KG từ KB4Rec/Freebase |
| Wikidata | KG chính để lấy triple **có ngày tháng** | Liên kết qua thuộc tính P646 (Freebase ID) |

## 2. Collaborative Knowledge Graph (CKG)
KGAT gộp tương tác và KG thành **một đồ thị**:
- Mỗi tương tác dương $(u, i)$ thành triple $(u, \text{Interact}, i)$.
- Item được **căn chỉnh** với entity trong KG (item 313 ≡ entity "Titanic").
- Kết quả: $\mathcal{G} = \{(h, r, t)\}$ chứa cả user, item và các entity khác.

Đường đi $u_1 \xrightarrow{\text{Interact}} \text{Titanic} \xrightarrow{\text{cast}} \text{DiCaprio} \xrightarrow{\text{cast}^{-1}} \text{The Beach}$ giải thích được *vì sao* gợi ý *The Beach* cho $u_1$. Đây là điểm mạnh về khả năng giải thích của KGAT.

**Quan hệ ngược:** với mỗi $(h, r, t)$, KGAT thêm $(t, r^{-1}, h)$ để thông tin lan được theo cả hai chiều.

## 3. Knowledge Graph Embedding (KGE)
Mục tiêu: biểu diễn mỗi entity và relation bằng vector sao cho triple **đúng** có điểm tốt hơn triple **sai**.

### TransE (Bordes et al., 2013)
$$\mathbf{e}_h + \mathbf{e}_r \approx \mathbf{e}_t, \qquad g(h,r,t) = \lVert \mathbf{e}_h + \mathbf{e}_r - \mathbf{e}_t \rVert_2^2$$
Quan hệ là một **phép tịnh tiến**. Điểm $g$ càng nhỏ thì triple càng đúng.

Điểm yếu: quan hệ 1-N. *Titanic* có nhiều diễn viên, nên $\mathbf{e}_{\text{Titanic}} + \mathbf{e}_{\text{cast}}$ phải gần **tất cả** các diễn viên cùng lúc, tức là mọi diễn viên của Titanic bị ép lại gần nhau.

### TransR (Lin et al., 2015) — KGAT dùng
Mỗi relation có không gian riêng và một ma trận chiếu $\mathbf{W}_r \in \mathbb{R}^{k \times d}$:
$$g(h,r,t) = \lVert \mathbf{W}_r\mathbf{e}_h + \mathbf{e}_r - \mathbf{W}_r\mathbf{e}_t \rVert_2^2$$
Hai diễn viên có thể khác xa nhau trong không gian chung, nhưng sau khi chiếu vào không gian của quan hệ "cast" thì gần nhau. Điều đó giảm vấn đề 1-N.

### Huấn luyện KGE
Giống BPR: so sánh triple đúng $(h,r,t)$ với triple hỏng $(h,r,t')$ (thay tail ngẫu nhiên):
$$\mathcal{L}_{\text{KG}} = \sum_{(h,r,t,t')} -\ln\sigma\big(g(h,r,t') - g(h,r,t)\big)$$

## 4. Các họ mô hình gợi ý dùng KG

| Họ | Cách dùng KG | Đại diện |
|---|---|---|
| Embedding-based | Học KGE rồi cộng vào embedding item | **CKE** (Zhang et al., 2016): TransR + MF |
| Path-based | Liệt kê đường đi user→item làm đặc trưng | PER, KPRN |
| Propagation-based (GNN) | Lan truyền thông tin trên đồ thị nhiều bước | RippleNet, KGCN, **KGAT** |

CKE là baseline chính khi so với KGAT (paper KGAT có so). Hiểu CKE trước sẽ giúp hiểu KGAT dễ hơn: KGAT = CKE + lan truyền có attention.

## 5. Chất lượng KG — phải kiểm tra, không giả định
- **Liên kết sai:** KB4Rec ánh xạ "Clubland" (1998) với *Cloudland Canyon State Park*. Repo kiểm tra bằng cách đối chiếu tên và năm (`kg/linking.py`), loại liên kết nghi ngờ.
- **Thiếu dữ liệu:** không phải phim nào cũng có đủ 18 quan hệ. Phải báo cáo độ phủ theo từng quan hệ.
- **Entity quá phổ biến:** "English language" hay "United States of America" nối với gần hết phim, nên mang ít thông tin. Có thể lọc entity có bậc quá cao hoặc quá thấp, nhưng **chỉ dựa trên thông tin trước mốc** (chủ đề 6).
- **Ngày tháng:** Wikidata có nhiều mức chính xác (năm, tháng, ngày). Xem chủ đề 6.

## 6. Liên hệ với repo
| Khái niệm | Vị trí |
|---|---|
| Ánh xạ MovieLens → Freebase/YAGO | `data/raw/kb4rec/`, `scripts/03_link_entities.py` |
| Kiểm tra liên kết (tên + năm) | `src/tkgat/kg/linking.py` |
| 18 quan hệ lấy từ Wikidata | `scripts/04_fetch_wikidata.py` → `RELATIONS` |
| Ngày của fact | cột `point_in_time`, `start_time`, `tail_publication`, `tail_inception`, `tail_birth` trong `facts.tsv` |
| Quyết định "KG theo mốc" | `docs/decisions/0004-*.md` |

## 7. Tự kiểm tra
1. Viết 3 triple về một bộ phim bạn thích. Relation nào là 1-N?
2. CKG khác KG thông thường ở điểm nào?
3. Vì sao cần thêm quan hệ ngược $r^{-1}$?
4. Viết hàm điểm của TransE và TransR. Vì sao TransR xử lý quan hệ 1-N tốt hơn?
5. Loss của KGE giống loss BPR ở điểm nào?
6. CKE kết hợp KG với MF như thế nào?
7. Nêu 3 lỗi chất lượng KG có thể gặp với dữ liệu của nhóm.
8. Vì sao entity "English language" ít có ích cho gợi ý?

## 8. Tài liệu đọc
- Bordes et al. *Translating Embeddings for Modeling Multi-relational Data.* NeurIPS 2013 (TransE).
- Lin, Liu, Sun, Liu, Zhu. *Learning Entity and Relation Embeddings for Knowledge Graph Completion.* AAAI 2015 (TransR).
- Zhang, Yuan, Lian, Xie, Ma. *Collaborative Knowledge Base Embedding for Recommender Systems.* KDD 2016 (CKE).
- Wang, Q., Mao, Wang, B., Guo. *Knowledge Graph Embedding: A Survey of Approaches and Applications.* IEEE TKDE 2017 — đọc mục 3.1.
- Zhao et al. *KB4Rec: A Data Set for Linking Knowledge Bases with Recommender Systems.* Data Intelligence 2019.
- Guo et al. *A Survey on Knowledge Graph-Based Recommender Systems.* IEEE TKDE 2020 — đọc phần phân loại.
