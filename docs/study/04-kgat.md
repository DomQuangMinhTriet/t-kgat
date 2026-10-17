# Chủ đề 4 — KGAT chi tiết

> Người trình bày: **Triết** · Ngày: T4 21/10 · Paper: Wang, He, Cao, Liu, Chua. *KGAT: Knowledge Graph Attention Network for Recommendation.* KDD 2019 · Code gốc: github.com/xiangwang1223/knowledge_graph_attention_network

Đọc trước: chủ đề 1 (BPR) và 3 (CKG, TransR). KGAT có thể tóm gọn là **CKE + lan truyền nhiều bước trên CKG có attention**.

## 1. Kiến trúc tổng quát (3 khối)
1. **Embedding layer:** học embedding cho mọi node (user, item, entity) bằng TransR trên CKG.
2. **Attentive embedding propagation layers:** $L$ lớp, mỗi lớp mỗi node gom thông tin từ hàng xóm, có trọng số attention.
3. **Prediction layer:** nối embedding của mọi lớp, rồi lấy tích vô hướng user·item.

## 2. Khối 1 — Embedding với TransR
$$g(h,r,t) = \lVert \mathbf{W}_r\mathbf{e}_h + \mathbf{e}_r - \mathbf{W}_r\mathbf{e}_t \rVert_2^2, \qquad \mathcal{L}_{\text{KG}} = \sum -\ln\sigma\big(g(h,r,t') - g(h,r,t)\big)$$
Giống chủ đề 3. Triple hỏng $(h,r,t')$ được tạo bằng cách thay tail ngẫu nhiên.

## 3. Khối 2 — Lan truyền có attention (trái tim của KGAT)

**Bước a — Gom thông tin hàng xóm (ego-network) của $h$:**
$$\mathbf{e}_{\mathcal{N}_h} = \sum_{(h,r,t)\in\mathcal{N}_h} \pi(h,r,t)\, \mathbf{e}_t$$

**Bước b — Attention theo quan hệ (knowledge-aware attention):**
$$\pi(h,r,t) = (\mathbf{W}_r\mathbf{e}_t)^\top \tanh\big(\mathbf{W}_r\mathbf{e}_h + \mathbf{e}_r\big)$$
$$\pi(h,r,t) \leftarrow \frac{\exp \pi(h,r,t)}{\sum_{(h,r',t')\in\mathcal{N}_h}\exp \pi(h,r',t')}$$
- Trực giác: hàng xóm $t$ nào "khớp" với $h$ trong không gian của quan hệ $r$ thì được chú ý nhiều hơn.
- Softmax chuẩn hóa trên **toàn bộ** hàng xóm của $h$, nên các trọng số cộng lại bằng 1.
- **Chỗ T-KGAT can thiệp** (chủ đề 5): thêm $-\gamma\,\delta_r\,\Delta t$ vào **trước** softmax.

**Bước c — Aggregator (kết hợp $\mathbf{e}_h$ với $\mathbf{e}_{\mathcal{N}_h}$).** Paper thử 3 loại:

| Aggregator | Công thức | Ghi chú |
|---|---|---|
| GCN | $\text{LeakyReLU}\big(\mathbf{W}(\mathbf{e}_h + \mathbf{e}_{\mathcal{N}_h})\big)$ | Cộng rồi biến đổi |
| GraphSage | $\text{LeakyReLU}\big(\mathbf{W}(\mathbf{e}_h \Vert \mathbf{e}_{\mathcal{N}_h})\big)$ | Nối rồi biến đổi |
| **Bi-Interaction** (mặc định) | $\text{LeakyReLU}\big(\mathbf{W}_1(\mathbf{e}_h + \mathbf{e}_{\mathcal{N}_h})\big) + \text{LeakyReLU}\big(\mathbf{W}_2(\mathbf{e}_h \odot \mathbf{e}_{\mathcal{N}_h})\big)$ | Thêm tương tác từng phần tử $\odot$; tốt nhất trong paper |

**Bước d — Nhiều lớp (high-order):** lớp $l$ dùng output lớp $l-1$:
$$\mathbf{e}_h^{(l)} = f\big(\mathbf{e}_h^{(l-1)}, \mathbf{e}_{\mathcal{N}_h}^{(l-1)}\big)$$
Sau $L$ lớp, $\mathbf{e}_u^{(L)}$ chứa thông tin từ node cách $u$ tối đa $L$ bước. Ví dụ $L=3$: user → phim đã xem → đạo diễn → phim khác của đạo diễn đó. Paper dùng $L = 3$, kích thước lớp [64, 32, 16].

## 4. Khối 3 — Dự đoán
$$\mathbf{e}_u^* = \mathbf{e}_u^{(0)} \Vert \mathbf{e}_u^{(1)} \Vert \cdots \Vert \mathbf{e}_u^{(L)}, \qquad \hat{y}(u,i) = {\mathbf{e}_u^*}^\top \mathbf{e}_i^*$$
Nối (concat) mọi lớp để giữ thông tin ở mọi bậc (layer-aggregation).

## 5. Huấn luyện
$$\mathcal{L}_{\text{KGAT}} = \mathcal{L}_{\text{KG}} + \mathcal{L}_{\text{CF}} + \lambda\lVert\Theta\rVert_2^2$$
- $\mathcal{L}_{\text{CF}}$ là loss BPR trên $\hat{y}$ (chủ đề 1).
- Paper **luân phiên** cập nhật: một bước cho $\mathcal{L}_{\text{KG}}$, một bước cho $\mathcal{L}_{\text{CF}}$, và tính lại attention sau mỗi epoch.
- Có dropout trên message (node dropout / message dropout).

## 6. Thiết lập thí nghiệm của paper (cần biết để so sánh công bằng)

| Mục | Paper KGAT | Đề tài T-KGAT |
|---|---|---|
| Dữ liệu | Amazon-book, Last-FM, Yelp2018 | MovieLens-100K (chính), 1M (phụ) |
| KG | Freebase qua KB4Rec, lấy 2 bước (2-hop) | Wikidata có ngày tháng, liên kết qua KB4Rec |
| Lọc | 10-core | 5-core trên train (chính), 10-core (độ nhạy) |
| Chia dữ liệu | **Ngẫu nhiên** 80/20, lấy 10% train làm val | **Theo mốc thời gian chung** 80/10/10 |
| Metric | Recall@20, NDCG@20, xếp hạng toàn bộ | Giống |
| Baseline | FM, NFM, CKE, CFKG, RippleNet, GC-MC, BPR-MF | MostPop, BPR-MF, CKE, KGAT |

Hệ quả: con số trong paper **không so trực tiếp** được với con số của nhóm vì khác dữ liệu và cách chia. Nhóm phải chạy lại KGAT gốc trên đúng protocol của mình.

## 7. Điểm yếu của KGAT mà đề tài nhắm tới
1. **Không có thời gian:** một rating năm 1997 và một rating hôm qua có cùng vai trò. Sở thích thay đổi theo thời gian thì KGAT không thấy.
2. **KG tĩnh:** dùng fact bất kể nó xuất hiện khi nào, có thể đưa "tri thức tương lai" vào (rò rỉ, chủ đề 2 và 6).
3. **Chia ngẫu nhiên:** kết quả gốc có thể lạc quan hơn thực tế (Ji et al., TOIS).

T-KGAT trả lời (1) bằng suy giảm theo $\Delta t$ trong attention và (2) bằng KG theo mốc.

## 8. Liên hệ với repo
Code KGAT chưa có trong repo; tuần 2 nhóm sẽ viết `src/tkgat/models/kgat.py` theo đúng các phương trình ở trên, rồi đối chiếu với code gốc. Những phần đã có và được tái sử dụng:

| Phần | Vị trí |
|---|---|
| Vòng huấn luyện BPR, dừng sớm theo val | `src/tkgat/train/bpr.py` |
| Metric xếp hạng toàn bộ | `src/tkgat/eval/metrics.py` |
| Mã hóa id user/item/entity | `src/tkgat/data/encode.py` |
| Thiết kế thời gian trong attention | `docs/decisions/0005-time-in-attention.md` |

## 9. Tự kiểm tra
1. Vẽ sơ đồ 3 khối của KGAT và dòng dữ liệu đi qua.
2. Viết công thức attention $\pi(h,r,t)$. Vì sao có $\tanh$ và $\mathbf{W}_r$?
3. Softmax trong KGAT chuẩn hóa trên tập nào?
4. Bi-Interaction khác GCN aggregator ở điểm nào? Vì sao nó tốt hơn?
5. Với $L=2$, user nhận được thông tin từ những node nào? Cho ví dụ cụ thể với MovieLens.
6. Vì sao phải concat embedding mọi lớp thay vì chỉ lấy lớp cuối?
7. Loss tổng gồm những thành phần nào? Huấn luyện luân phiên nghĩa là gì?
8. Vì sao không so trực tiếp số của paper với số của nhóm?
9. Nêu 2 điểm yếu của KGAT liên quan đến thời gian.
10. Thời gian sẽ được đưa vào công thức nào, ở vị trí nào?

## 10. Tài liệu đọc
- Wang, He, Cao, Liu, Chua. *KGAT: Knowledge Graph Attention Network for Recommendation.* KDD 2019 — **đọc toàn bộ**, đặc biệt mục 3 và 4.
- Veličković et al. *Graph Attention Networks.* ICLR 2018 — nguồn gốc của attention trên đồ thị.
- Wang, Zhang, Wang, Zhao, Li, Xie, Guo. *RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems.* CIKM 2018.
- He et al. *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.* SIGIR 2020 — đọc mục 2 để thấy phần nào của GNN thực sự có ích.
