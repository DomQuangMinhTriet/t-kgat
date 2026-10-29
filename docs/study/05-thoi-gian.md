# Chủ đề 5 — Gợi ý có yếu tố thời gian

> Người trình bày: **Triết** · Ngày: T5 22/10 · Đọc kèm: `docs/decisions/0005-time-in-attention.md`, `docs/decisions/0004-kg-temporal-rules.md`

## 1. Vì sao thời gian quan trọng
- **Sở thích trôi (preference drift):** một user thích phim hoạt hình năm 1996 có thể chuyển sang phim tâm lý năm 1998.
- **Độ phổ biến thay đổi:** *Titanic* được xem nhiều sau Oscar tháng 3/1998, nhưng không phải trước đó.
- **Tri thức thay đổi:** tại một thời điểm, phần tiếp theo của một bộ phim có thể chưa ra mắt, giải thưởng có thể chưa được trao.

Mô hình tĩnh (MF, KGAT) coi mọi tương tác như nhau bất kể thời điểm. Đề tài đặt câu hỏi: **nếu cho mô hình biết thời gian thì gợi ý có tốt hơn không, và tốt hơn ở đâu?**

## 2. Ba cách đưa thời gian vào mô hình

| Cách | Ý tưởng | Đại diện |
|---|---|---|
| Thời gian là tham số | Bias, embedding thay đổi theo thời gian | timeSVD++ (Koren, 2009) |
| Trọng số suy giảm (decay) | Tương tác càng cũ càng ít ảnh hưởng: $w = e^{-\gamma\Delta t}$ | Ding & Li 2005; nhiều mô hình CF có decay |
| Mô hình chuỗi | Xem lịch sử là một chuỗi có thứ tự, có khoảng thời gian | GRU4Rec, SASRec, **TiSASRec** (Li et al., 2020) |
| KG theo thời gian | Mỗi fact có thời gian hiệu lực $(h, r, t, \tau)$ | TTransE, HyTE; Temporal KG |

### timeSVD++ (Koren, KDD 2009)
$$\hat{r}_{ui}(t) = \mu + b_i + b_{i,\text{Bin}(t)} + b_u + \alpha_u \cdot \text{dev}_u(t) + \mathbf{q}_i^\top \mathbf{p}_u(t)$$
Bias của item thay đổi theo "thùng thời gian"; bias và sở thích user trôi theo thời gian. Bài học: trên Netflix Prize, mô hình hóa thời gian giúp **đáng kể**. Nhưng Netflix kéo dài 7 năm, còn MovieLens-100K chỉ có 7 tháng.

### TiSASRec (Li, Wang, McAuley, WSDM 2020)
Self-attention trên chuỗi, cộng thêm embedding của **khoảng thời gian** giữa hai tương tác vào phép tính attention. Đây là ví dụ gần nhất với ý tưởng của T-KGAT: thời gian điều chỉnh trọng số attention.

## 3. Thiết kế T-KGAT (quyết định 0005)

**Biến thể chính — suy giảm trên cạnh user–item:**
$$\pi_{\text{time}}(h,r,t) = \pi(h,r,t) - \gamma\,\delta_r\,\Delta t_{\text{norm}}$$
rồi mới softmax như KGAT gốc.
- $\delta_r = 1$ nếu $r$ là Interact, 0 nếu là quan hệ KG. Chỉ cạnh tương tác bị suy giảm.
- $\Delta t$ = mốc dự đoán ($t_1$ cho train/val, $t_2$ cho test) trừ thời điểm tương tác, chuẩn hóa về khoảng hợp lý.
- $\gamma \geq 0$ là siêu tham số chọn theo val (lưới {1, 3, 10}), không học bằng gradient (quyết định 0009). **$\gamma = 0$ thì quay về đúng KGAT**, nên so sánh với KGAT là so sánh lồng nhau, rất sạch.

**Vì sao trừ trước softmax lại là "suy giảm mũ":**
$$\text{softmax}(\pi - \gamma\Delta t)_t = \frac{e^{\pi_t}\,e^{-\gamma\Delta t_t}}{\sum_{t'} e^{\pi_{t'}}\,e^{-\gamma\Delta t_{t'}}}$$
Tức trọng số attention gốc được nhân với $e^{-\gamma\Delta t}$ rồi chuẩn hóa lại. Tương tác cũ hơn bị giảm theo hàm mũ **so với các hàng xóm khác của cùng node**. Hệ quả: nếu mọi tương tác của một user có cùng $\Delta t$, suy giảm triệt tiêu **ở phía user** (softmax của node user), không có tác dụng gì. Phía item thì khác: các cạnh vào một item đến từ nhiều user, ở nhiều thời điểm khác nhau (độ trải thời gian của cạnh train theo item có trung vị 45 ngày ở ML-100K, 67 ngày ở ML-1M), nên suy giảm vẫn có thể có tác dụng ở đó. Điều này chỉ đúng nếu cạnh ngược của Interact (item → user) cũng bị suy giảm; điểm này sẽ chốt khi viết `models/kgat.py`.

**Biến thể so sánh — thời gian trên triple KG (quyết định 0004):** KG cho train/val được dựng ở $t_1$, KG cho test ở $t_2$; chỉ giữ fact đã đúng trước mốc. Đây là cách giải quyết rò rỉ tri thức tương lai, đồng thời là thí nghiệm "thời gian của tri thức có giúp ích không".

## 4. Cảnh báo về dữ liệu MovieLens (phải nói rõ khi báo cáo)
- Timestamp trong MovieLens, nhóm hiểu là **lúc ghi rating**, không phải lúc xem phim. Đây là giả định của nhóm: README của GroupLens chỉ nói đó là số giây tính từ epoch (quyết định 0008). Giả định này khớp với dữ liệu, vì nhiều user chấm hàng loạt trong một lần ngồi nhập.
- Số liệu nhóm đo (`docs/data-report-numbers.md`): **77,9%** rating của ML-100K (72,4% với ML-1M) nằm trong 24 giờ đầu kể từ rating đầu tiên của user; **53,0%** user của ML-100K có toàn bộ cạnh train trải trong chưa đầy 1 giờ; một phiên nhập rating (nghỉ quá 60 phút thì kết thúc) có độ dài trung vị 13,4 phút ở ML-100K.
- Hệ quả: với nhiều user, $\Delta t$ giữa các cạnh của họ gần bằng 0, nên suy giảm ở phía user (mục 3) gần như không có tác dụng; phía item vẫn có thể có. Một kết quả "không cải thiện" trên 100K **là kết quả hợp lệ** và phải giải thích được bằng số liệu này.
- **Thí nghiệm đối chứng bắt buộc — hoán vị timestamp:** xáo trộn timestamp giữa các tương tác của cùng user rồi chạy lại. Nếu T-KGAT vẫn tốt như cũ, cải thiện **không đến từ thông tin thời gian** (có thể chỉ do thêm tham số hoặc do regularization).
- ML-1M kéo dài gần 3 năm nên có nhiều khả năng có tín hiệu thời gian hơn. Đó là lý do chạy song song.

## 5. Liên hệ với repo
| Phần | Vị trí |
|---|---|
| Mốc $t_1$, $t_2$ | `src/tkgat/data/split.py` (`t1`, `t2` trong `TemporalSplit`) |
| Phân tích phiên nhập rating | `notebooks/01_eda.ipynb` (mục 3b), số liệu ở `docs/data-report-numbers.md` |
| Thiết kế suy giảm | `docs/decisions/0005-time-in-attention.md` |
| KG theo mốc | `docs/decisions/0004-kg-temporal-rules.md`, `data/raw/wikidata/facts.tsv` |
| Baseline thời gian đơn giản | `MostPop(window_days=...)` trong `src/tkgat/models/baselines.py` |

## 6. Tự kiểm tra
1. Nêu 3 lý do thời gian ảnh hưởng tới gợi ý, mỗi lý do một ví dụ MovieLens.
2. timeSVD++ đưa thời gian vào những thành phần nào?
3. Chứng minh softmax$(\pi - \gamma\Delta t)$ tương đương nhân trọng số với $e^{-\gamma\Delta t}$ rồi chuẩn hóa.
4. Vì sao chỉ suy giảm cạnh Interact mà không suy giảm cạnh KG?
5. $\gamma = 0$ thì T-KGAT thành gì? Vì sao tính chất này có ích khi so sánh?
6. Nếu mọi tương tác của user $u$ có cùng timestamp, suy giảm ảnh hưởng thế nào tới $u$?
7. Vì sao timestamp MovieLens không phải thời điểm xem phim? Số liệu nào chứng minh?
8. Mô tả thí nghiệm hoán vị timestamp và cách đọc kết quả.
9. MostPop trong cửa sổ 30 ngày gần nhất là baseline cho câu hỏi nào?
10. Biến thể "thời gian trên triple KG" khác biến thể chính ở đâu?

## 7. Tài liệu đọc
- Koren. *Collaborative Filtering with Temporal Dynamics.* KDD 2009 (timeSVD++).
- Li, Wang, McAuley. *Time Interval Aware Self-Attention for Sequential Recommendation.* WSDM 2020 (TiSASRec).
- Campos, Díez, Cantador. *Time-aware Recommender Systems: A Comprehensive Survey and Analysis of Existing Evaluation Protocols.* UMUAI 2014 — đọc phần về protocol đánh giá.
- Leblay, Chekol. *Deriving Validity Time in Knowledge Graph.* WWW Companion 2018 (TTransE).
- Dasgupta, Ray, Talukdar. *HyTE: Hyperplane-based Temporally Aware Knowledge Graph Embedding.* EMNLP 2018.
- Harper, Konstan. *The MovieLens Datasets: History and Context.* ACM TiiS 2015 — đọc phần về cách thu thập rating.
