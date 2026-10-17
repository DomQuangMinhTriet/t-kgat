# Chủ đề 7 — Kiểm định thống kê cho thí nghiệm gợi ý

> Người trình bày: **cả hai** · Ngày: T7 24/10 · Đọc kèm code: `src/tkgat/eval/stats.py`, `src/tkgat/registry.py`, `configs/model/bprmf.yaml` (seeds)

## 1. Vì sao cần thống kê
Kết quả "T-KGAT Recall@20 = 0,162, KGAT = 0,158" chưa nói được gì nếu không biết:
- Chạy lại với seed khác thì chênh lệch còn không?
- Chênh lệch có lớn hơn dao động ngẫu nhiên giữa các user không?

Với test chỉ có **51 user** (ML-100K), dao động rất lớn: BPR-MF chạy thử cho 0,156 ± 0,026 giữa các seed. Một chênh lệch 0,004 nằm gọn trong nhiễu.

## 2. Hai nguồn biến thiên
| Nguồn | Ý nghĩa | Cách xử lý |
|---|---|---|
| Ngẫu nhiên khi huấn luyện | Khởi tạo, lấy mẫu âm, thứ tự batch | Chạy **5 seed** (2019–2023), báo cáo trung bình ± độ lệch chuẩn |
| Mẫu user | Test chỉ là một mẫu user | Kiểm định **cặp theo user**, khoảng tin cậy bootstrap |

## 3. Kiểm định cặp theo user
Hai mô hình A và B được chấm trên **cùng** tập user, nên mỗi user cho một cặp $(m_A(u), m_B(u))$. Xét hiệu $d_u = m_A(u) - m_B(u)$.

**Paired t-test:**
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$
Giả định $d_u$ xấp xỉ phân phối chuẩn. Recall theo user thường lệch và có nhiều số 0, nên giả định này hay bị vi phạm.

**Wilcoxon signed-rank (mặc định của nhóm):** xếp hạng $|d_u|$, cộng hạng của các $d_u > 0$. Không giả định phân phối chuẩn, nên phù hợp với metric theo user. Lưu ý: user có $d_u = 0$ bị bỏ ra.

**Khoảng tin cậy bootstrap:** lấy mẫu có hoàn lại $n$ user, tính $\bar{d}$, lặp 10.000 lần, lấy phân vị 2,5% và 97,5%. Nếu khoảng không chứa 0 thì chênh lệch có ý nghĩa. Ưu điểm: báo cáo được **độ lớn** chênh lệch, không chỉ p-value.

## 4. So sánh nhiều lần
Nhóm so T-KGAT với nhiều mô hình (MostPop, BPR-MF, CKE, KGAT) trên 2 metric và 2 dữ liệu. Với 16 kiểm định ở mức 0,05, xác suất có ít nhất một kết quả "có ý nghĩa" do may mắn là $1 - 0{,}95^{16} \approx 56\%$.

**Hiệu chỉnh Holm–Bonferroni:** sắp p-value tăng dần $p_{(1)} \le \dots \le p_{(m)}$; bác bỏ $H_{(k)}$ nếu $p_{(j)} \le \frac{\alpha}{m - j + 1}$ với mọi $j \le k$. Chặt như Bonferroni về kiểm soát lỗi nhưng mạnh hơn.

**Quy ước của nhóm:** gộp p-value của mọi so sánh với mô hình chính rồi hiệu chỉnh Holm một lần.

## 5. Nhiều dữ liệu
Khi so trên nhiều dataset, Demšar (2006) khuyên dùng Wilcoxon **trên các dataset** hoặc Friedman + Nemenyi. Nhóm chỉ có 2 dataset nên không đủ cho kiểm định đó; báo cáo riêng từng dataset và nói rõ giới hạn.

## 6. Báo cáo kết quả đúng cách
Một dòng trong bảng kết quả cần có:
- Trung bình ± độ lệch chuẩn qua 5 seed.
- Chênh lệch so với baseline mạnh nhất, kèm khoảng tin cậy bootstrap 95%.
- p-value Wilcoxon đã hiệu chỉnh Holm.
- Đánh dấu rõ cấu hình nào được **chọn trên val**; test chỉ chấm một lần.

**Kết quả âm tính vẫn là kết quả:** nếu T-KGAT không hơn KGAT một cách có ý nghĩa trên ML-100K, báo cáo điều đó cùng giải thích (phiên nhập rating, chủ đề 5) và thí nghiệm đối chứng. Như vậy tốt hơn nhiều so với chọn ra một seed đẹp.

## 7. Liên hệ với repo
| Phần | Vị trí |
|---|---|
| Wilcoxon, paired t, bootstrap, Holm | `src/tkgat/eval/stats.py` |
| 5 seed, lưới siêu tham số | `configs/model/bprmf.yaml` |
| Ghi mọi lần chạy (commit, config, seed, metric) | `src/tkgat/registry.py` → `results/registry.csv` |
| Metric theo từng user (đầu vào của kiểm định) | `src/tkgat/eval/metrics.py` |

## 8. Tự kiểm tra
1. Vì sao 0,162 so với 0,158 chưa đủ để kết luận?
2. Nêu 2 nguồn biến thiên và cách xử lý từng nguồn.
3. Vì sao dùng kiểm định **cặp**? Nếu dùng kiểm định độc lập thì sai ở đâu?
4. Khi nào Wilcoxon phù hợp hơn paired t-test?
5. Mô tả cách tính khoảng tin cậy bootstrap cho chênh lệch Recall.
6. Tính xác suất có ít nhất một kết quả dương tính giả với 10 kiểm định ở mức 0,05.
7. Áp dụng Holm cho p-value {0,01; 0,02; 0,04} với α = 0,05. Kết quả nào được bác bỏ?
8. Vì sao không chọn seed có kết quả tốt nhất để báo cáo?
9. Một kết quả âm tính nên được trình bày thế nào?

## 9. Tài liệu đọc
- Demšar. *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 2006.
- Dror, Baumer, Shlomov, Reichart. *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing.* ACL 2018 — phần chọn kiểm định, rất dễ đọc.
- Sakai. *Statistical Significance, Power, and Sample Sizes: A Systematic Review of SIGIR and TOIS, 2006–2015.* SIGIR 2016.
- Holm. *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian Journal of Statistics 1979.
- Efron, Tibshirani. *An Introduction to the Bootstrap.* 1993 — chương 1–6 nếu cần nền tảng.
