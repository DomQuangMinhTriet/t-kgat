# Báo cáo dữ liệu: MovieLens, liên kết Wikidata và KG theo thời gian

Dự án T-KGAT — Temporal Knowledge Graph Attention Network for Recommendation. Báo cáo dành cho cô Nguyễn Ngọc Minh Châu; nhóm thực hiện: Đởm Quang Minh Triết, Nguyễn Đăng Phôn.

Mọi ngày tính theo UTC. Mọi con số do `scripts/07_report_numbers.py` sinh từ các file của pipeline (bảng đầy đủ kèm nguồn: `docs/data-report-numbers.md`); chỗ nguồn chưa tồn tại ghi là "chưa có". Số item kèm cơ sở đếm (Phụ lục A); phần chi tiết nằm ở Phụ lục.

## Tóm tắt

- **Dữ liệu.** ML-100K: 943 user, 1.682 item được rating, 100.000 rating, từ 20/09/1997 đến 22/04/1998. ML-1M: 6.040 user, 3.706 item được rating, 1.000.209 rating, từ 25/04/2000 đến 28/02/2003, nhưng 90,5% rating nằm trong năm 2000. Cả 9 file gốc khớp SHA-256.
- **Timestamp.** Nhóm hiểu timestamp là lúc user ghi rating (README của GroupLens chỉ nói số giây kể từ epoch). 77,9% rating của ML-100K và 72,4% của ML-1M nằm trong 24 giờ đầu của user, nên các cạnh của một user gần như trùng thời điểm, còn các cạnh của một item trải trên nhiều tháng (mục 3). Báo cáo chưa kiểm tra nguyên nhân.
- **Chia theo mốc t1/t2** giữ lại 12,6% (val) và 13,3% (test) positive của ML-100K, tức chỉ 66 và 51 user; ML-1M giữ lại 21,0% và 75,7% với 480 và 952 user (mục 4).
- **KG theo thời gian** chỉ bỏ 1,3% (ML-100K) và 1,5% (ML-1M) số fact tại t1, nhưng các item mất fact nhận 53,1% và 48,1% positive val/test của ML-100K, 41,5% và 43,9% của ML-1M (mục 6). Từ quyết định 0011, KG gồm fact Wikidata và metadata MovieLens (thể loại, năm, thập niên).
- **Baseline** (Recall@20 trên test). ML-1M: MostPop 0,0902; MostPop recent (cửa sổ 7 ngày) 0,1060; BPR-MF 0,1194 ± 0,0031. ML-100K (51 user test): MostPop 0,1555; MostPop recent (cửa sổ 14 ngày) 0,3372; BPR-MF 0,1471 ± 0,0295 (mục 7).

## 1. Hai bộ dữ liệu

Nhóm dùng MovieLens-100K (bộ chính, Phôn chạy) và MovieLens-1M (cỡ lớn hơn, Triết chạy song song, cùng quy trình và mã). Liên kết dùng KB4Rec (MovieLens-20M → Freebase/YAGO); fact có ngày lấy từ Wikidata (bản chụp ngày 17/10/2026: 119.748 fact, 18 quan hệ, 3.725 phim). 9 file gốc kiểm bằng SHA-256: 9 khớp, 0 thiếu, 0 lệch.

| | ML-100K | ML-1M |
|---|---|---|
| User (có ít nhất một rating) | 943 | 6.040 |
| Item được rating | 1.682 | 3.706 |
| Rating | 100.000 | 1.000.209 |
| Mật độ = rating / (user × item được rating) | 6,3% | 4,5% |
| Khoảng thời gian (UTC) | 20/09/1997 – 22/04/1998 | 25/04/2000 – 28/02/2003 |
| Số ngày giữa rating đầu và cuối (làm tròn) | 215 | 1.039 |
| Trung vị số rating mỗi user | 65 | 96 |
| Rating ≥ 4 (positive), trên mọi rating gốc | 55,4% | 57,5% |
| Năm có nhiều rating nhất | 1997 (52,9%) | 2000 (90,5%) |

*Bảng 1. Quy mô và khoảng thời gian của hai bộ dữ liệu gốc (trước làm sạch).*

ML-1M kéo dài tới năm 2003 nhưng gần như toàn bộ rating nằm trong năm 2000, nên các cửa sổ val và test của ML-1M có độ dài rất khác nhau (mục 4).

## 2. Làm sạch

Chỉ ML-100K được làm sạch: bỏ item 267 (không có tiêu đề nên không liên kết được) cùng 9 rating; gộp 18 item trùng tiêu đề (608 rating được gán lại); cặp (user, item) trùng giữ lần sớm nhất (quyết định 0002), loại 307 cặp. ML-100K còn 1.663 item trong danh mục sau làm sạch (từ 1.682 item được rating), 99.991 rating sau làm sạch và 99.684 sau khử trùng. ML-1M không cần làm sạch, không có cặp trùng và giữ nguyên 1.000.209 rating. Mục 1 dùng dữ liệu gốc nên chưa trừ các phần này.

## 3. Rating theo phiên và hệ quả cho Δt

**Timestamp.** README của GroupLens (cả hai bộ) chỉ mô tả timestamp là số giây kể từ epoch (UTC), không nói đó là thời điểm nào của rating. Nhóm hiểu đó là lúc user ghi rating; các số dưới đây nhất quán với cách hiểu này nhưng không chứng minh nó. Số rating mỗi tuần có những tuần tăng đột biến (Phụ lục B); báo cáo chưa kiểm tra nguyên nhân.

- **24 giờ đầu của user:** 77,9% rating của ML-100K và 72,4% của ML-1M nằm trong 24 giờ kể từ rating đầu của user (theo ngày lịch UTC: 73,5% và 66,0%); 63,5% và 58,7% user chỉ hoạt động đúng một ngày.
- **Phiên** (nghỉ quá 60 phút): 2.692 và 24.413 phiên; trung vị mỗi user chỉ có 1 phiên; trung vị thời lượng phiên có từ 2 rating là 13,4 và 13,1 phút; ngưỡng 30 phút cho 2.793 và 25.163 phiên. Hai rating liên tiếp của một user cách nhau dưới 1 phút ở 76,8% và 82,9% trường hợp.

**Δt giữa các cạnh của cùng một node.** T-KGAT trừ một số hạng phụ thuộc Δt vào điểm attention trước softmax, và softmax chỉ so sánh các cạnh của cùng một node; vì vậy đo độ trải thời gian của các cạnh train (trước t1) trong một node (Hình 1, Bảng 2).

![Hình 1. Phân phối độ trải thời gian max(ts) − min(ts) của các cạnh train trong một user (trái) và trong một item (phải); trục x là thang log, độ trải bằng 0 được vẽ tại 1 phút.](figures/fig2_do_trai_thoi_gian.png)

| Độ trải của các cạnh train trong một node | ML-100K | ML-1M |
|---|---|---|
| **User:** nhỏ hơn 1 giờ | 53,0% | 53,4% |
| User: nhỏ hơn 1 ngày | 68,1% | 71,1% |
| User: nhỏ hơn 30 ngày | 80,9% | 90,1% |
| User: trung vị độ lệch chuẩn của (t1 − ts) | khoảng 14 phút | khoảng 14 phút |
| **Item:** nhỏ hơn 1 giờ | 0,0% | 0,0% |
| Item: nhỏ hơn 1 ngày | 0,0% | 0,0% |
| Item: nhỏ hơn 30 ngày | 0,3% | 0,4% |
| Item: trung vị độ lệch chuẩn của (t1 − ts) | 45 ngày | 67 ngày |

*Bảng 2. Độ trải thời gian của các cạnh train theo user và theo item.*

Trong một user, các Δt gần như bằng nhau, nên phép trừ phụ thuộc Δt khó làm đổi thứ tự attention ở phía user; ở phía item, các cạnh trải trên nhiều tháng nên Δt có thể có tác dụng. Đây mới là **giả thuyết** cho RQ3: nhóm sẽ kiểm bằng thí nghiệm hoán vị timestamp (bắt buộc cho mọi biến thể có thời gian); số liệu hiện có chưa đủ để kết luận.

## 4. Chia theo mốc t1/t2 và bằng chứng không rò rỉ

**Quy trình** (quyết định 0002): khử trùng cặp (user, item); t1 và t2 là phân vị 80% và 90% thời gian của mọi timestamp (train trước t1, val từ t1 đến trước t2, test từ t2); positive là rating ≥ 4; 5-core tính trên train; val và test chỉ giữ user và item đã có trong train; mask khi xếp hạng chỉ chứa tương tác trước mốc; chọn mô hình trên val, test chỉ chấm một lần. ML-100K: t1 = 07/03/1998, t2 = 31/03/1998; ML-1M: t1 = 02/12/2000, t2 = 29/12/2000. Train sau 5-core có 43.018 positive, 745 user, 932 item ở ML-100K; 461.858 positive, 5.350 user, 3.009 item ở ML-1M.

| Cửa sổ | Positive trước lọc | Loại vì user mới | Loại vì item mới | Còn lại | Tỷ lệ còn lại | User còn lại |
|---|---|---|---|---|---|---|
| ML-100K val | 5.648 | 4.886 | 50 | 712 | 12,6% | 66 |
| ML-100K test | 5.612 | 4.805 | 60 | 747 | 13,3% | 51 |
| ML-1M val | 57.812 | 45.605 | 74 | 12.133 | 21,0% | 480 |
| ML-1M test | 54.452 | 12.911 | 343 | 41.198 | 75,7% | 952 |

*Bảng 3. Positive của val và test trước và sau khi loại user hoặc item không có trong train.*

"User mới" và "item mới" là user và item không có trong train sau 5-core (không nhất thiết mới đăng ký); positive bị loại vì user mới được đếm trước.

![Hình 2. Tỷ lệ positive của mỗi cửa sổ còn lại hoặc bị loại vì user mới hoặc item mới.](figures/fig3_positive_con_lai.png)

- **ML-100K:** phần lớn positive bị loại vì user mới; 99,8% (val) và 99,6% (test) trong số đó thuộc user có rating đầu tiên từ t1 trở đi (116 trong 117 và 96 trong 97 user bị loại), phù hợp với mục 3. Chỉ còn 66 và 51 user nên kết quả sẽ nhiễu: mọi số kèm độ lệch chuẩn qua seed và cần kiểm định cặp.
- **ML-1M:** val dài 27 ngày: 99,0% positive bị loại vì user mới thuộc user có rating đầu tiên từ t1 trở đi (611 trong 622 user). Test dài 791 ngày: 83,7% thuộc user như vậy (195 trong 233 user) và 16,3% thuộc user đã có rating trước t1 nhưng không vào train (chưa phân tách lý do); 75,7% positive của test thuộc user có trong train. Mỗi cửa sổ chứa khoảng 10% số rating nhưng độ dài và thành phần user khác hẳn nhau, nên điểm val và test của ML-1M không so trực tiếp được. ML-100K: val 25 ngày, test 22 ngày (làm tròn đến ngày).
- **Khi kết luận** nên dựa chủ yếu vào ML-1M: test của ML-1M có 952 user, của ML-100K chỉ 51 user.

**Bằng chứng không rò rỉ.** `check_split` chạy trên split đã lưu của cả hai bộ trả về 0 vấn đề (thứ tự thời gian, không có user hoặc item lạ trong val/test, mask chỉ chứa tương tác trước mốc); CI chạy các kiểm tra này cùng các test cố ý tạo từng loại rò rỉ.

## 5. Liên kết MovieLens → Freebase/Wikidata

KB4Rec liên kết MovieLens-20M với Freebase và YAGO. ML-1M dùng chung không gian id với MovieLens-20M; ML-100K được ánh xạ sang ML-1M theo tiêu đề và năm (quyết định 0003, quy tắc ở Phụ lục C), rồi sang Freebase; phim được tìm trên Wikidata qua mã Freebase. Cơ sở: item trong danh mục sau làm sạch. ML-100K: 1.617 trong 1.663 item (97,2%) có liên kết Freebase, 99,5% rating thuộc các item đó, 83,1% item có liên kết YAGO. ML-1M: 3.811 trong 3.883 item (98,2%), 99,7% rating, 84,1% item có liên kết YAGO.

**Tỷ lệ lỗi liên kết:** 0/50 sai trong mẫu ngẫu nhiên 50 phim ML-100K (40 xác nhận tự động bằng hai nguồn, 10 kiểm tay); khoảng tin cậy Wilson 95% của tỉ lệ sai là [0,0%; 7,1%]. Mẫu nhỏ nên chưa loại trừ được tỉ lệ sai đến 7,1%. 97,2% (1.617/1.663) là độ phủ, không phải độ đúng.

## 6. KG theo thời gian

**Quy tắc** (quyết định 0004 và 0006). KG tại mốc t chỉ giữ fact đã đúng trước t (so sánh chặt): fact có ngày statement giữ nếu ngày đó trước t; fact trỏ tới thực thể chưa tồn tại tại t (phần tiếp theo, hãng ra đời sau) bị loại; fact không có ngày nào, như phim → thể loại, được giữ theo mặc định. Ngày chỉ có năm đọc là 31/12, chỉ có tháng đọc là ngày cuối tháng. KG của train và val dựng tại t1, của test tại t2. Ba biến thể: **default**, **drop_undated** (bỏ fact không có ngày nào) và **static** (giữ mọi fact như bài báo KGAT; chỉ để đo rò rỉ qua KG, không phải kết quả đúng theo thời gian).

**Các fact có ngày đến đâu.** Cơ sở: fact Wikidata của item train, 37.103 fact (927 item) ở ML-100K và 104.782 fact (2.975 item) ở ML-1M; ngoài ra có 3.654 và 11.277 fact metadata MovieLens (quyết định 0011), mang ngày công chiếu của phim và đi qua cùng các quy tắc. Chỉ 7,6% và 5,8% số fact có ngày của chính statement, và 97,1% và 97,8% trong số đó thuộc `award_received` hoặc `nominated_for`, đúng hai quan hệ có thể kéo thông tin tương lai vào KG; 18,1% (cả hai bộ) không có ngày nào. Ngày chỉ có năm là một giới hạn đã biết: 44,4% trong 6.295 giá trị `point_in_time` (Phụ lục D).

| Biến thể | ML-100K tại t1 | ML-100K tại t2 | ML-1M tại t1 | ML-1M tại t2 |
|---|---|---|---|---|
| default | 40.210 | 40.230 | 114.288 | 114.288 |
| drop_undated | 33.490 | 33.510 | 95.367 | 95.367 |
| static (tham chiếu) | 40.757 | 40.757 | 116.059 | 116.059 |

*Bảng 4. Số triple của KG theo biến thể và mốc (Wikidata và metadata MovieLens, quyết định 0011).*

- **Fact bị lọc tại t1** (static trừ default): 547 fact (1,3%) ở ML-100K và 1.771 fact (1,5%) ở ML-1M; đích chưa tồn tại là 433 và 1.069 fact, ngày sau mốc là 114 và 702 fact (kể cả fact metadata MovieLens của phim mà năm phát hành, đọc là 31/12, rơi sau mốc; quyết định 0011). Biến thể drop_undated bỏ thêm 6.720 và 18.921 fact không có ngày nào.
- **Từ t1 đến t2:** KG default của ML-100K tăng 20 triple, trong đó 19 là fact `award_received` được trao giữa hai mốc; ML-1M không đổi vì không có fact nào mang ngày rơi vào khoảng từ 02/12/2000 đến 29/12/2000. Số thực thể chỉ xuất hiện trong KG t2 là 1 (ML-100K) và 0 (ML-1M); cách xử lý sẽ chọn ở giai đoạn KGAT.

**Fact bị loại tập trung vào phim nổi tiếng.** 10% item phổ biến nhất trong train (93 và 301 item) nắm 15,3% và 14,6% số fact, nhưng chiếm 44,3% và 35,8% số fact giải thưởng và đề cử "tương lai" bị loại tại t1 (97 fact trên 25 phim ở ML-100K; 260 fact trên 74 phim ở ML-1M). Ví dụ *Titanic* và *Toy Story*: Phụ lục D.

![Hình 3. Trái: fact bị loại tại t1 theo quan hệ ở ML-100K, tách theo lý do. Phải: tỷ lệ positive val/test nằm trên các item mất ít nhất một fact.](figures/fig4_fact_bi_loai.png)

**Phần tương tác bị ảnh hưởng.** Một item "bị lọc" là item có số triple khác nhau giữa static và default tại cùng một mốc: 344 item (36,9% item train) ở ML-100K và 945 item (31,4% item train) ở ML-1M. Các item này nhận 53,1% positive val và 48,1% positive test của ML-100K, 41,5% và 43,9% của ML-1M: dù chỉ một phần nhỏ fact bị lọc, phần lớn tương tác dùng để chọn mô hình và chấm điểm nằm trên item mà KG tĩnh chứa thông tin tương lai. Mức ảnh hưởng lên Recall@20 **chưa được đo**; chỉ đo được khi chạy KGAT với hai biến thể.

## 7. Baseline

Baseline chính thức trên ML-1M (val 480 user, test 952 user) và ML-100K (val 66 user, test 51 user). Cấu hình chọn trên val; chỉ điểm test của cấu hình đã chọn được ghi lại. MostPop không ngẫu nhiên nên chạy một lần (không có độ lệch chuẩn); BPR-MF chạy 5 seed (2019–2023), báo cáo trung bình ± độ lệch chuẩn mẫu.

| Mô hình (bộ dữ liệu) | Cấu hình chọn trên val | Recall@20 (test) | NDCG@20 (test) | Số seed |
|---|---|---|---|---|
| MostPop (ML-1M) | toàn bộ train | 0,0902 | 0,1942 | 1 |
| MostPop recent (ML-1M) | cửa sổ 7 ngày | 0,1060 | 0,2034 | 1 |
| BPR-MF (ML-1M) | dim = 64, l2 = 0,0001, lr = 0,001 | 0,1194 ± 0,0031 | 0,2208 ± 0,0069 | 5 |
| MostPop (ML-100K) | toàn bộ train | 0,1555 | 0,1451 | 1 |
| MostPop recent (ML-100K) | cửa sổ 14 ngày | 0,3372 | 0,2870 | 1 |
| BPR-MF (ML-100K) | dim = 32, l2 = 0,0001, lr = 0,001 | 0,1471 ± 0,0295 | 0,1549 ± 0,0170 | 5 |

*Bảng 5. Baseline trên ML-1M và ML-100K (ML-100K chỉ có 51 user test).*

Trên ML-1M, MostPop recent cao hơn MostPop và BPR-MF cao hơn cả hai ở Recall@20 và NDCG@20. Trên ML-100K, MostPop recent cao hơn hẳn (Recall@20 0,3372 so với 0,1555 của MostPop). Kiểm định Wilcoxon theo từng user trên 51 user test (Recall@20, hiệu chỉnh Holm trong cùng bộ dữ liệu và metric; `results/significance.csv`): BPR-MF so với MostPop có p = 0,513, chưa có bằng chứng hai mô hình khác nhau; BPR-MF so với MostPop recent có p = 0,029, BPR-MF thấp hơn; MostPop recent so với MostPop có p = 0,0071. Mẫu chỉ 51 user nên các p này nhạy với nhiễu. Kiểm định cho ML-1M cũng nằm ở file đó, chưa đưa vào báo cáo này; MostPop chỉ có một lần chạy nên chưa nói được về độ ổn định. Cửa sổ 7 ngày của MostPop recent ở ML-1M nằm giữa lưới tìm kiếm {1, 3, 5, 7, 14, 30} ngày, nên biên của lưới không còn là vấn đề (Phụ lục E); ở ML-100K cửa sổ được chọn là 14 ngày, cũng không nằm ở biên của lưới. Sau này mô hình có yếu tố thời gian phải vượt được MostPop recent, không chỉ MostPop.

## 8. Hạn chế và việc tiếp theo

**Hạn chế đã biết**
1. Wikidata là bản chụp ngày 17/10/2026, không phải tri thức có sẵn vào thời điểm của các rating; mọi kết quả dùng KG phải nêu điều này.
2. Ngày chỉ có năm: 44,4% giá trị `point_in_time` và phần lớn các ngày khác kết thúc bằng `-01-01`; đọc là 31/12 là thận trọng nhưng có thể bỏ nhầm fact (quyết định 0006 A).
3. Fact không có ngày nào (18,1%) được giữ theo mặc định; biến thể drop_undated đo ảnh hưởng của giả định này.
4. ML-100K chỉ còn 66 user val và 51 user test; val của ML-1M chỉ dài 27 ngày.
5. Timestamp được hiểu là lúc user ghi rating (README của GroupLens chỉ nói số giây kể từ epoch), nên các suy luận về Δt mới là giả thuyết.
6. Độ chính xác liên kết mới đo trên mẫu 50 phim (0/50 sai, khoảng Wilson 95% của tỉ lệ sai [0,0%; 7,1%]); trong 50 phim đó 40 phim chỉ được máy xác nhận bằng hai nguồn, 10 phim được kiểm tay.
7. Chưa đo ảnh hưởng của rò rỉ qua KG lên Recall@20, vì KGAT chưa được chạy.

**Câu hỏi xin ý kiến cô**
1. Với ngày chỉ có năm, giữ quy tắc đọc là 31/12, hay tải lại Wikidata kèm độ chính xác cho mọi cột ngày?
2. Khi kết luận, nhóm dựa chủ yếu vào ML-1M và dùng ML-100K như kiểm tra phụ; cô thấy cách này có hợp lý không?
3. Cô có chấp nhận biến thể static làm tham chiếu để đo rò rỉ qua KG, không đưa vào kết quả chính?
4. Với fact không có ngày nào, cô muốn giữ mặc định (default) hay bỏ (drop_undated) làm cấu hình chính?

**Việc tiếp theo.** Đưa kiểm định Wilcoxon giữa các baseline (`results/significance.csv`) và phân tích độ nhạy theo cách chia (`docs/sensitivity-val.md`) vào báo cáo; mở rộng mẫu kiểm tay liên kết nếu cần khoảng tin cậy hẹp hơn; sau khi cô duyệt quy trình, chạy KGAT và T-KGAT với các biến thể KG ở trên.

## Phụ lục

### A. Ba cơ sở đếm item

| Cơ sở đếm item | ML-100K | ML-1M |
|---|---|---|
| Item được rating (dữ liệu gốc) | 1.682 | 3.706 |
| Item trong danh mục liên kết, sau làm sạch (kể cả item chưa có rating) | 1.663 | 3.883 |
| Item trong train, sau 5-core | 932 | 3.009 |

*Bảng A1. Ba cơ sở đếm item.*

### B. Số rating theo tuần

![Hình A1. Số rating mỗi tuần, tuần tính theo UTC; vùng xám đậm là cửa sổ val (từ t1 đến trước t2), vùng xám nhạt là test (từ t2). Panel ML-1M dùng thang log.](figures/fig1_rating_theo_tuan.png)

Số rating mỗi tuần dao động mạnh với những tuần tăng đột biến; ở ML-1M, trục log cho thấy lượng rating giảm dần sau các tuần đó. Báo cáo chưa kiểm tra nguyên nhân của các đột biến.

### C. Quy tắc ánh xạ ML-100K → ML-1M và một lỗi đã biết của KB4Rec

Theo thứ tự ưu tiên: khớp tiêu đề và năm cho 1.628 item, tiêu đề phụ (a.k.a.) và năm cho 13, chứa tiêu đề và năm cho 4, tiêu đề và năm lệch một năm cho 2; 16 item không khớp được. Đã biết một lỗi của KB4Rec: phim *Clubland* bị liên kết nhầm sang Cloudland Canyon State Park.

### D. Chi tiết KG theo thời gian

**Độ chính xác của ngày.** Chỉ `point_in_time` có độ chính xác kèm theo: trong 6.295 giá trị, 44,4% chỉ biết năm. Các cột ngày khác mất độ chính xác khi tải về, và Wikidata ghi ngày chỉ có năm thành `YYYY-01-01`: 58,3% ngày phát hành phim, 45,1% ngày thành lập của thực thể đích và 4,8% ngày sinh kết thúc bằng `-01-01`. Quyết định 0006 đọc các ngày này là năm-only (31/12), nên một fact thật sự xảy ra ngày 1/1 sẽ bị đặt muộn hơn: thận trọng, không làm lọt fact tương lai, nhưng có thể bỏ nhầm. Ngày phát hành của chính phim không dùng để loại fact của phim đó, vì mọi item của KG đã được rating trước t1.

**Số fact "tương lai" theo mốc.** Số fact `award_received` "tương lai" giảm từ 52 xuống 33 giữa t1 và t2 ở ML-100K.

**Hai ví dụ.** *Titanic (1997)* có 136 fact; KG tại t1 thiếu 28 fact trong số đó, gồm ba nhóm:

- 11 giải thưởng, đều là giải Oscar, được trao ngày 23/03/1998, sau t1 (07/03/1998) nhưng trước t2 (31/03/1998): chúng không có trong KG của train/val nhưng có trong KG của test.
- 15 giải hoặc đề cử bị loại ở cả hai mốc, trong đó 13 giải chỉ ghi năm.
- 2 fact khác, không phải giải thưởng: `distributed_by → Disney+` (ngày tồn tại của đích: 12/11/2019) và `cast_member → Lewis Abernathy` (ngày tồn tại của đích: 31/12/2000). Cả hai bị loại vì thực thể đích chưa tồn tại tại hai mốc. Ngày của Lewis Abernathy là ngày sinh mà Wikidata ghi là `2000-01-01`, và quy tắc năm-only đọc thành 31/12; với một người tham gia phim năm 1997 thì ngày này đáng ngờ, nhưng báo cáo chưa kiểm tra lại nguồn.

*Toy Story (1995)* mất 2 fact tại t1, trong đó có liên kết `followed_by → Toy Story 2`, phim ra mắt ngày 24/11/1999, sau toàn bộ dữ liệu ML-100K: một KG tĩnh sẽ cho mô hình biết trước phim này có phần tiếp theo.

### E. Cửa sổ của MostPop recent

Lưới ban đầu {7, 14, 30} ngày chọn 7 ngày, tức giá trị nhỏ nhất của lưới, và val giảm đều khi cửa sổ dài ra, nên giá trị tốt nhất có thể còn nhỏ hơn. Quyết định 0007 mở rộng lưới thành {1, 3, 5, 7, 14, 30} ngày. Recall@20 trên val theo cửa sổ 1, 3, 5, 7, 14 và 30 ngày là 0,0748; 0,1130; 0,1289; 0,1336; 0,1082 và 0,0986. 7 ngày vẫn được chọn và nay nằm giữa lưới; điểm test không đổi.

### F. Ghi chú về registry

Khóa của một lần chạy trong `results/registry.csv` nay gồm cả `split_sha256`. Registry có 78 dòng gắn nhãn `-dirty`, thuộc hai nhóm. 19 dòng nhãn `08f5e63-dirty` thật ra là commit sạch `08f5e63` bị gắn nhãn sai do một lỗi đã sửa. 59 dòng nhãn `0231c86-dirty` được chạy khi một file code phân tích bị sửa trong lúc chạy nền; code huấn luyện đã được commit và không đổi. Các dòng cũ không bị sửa.
