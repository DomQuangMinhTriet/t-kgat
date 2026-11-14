# 13 — Hướng dẫn present kết quả Last-FM (`lastfm-u25` và `lastfm-u25-latest`) với cô, 14/11/2026 lúc 13:00

> **Cập nhật 14/11/2026 (trước buổi họp): Stage 2 có lỗi thiết kế chuẩn hóa.** `dt_norm` của cạnh KG chia cho cửa sổ của **tương tác** train (`dt_den`, quyết định 0009 mục 3) rồi bị chặn ở 1 (mục 10), nên fact cũ hơn cửa sổ đều có `dt_norm` = 1. `lastfm-u25` bị một phần (55,7% triple bị chặn ở 1), `lastfm-u25-latest` gần như toàn bộ (80,1%, 0,0% trong cửa sổ). Code làm đúng quyết định 0009; lỗi nằm ở thiết kế, không phải ở dữ liệu. **Hệ quả:** kết quả Stage 2 của cả hai bản không dùng để kết luận "fact cũ hay mới"; **Stage 1 không bị ảnh hưởng** (cạnh tương tác luôn nằm trong cửa sổ). Đề xuất: quyết định mới chuẩn hóa Δt của fact theo ngày của fact, báo là thêm sau khi đã thấy test, rồi chạy lại Stage 2. Các câu cũ bên dưới đã sửa theo ghi chú này. Notebook trình bày: bản rút gọn `notebooks/00_present_lastfm_short.ipynb` (bản đầy đủ `00_present_lastfm.ipynb` để tra khi cô hỏi sâu). PR của bản `latest` là **#44** (thay #43 đã đóng).

Kịch bản trình bày theo bố cục đề xuất trong `12-hoc-va-bao-cao-lastfm.md` (khoảng 24 phút, chưa tính phần cô hỏi). Mạch: **`lastfm-u25` → ba vấn đề của `lastfm-u25` → bản mới `lastfm-u25-latest` (Phôn) → kế hoạch và ba câu hỏi**. Mỗi đoạn có: thời lượng, người nói, phần của notebook cần mở, lời nói mẫu, câu chuyển sang đoạn sau. Lời nói mẫu là gợi ý: hiểu ý và nói bằng lời của mình, không đọc. Nội dung để học ở `12-hoc-va-bao-cao-lastfm.md`; lý thuyết ở `11-tong-hop-ly-thuyet-lastfm.md`.

Ký hiệu: **[Mở]** = phần (mục) của notebook `t-kgat/notebooks/00_present_lastfm.ipynb`; **[Chuyển]** = câu chuyển; *[ghi chú]* = không đọc.

**Lưu ý trước khi dùng:** (1) thời lượng 24 phút và cách chia người nói giữa Triết và Phôn là **đề xuất**, chưa được cô hay Phôn xác nhận; (2) present bằng notebook `00_present_lastfm.ipynb`, không làm slide; notebook giữ output đã chạy (như bản `lastfm-u25`), và kết quả KGAT/T-KGAT của **cả hai bản** còn sơ bộ; (3) phần 1 đến 3 chỉ `lastfm-u25`, phần 4 chỉ `lastfm-u25-latest`; **không trộn số**; (4) số của `lastfm-u25-latest` chưa có trong `results/registry.csv`, quyết định 0015 là nháp của Phôn, code của Phôn chưa vào repo: phải nói rõ ở phần 4.

---

## A. Trước buổi present

### Bản đồ kịch bản và notebook
Notebook có 7 mục (66 ô, trong đó 31 ô code), mỗi mục ghi người trình bày (đề xuất). Mọi số trong notebook được in từ file kết quả của repo, không gõ tay.

| Mục notebook | Nội dung | Khớp đoạn kịch bản |
|---|---|---|
| 1. Vì sao Last-FM; protocol; ba RQ | Context của 0012 (nguyên văn), bảng 0012–0014, RQ | Phần 0 |
| 2. Dữ liệu `lastfm-u25` | Nguồn; bảng chia; cold item; Δt của cạnh train; cạnh theo tuần quanh 01/04/2014; KG, thang Δt Stage 2; `check_split` và hai thử phá trong bộ nhớ | 1a, 1b, 1c, 1d |
| 3. Huấn luyện | Quy trình; bảng BPR-MF; lưới và điểm chọn trên val; learning curve; giờ chạy theo máy; cổng đo và kết quả thực tế của 0014 | 2a, 2b |
| 4. Kết quả `lastfm-u25` | Bảng test; kiểm định cặp; baseline; bốn nhận xét | 3a, 3b |
| 5. Minh bạch và hạn chế | Danh sách phải báo rõ; tám hạn chế | 3c |
| 6. Ba vấn đề và bản `latest` | 6.0 nguồn số; 6.1 ba vấn đề A, B, C; 6.2 bảng so sánh hai bản; 6.3 Stage 1; 6.4 Stage 2; 6.5 kết quả, kiểm định, hình, đặt cạnh u25, nhận xét; 6.6 hạn chế | 4a đến 4f |
| 7. Kế hoạch và ba câu hỏi | Ba phương án dữ liệu A, B, C; Stage 2 và phân tích bổ sung; kết quả sơ bộ | Phần 5 |

### Tối trước ngày báo cáo (13/11, sau rehearsal)
- [ ] Mở notebook, **Kernel → Restart & Run All** từ môi trường của `t-kgat` (`.venv`): 31 ô code chạy hết, không lỗi (đã chạy ngày 13/11, 0 lỗi). Đối chiếu vài số với Phụ lục A của tài liệu 12.
- [ ] Ô 6.0 phải in: registry của repo có **0** dòng `lastfm-u25-latest`, file `0015-*.md` trong repo là **0**, registry gộp của Phôn 323 dòng / 132 dòng riêng biệt / commit `873ca7f` / 0 dòng `-dirty`. Nếu khác, dừng và hỏi.
- [ ] **Hai con số 69,7% và 64,1%** (nhật ký 12/11: "cặp val/test đã được nghe lần đầu trước khi tập bắt đầu") **không tái lập được: không nói, không đưa lên màn hình.** Lưu ý: `docs/data-report-lastfm-latest-numbers.md` có khóa `val_kept_share` = 69,7% (phần positive val còn lại sau khi lọc cold user và cold item của latest): trùng giá trị nhưng **khác ý nghĩa**. Notebook không hiển thị chỉ tiêu này; nếu cô hỏi về nó, nói rõ nó là gì.
- [ ] Số có chú thích **(tính thêm)** trong tài liệu 12 (Δt của cạnh train, giờ chạy, epoch tốt nhất): notebook tự tính lại từ `train.csv` và registry, nên trình bày được; nói rõ nguồn nếu cô hỏi.
- [ ] Xác nhận với Phôn: ai nói 6.1 (đề xuất: Phôn), thời lượng từng mục, và Phôn nắm được vì sao 0015 và code chưa vào repo.
- [ ] Mở sẵn: notebook đã chạy (đừng chạy lại giữa buổi), `docs/rq3-table.md`, `docs/rq3-table-lastfm-latest.md`, `docs/data-report-lastfm.md`, `docs/lastfm-latest-provenance.md`, tài liệu 12 (Phụ lục A, thẻ số).
- [ ] Xuất notebook đã chạy ra HTML hoặc PDF làm bản dự phòng (File → Save and Export As), chép vào USB hoặc Drive.
- [ ] Tập dượt 2 lượt, lượt 2 đổi vai; bấm giờ từng phần.

### Sáng ngày báo cáo (14/11)
- [ ] Máy chiếu, cáp, sạc; phóng chữ.
- [ ] Tắt thông báo, đóng ứng dụng không cần.
- [ ] Giấy ghi góp ý của cô; người không nói thì ghi, đặc biệt câu trả lời cho ba câu hỏi ở phần 5.

---

## B. Kịch bản

### Phần 0 — Mở đầu (1,5 phút) · Phôn
**[Mở]** mục 1.

**Phôn:** "Thưa cô, hôm nay nhóm báo cáo kết quả trên bộ dữ liệu Last-FM, theo hướng cô gợi ý ngày 01/11: lặp lại các thí nghiệm của MovieLens trên một dữ liệu khác. Nhóm làm năm phần: dữ liệu và kết quả của `lastfm-u25`, ba vấn đề nhóm tự thấy ở bản đó, một bản dữ liệu mới của bạn Phôn, và cuối cùng là kế hoạch kèm ba câu hỏi xin ý kiến cô."

"Một điểm nhóm xin nói ngay: quyết định chuyển sang Last-FM được đưa ra sau khi nhóm đã xem điểm test của MovieLens. Nhóm làm theo hướng cô chỉ, không chọn dữ liệu để cải thiện điểm số, và giữ nguyên quy trình: cùng mốc chia theo thời gian, cùng k-core chỉ trên train, cùng kiểm tra rò rỉ, cùng lưới tham số, cùng 5 seed."

**[Chuyển]** "Phần dữ liệu `lastfm-u25` do bạn Triết trình bày."

---

### Phần 1 — Dữ liệu `lastfm-u25` (5,5 phút) · Triết

**1a. Nguồn và cách chia (2 phút).** **[Mở]** mục 2, bảng chia.

**Triết:** "Last-FM là dữ liệu nghe nhạc, không có điểm số; mỗi cặp người dùng–bài hát là một tương tác dương. Nhóm ghép ba nguồn: tập cặp và đồ thị tri thức mà tác giả KGAT công bố, timestamp lấy từ LFM-1b bằng lần nghe sớm nhất của từng cặp, và ngày của fact lấy từ bản chụp Freebase năm 2015."

"Chia dữ liệu như MovieLens: hai mốc là phân vị 80% và 90% của mọi timestamp, 5-core chỉ trên train, val và test chỉ giữ người dùng và bài hát đã có trong train. Do máy chỉ đủ chạy một phần, nhóm dùng 25% người dùng, tức 5.892 trong 23.566. Kết quả: train 236.053 cặp, val 23.708 cặp của 2.319 người dùng, test 22.419 cặp của 1.608 người dùng. Con số 1.608 là điểm khác lớn so với MovieLens, nơi test chỉ có 51 và 95 người dùng, nên kiểm định ở Last-FM có sức mạnh thật."

**1b. Cái giá và đặc điểm thời gian (1,5 phút).** **[Mở]** mục 2, cold item và Δt của cạnh train.

**Triết:** "Cái giá của việc lấy 25% người dùng là 28,8% số cặp val và 28,1% số cặp test bị loại vì bài hát chưa có trong train; trên bộ đầy đủ chỉ khoảng 3 và 5 phần trăm. Vì vậy nhóm chỉ nói kết quả cho mẫu 25% này."

"Về thời gian, khác MovieLens nơi rating được nhập theo phiên, thời gian của một người dùng ở `lastfm-u25` trải rất rộng: trung vị độ lệch chuẩn khoảng 250 ngày, chỉ 0,5% người dùng có mọi tương tác trong một giờ, so với 53% ở ML-100K."
*[ghi chú: các số Δt_norm là notebook tự tính từ `train.csv`; nói rõ nếu cô hỏi nguồn.]*

**1c. KG và ngày của fact (1 phút).** **[Mở]** mục 2, phần KG.

**Triết:** "KG của `lastfm-u25` có hơn 152 nghìn triple, 44 nghìn thực thể, 9 quan hệ. Ngày của fact lấy qua ngày tồn tại của đích trong Freebase 2015; 19,8% triple không có ngày. Bộ lọc theo thời gian gần như không cắt gì, chỉ 3 triple, và KG tại hai mốc là một, nên Last-FM không đo được hiện tượng fact tương lai rơi giữa hai mốc; câu hỏi đó vẫn trả lời bằng MovieLens. Thang Δt của Stage 2 trên `lastfm-u25` có độ phân giải một phần: 24,4% triple nằm trong cửa sổ, so với dưới 1% ở MovieLens; phần còn lại bị chặn ở 1, lát nữa nhóm sẽ nói nguyên nhân."

**1d. Hạn chế của tập cặp (1 phút).** **[Mở]** mục 2, biểu đồ cạnh theo tuần và đoạn Pair selection.

**Triết:** "Có một hạn chế nhóm phải nói rõ vì các kiểm tra tự động không thấy được: tập cặp mà KGAT công bố chỉ chứa các cặp có ít nhất một lần nghe từ 01/04/2014. Tức tập cặp được chọn bằng thông tin sau mốc, và có thể thiên về các cặp được nghe lâu dài. Dấu vết thấy ngay trong biểu đồ số cạnh theo tuần. Nhóm chưa đo ảnh hưởng lên kết quả; bạn Phôn sẽ trình bày một bản dữ liệu mới liên quan đến điểm này."

**[Chuyển]** "Phần huấn luyện do bạn Phôn và mình trình bày."

---

### Phần 2 — Huấn luyện mô hình (4 phút) · Phôn, rồi Triết

**2a. Baseline và quy trình chung (1,5 phút) · Phôn.** **[Mở]** mục 3, bảng BPR-MF và hình `fig5_bprmf_curve_lastfm-u25.png`.

**Phôn:** "Nhóm có ba baseline: MostPop, MostPop gần đây chỉ đếm vài ngày cuối train, và BPR-MF. Mọi mô hình theo cùng quy trình: mỗi điểm trong lưới chạy 5 seed, đánh giá Recall@20 trên val mỗi 5 epoch, dừng sớm khi val không tăng, chọn điểm lưới có val trung bình cao nhất, rồi chấm test đúng một lần. Với BPR-MF điểm được chọn là chiều 64, l2 bằng 0,001; MostPop gần đây chọn cửa sổ 14 ngày. Mọi lần chạy ghi vào một registry chỉ thêm dòng, có mã commit; với `lastfm-u25` có 139 dòng cùng một commit, không dòng nào chạy khi code chưa sạch."

**2b. KGAT, T-KGAT và tính toán (2,5 phút) · Triết.** **[Mở]** mục 3, lưới, learning curve, giờ chạy và 0014.

**Triết:** "KGAT gộp tương tác và KG thành một đồ thị, dùng attention để quyết định hàng xóm nào quan trọng, học bằng hai loss: BPR và TransR. T-KGAT chỉ khác một chỗ: trước khi chuẩn hóa attention, trừ γ nhân Δt cho cạnh người dùng–bài hát; γ bằng 0 là đúng KGAT; γ chọn trên val trong 1, 3, 10. Để biết cải thiện có nhờ thời gian không, nhóm chạy bản hoán vị timestamp. Stage 2 đặt Δt lên fact của KG, cũng có bản hoán vị ngày fact."

"Về tính toán: KGAT trên 25% người dùng mất khoảng 20 giây mỗi epoch trên laptop. Stage 2 chạy trên Kaggle theo gợi ý của cô ngày 02/11. Có ba chi tiết nhóm xin báo ở đây vì chưa báo cô: GPU Kaggle là Tesla T4; cổng đo thời gian ban đầu viết là 12 giây mỗi epoch, đo được 13,81 giây, và nhóm nâng cổng lên 14 giây sau khi thấy số đo, tức là thay đổi có chủ đích chứ không phải lần qua cổng đã đăng ký trước; và ước lượng giờ chạy là 9,9 giờ nhưng thực tế khoảng 12,3 giờ. Hệ quả là so Stage 2 với KGAT trên `lastfm-u25` là so giữa hai máy."

*[Nếu thiếu giờ: bỏ giải thích Stage 2 và hoán vị chi tiết, giữ "T-KGAT = KGAT trừ γ·Δt", ý hoán vị, và ba chi tiết Kaggle.]*

**[Chuyển]** "Phần tiếp theo là kết quả của `lastfm-u25`."

---

### Phần 3 — Kết quả `lastfm-u25`, minh bạch và hạn chế (4,5 phút) · Phôn, rồi Triết

**3a. Baseline (1 phút) · Phôn.** **[Mở]** mục 4, bảng test (hàng baseline) và bảng baseline.

**Phôn:** "Tất cả là Recall@20 trên 1.608 người dùng test. BPR-MF đạt 0,0520, gần gấp đôi MostPop với 0,0238, và thắng có ý nghĩa mạnh sau kiểm định Wilcoxon cặp và hiệu chỉnh Holm. Ngược với ML-100K, MostPop gần đây với cửa sổ 14 ngày đạt 0,0211, thấp hơn MostPop thường. Giả thuyết của nhóm, chưa kiểm: cửa sổ neo ở cuối train, còn test bắt đầu khoảng 39 ngày sau và kéo dài khoảng 86 ngày."

**3b. KGAT và T-KGAT (2,5 phút) · Triết.** **[Mở]** mục 4, bảng kiểm định cặp và nhận xét 2 đến 4.

**Triết:** "Đây là kết quả sơ bộ. KGAT đạt 0,0534, KGAT không KG 0,0518; hiệu 0,0016, p Holm 0,80 cho Recall và 0,51 cho NDCG. Chưa có bằng chứng KG giúp trên `lastfm-u25`. KGAT chỉ nhỉnh hơn BPR-MF 0,0013, nhỏ hơn độ lệch chuẩn qua seed của BPR-MF, và cặp này chưa nằm trong họ kiểm định."

"Về thời gian: T-KGAT Stage 1 đạt 0,0529, KGAT 0,0534, bản hoán vị 0,0534. T-KGAT không khác KGAT, p Holm 0,42; không khác bản hoán vị, p Holm 0,80. Δt trên `lastfm-u25` có biến thiên thật và test có 1.608 người dùng, nên không thể giải thích âm tính này bằng việc thời gian gần như bằng không hay mẫu quá nhỏ. Nhưng p lớn không chứng minh bằng nhau. Stage 2: 0,0513, thấp hơn KGAT và bản hoán vị, không có ý nghĩa, p Holm 0,28; kèm lưu ý so sánh với KGAT là giữa hai máy. Và một phát hiện nhóm xin báo cô: Stage 2 có lỗi thiết kế chuẩn hóa. `dt_norm` của fact chia cho cửa sổ của tương tác nên phần lớn fact bị chặn ở 1, ở đây 55,7%. Vì vậy kết quả Stage 2 không dùng để kết luận về thời gian của KG. Stage 1 không bị ảnh hưởng."

**3c. Minh bạch và hạn chế (1 phút) · Triết.** **[Mở]** mục 5.

**Triết:** "Nhóm tóm các điểm cần báo rõ. Một: quyết định chuyển sang Last-FM đưa ra sau khi đã xem test của MovieLens. Hai: KGAT và T-KGAT là sơ bộ, chạy trước khi cô duyệt giao thức chia theo mốc, chưa có tag đóng băng, chưa tái lập độc lập. Ba: tập cặp có thiên lệch chọn mẫu chưa đo; chỉ 25% người dùng; Freebase là bản chụp 2015; KG tại hai mốc giống nhau; một số điểm được chọn nằm ở mép lưới. Bốn: các phân tích bổ sung nhóm đã đề xuất chưa chạy; mỗi cái cần quyết định riêng và sẽ được ghi là thêm sau khi đã thấy test."

**[Chuyển]** "Từ hạn chế về tập cặp, bạn Phôn sẽ nói về ba vấn đề của `lastfm-u25` và bản dữ liệu mới."

---

### Phần 4 — Ba vấn đề của `lastfm-u25` và bản `lastfm-u25-latest` (6 phút) · Phôn
**[Mở]** mục 6.

**4a. Ba vấn đề (1 phút).** **[Mở]** 6.1.

**Phôn:** "Nhóm gom hạn chế của `lastfm-u25` thành ba vấn đề. Một, tập cặp chỉ chứa cặp còn được nghe từ 01/04/2014, tức được chọn bằng thông tin sau mốc. Hai, dấu vết của nó: số cạnh train theo tuần nhảy từ 2.494 lên 12.416, gấp 5 lần, và 21,7% cạnh train có timestamp từ 01/04/2014 dù cửa sổ train dài 3.361 ngày. Ba, `check_split` báo 0 vấn đề, tức các kiểm tra tự động không thấy được hai điều trên."

**4b. Bản latest là gì (1 phút).** **[Mở]** 6.0 và 6.2.

**Phôn:** "Bản `lastfm-u25-latest` chỉ đổi một thứ: timestamp của cặp là lần nghe muộn nhất thay vì sớm nhất. Cùng 370.979 cặp, cùng người dùng, nhưng t1 và t2 dời sang 23/06 và 19/07/2014; cửa sổ train chỉ 83 ngày, val 26 ngày, test 45 ngày; test có 1.341 người dùng. Em xin nói rõ tình trạng: quyết định 0015 là bản nháp của em, chưa được duyệt; code xuất dữ liệu và các dòng registry nằm ngoài repo, với commit riêng; PR 43 chỉ chứa bảng tóm tắt và nguồn gốc. Nên kết quả của bản này là sơ bộ và chưa tái lập được chỉ từ repo."

**4c. Stage 1 (1 phút).** **[Mở]** 6.3.

**Phôn:** "Bước nhảy biến mất: train bắt đầu đúng 01/04/2014. Nhưng thiên lệch không mất mà đổi hình thức. Với timestamp là lần nghe muộn nhất, một cặp nằm ở train khi và chỉ khi người dùng không nghe cặp đó từ t1 trở đi; tức train chỉ chứa các cặp đã dừng. Đây là suy luận từ định nghĩa, ảnh hưởng lên kết quả chưa đo, và `check_split` vẫn báo 0 vấn đề. Về Δt: cửa sổ chỉ 83 ngày nên một đơn vị Δt chuẩn hóa là 83 ngày; trung vị 0,73; 16,2% cạnh nằm trong 30 ngày cuối; độ lệch chuẩn thời gian theo người dùng vẫn 9,2 ngày."

**4d. Stage 2 (1 phút).** **[Mở]** 6.4.

**Phôn:** "Stage 2 trên bản này không có độ phân giải: 0,0% triple nằm trong cửa sổ, 80,1% bị chặn ở 1, 19,9% không có ngày. Lý do: fact có ngày mới nhất là 31/12/2013, sớm hơn ngày đầu của cửa sổ train 91 ngày. Nên mọi fact có ngày có cùng Δt chuẩn hóa bằng 1; Stage 2 chỉ còn phân biệt có ngày hay không có ngày, và bản hoán vị ngày fact chỉ xáo xem triple nào là không có ngày. Nguyên nhân là cách chuẩn hóa: cửa sổ lấy theo tương tác rồi áp cho fact, như Triết vừa nói; bản `lastfm-u25` cũng bị, một phần."

**4e. Kết quả (1,5 phút).** **[Mở]** 6.5, bảng, kiểm định, hình, bảng đặt cạnh `lastfm-u25`, nhận xét.

**Phôn:** "Recall@20 trên 1.341 người dùng test, 5 seed; mọi mô hình học sâu chạy trên cùng một loại máy, Tesla T4. KGAT 0,0673, KGAT không KG 0,0633: ở Recall hiệu có ý nghĩa sau Holm, p 0,028; ở NDCG thì không, p 0,225. Khác `lastfm-u25`, nơi cả hai không có ý nghĩa; nhóm coi là sơ bộ. T-KGAT 0,0676 không khác KGAT, p Holm 0,733. T-KGAT thật hơn bản hoán vị có ý nghĩa, p 0,033 và 0,0046, nhưng bản hoán vị thấp hơn KGAT 0,0021, nên điều đó không chứng minh thời gian giúp. Stage 2: không so sánh nào có ý nghĩa và, như vừa nói, không có độ phân giải. Nhóm không so điểm tuyệt đối với `lastfm-u25` vì test khác nhau."

**4f. Hạn chế và chuyển (0,5 phút).** **[Mở]** 6.6.

**Phôn:** "Tóm lại về bản này: không loại được vấn đề thứ nhất và thiên lệch đổi hình thức; phép chia rất lệch về độ dài; Stage 2 không đo được; chưa tái lập từ repo; MostPop gần đây chưa chạy."

**[Chuyển]** "Từ đó nhóm có ba câu hỏi xin ý kiến cô."

---

### Phần 5 — Kế hoạch và câu hỏi (2,5 phút) · Cả hai
**[Mở]** mục 7.

**Phôn:** "Kế hoạch tiếp theo: chốt bản dữ liệu chính thức; nếu cô chọn bản của em, đưa code xuất dữ liệu và quyết định 0015 vào repo bằng một PR riêng để duyệt, rồi thêm dòng registry theo một quyết định riêng; hoàn tất tái lập độc lập và đóng băng kết quả; chọn một số phân tích bổ sung theo ý cô; rồi viết khóa luận."

**Triết:** "Nhóm xin ý kiến cô ba câu. Một: bản dữ liệu chính thức. Phương án A là giữ `lastfm-u25`, đã chạy đủ, giữ hai vấn đề đầu kèm lời báo. Phương án B là dùng `lastfm-u25-latest`: hết bước nhảy, nhưng tập cặp vẫn thiên lệch theo hình thức khác, Stage 2 không có độ phân giải, và cần duyệt quyết định 0015 cùng đưa code vào repo. Phương án C là dựng lại tập cặp từ sự kiện thô của LFM-1b, không giới hạn bởi cửa sổ tháng 4/2014, xử lý gốc của vấn đề nhưng cần quyết định mới và thêm thời gian tính toán. Hai: nhóm có nên sửa cách chuẩn hóa Δt của Stage 2 theo ngày của fact, bằng một quyết định mới và báo là thêm sau khi đã thấy test, rồi chạy lại Stage 2, trên bản nào; và trong các phân tích bổ sung đã đề xuất cô ưu tiên cái nào. Ba: giữ kết quả của bản được chọn làm chính thức sau khi tái lập độc lập và gắn tag, hay cô muốn chỉnh giao thức rồi chạy lại."

**Cả hai (chốt):** "Tóm lại: quy trình đã sạch rò rỉ và kiểm tra được; trên `lastfm-u25` chưa có bằng chứng KG hay thời gian giúp; nhóm tự thấy ba vấn đề của tập dữ liệu, thử một bản mới nhưng nó chỉ đổi hình thức thiên lệch; mọi kết quả là sơ bộ và được báo kèm đủ hạn chế. Nhóm cảm ơn cô và xin nghe góp ý."

---

## C. Khi bị hỏi giữa chừng

| Tình huống | Cách xử lý |
|---|---|
| Cô hỏi phần của người kia | Người đang nói trả lời nếu chắc; nếu không thì "Bạn X phụ trách phần này, xin phép bạn trả lời" |
| Không biết câu trả lời | "Em chưa kiểm tra điều này; nhóm sẽ kiểm và báo lại cô." Ghi vào giấy. Không đoán số |
| Cô hỏi một con số | Mở thẻ số (tài liệu 12, Phụ lục A) hoặc `docs/rq3-table.md` / `docs/rq3-table-lastfm-latest.md`; nói nguồn |
| Cô hỏi nguồn của số "tính thêm" | "Notebook tính từ file train của `lastfm-u25`; một phần đã nằm trong báo cáo dữ liệu, phần còn lại sẽ đưa vào script nguồn" |
| Cô nghi rò rỉ | Nêu 6 kiểm tra của `check_split`, `check_kg`, test chấm một lần, registry có commit; **nói thêm** thiên lệch chọn mẫu của tập cặp là thứ các kiểm tra này không bắt được, và bản `latest` chỉ đổi hình thức của nó |
| Cô hỏi vì sao Recall của `latest` cao hơn | "Không so điểm tuyệt đối: test khác (1.341 so với 1.608 người dùng, 45 so với 86 ngày, cold item khác) và phép chia khác. Chưa tách được nguyên nhân; nhóm không dùng nó làm bằng chứng" |
| Cô hỏi số của `latest` có trong registry không | "Chưa: registry chỉ thêm dòng và gắn commit của repo; số của bạn Phôn ở commit `873ca7f` của lịch sử riêng; thêm dòng cần quyết định riêng. PR 43 có bảng tóm tắt và nguồn gốc ở `docs/lastfm-latest-provenance.md`" |
| Cô hỏi 0015 | "Là bản nháp của bạn Phôn, chưa accepted, chưa vào repo" |
| Cô hỏi 69,7% hoặc 64,1% | "Nhóm đã đo một lần nhưng chưa tái lập được nên không dùng." Không nêu thêm số. (Nếu cô hỏi về phần val còn lại của `latest`: đó là phần positive sau khi lọc cold user và cold item) |
| Cô hỏi vì sao chọn lưới/γ ở mép | Lưới giữ từ MovieLens để đồng bộ giao thức; chênh lệch val giữa các γ cùng cỡ nhiễu; chưa mở rộng; nằm trong danh sách hạn chế |
| Cô hỏi vì sao Stage 2 của `lastfm-u25` trên máy khác | Laptop quá chậm; cổng 14 giây nâng từ 12 sau khi thấy số đo; số ghi kèm cột `device`; so Stage 2 với KGAT là hai máy. Trên `latest` mọi mô hình học sâu cùng Tesla T4 |
| Cô cho rằng kết quả âm là thất bại | "Biên bản 07/08 ghi kết quả âm hợp lệ nếu quy trình đúng; nhóm có đối chứng hoán vị và nói rõ khi thang Δt không có độ phân giải" |
| Cô hỏi có nên tin số 0,05 hay 0,06 so với paper | "Không so trực tiếp: nhóm chia theo thời gian, dùng 5-core, 25% người dùng" |
| Hết giờ | Bỏ chi tiết KGAT ở 2b, đoạn Δt của 1b, và 4c (giữ câu "thiên lệch đổi hình thức"); giữ nhận xét, hạn chế và ba câu hỏi |
| Máy chiếu lỗi | Mở bản PDF; nếu vẫn lỗi, dùng `docs/rq3-table.md`, `docs/rq3-table-lastfm-latest.md` và các hình trong repo |

## D. Bảy câu cô gần như chắc chắn hỏi (đáp án trong tài liệu 12)
1. Sao chắc không rò rỉ, và thiên lệch của tập cặp là gì? (12, phần 1.6 và 4.1)
2. Sao chỉ lấy 25% người dùng, và mất gì? (12, phần 0 và 1.3)
3. Thời gian có thật sự không giúp, hay do dữ liệu? (12, nhận xét 3 của phần 3.2, phần 1.4)
4. KG có giúp không, và vì sao hai bản cho hai kết luận khác nhau? (12, nhận xét 2 và 4.6)
5. Bản `latest` khác gì, có hết thiên lệch chưa? (12, phần 4.2)
6. Sao Stage 2 trên `latest` không đo được? (12, phần 4.5)
7. Sao Stage 2 chạy trên Kaggle và cổng 14 giây là gì? (12, phần 2.4)

## E. Nói và không nói

| Nên nói | Tránh nói |
|---|---|
| "Chưa có bằng chứng KG hay thời gian giúp trên `lastfm-u25`" | "KG/thời gian không giúp gì" (p lớn không chứng minh bằng nhau) |
| "Trên `latest`, KGAT hơn không KG ở Recall@20 (p Holm 0,028), NDCG@20 thì không; sơ bộ" | "KG giúp trên Last-FM" |
| "T-KGAT thật hơn hoán vị, nhưng hoán vị thấp hơn KGAT nên chưa chứng minh thời gian giúp" | "Thời gian giúp (p Holm 0,033)" |
| "Latest hết bước nhảy nhưng không loại được thiên lệch; hình thức đổi" | "Latest sửa được thiên lệch" |
| "Stage 2 trên `latest` không có độ phân giải: 0,0% triple trong cửa sổ" | Rút kết luận về fact cũ hay mới từ Stage 2 trên `latest` |
| "Kết quả sơ bộ, chưa tái lập độc lập; số của `latest` chưa tái lập được chỉ từ repo" | Trình bày như kết quả cuối |
| "Không so điểm tuyệt đối giữa `lastfm-u25` và `lastfm-u25-latest`" | "Latest tốt hơn vì Recall cao hơn" |
| "Chỉ nói cho mẫu 25% người dùng; `lastfm-u25` mất khoảng 28% val/test vì bài hát mới" | Gọi `lastfm-u25` là "Last-FM" |
| "Stage 2 của `lastfm-u25` so với KGAT là hai máy; so với bản hoán vị là cùng máy" | So Stage 2 với KGAT mà không nói khác máy |
| "Cổng 14 giây được nâng từ 12 sau khi thấy số đo" | "Qua cổng Kaggle đã đăng ký" |
| "MostPop gần đây không thắng MostPop thường ở test của `lastfm-u25`; giải thích còn là giả thuyết" | Khẳng định nguyên nhân |
| "0015 là bản nháp, chưa vào repo" | "Đã được duyệt" |

## F. Sau buổi present
- [ ] Ghi biên bản lên Notion ngay trong ngày: góp ý của cô, **lựa chọn A, B hay C**, câu trả lời cho hai câu còn lại, việc mới.
- [ ] Mỗi quyết định mới của cô thành một file `docs/decisions/` (hỏi Phôn duyệt). Nếu cô chọn B: PR riêng đưa code xuất của Phôn và quyết định 0015 vào repo, rồi quyết định cách thêm dòng registry; nếu chọn C: quyết định mới cho việc dựng lại tập cặp.
- [ ] Cập nhật `nhat-ky-thay-doi.md` và kế hoạch theo góp ý.
- [ ] Nếu cô duyệt phân tích bổ sung: viết quyết định trước khi chạy; mọi số thêm được ghi là "thêm sau khi đã thấy test".
