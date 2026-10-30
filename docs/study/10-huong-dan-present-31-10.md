# 10 — Hướng dẫn present ngày 31/10/2026

Kịch bản trình bày theo bố cục đã chốt (khoảng 24 phút, chưa tính phần cô hỏi). Mỗi đoạn có: thời lượng, người nói, mục notebook hoặc hình cần mở, lời nói mẫu, câu chuyển sang đoạn sau. Lời nói mẫu là gợi ý: hiểu ý và nói bằng lời của mình, không đọc. Nội dung để học nằm ở `docs/study/09-hoc-va-bao-cao-31-10.md`; lý thuyết ở `docs/study/08-tong-hop-ly-thuyet.md`.

Ký hiệu: **[Mở]** = mục notebook hoặc hình cần mở; **[Chuyển]** = câu chuyển; *[ghi chú]* = không đọc.

---

## A. Trước buổi present

**Tối 30/10**
- [ ] Notebook `notebooks/00_present_2026-10-31.ipynb` chạy từ đầu đến cuối không lỗi trên laptop, với số mới nhất (sau khi merge kết quả Stage 2 và bảng RQ3).
- [ ] Xuất bản HTML dự phòng (`E:\KLTN HCMUS\present-backup\`), chép thêm vào USB hoặc Google Drive.
- [ ] Mở sẵn trong một thư mục: 8 hình ở `docs/figures/`, `docs/rq3-table.md`, tài liệu 09 (Phụ lục A, thẻ số).
- [ ] Tập dượt 2 lượt, lượt 2 đổi vai; bấm giờ từng phần.

**Sáng 31/10, 07:30**
- [ ] Máy chiếu, cáp, sạc; độ phân giải màn hình; phóng chữ notebook (Ctrl +) cho người ngồi xa đọc được.
- [ ] Mở notebook (đã chạy sẵn, không chạy lại trước mặt cô) và bản HTML ở tab bên cạnh.
- [ ] Tắt thông báo, đóng ứng dụng không cần.
- [ ] Chuẩn bị giấy ghi góp ý của cô; phân công: người không nói thì ghi.

---

## B. Kịch bản

### Phần 0 — Mở đầu (2 phút) · Phôn
**[Mở]** notebook mục 1, 2, 3.

**Phôn:** "Thưa cô, hôm nay nhóm báo cáo bốn phần: cách chuẩn bị dữ liệu, cách huấn luyện mô hình, kết quả và nhận xét, cuối cùng là kế hoạch và ba câu hỏi xin ý kiến cô. Buổi 17/10 cô góp ý ba điểm: thiếu tập validation nên có rò rỉ, chưa khám phá dữ liệu đủ kỹ, và Knowledge Graph chưa tôn trọng chiều thời gian. Khi rà lại, nhóm thấy kết quả cũ mắc ba lỗi rò rỉ: chọn epoch tốt nhất trên test, lọc k-core trên toàn bộ dữ liệu, và dựng KG từ metadata hiện tại. Vì vậy nhóm làm lại từ đầu."

"Nhóm đặt đề tài theo ngành Hệ thống thông tin: đây là bài toán tích hợp và quản lý dữ liệu cho hệ gợi ý, còn T-KGAT là mô hình để đo giá trị của dữ liệu. Có ba câu hỏi nghiên cứu: dữ liệu và KG có đặc điểm gì theo thời gian; khi đã loại rò rỉ thì KG còn giúp không; và thêm thời gian vào attention có giúp không, có thật sự nhờ thời gian không."

**[Chuyển]** "Phần đầu, bạn Triết trình bày cách nhóm chuẩn bị dữ liệu."

---

### Phần 1 — Chuẩn bị dữ liệu (8 phút) · Triết

**1a. EDA (2 phút).** **[Mở]** mục 4, hình `fig1_rating_theo_tuan.png`, `fig2_do_trai_thoi_gian.png`.

**Triết:** "Trước khi làm gì với mô hình, nhóm khám phá dữ liệu. ML-100K có 100 nghìn rating trong 215 ngày; ML-1M có hơn 1 triệu rating trong khoảng 1.000 ngày nhưng 90,5% rating rơi vào năm 2000. Phát hiện quan trọng nhất: rating được nhập theo phiên. 77,9% rating của ML-100K nằm trong 24 giờ đầu kể từ rating đầu tiên của user; 53% user có toàn bộ rating train trong chưa đầy một giờ. Ngược lại, rating của một phim trải ra trung vị 45 ngày."

"Phát hiện này có ba hệ quả mà cô sẽ thấy lại ở các phần sau: chia theo thời gian làm mất nhiều user; Δt giữa các rating của một user gần như bằng 0, nên tác dụng của thời gian chỉ có thể đến từ phía phim; và vì thế thí nghiệm hoán vị timestamp là bắt buộc."

**1b. Làm sạch và chia dữ liệu (2 phút).** **[Mở]** mục 7, hình `fig3_positive_con_lai.png`.

**Triết:** "Nhóm gộp 18 tiêu đề trùng, bỏ item 267 'unknown', gộp cặp user–item trùng. Sau đó chia theo hai mốc chung cho mọi user: t1, t2 là phân vị 80% và 90% của mọi timestamp. Rating từ 4 trở lên là tương tác dương. Lọc 5-core chỉ trên train. Val và test chỉ giữ user và phim đã có trong train."

"Cái giá là ML-100K chỉ còn giữ khoảng 13% số dương ở test, tức 51 user, vì gần như toàn bộ phần bị loại là của user mới vào sau t1. ML-1M giữ 75,7%, có 952 user test. Vì vậy kết luận chính nhóm dựa vào ML-1M."

**1c. Liên kết thực thể và KG theo thời điểm (2 phút).** **[Mở]** mục 5, 6, hình `fig4_fact_bi_loai.png`.

**Triết:** "Để có KG, nhóm nối MovieLens sang Freebase qua KB4Rec rồi sang Wikidata. Độ phủ là 97,2% số phim. Độ phủ không phải độ chính xác, nên nhóm kiểm 50 phim ngẫu nhiên: không phim nào sai, nhưng khoảng tin cậy Wilson 95% vẫn cho tỷ lệ sai đến khoảng 7%, nên nhóm không nói 'đúng 100%'. KG hiện có fact Wikidata và metadata MovieLens như thể loại, năm phát hành."

"Theo góp ý của cô, một fact chỉ được dùng nếu nó đã đúng trước mốc. Ngày chỉ có năm thì coi là 31/12 cho chắc. KG cho train và val dựng tại t1, cho test dựng tại t2. Ví dụ *Titanic*: 11 giải Oscar ngày 23/03/1998 nằm giữa t1 và t2, nên chỉ có trong KG của test. Bộ lọc chỉ bỏ khoảng 1,3% fact ở ML-100K, nhưng chúng tập trung ở phim nổi tiếng: những phim bị mất fact mang khoảng một nửa số dương của val và test."

**1d. Bằng chứng không rò rỉ (2 phút).** **[Mở]** mục 7 (bảng kiểm tra).

**Triết:** "Toàn bộ quy trình được kiểm tra tự động. Hàm `check_split` kiểm 6 điều: thứ tự thời gian, không cặp nào ở hai phần, val và test chỉ có user và phim đã biết, 5-core đúng trên riêng train, mask chỉ dùng quá khứ, và đáp án không trùng mask. Hàm `check_kg` từ chối lưu KG có fact sau mốc. Các kiểm tra chạy trên GitHub mỗi lần đẩy code."

**[Chuyển]** "Dữ liệu đã sẵn sàng; bạn Phôn trình bày cách nhóm huấn luyện mô hình."

---

### Phần 2 — Huấn luyện mô hình (6 phút) · Phôn, rồi Triết

**2a. Baseline và quy trình chung (3 phút) · Phôn.** **[Mở]** mục 9, hình `fig5_bprmf_curve_ml-1m.png`, bảng siêu tham số (tài liệu 09, mục 2.3).

**Phôn:** "Nhóm có ba baseline: MostPop gợi ý phim phổ biến nhất trong train; MostPop gần đây chỉ đếm vài ngày cuối của train, số ngày chọn trên val; và BPR-MF, matrix factorization học xếp hạng."

"Mọi mô hình theo cùng một quy trình: mỗi điểm trong lưới tham số chạy 5 seed; đánh giá Recall@20 trên val mỗi 5 epoch, dừng sớm khi val không tăng nữa; chọn điểm lưới có val trung bình cao nhất; rồi mới chấm test, đúng một lần. Đây là learning curve của BPR-MF: epoch tốt nhất được chọn theo val, test không tham gia. Bảng này liệt kê batch size, số epoch, learning rate, l2 và điểm được chọn. Mọi lần chạy đều ghi vào một file registry chỉ thêm dòng, có mã commit; số trong báo cáo đều đọc từ file này, không gõ tay."

**2b. KGAT và T-KGAT (3 phút) · Triết.** **[Mở]** mục 11, hình `rq3_curves_ml-1m-u10.png`.

**Triết:** "KGAT gộp tương tác và KG thành một đồ thị, lan truyền thông tin qua ba lớp, dùng attention để quyết định hàng xóm nào quan trọng, và học bằng hai loss: BPR cho gợi ý và TransR cho KG. Nhóm viết lại bằng PyTorch, chạy trên GPU laptop, cấu hình như paper và lưới learning rate × l2."

"T-KGAT chỉ khác một chỗ: trước khi chuẩn hóa attention, trừ đi γ nhân Δt cho cạnh user–phim, Δt là độ cũ của rating so với cuối train. Cạnh cũ được chú ý ít hơn. Khi γ = 0, T-KGAT đúng bằng KGAT. γ chọn trên val trong {1, 3, 10}."

"Để biết phần cải thiện có thật sự nhờ thời gian không, nhóm chạy thêm bản hoán vị: cùng mô hình, cùng γ, nhưng timestamp bị xáo trộn. Nếu bản hoán vị tốt bằng bản thật thì không phải nhờ thời gian. Nhóm còn có Stage 2, đặt Δt lên các fact của KG thay vì lên rating."

*[Nếu thiếu giờ: bỏ câu về hai loss và Stage 2, giữ "T-KGAT = KGAT trừ γ·Δt trước softmax" và ý hoán vị.]*

**[Chuyển]** "Phần tiếp theo là kết quả. Bạn Phôn bắt đầu với baseline."

---

### Phần 3 — Kết quả và nhận xét (6 phút) · Phôn, rồi Triết

**3a. Baseline (1,5 phút) · Phôn.** **[Mở]** mục 8.

**Phôn:** "Trên ML-1M, 952 user test, BPR-MF đạt Recall@20 0,1194, hơn MostPop gần đây 0,1060 một cách có ý nghĩa sau kiểm định Wilcoxon và hiệu chỉnh Holm. Trên ML-100K thì MostPop gần đây đạt 0,3372, cao hơn mọi mô hình. Nhóm giải thích bằng dữ liệu: test ML-100K chỉ 22 ngày và 51 user, nên phim đang hot gần mốc cắt chiếm ưu thế. Vì vậy mô hình có thời gian phải vượt được MostPop gần đây, không chỉ MostPop."

**3b. KGAT và T-KGAT (3 phút) · Triết.** **[Mở]** mục 11 (bảng và kiểm định).

**Triết:** "Đây là kết quả sơ bộ trên ML-100K và ml-1m-u10, tức 10% user của ML-1M. Nhận xét thứ nhất về KG: KGAT nhỉnh hơn bản không có KG, nhưng chỉ sát ngưỡng ý nghĩa, NDCG trên ml-1m-u10 có p bằng 0,049; Recall thì không có ý nghĩa."

"Nhận xét thứ hai về thời gian: T-KGAT không khác bản hoán vị ở cả hai bộ dữ liệu. Trên ml-1m-u10, T-KGAT có hơn KGAT ở NDCG, nhưng bản hoán vị cũng đạt mức đó, nên phần hơn không phải nhờ thời gian. Kết luận hiện tại: chưa có bằng chứng thời gian giúp ích. Nhóm giải thích bằng chính phát hiện ở phần dữ liệu: rating nhập theo phiên nên Δt gần như không mang thông tin."

"Với Stage 2, đặt Δt lên các fact của KG: kết quả cũng không khác bản hoán vị ngày fact, và không khác KGAT có ý nghĩa. Nhóm phát hiện Δt chuẩn hóa của fact bị chặn ở 1, và chỉ 0,7% fact có ngày nằm dưới mức đó, nên Stage 2 thực chất giảm trọng số fact có ngày so với fact không ngày, chưa đo được fact cũ hay mới. Nhóm đề xuất thiết kế lại cách chuẩn hóa này."

**3c. Minh bạch và hạn chế (1,5 phút) · Triết.** **[Mở]** phụ lục của bảng RQ3; slide hạn chế (tài liệu 09, mục 3.4).

**Triết:** "Có một điểm nhóm phải báo rõ: KG được thêm metadata MovieLens sau khi đã xem điểm test của KG cũ. Nhóm ghi quyết định 0011: KG mới là chính thức, chạy lại toàn bộ với cùng lưới và seed, lấy kết quả mới dù tốt hay xấu; kết quả cũ để ở phụ lục để cô đối chiếu."

"Các hạn chế: Wikidata là bản chụp năm 2026; timestamp được giả định là lúc nhập rating; test dưới 100 user ở hai bộ nhỏ nên p lớn chưa chứng minh không khác biệt; KGAT trên GPU không tái lập từng bit; ml-1m-u10 chưa phải ML-1M đầy đủ; KGAT chạy trước khi cô duyệt quy trình nên nhóm ghi là sơ bộ; và nhóm chưa kiểm tra tái lập độc lập trên máy khác, sẽ làm đầu giai đoạn 4."

**[Chuyển]** "Cuối cùng là kế hoạch và các câu hỏi nhóm xin ý kiến cô."

---

### Phần 4 — Kế hoạch và câu hỏi (2 phút) · Cả hai
**[Mở]** mục 10.

**Phôn:** "Từ 02/11, giai đoạn 4 làm các biến thể KG, KG 2-hop, baseline CKE và một lần chạy theo quy trình gốc của paper làm mốc. Giai đoạn 5 làm thời gian trên KG và ba hướng cô gợi ý ngày 18/09. Giai đoạn 6 viết khóa luận."

**Triết:** "Nhóm xin ý kiến cô ba câu. Một: kết quả KGAT và T-KGAT đã chạy sớm, giữ làm chính thức hay chạy lại sau khi cô duyệt quy trình? Hai: chạy ML-1M đầy đủ, khoảng 29 giờ GPU mỗi cấu hình, hay dùng ml-1m-u10 và ghi là giới hạn? Ba: bước tiếp theo nên ưu tiên thiết kế lại Stage 2 hay ba hướng cô gợi ý?"

**Cả hai (chốt):** "Tóm lại: quy trình đã sạch rò rỉ và kiểm tra được, dữ liệu được hiểu kỹ, và kết quả âm của câu hỏi thời gian được báo cáo trung thực kèm giải thích. Nhóm cảm ơn cô và xin nghe góp ý."

---

## C. Khi bị hỏi giữa chừng

| Tình huống | Cách xử lý |
|---|---|
| Cô hỏi phần của người kia | Người đang nói trả lời nếu chắc; nếu không thì "Bạn X phụ trách phần này, xin phép bạn trả lời" |
| Không biết câu trả lời | "Em chưa kiểm tra điều này; nhóm sẽ kiểm và báo lại cô." Ghi vào giấy. Không đoán số |
| Cô hỏi một con số | Mở `docs/rq3-table.md` hoặc thẻ số (tài liệu 09, Phụ lục A); nói nguồn |
| Cô nghi rò rỉ | Nêu 6 kiểm tra của `check_split`, `check_kg`, test chấm một lần, registry có commit |
| Cô cho rằng kết quả âm là thất bại | "Biên bản 07/08 ghi kết quả âm hợp lệ nếu quy trình đúng; nhóm có đối chứng hoán vị và giải thích bằng dữ liệu" |
| Hết giờ | Bỏ chi tiết KGAT ở 2b và hình ở 1c; giữ ba nhận xét và ba câu hỏi |
| Máy chiếu hoặc notebook lỗi | Mở bản HTML; nếu vẫn lỗi, dùng `docs/rq3-table.md` và hình trên GitHub |

## D. Năm câu cô gần như chắc chắn hỏi (đáp án trong tài liệu 09)
1. Sao chắc không rò rỉ? (09, phần 1.6)
2. Sao test ML-100K chỉ 51 user? (09, phần 1.3)
3. Sao MostPop gần đây thắng trên ML-100K? (09, nhận xét 1)
4. T-KGAT khác KGAT chỗ nào, hoán vị để làm gì? (09, phần 2.1, nhận xét 3)
5. Sao số KGAT đổi so với trước? (09, phần 3.3)

## E. Nói và không nói

| Nên nói | Tránh nói |
|---|---|
| "Chưa có bằng chứng thời gian giúp ích" | "T-KGAT tốt hơn KGAT" |
| "KG giúp sát ngưỡng ý nghĩa, ở NDCG trên ml-1m-u10" | "KG giúp có ý nghĩa" |
| "0/50 sai, cận trên khoảng 7%" | "Liên kết đúng 100%" |
| "ml-1m-u10, 10% user của ML-1M" | Gọi ml-1m-u10 là "ML-1M" |
| "Kết quả sơ bộ" (KGAT, T-KGAT) | Trình bày như kết quả cuối |
| "Stage 2 giảm trọng số fact có ngày" | "Stage 2 đo tác dụng thời gian trên KG" |

## F. Sau buổi present
- [ ] Ghi biên bản lên Notion ngay trong ngày: góp ý của cô, câu trả lời cho ba câu hỏi, việc mới.
- [ ] Mỗi quyết định mới của cô thành một file `docs/decisions/` (hỏi Phôn duyệt).
- [ ] Cập nhật kế hoạch (Claude Doc) theo góp ý.
