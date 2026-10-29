# Chủ đề 6: Tích hợp dữ liệu, liên kết thực thể và tính đúng theo thời điểm

Người trình bày: Phôn · Người hỏi vặn: Triết · Ngày teach-back: ../10/2026

Bản nháp soạn từ `docs/study/06-tich-hop-du-lieu.md`, quyết định 0003, 0004, 0006, 0011 và `notebooks/02_entity_linking.ipynb`; Phôn đọc lại, sửa bằng lời của mình và điền phần teach-back.

## Ý chính
- Đây là phần của đề tài gắn với chuyên ngành Hệ thống thông tin: ghép dữ liệu từ nhiều nguồn, đo chất lượng, và bảo đảm dữ liệu "đúng tại thời điểm" như trong data warehouse.
- Chuỗi tích hợp: MovieLens (id, tên, năm) → KB4Rec (MovieLens → Freebase) → Wikidata qua thuộc tính P646 (Freebase ID) → fact có ngày. Ba nguồn, hai bước nối; mỗi bước phải được đo và ghi lại (checksum, ngày tải, commit).
- ML-100K không có trong KB4Rec nên đi vòng: ML-100K → ML-1M theo tên và năm → ML-1M dùng chung id với ML-20M → Freebase. Các khớp không chính xác (tên phụ, chứa chuỗi, năm lệch 1) nhóm đều kiểm lại.
- Làm sạch trước khi nối: 18 tiêu đề trùng gộp về một id, item 267 "unknown" bị loại, cặp (user, item) trùng giữ lần đầu, và checksum của file thô để chắc chắn mọi người dùng cùng một bản dữ liệu.
- Kiểm tra liên kết là bắt buộc, không tin tuyệt đối bảng có sẵn. Hai chỉ số khác nhau: độ phủ (bao nhiêu phim có liên kết) và độ chính xác (bao nhiêu liên kết đúng). Độ phủ cao không chứng minh độ chính xác.
- Mô hình dữ liệu Wikidata: statement = (item, property, value) cùng qualifier. Ngày nằm ở qualifier (P585 thời điểm, P580 bắt đầu) hoặc ở chính entity (P577 ngày công chiếu, P571 thành lập, P569 sinh). `timePrecision` 9 = năm, 10 = tháng, 11 = ngày; "1998" ở độ chính xác 9 được lưu là `1998-01-01` nhưng **không** có nghĩa là ngày 1/1.
- Quy tắc bảo thủ (quyết định 0006): chỉ có năm thì coi là 31/12 của năm đó, chỉ có tháng thì coi là ngày cuối tháng. Lý do: một fact chỉ được dùng khi chắc chắn nó đã xảy ra trước mốc; coi là đầu năm thì có nguy cơ đưa tri thức tương lai vào.
- Tính đúng theo thời điểm: để tái tạo trạng thái tại thời điểm t chỉ được dùng dữ liệu đã biết tại t. Với KG: fact có ngày thì giữ nếu ngày < mốc; fact trỏ tới thực thể ra đời sau mốc thì bỏ; fact không ngày thì giữ nếu hai đầu đã tồn tại (mặc định) và có biến thể bỏ hết fact không ngày để đo ảnh hưởng. KG cho train/val dựng tại t1, cho test dựng tại t2.
- Giới hạn phải nói rõ: Wikidata là ảnh chụp 2026. Lọc theo ngày sự kiện không có nghĩa là KG lúc đó đã có fact đó; thứ ta lọc được là *ngày sự kiện xảy ra*, không phải *ngày fact được nhập vào KG*.
- Từ quyết định 0011, KG còn có metadata từ danh mục MovieLens (thể loại, năm, thập niên). Các fact này được gán ngày là ngày công chiếu và đi qua cùng các bộ lọc thời gian.

## Công thức
Wilson 95% cho tỷ lệ lỗi liên kết k/n: p = k/n; khoảng = (p + z²/2n ± z·√(p(1−p)/n + z²/4n²)) / (1 + z²/n) với z = 1,96. Khi k = 0 cận dưới đúng bằng 0, và cận trên phụ thuộc n.

Ví dụ số: 0 lỗi trong 50 liên kết cho khoảng ≈ [0%; 7,1%], tức tỷ lệ lỗi thật có thể đến khoảng 7% mà mẫu vẫn không bắt được lỗi nào. Nếu mẫu là 200 thì cận trên giảm còn khoảng 1,9%.

## Dữ liệu của nhóm
- Độ phủ KB4Rec: 97,2% số phim và 99,5% số rating của ML-100K; ML-1M 98,2% và 99,7% (`data/processed/links/linking_report.json`). Phim chưa liên kết là phim ít người xem.
- Kiểm tay mẫu 50 phim ngẫu nhiên của ML-100K (seed 2019): 0/50 sai, Wilson 95% [0,0%; 7,1%]. 40 phim có hai nguồn độc lập cùng khớp (năm Wikidata và tên YAGO), 10 phim Triết mở trang Wikidata kiểm tay. Mẫu nhỏ, nên báo cáo cận trên chứ không báo "0%".
- Lỗi đã biết của KB4Rec: *Clubland* (1998, ML-1M id 2588) mang mã YAGO của Cloudland Canyon State Park (quyết định 0003).
- Wikidata (bản chụp 17/10/2026): 119.748 fact, 18 quan hệ, 3.725 trong 3.811 phim ML-1M đã liên kết. Ngày statement gần như chỉ có ở `award_received` và `nominated_for`.
- Ví dụ point-in-time: 11 giải Oscar của *Titanic* (23/03/1998) nằm giữa t1 (07/03/1998) và t2 (31/03/1998) của ML-100K, nên chỉ có trong KG của test.
- Số triple của KG sau 0011 và tỷ lệ fact có ngày được sinh bởi `scripts/07_report_numbers.py` vào `docs/data-report-numbers.md`; thống kê ngày chỉ tính trên fact Wikidata, metadata MovieLens có dòng riêng.

## Nằm ở đâu trong code
- Làm sạch: `src/tkgat/data/clean.py`; liên kết: `src/tkgat/kg/linking.py`, `scripts/03_link_entities.py`
- Lấy fact từ Wikidata: `scripts/04_fetch_wikidata.py`
- Đọc ngày và lọc theo mốc: `src/tkgat/kg/temporal.py`; build và kiểm tra KG: `scripts/06_build_kg.py` (`check_kg` từ chối lưu nếu có fact sau mốc)
- Khoảng Wilson: `src/tkgat/analysis/report.py` (`wilson_interval`); notebook `notebooks/02_entity_linking.ipynb`
- Quyết định: `docs/decisions/0003-entity-linking.md`, `0004-kg-temporal-rules.md`, `0006-kg-date-handling.md`, `0011-movielens-metadata-in-kg.md`

## 3 câu tự kiểm tra
1. Vì sao 0/50 liên kết sai chưa chứng minh tỷ lệ lỗi bằng 0? Vì mẫu nhỏ: khoảng Wilson 95% cho 0/50 là [0%; 7,1%], tỷ lệ lỗi thật có thể đến khoảng 7%.
2. Vì sao ngày chỉ có năm được coi là 31/12? Vì chỉ dùng fact khi chắc chắn đã xảy ra trước mốc; coi là 1/1 có thể cho một fact đến muộn vào KG của một mốc giữa năm.
3. Quy tắc theo mốc có giải quyết hết vấn đề tri thức tương lai không? Không: nó lọc theo ngày sự kiện, nhưng Wikidata là ảnh chụp 2026 nên có thể chứa thông tin mà năm 1998 chưa ai nhập; giới hạn này phải nêu mỗi khi báo cáo kết quả có KG.

## Teach-back với Triết (điền sau buổi giảng)
Ngày: ..; 3 câu Triết hỏi vặn: ..; trả lời thế nào: ..
