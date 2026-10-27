# Chủ đề 3: Knowledge Graph và Knowledge Graph Embedding

Người trình bày: Phôn · Người hỏi vặn: Triết · Ngày teach-back: ../10/2026

Bản nháp do Triết chuẩn bị sau khi học chủ đề này bằng hỏi đáp; Phôn sửa lại bằng lời của mình và điền phần teach-back.

## Ý chính
- KG là tập triple (h, r, t): thực thể đầu, quan hệ, thực thể đuôi. Ví dụ: (Tenet, director, Christopher Nolan), (Tenet, cast_member, Robert Pattinson). Tên thực thể phải cụ thể ("Batman Begins (2005)", không phải "Batman"), vì tên mơ hồ dẫn đến liên kết sai.
- KG giúp gợi ý ở hai phía. Phía item: phim ít được đánh giá có embedding yếu, nhưng nối qua đạo diễn, diễn viên, thể loại chung với phim khác thì nhận thêm thông tin; hai phim không có user chung vẫn có thể gần nhau. Phía user: user ít lịch sử vẫn có vài phim đã xem, nên đi qua thuộc tính để tới nhiều phim mới.
- Quan hệ 1-N: một phim có nhiều diễn viên (`cast_member`); nhìn từ đạo diễn, `director` ngược cũng là 1-N.
- CKG (collaborative KG) của KGAT gộp tương tác và KG vào một đồ thị, bằng hai bước: mỗi tương tác dương (rating >= 4) thành triple (u, Interact, i), và item được căn chỉnh với entity trong KG. Quan hệ ngược (t, r⁻¹, h) được thêm để thông tin đi hai chiều. Ví dụ đường đi: user A → Interact → phim B → genre → Hành động ← genre⁻¹ ← phim C.
- KGE (embedding KG): biểu diễn entity và relation bằng vector sao cho triple đúng có điểm tốt hơn triple sai.
- TransE: quan hệ là phép tịnh tiến, e_h + e_r ≈ e_t. Điểm yếu với 1-N: mọi tail của cùng (h, r) bị kéo về cùng một điểm, nên các diễn viên của một phim bị ép gần nhau.
- TransR (KGAT dùng): mỗi relation có ma trận chiếu W_r riêng, đưa entity sang không gian của relation đó. Hai diễn viên xa nhau trong không gian chung vẫn gần nhau sau khi chiếu, nên giảm vấn đề 1-N.
- Ba họ mô hình dùng KG: embedding-based (CKE: TransR cộng MF, không lan truyền), path-based (PER, KPRN: dùng các đường đi làm đặc trưng), propagation-based (RippleNet, KGCN, KGAT: lan truyền nhiều bước). KGAT dùng cả TransR như CKE, cộng lan truyền nhiều bước có attention. CKE là baseline KG chính vì so với KGAT cho biết lan truyền có attention có thêm giá trị so với chỉ nhúng KG.

## Công thức
TransE: g(h, r, t) = ‖e_h + e_r − e_t‖². g nhỏ nghĩa là triple có khả năng đúng.

TransR: g(h, r, t) = ‖W_r e_h + e_r − W_r e_t‖², với e_h, e_t ∈ ℝᵈ, e_r ∈ ℝᵏ, W_r ∈ ℝᵏˣᵈ.

Loss KGE: L = Σ −ln σ(g(h, r, t') − g(h, r, t)), triple hỏng (h, r, t') có được bằng cách thay tail ngẫu nhiên. Giống BPR ở chỗ so sánh theo cặp; khác ở dấu: g là khoảng cách nên đúng là g nhỏ, vì vậy viết g(sai) − g(đúng). Ví dụ g(đúng) = 1, g(sai) = 4: hiệu 3, loss ≈ 0,049. Hai g gần nhau (hiệu 0) thì loss = ln 2 ≈ 0,693.

## Dữ liệu của nhóm
- Đường liên kết cho ML-100K: ML-100K → ML-1M theo tiêu đề và năm (quyết định 0003; 1.647 trong 1.663 item khớp, 16 item không khớp, các khớp không chính xác đều kiểm tay) → ML-1M dùng chung id với ML-20M nên tra KB4Rec ra Freebase → tìm phim trên Wikidata qua thuộc tính P646. KB4Rec chỉ liên kết ML-20M, không có ML-100K.
- Wikidata (bản chụp 17/10/2026): 119.748 fact, 18 quan hệ, 3.725 phim.
- Độ phủ liên kết ML-100K: 1.617 trong 1.663 item (97,2%) có liên kết Freebase. Đây là độ phủ, chưa phải độ chính xác: tỷ lệ liên kết sai chưa đo vì kiểm tay một mẫu ngẫu nhiên (việc 15 của Phôn) chưa xong.
- Lỗi đã biết của KB4Rec: *Clubland* bị nối sang *Cloudland Canyon State Park* (phát hiện khi kiểm tay).
- Entity quá phổ biến ("English language", "United States of America") nối với gần hết phim nên ít thông tin; chỉ được lọc theo thông tin trước mốc.
- Tri thức tương lai: *Toy Story* có `followed_by → Toy Story 2` (24/11/1999), sau toàn bộ dữ liệu ML-100K. KG tĩnh sẽ cho mô hình biết trước; vì vậy KG được dựng theo mốc (quyết định 0004).

## Nằm ở đâu trong code
- Kiểm tra và quy tắc ánh xạ: `src/tkgat/kg/linking.py`, `scripts/03_link_entities.py`
- Lấy 18 quan hệ từ Wikidata: `scripts/04_fetch_wikidata.py` (`RELATIONS`)
- Fact có ngày: `data/raw/wikidata/facts.tsv`; luật theo mốc: `src/tkgat/kg/temporal.py`, `docs/decisions/0004-kg-temporal-rules.md`

## 3 câu tự kiểm tra
1. TransR sửa điểm yếu 1-N của TransE thế nào? Mỗi quan hệ có không gian riêng qua W_r; các tail xa nhau trong không gian chung vẫn gần nhau sau khi chiếu.
2. Loss KGE nhỏ khi nào? Khi g(sai) lớn hơn nhiều g(đúng), vì g nhỏ nghĩa là đúng; hiệu bằng 0 cho loss ln 2.
3. 97,2% item có liên kết Freebase có phải độ chính xác không? Không, đó là độ phủ; số liên kết sai chưa đo vì kiểm tay một mẫu (việc 15) chưa xong.

## Teach-back với Triết (điền sau buổi giảng)
Ngày: ..; 3 câu Triết hỏi vặn: ..; trả lời thế nào: ..
