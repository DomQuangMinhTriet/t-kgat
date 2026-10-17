# Meeting Notes - [17/10/2026]

## Meeting Notes - [17/10/2026]

### Meeting Notes - Khoá luận tốt nghiệp

Ngày: @October 17, 2026
Thời gian: 9:00 a.m.
Hình thức: Online (Google Meet)
Địa điểm / Link: Meet

## Thông tin chung

Mục tiêu buổi họp:

- Báo cáo tiến độ sau buổi 18/09/2026: kết quả mở rộng Knowledge Graph (Section08) và tiến độ track MovieLens.
- Nhận góp ý của GVHD về dữ liệu, quy trình đánh giá và hướng đi tiếp theo.

Thành phần tham dự:

- GVHD: cô Nguyễn Ngọc Minh Châu
- Sinh viên:
  - Nguyễn Đăng Phôn - 23127451
  - Đởm Quang Minh Triết - 23127131
- Khác: Không có

Tóm tắt nhanh

- Kết quả trình bày chưa đạt yêu cầu. Nguyên nhân chính nằm ở dữ liệu và quy trình đánh giá, không nằm ở mô hình.
- Cô Châu chỉ ra: thiếu tập validation (dẫn tới data leakage), chưa khám phá dữ liệu đủ kỹ, và Knowledge Graph chưa tôn trọng chiều thời gian (dùng thông tin tương lai để dự đoán hiện tại).
- Cô nhấn mạnh đề tài là bài toán về dữ liệu, đúng tinh thần chuyên ngành Hệ thống thông tin.
- Sau buổi họp, nhóm thống nhất làm lại từ đầu trên MovieLens-100K, với repo mới và quy trình chuẩn hóa. Cô đồng ý dùng MovieLens-100K làm dữ liệu chính.

## Tiến độ hiện tại

Đã hoàn thành

- Trình bày kết quả giai đoạn trước (Foursquare NYC, Amazon CDs_and_Vinyl, mở rộng KG ở Section08) và track MovieLens.
- Nhận đầy đủ góp ý của GVHD và thống nhất hướng làm lại.

Đang thực hiện / Chuẩn bị thực hiện

- Xây kế hoạch làm lại 13 tuần (19/10/2026 → 17/01/2027), mốc kiểm tra đầu tiên là buổi present 31/10/2026.
- Tạo repo GitHub mới, phân chia rõ ràng, có kiểm thử rò rỉ dữ liệu tự động.
- Ôn lại lý thuyết theo hình thức dạy lại cho nhau (teach-back).

## Vướng mắc / Rủi ro

- Kết quả của các giai đoạn trước không dùng tiếp được vì cùng mắc lỗi rò rỉ:
  - Không có tập validation: epoch tốt nhất được chọn trên tập test.
  - Lọc k-core trên toàn bộ dữ liệu trước khi chia, nên đã dùng cả tương tác của giai đoạn test.
  - KG dựng từ metadata hiện tại cho các tương tác trong quá khứ.
- Các kết quả trước đó mới chạy 1 seed. Khi kiểm chứng 3 seed, cải thiện của T-KGAT không còn đứng vững.
- KB4Rec chỉ liên kết sẵn MovieLens-20M với Freebase. Với MovieLens-100K phải ghép phim qua tên + năm, và chưa xác nhận được cách tải phần đồ thị.
- MovieLens-100K chỉ trải khoảng 7 tháng (19/09/1997 → 22/04/1998), nên tín hiệu thời gian có thể yếu.

## Nội dung trao đổi (theo chủ đề)

### 1. Khám phá dữ liệu

- Cần kiểm tra và khám phá dữ liệu kỹ hơn. Đề tài làm về dữ liệu, nên dữ liệu phải được hiểu rõ trước khi nói đến mô hình.

### 2. Mở rộng metadata và Knowledge Graph

- Cần tìm kỹ hơn các thông tin mở rộng (metadata, Knowledge Graph) và kiểm chứng xem chúng có thật sự làm kết quả tốt hơn không.

### 3. Data leakage

- Phải đảm bảo không có data leakage trong toàn bộ quy trình.

### 4. Tập validation

- Kết quả hiện tại thiếu tập validation. Phải có tập val, nếu không sẽ bị data leakage (dùng test để chọn mô hình).

### 5. Knowledge Graph theo chiều thời gian

- Với MovieLens, phần knowledge base khi map sang bài toán gợi ý phải bao hàm theo chiều thời gian: không dùng dữ liệu tương lai để dự đoán kết quả hiện tại.

## Quyết định / Chốt (Decisions)

- Làm lại từ đầu toàn bộ thực nghiệm. Kết quả cũ chỉ giữ làm bài học, không dùng trong báo cáo.
- Dữ liệu chính: MovieLens-100K (cô đồng ý). Triết mở rộng sang MovieLens-1M sau, dùng cùng pipeline.
- Knowledge Graph lấy từ KB4Rec/Freebase, ghép phim qua MovieLens-20M theo tên + năm. Nếu Freebase thiếu ngày thì bổ sung ngày của fact từ Wikidata.
- Chia train/validation/test theo mốc thời gian chung cho mọi user. Mọi lựa chọn epoch và tham số dựa trên validation; test chỉ chấm một lần cho cấu hình cuối.
- Knowledge Graph được dựng theo mốc: chỉ giữ fact đã tồn tại trước mốc cắt.
- Bỏ phần "Confidence" khỏi phạm vi; đề tài tập trung vào yếu tố thời gian.
- Tạo repo GitHub mới (private, trong một Organization chung); repo cũ được archive.
- Buổi báo cáo tiếp theo: 31/10/2026, trình bày bằng notebook (không làm slide). Nội dung gồm báo cáo dữ liệu, quy trình chống rò rỉ và baseline đơn giản. KGAT chỉ chạy lại sau khi cô duyệt quy trình.
- Ngôn ngữ: báo cáo và notebook present viết tiếng Việt; code, comment, commit, README viết tiếng Anh.

## Việc cần làm (Actions và Milestones)

### A. Dựng nền (19–21/10)

- Tạo repo, CI chạy pytest, bảo vệ nhánh main, bảng GitHub Projects.
- Viết CONTRIBUTING, mẫu Issue/PR; archive repo cũ; đăng ký Kaggle để có GPU.
- Kiểm tra cách tải phần đồ thị của KB4Rec; tải MovieLens-100K, bảng phim MovieLens-20M và KB4Rec, ghi checksum.

### B. Khám phá dữ liệu và KG (19–28/10)

- EDA rating theo thời gian; thử nhiều cặp mốc T1, T2 và đếm user, item, tương tác ở val/test.
- Ghép phim MovieLens-100K ↔ MovieLens-20M ↔ Freebase; đo độ phủ; kiểm tra tay 50 phim.
- Hồ sơ KG theo thời gian: tỷ lệ fact có ngày, số fact "tương lai" so với từng mốc; dựng KG theo mốc.
- Báo cáo dữ liệu tổng hợp (4–6 trang).

### C. Quy trình chống rò rỉ và baseline (22–29/10)

- Chia train/val/test theo mốc; lọc k-core chỉ trên train; bộ test rò rỉ tự động chạy trong CI.
- Bộ đánh giá Recall/NDCG@10/20.
- Baseline MostPop và BPR-MF: dừng sớm theo val, 5 seed, chấm test một lần.

### D. Ôn lý thuyết (19/10–08/11)

- Teach-back: đánh giá offline và rò rỉ dữ liệu; BPR; KG và KGE (trước 31/10); các chủ đề còn lại sau 31/10.

### E. Chuẩn bị buổi 31/10

- Notebook present chạy sẵn; tập dượt 2 lượt, có đảo vai.

## Phân công

| Người phụ trách | Nhiệm vụ chính |
|---|---|
| Đởm Quang Minh Triết | Repo và CI; tải dữ liệu; EDA rating; chia dữ liệu và test rò rỉ; dựng KG theo mốc; báo cáo dữ liệu; mở rộng MovieLens-1M (sau 31/10). |
| Nguyễn Đăng Phôn | CONTRIBUTING và quy ước; liên kết thực thể (MovieLens ↔ Freebase/Wikidata); tra ngày fact trên Wikidata; bộ đánh giá; baseline MostPop và BPR-MF. |
| Cả hai | Duyệt chéo mọi Pull Request; teach-back; notebook present; tập dượt. |

## Nội dung cho buổi họp tiếp theo (31/10/2026)

- Góp ý ngày 17/10 và việc đã làm cho từng góp ý.
- Báo cáo dữ liệu MovieLens-100K: phân bố theo thời gian, cách chọn mốc T1, T2.
- Liên kết Knowledge Graph: độ phủ, kết quả kiểm tra tay, tỷ lệ fact có ngày, số fact "tương lai" bị loại.
- Quy trình chia train/val/test và bằng chứng không rò rỉ (test tự động).
- Kết quả baseline MostPop và BPR-MF trên quy trình mới (trung bình ± độ lệch chuẩn qua 5 seed).
- Kế hoạch từ 02/11 và các câu hỏi xin ý kiến cô: xử lý fact không có ngày; tên đề tài có cần bỏ chữ "Confidence" không.
