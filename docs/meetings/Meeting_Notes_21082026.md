# Meeting Notes - [21/08/2026]

## Meeting Notes - [21/08/2026]

### Meeting Notes - Khoá luận tốt nghiệp

Ngày: @August 21, 2026
Thời gian: 8:00-9:00 a.m.
Hình thức: Online (Google Meet)
Địa điểm / Link: Meet

## Thông tin chung

Mục tiêu buổi họp:

- Báo cáo tiến độ track Foursquare NYC (T-KGAT) của Triết sau khi đổi dataset từ Last.fm-1K (16/8/2026).
- Trình bày kết quả Run 1 (random), Run 2 (chronological) và Run 3 (T-KGAT, grid γ ∈ {0.5, 1.0, 2.0}).
- Trình bày phát hiện rò rỉ dữ liệu (data leakage) do check-in lặp lại và cách khắc phục.
- Ghi nhận tình hình phần MovieLens của Phôn (gặp sự cố kỹ thuật, chưa trình bày được).
- Chốt các yêu cầu bổ sung của GVHD để chuẩn bị cho buổi báo cáo ngày 18/09/2026.

Thành phần tham dự:

- GVHD: cô Nguyễn Ngọc Minh Châu
- Sinh viên:
  - Nguyễn Đăng Phôn - 23127451
  - Đởm Quang Minh Triết - 23127131
- Khác: Không có

Tóm tắt nhanh

- Triết trình bày toàn bộ quá trình đổi dataset (Last.fm-1K → Foursquare NYC), lý do kỹ thuật (KGAT lan truyền toàn đồ thị mỗi batch, không khả thi trên CPU-only với Last.fm-1K), pipeline tiền xử lý, phát hiện và khắc phục lỗi rò rỉ dữ liệu, cùng kết quả Run 1/Run 2/Run 3.
- Phôn gặp sự cố kỹ thuật trong buổi họp nên chưa trình bày tốt phần MovieLens; sẽ trình bày lại toàn bộ ở buổi tiếp theo kèm các nâng cấp mới được yêu cầu.
- Cô Châu ghi nhận kết quả T-KGAT trên Foursquare NYC và giao thêm các yêu cầu để hoàn thiện trước báo cáo ngày 18/09/2026.

## Tiến độ hiện tại

Đã hoàn thành

- Phát hiện track Last.fm-1K không khả thi về hiệu năng (1 epoch ≥25 phút CPU-only, 400 epoch ≈ 7-8 ngày) và quyết định đổi sang dataset Foursquare NYC check-in (Yang et al., IEEE TSMC 2015), thoả 2 ràng buộc đã chốt ngày 07/08 (khác domain MovieLens, có timestamp gốc theo interaction).
- Hoàn tất pipeline tiền xử lý Foursquare NYC: raw 227,428 dòng → k-core=10 → 147,938 interaction, 1,083 user, 5,135 venue, 318 category (đã chuẩn hoá category theo mode).
- Phát hiện và khắc phục lỗi rò rỉ dữ liệu: check-in lặp lại nhiều lần trên cùng 1 cặp (user, venue) (trung bình 4.09 check-in/cặp) khiến 98.9% user bị overlap giữa train/test nếu chia theo dòng thô. Đã sửa bằng cách dedupe về cặp (user, item) duy nhất (giữ timestamp sớm nhất) trước khi chia split — xác nhận 0% overlap sau khi sửa.
- Chạy xong Run 1 - KGAT-Random (Recall@20 = 0.0632) và Run 2 - KGAT-Chronological (Recall@20 = 0.0435) — đúng lý thuyết: chronological khó hơn random do không còn rò rỉ thông tin tương lai (~31% thấp hơn).
- Chạy xong Run 3 - T-KGAT với grid γ ∈ {0.5, 1.0, 2.0} trên chronological split: cả 3 giá trị đều vượt Run 2 trên mọi metric; γ=2.0 tốt nhất toàn diện (Recall@20 +7.3%, NDCG@20 +12.4%, Hit Ratio@20 +15.3%), hội tụ sớm hơn (epoch 19) và train nhanh hơn (460.9s).
- Hoàn thiện notebook báo cáo (`foursquare_tkgat_report.ipynb`) và tài liệu hướng dẫn present kèm giải thích lý thuyết và kịch bản demo.

Đang thực hiện / Chuẩn bị thực hiện

- Phôn: khắc phục sự cố kỹ thuật, chuẩn bị trình bày lại toàn bộ tiến độ MovieLens ở buổi báo cáo tiếp theo, kèm các nâng cấp mới (mục dưới).
- Triết: bổ sung baseline Matrix Factorization (MF) trên cùng chronological split, thay thế vị trí Run 1 hiện tại, để so sánh chính thức là MF vs Run 2 vs Run 3.
- Triết: nghiên cứu chi tiết hơn công thức time-attention — hiện mới dừng ở grid 3 mốc γ rời rạc, chưa có phân tích sâu để chọn/tối ưu γ.
- Triết: tìm hiểu thêm hướng thử nghiệm mới cho phép các cặp (user, item) lặp lại (không dedupe), coi là trường hợp phụ thuộc khoảng thời gian (time-window) — cần làm rõ thêm thiết kế cụ thể trước khi triển khai, chưa thay đổi pipeline dedupe hiện tại (vẫn giữ nguyên cho các kết quả Run 1/2/3 đã có).
- Cả nhóm: thống nhất tên gọi/ký hiệu công thức time-attention giữa 2 track (Triết và Phôn) trước khi tổng hợp báo cáo chung.

## Vướng mắc / Rủi ro

- Phôn gặp lỗi kỹ thuật trong buổi 21/08 nên chưa trình bày đầy đủ phần MovieLens; cần xử lý sớm để không ảnh hưởng tiến độ trước 18/09.
- Công thức time-attention hiện tại (`a_time(h,r,t) = a(h,r,t) − δ_r·γ·Δt_norm`) mới dừng ở mức đề xuất baseline kèm grid γ thô (0.5/1.0/2.0), chưa có cơ sở lý thuyết chi tiết để chọn γ — GVHD nhận xét cần nghiên cứu kỹ hơn.
- Chưa có ablation so sánh với CF thuần (Matrix Factorization) — hạn chế nhóm đã tự nêu trong báo cáo trước, nay trở thành yêu cầu cụ thể từ GVHD.
- Kết quả hiện tại mới chạy 1 seed/cấu hình, chưa có kiểm định ý nghĩa thống kê.
- Hướng "cho phép lặp lại, không dedupe" cần làm rõ thêm cách triển khai cụ thể (theo time-window) — nếu áp dụng sai phạm vi có thể mâu thuẫn với cơ chế chống leakage đã xác nhận đúng ở Run 1/2/3.

## Nội dung trao đổi (theo chủ đề)

### Báo cáo tiến độ Foursquare NYC / T-KGAT (Triết)

- Trình bày lý do đổi dataset, pipeline tiền xử lý (k-core, chuẩn hoá category), phát hiện và khắc phục rò rỉ dữ liệu, và kết quả Run 1/Run 2/Run 3.

### Sự cố phần MovieLens (Phôn)

- Phôn gặp lỗi kỹ thuật, chưa trình bày được đầy đủ; sẽ trình bày lại toàn bộ ở buổi báo cáo tiếp theo.

### Bổ sung baseline Matrix Factorization

- GVHD yêu cầu chạy thêm 1 baseline Matrix Factorization (MF) trên dataset của Triết, chia train/test theo chronological split (không phải random), thay thế vị trí Run 1 hiện tại.
- So sánh chính thức từ nay sẽ là MF (baseline CF thuần) vs Run 2 (KGAT-Chronological) vs Run 3 (T-KGAT), nhằm làm rõ đóng góp thật của Knowledge Graph so với CF thuần, không chỉ so với random split.

### Đào sâu công thức time-attention

- GVHD nhận xét công thức time-attention hiện mới dừng ở mức thử 3 mốc γ rời rạc, cần nghiên cứu chi tiết hơn (cơ sở chọn γ, có thể cân nhắc decay function khác hoặc learned time embedding) trước báo cáo 18/09.
- Cần thống nhất tên gọi/ký hiệu công thức với Phôn để đồng bộ giữa 2 track khi tổng hợp báo cáo chung.

### Định hướng xử lý interaction cũ

- Hướng đi thống nhất: không loại bỏ các interaction cũ khỏi dữ liệu huấn luyện, chỉ giảm trọng số attention của chúng theo thời gian — đúng tinh thần công thức T-KGAT hiện tại (trừ vào attention logit trước softmax), giữ nguyên toàn bộ lịch sử tương tác.

### Thử nghiệm cho phép lặp lại (không dedupe)

- Đề xuất hướng thử nghiệm mới trên dataset của Triết: cho phép các cặp (user, item) lặp lại thay vì dedupe về 1 cặp duy nhất, xem đây là trường hợp phụ thuộc vào khoảng thời gian (time-window) cụ thể.
- Cần làm rõ thêm thiết kế chi tiết (cách chia theo time-window, cách đảm bảo không rò rỉ dữ liệu) trước khi triển khai; chưa áp dụng thay cho pipeline dedupe đã dùng cho Run 1/2/3.

## Quyết định / Chốt (Decisions)

- Ghi nhận kết quả Run 1/Run 2/Run 3 trên Foursquare NYC của Triết là hợp lệ, đạt mục tiêu thực nghiệm cốt lõi của đề tài (T-KGAT cải thiện toàn diện so với KGAT chuẩn trên cùng chronological split).
- Bổ sung 1 baseline Matrix Factorization (MF), chia train/test theo chronological split, thay thế vị trí Run 1 (random/original) trong bảng so sánh chính thức; so sánh mới sẽ là MF vs Run 2 vs Run 3.
- Tiếp tục giữ hướng đi: không bỏ interaction cũ khỏi dữ liệu, chỉ giảm attention của chúng theo thời gian — không đổi kiến trúc T-KGAT hiện tại.
- Cần nghiên cứu sâu hơn công thức time-attention, không dừng ở việc chỉ thử 3 mốc γ rời rạc.
- Thống nhất tên gọi/ký hiệu công thức time-attention với Phôn trước khi tổng hợp báo cáo chung.
- Phôn trình bày lại toàn bộ phần MovieLens (kèm các nâng cấp mới) trong buổi báo cáo tiếp theo.
- Mốc báo cáo tiếp theo với GVHD: 18/09/2026.

## Việc cần làm (Actions và Milestones)

### A. Triết — Matrix Factorization baseline

- Cài đặt/chạy 1 mô hình Matrix Factorization (ví dụ BPR-MF hoặc MF thuần) trên dataset Foursquare NYC (k-core=10 đã có).
- Chia train/test theo chronological split, dùng cùng mốc cutoff đã dùng cho Run 2/Run 3 để đảm bảo so sánh công bằng.
- Ghi nhận Recall@20, NDCG@20, Precision@20, Hit Ratio@20 và training time.
- Thay thế Run 1 (random) bằng Run MF trong bảng so sánh tổng hợp; giữ nguyên Run 2, Run 3 làm đối chứng.

### B. Triết — Nghiên cứu sâu công thức time-attention

- Rà soát lại cơ sở lý thuyết cho việc chọn γ (hiện mới grid thô 0.5/1.0/2.0), cân nhắc decay function khác hoặc learned time embedding nếu cần.
- Thống nhất tên gọi/ký hiệu công thức với Phôn (rà lại `δ_r`, `γ`, `Δt_norm` và các ký hiệu tương ứng bên track MovieLens).

### C. Triết — Thử nghiệm hướng "cho phép lặp lại" (time-window)

- Làm rõ thiết kế cụ thể: cách định nghĩa time-window, cách tránh rò rỉ dữ liệu khi cho phép lặp lại — cần trao đổi thêm trước khi code.
- Triển khai như một biến thể thử nghiệm bổ sung, không thay đổi pipeline dedupe hiện tại của Run 1/2/3 đã chạy.

### D. Phôn — MovieLens

- Khắc phục lỗi kỹ thuật, hoàn thiện lại toàn bộ pipeline và kết quả trên MovieLens.
- Trình bày lại đầy đủ trong buổi báo cáo tiếp theo, kèm các nâng cấp mới (baseline MF, thống nhất tên công thức time-attention).

### E. Cả nhóm

- Thống nhất tên gọi/ký hiệu công thức time-attention giữa 2 track trước ngày 18/09.
- Chuẩn bị báo cáo tổng hợp so sánh Foursquare NYC (Triết) và MovieLens (Phôn) cho buổi 18/09/2026.

## Phân công

| Người phụ trách | Dataset | Nhiệm vụ |
|---|---|---|
| Nguyễn Đăng Phôn | MovieLens | Khắc phục sự cố kỹ thuật, hoàn thiện lại pipeline và kết quả, trình bày lại toàn bộ trong buổi tiếp theo kèm nâng cấp mới (MF baseline, thống nhất tên công thức time-attention). |
| Đởm Quang Minh Triết | Foursquare NYC | Chạy thêm baseline Matrix Factorization (chronological split) thay Run 1; nghiên cứu sâu hơn công thức time-attention và thống nhất tên gọi với Phôn; làm rõ và thử nghiệm hướng cho phép lặp lại theo time-window. |

## Nội dung cho buổi họp tiếp theo (18/09/2026)

- Phôn trình bày lại toàn bộ tiến độ MovieLens, bao gồm các nâng cấp mới.
- Triết trình bày kết quả baseline Matrix Factorization so với Run 2 và Run 3 trên Foursquare NYC.
- Trình bày nghiên cứu chi tiết hơn về công thức time-attention (cơ sở chọn γ / decay function), đã thống nhất tên gọi giữa 2 track.
- Cập nhật tiến độ hướng thử nghiệm "cho phép lặp lại theo time-window" nếu đã có thiết kế cụ thể.
- Trao đổi các vấn đề phát sinh khác trong quá trình thực hiện.
