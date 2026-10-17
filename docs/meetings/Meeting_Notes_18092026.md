# Meeting Notes - [18/09/2026]

## Meeting Notes - [18/09/2026]

### Meeting Notes - Khoá luận tốt nghiệp

Ngày: @September 18, 2026
Thời gian: 8:00-9:00 a.m.
Hình thức: Online (Google Meet)
Địa điểm / Link: Meet

## Thông tin chung

Mục tiêu buổi họp:

- Phôn trình bày lại toàn bộ tiến độ MovieLens sau khi khắc phục sự cố kỹ thuật ở buổi 21/08.
- Triết trình bày kết quả đào sâu trên Foursquare NYC (Section06: baseline MF, mở rộng grid gamma/decay function, thử nghiệm time-window) và kết quả áp dụng lại toàn bộ pipeline trên dataset thứ hai — Amazon CDs_and_Vinyl (Section07) — để kiểm chứng chéo.
- Cô Châu nhận xét kết quả và giao thêm định hướng nghiên cứu tiếp theo cho cả 2 track.

Thành phần tham dự:

- GVHD: cô Nguyễn Ngọc Minh Châu
- Sinh viên:
  - Nguyễn Đăng Phôn - 23127451
  - Đởm Quang Minh Triết - 23127131
- Khác: Không có

Tóm tắt nhanh

- Phôn đã khắc phục lỗi kỹ thuật của buổi 21/08 và trình bày lại đầy đủ tiến độ MovieLens.
- Triết trình bày kết quả 3 yêu cầu đã giao ở buổi 21/08 (baseline MF, đào sâu grid gamma/decay function, thử nghiệm time-window) trên Foursquare NYC (Section06), đồng thời chủ động áp dụng lại toàn bộ pipeline trên dataset thứ hai — Amazon CDs_and_Vinyl (Section07) — để kiểm chứng chéo. Phát hiện quan trọng nhất: độ phong phú của Knowledge Graph là yếu tố quyết định KGAT/T-KGAT có vượt qua CF thuần (Matrix Factorization) hay không.
- Cô Châu giao thêm định hướng cho cả 2 track: mở rộng khai thác các mối quan hệ trong Knowledge Graph (đặc biệt là giá trị rating), thử đưa timestamp vào như một relation, thử nghiệm trước trên dataset dung lượng thấp trước khi mở rộng, đồng thời tạm thời tập trung vào hướng "không lặp lại" để thống nhất cách báo cáo giữa 2 người.

## Tiến độ hiện tại

Đã hoàn thành

- Phôn: khắc phục lỗi kỹ thuật gặp phải ở buổi 21/08, hoàn thiện và trình bày lại toàn bộ pipeline MovieLens.
- Triết — Section06 (Foursquare NYC deep-dive): hoàn thành đúng 3 yêu cầu từ buổi 21/08 — (A) baseline Matrix Factorization vượt trội tuyệt đối so với KGAT/T-KGAT (Recall gấp ~2x, train nhanh ~12 lần); (B) mở rộng grid γ (9 điểm, 0.25→4.0) + so sánh decay function, chưa tìm được γ tối ưu tuyệt đối, exponential decay lợi cho NDCG/HitRatio nhưng đánh đổi Recall; (C) thử nghiệm time-window (τ=24h, dựa số liệu thực tế) — T-KGAT vẫn cải thiện so KGAT.
- Triết — Section07 (Amazon CDs_and_Vinyl): áp dụng lại toàn bộ pipeline trên dataset thứ hai có Knowledge Graph phong phú hơn (3 quan hệ thay vì 1) để kiểm chứng chéo — kết quả đối lập Foursquare: KGAT/T-KGAT vượt MF (trừ NDCG); time-window không áp dụng được (đo thực tế 0% cặp lặp lại).
- Kết luận nghiên cứu chính: độ phong phú của Knowledge Graph quyết định KGAT/T-KGAT có vượt CF thuần (MF) hay không — KG nghèo (Foursquare) MF thắng, KG phong phú hơn (Amazon) KGAT/T-KGAT thắng đa số metric; T-KGAT cải thiện nhất quán về hướng so với KGAT chuẩn nhưng mạnh/yếu khác nhau tuỳ domain.

Đang thực hiện / Chuẩn bị thực hiện

- Cả nhóm: mở rộng và khai thác nhiều hơn các mối quan hệ trong Knowledge Graph, đặc biệt là đưa giá trị của rating vào mô hình (hiện interaction đang được coi là tín hiệu ngầm định/nhị phân, chưa dùng giá trị rating cụ thể).
- Cả nhóm: tìm hiểu cách đưa timestamp vào như một relation trong Knowledge Graph — hướng mới, khác với cách dùng hiện tại (timestamp chỉ đang dùng để điều chỉnh attention qua công thức T-KGAT) — cần thiết kế cụ thể.
- Cả nhóm: nên thử nghiệm các hướng mở rộng trên dataset/phiên bản dung lượng thấp trước khi chạy trên tập dữ liệu lớn hơn.
- Cả nhóm: tạm thời tập trung vào hướng "không lặp lại" (dedupe) để thống nhất cách báo cáo và so sánh giữa 2 track; hướng "cho phép lặp lại theo time-window" (đã có kết quả bước đầu ở Section06) sẽ được quay lại tìm hiểu sâu hơn sau.
- Triết: rà soát lại 2 dataset đang dùng (Foursquare NYC, Amazon CDs_and_Vinyl) xem còn khai thác thêm được thông tin/attribute nào không; gợi ý cụ thể từ cô: với Amazon, có thể dùng LLM để lọc dữ liệu bẩn từ metadata thô và trích xuất thêm attribute mới, làm phong phú thêm Knowledge Graph.

## Vướng mắc / Rủi ro

- Tốc độ training chậm do dataset lớn — đặc biệt rõ ở Amazon CDs_and_Vinyl (cùng quy mô user/item với Foursquare nhưng cần nhiều epoch hơn nhiều để hội tụ; tổng thời gian train của Section06+07 ước tính khoảng 15-20 giờ compute, CPU-only).
- Kết quả cải thiện của T-KGAT so với KGAT hiện tại chưa đủ để khẳng định chắc chắn: mới chạy 1 seed/cấu hình, chưa kiểm định ý nghĩa thống kê; ở Amazon, đường cong theo γ dao động không đơn điệu, một số giá trị γ còn cho kết quả kém hơn cả baseline không-temporal — cần tìm hiểu và chứng minh thêm.
- Grid γ chưa xác định được chặn trên tối ưu tuyệt đối (Recall ở Foursquare còn tăng ở γ=4.0, là đầu trên của grid đã thử).

## Nội dung trao đổi (theo chủ đề)

### Báo cáo lại MovieLens (Phôn)

- Trình bày lại toàn bộ sau khi khắc phục sự cố kỹ thuật ở buổi 21/08, kết quả có phạm vi tương tự Section05/06 (Foursquare) của Triết.

### Báo cáo đào sâu Foursquare NYC — Section06 (Triết)

- Trình bày kết quả đúng 3 yêu cầu từ buổi 21/08: baseline Matrix Factorization, mở rộng grid gamma/decay function, thử nghiệm time-window.
- Điểm nhấn: MF vượt trội tuyệt đối so với KGAT/T-KGAT trên Foursquare — nguyên nhân là do Knowledge Graph ở đây chỉ có 1 quan hệ (`hasCategory`), không đủ phong phú để bù lại chi phí kiến trúc phức tạp hơn.

### Kiểm chứng chéo trên Amazon CDs_and_Vinyl — Section07 (Triết)

- Áp dụng lại toàn bộ pipeline (gốc + 3 phần mở rộng A/B/C) trên dataset thứ hai, có Knowledge Graph phong phú hơn (3 quan hệ thay vì 1).
- Kết quả đối lập hoàn toàn với Foursquare: KGAT/T-KGAT vượt MF (trừ NDCG) — củng cố giả thuyết "độ phong phú của Knowledge Graph quyết định KGAT có đáng dùng hơn CF thuần hay không".
- Time-window không áp dụng được trên Amazon (0% cặp lặp lại) — quyết định bỏ thí nghiệm dựa trên số liệu đo thực tế, không phải bỏ qua tuỳ tiện.

### Định hướng mở rộng Knowledge Graph (cả 2 track)

- Cô Châu nhận xét cần mở rộng và khai thác nhiều hơn các mối quan hệ trong Knowledge Graph, đặc biệt là giá trị của rating.
- Gợi ý thử đưa timestamp vào như một relation trong đồ thị, thay vì chỉ dùng để điều chỉnh attention như hiện tại.
- Nên thử nghiệm trước trên dataset/phiên bản dung lượng thấp trước khi mở rộng ra dataset lớn hơn.

### Thống nhất hướng "không lặp lại" giữa 2 track

- Cô Châu nhắc cả 2 tạm thời tập trung vào hướng dedupe (không lặp lại) để thống nhất cách báo cáo và so sánh giữa MovieLens (Phôn) và các dataset của Triết; hướng "cho phép lặp lại theo time-window" (đã thử nghiệm bước đầu ở Section06) sẽ được quay lại tìm hiểu sâu hơn sau.

### Định hướng khai thác thêm dữ liệu (riêng Triết)

- Rà soát lại 2 dataset hiện có (Foursquare NYC, Amazon CDs_and_Vinyl) xem còn khai thác được thêm attribute/thông tin gì không.
- Gợi ý cụ thể: với Amazon CDs_and_Vinyl, có thể dùng LLM để lọc dữ liệu bẩn (dirty data) từ metadata thô và trích xuất thêm attribute mới, làm phong phú thêm Knowledge Graph.

## Quyết định / Chốt (Decisions)

- Ghi nhận kết quả Section06 (Foursquare deep-dive) và Section07 (Amazon CDs_and_Vinyl) của Triết là hợp lệ, trả lời đầy đủ 3 yêu cầu từ buổi 21/08, đồng thời đưa ra phát hiện nghiên cứu mới về vai trò của độ phong phú Knowledge Graph.
- Ghi nhận Phôn đã hoàn thành trình bày lại toàn bộ tiến độ MovieLens sau sự cố kỹ thuật.
- Cả 2 track tạm thời thống nhất tập trung vào hướng "không lặp lại" (dedupe) để đồng bộ cách báo cáo; hướng time-window sẽ quay lại tìm hiểu sau.
- Cả 2 track cần mở rộng khai thác các mối quan hệ trong Knowledge Graph (đặc biệt giá trị rating) và thử đưa timestamp vào như một relation; nên thử nghiệm trước trên dataset dung lượng nhỏ.
- Triết cần rà soát thêm khả năng khai thác 2 dataset hiện có, cân nhắc dùng LLM để lọc dữ liệu bẩn / trích xuất thêm attribute cho Amazon CDs_and_Vinyl.
- Buổi báo cáo tiếp theo: chưa chốt lịch cụ thể, sẽ cập nhật sau.

## Việc cần làm (Actions và Milestones)

### A. Cả nhóm — Mở rộng Knowledge Graph

- Nghiên cứu cách đưa giá trị rating vào mô hình (thay vì chỉ coi interaction là tín hiệu nhị phân/ngầm định).
- Thiết kế cách đưa timestamp vào như một relation trong Knowledge Graph (khác với cách dùng attention hiện tại).
- Thử nghiệm trước trên dataset/phiên bản dung lượng thấp trước khi mở rộng sang tập dữ liệu lớn hơn.

### B. Cả nhóm — Thống nhất hướng báo cáo

- Tạm dừng phát triển thêm hướng "cho phép lặp lại theo time-window", tập trung vào hướng dedupe (không lặp lại) để 2 track dùng chung 1 protocol khi so sánh.
- Lưu lại hướng time-window (đã có thiết kế và kết quả bước đầu ở Section06) để quay lại tìm hiểu sau.

### C. Triết — Khai thác thêm dữ liệu

- Rà soát Foursquare NYC và Amazon CDs_and_Vinyl xem còn attribute/thông tin nào chưa khai thác.
- Thử nghiệm dùng LLM để lọc dữ liệu bẩn và trích xuất thêm attribute mới cho Amazon CDs_and_Vinyl.

### D. Phôn — MovieLens

- Tiếp tục áp dụng các định hướng mới (mở rộng quan hệ Knowledge Graph, giá trị rating, timestamp-as-relation, thống nhất hướng dedupe) lên MovieLens.

### E. Cả nhóm

- Chuẩn bị nội dung báo cáo tổng hợp cho buổi họp tiếp theo (chưa có lịch cụ thể).

## Phân công

| Người phụ trách | Dataset | Nhiệm vụ |
|---|---|---|
| Nguyễn Đăng Phôn | MovieLens | Hoàn thiện lại pipeline sau sự cố kỹ thuật; áp dụng các định hướng mới (mở rộng quan hệ Knowledge Graph, giá trị rating, timestamp-as-relation, thống nhất hướng dedupe, thử nghiệm trên dataset nhỏ trước khi mở rộng). |
| Đởm Quang Minh Triết | Foursquare NYC & Amazon CDs_and_Vinyl | Đã hoàn thành Section06 (MF, mở rộng gamma/decay, time-window) và Section07 (kiểm chứng chéo); tiếp tục rà soát khai thác thêm dữ liệu (gợi ý dùng LLM cho Amazon), áp dụng định hướng mới về relations/rating/timestamp. |

## Nội dung cho buổi họp tiếp theo

- Chưa có lịch cụ thể, sẽ cập nhật sau.
- Dự kiến nội dung: tiến độ mở rộng Knowledge Graph (rating, timestamp-as-relation) trên cả 2 track; kết quả thử nghiệm trên dataset dung lượng nhỏ trước khi mở rộng; tiến độ khai thác thêm dữ liệu của Triết (Amazon + LLM); cập nhật về việc thống nhất hướng "không lặp lại" giữa 2 track.
