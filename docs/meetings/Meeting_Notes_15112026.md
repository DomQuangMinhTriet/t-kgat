# Meeting Notes - [15/11/2026]

## Meeting Notes - [15/11/2026]

### Meeting Notes - Khoá luận tốt nghiệp

Ngày: @November 15, 2026
Thời gian: 8:00 p.m. (dời từ 13:00 ngày 14/11)
Hình thức: Online (Google Meet)
Địa điểm / Link: Meet

## Thông tin chung

Mục tiêu buổi họp:

- Báo cáo toàn bộ quy trình trên bộ dữ liệu Last-FM (`lastfm-u25`), kết quả KGAT và T-KGAT (giai đoạn 1 và 2), ba vấn đề nhóm tự phát hiện ở dữ liệu, và bản dữ liệu thử `lastfm-u25-latest`.
- Xin ý kiến cô về ba câu hỏi: bộ dữ liệu chính thức, cách sửa giai đoạn 2, và việc đóng băng kết quả.

Thành phần tham dự:

- GVHD: cô Nguyễn Ngọc Minh Châu
- Sinh viên:
  - Nguyễn Đăng Phôn - 23127451
  - Đởm Quang Minh Triết - 23127131
- Khác: Không có

Tóm tắt nhanh

- Cô góp ý tập trung vào **cách lấy dữ liệu** (không lấy ngẫu nhiên theo user mà cắt theo cửa sổ thời gian; dựng thẳng từ LFM-1b; tách bước cắt và bước xử lý) và **cách huấn luyện** (ít epoch hơn, learning rate giảm dần, huấn luyện lại trên train + val rồi mới áp dụng lên test).
- Về mô hình: **kết hợp cả hai giai đoạn** (Δt trên cạnh tương tác và trên triple KG) trong một mô hình; không phải quan hệ nào trong KG cũng có giá trị, nên cắt bớt quan hệ.
- Về timestamp của cặp: không đặt bài toán gợi ý sản phẩm mới; lấy **lần nghe sớm nhất** như cách làm theo KGAT. Bản `lastfm-u25-latest` (lần nghe muộn nhất) vì vậy dừng.
- Nhóm chốt phân công: Phôn dựng lại dữ liệu từ LFM-1b và viết công thức Δt chung cho hai giai đoạn; Triết làm phần tối ưu (cắt quan hệ, giảm epoch, lịch learning rate, huấn luyện lại trên train + val) trên một tập nhỏ dựng trước, rồi ghép với dữ liệu mới của Phôn.

## Nội dung đã báo cáo

- Dữ liệu `lastfm-u25`: nguồn (tập cặp KGAT, timestamp LFM-1b lần nghe sớm nhất, ngày fact từ Freebase 2015), cách chia t1 = 29/04/2014, t2 = 07/06/2014, 5-core trên train, số liệu train/val/test, KG 152.976 triple / 9 quan hệ, hạn chế của tập cặp (chỉ cặp còn nghe từ 01/04/2014).
- Kết quả (5 seed, chọn trên val, test một lần, Wilcoxon cặp + Holm): T-KGAT không khác KGAT, không khác đối chứng hoán vị; KGAT không khác KGAT không KG; giai đoạn 2 cũng không. Mọi kết quả KGAT/T-KGAT còn sơ bộ (chưa tag `v0.2-kgat`).
- Ba vấn đề nhóm tự thấy: (A) tập cặp chọn bằng thông tin sau mốc; (B) tập con 25% user bỏ 28% positive val/test vì cold item; (C) giai đoạn 2 có lỗi thiết kế chuẩn hóa (`dt_norm` của fact bị chặn ở 1 theo cửa sổ tương tác).
- Bản `lastfm-u25-latest` của Phôn: cùng user, timestamp là lần nghe muộn nhất; kết quả và lý do nó chỉ đổi hình thức thiên lệch.

## Góp ý của GVHD (ghi theo note tại buổi họp)

1. Lấy ngẫu nhiên (theo user) có thể ảnh hưởng kết quả → cắt theo cửa sổ thời gian: sắp xếp theo thời gian trước rồi mới cắt.
2. Đừng chạy một lèo "cắt dữ liệu + xử lý" trong một file → cắt riêng và xử lý riêng.
3. Hướng tới kết quả tốt nhất trong số lần chạy ít nhất.
4. Giảm số epoch xuống.
5. Learning rate giảm dần thay vì cố định.
6. Train + val → lấy trọng số mô hình: khi đã có siêu tham số thì cố định lại, xác định cần khoảng bao nhiêu epoch là tối ưu, rồi chạy trên train + val → mô hình mới → áp dụng trên test.
7. Gợi ý sản phẩm mới? Không có sản phẩm mới → lấy timestamp sớm nhất như KGAT.
8. Feature engineering: không phải relation nào cũng có giá trị.
9. Thử trên một bộ nhỏ, bỏ knowledge, để xem KG có đóng góp không.
10. Dữ liệu ít nhất để ra mô hình tốt nhất.
11. Kết hợp cả hai giai đoạn.
12. Lấy từ LFM-1b làm trực tiếp.

## Cách nhóm hiểu và việc tương ứng

| Góp ý | Việc trong repo | Quyết định bị thay |
|---|---|---|
| 1, 12 | Tập cặp dựng thẳng từ `lfm1b-tracks.inter`, giữ sự kiện trong cửa sổ N tháng cuối (sắp xếp theo thời gian rồi cắt), timestamp cặp = lần nghe sớm nhất; t1/t2 là quantile 0,8/0,9 trong cửa sổ | 0013 (tập con theo user), phần "tập cặp của KGAT" trong 0012 |
| 2 | Script cắt (`01b_cut_lastfm.py`) tách khỏi script chia (`02_build_splits.py`); file trung gian có SHA-256 | quy trình dựng dữ liệu của 0012 |
| 6 | Sau khi chọn điểm lưới và số epoch trên val: huấn luyện lại trên train ∪ val đúng số epoch đó, graph gồm cạnh train + val và KG tại t2, rồi chấm test một lần; áp dụng cho mọi mô hình | 0002 mục 7 |
| 7 | Giữ `timestamp_policy: earliest`; bản `latest` và nháp 0015 dừng, không đưa vào repo | 0015 (nháp) |
| 11 | Một mô hình T-KGAT có Δt trên cả cạnh tương tác lẫn triple KG, hai thang chuẩn hóa riêng, sửa lỗi chặn ở 1 | 0009 mục 7–10 |
| 8 | Phân tích 9 quan hệ, chọn tập giữ lại theo tiêu chí định trước, chỉ xem val; biến thể KG `pruned` | — (biến thể mới) |
| 3, 4, 5, 10 | Ngân sách epoch từ phân bố `best_epoch` của mọi run đã có; lịch learning rate chung cho mọi mô hình | `fixed` trong configs, trainer |
| 9 | Cổng trên tập nhỏ: BPR-MF, KGAT, KGAT không KG chỉ trên val trước khi chạy đủ | — |

Tất cả gom vào một quyết định mới (0016), ghi rõ các thay đổi được đưa ra **sau khi đã thấy điểm test** của ml-100k, ml-1m-u10 và lastfm-u25; các kết quả cũ giữ ở phụ lục.

## Phân công và mốc

| Ai | Việc | Mốc |
|---|---|---|
| Triết | Biên bản, đóng PR #42, dừng PR #44, quyết định 0016 | 17/11 |
| Triết | Tập nhỏ; bảng epoch; lịch learning rate; huấn luyện lại train + val; cắt quan hệ; cổng tập nhỏ | 21/11 (Triết bận 2–3 ngày đầu tuần) |
| Phôn | Script cắt LFM-1b thô (tách riêng), loader, chạy 02 → 06 → 07 trên dữ liệu mới | 24/11 |
| Phôn | Tài liệu công thức Δt chung hai giai đoạn (Triết hiện thực, Phôn review) | 20/11 |
| Cả hai | Ghép, chạy chính thức (laptop + Kaggle), bảng kết quả mới, tái lập, tag `v0.2-kgat`, báo cô | cuối tháng 11 |
| Cả hai | Bắt đầu viết khóa luận (chương 1–3) song song | từ 18/11 |

## Việc cần hỏi/xác nhận thêm với cô

- Mẫu khóa luận chính thức của khoa và mốc nộp bản thảo các chương đầu.
- Cách chọn độ dài cửa sổ thời gian (N tháng) và tiêu chí giữ user: nhóm đề xuất chọn trước theo số sự kiện và giờ chạy, không nhìn test.
