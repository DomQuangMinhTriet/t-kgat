# Chủ đề 7: Kiểm định thống kê cho thí nghiệm gợi ý

Người trình bày: Triết và Phôn · Ngày teach-back: ../10/2026

Bản nháp soạn từ `docs/study/07-thong-ke.md` và code `src/tkgat/eval/stats.py`, `src/tkgat/analysis/significance.py`, `src/tkgat/analysis/rq3.py`; cả hai đọc lại và điền phần teach-back.

## Ý chính
- Hai con số như 0,162 và 0,158 chưa nói được gì nếu không biết chạy lại có còn không và chênh lệch có lớn hơn dao động giữa các user không. Với test chỉ có 51 user (ML-100K), dao động rất lớn.
- Có hai nguồn biến thiên khác nhau. Ngẫu nhiên khi huấn luyện (khởi tạo, lấy mẫu âm, thứ tự batch) xử lý bằng 5 seed (2019–2023), báo cáo trung bình ± độ lệch chuẩn **mẫu** (ddof = 1; ddof mặc định của numpy là sai ở đây). Mẫu user (test chỉ là một mẫu) xử lý bằng kiểm định cặp theo user.
- Kiểm định cặp: hai mô hình được chấm trên cùng tập user, mỗi user cho một cặp điểm; xét hiệu d_u. Kiểm định độc lập bỏ qua việc cùng một user, nên coi phương sai của hiệu lớn hơn thực tế và kém nhạy.
- Mặc định là Wilcoxon signed-rank: xếp hạng |d_u|, cộng hạng của d_u > 0, không cần giả định chuẩn nên hợp với Recall theo user (lệch, nhiều số 0). User có d_u = 0 bị bỏ, vì vậy phải báo cả n_pairs lẫn n_nonzero. Nhóm đặt cảnh báo khi dưới 100 user hoặc dưới 30 chênh lệch khác 0; đây là ngưỡng **tự đặt** (heuristic), không phải chuẩn thống kê, và báo cáo phải ghi như vậy.
- Điểm theo user của một mô hình là trung bình qua các seed chung rồi mới so cặp. Hệ quả: p-value nói về quần thể user, không nói về độ ổn định giữa các seed. Nhóm ghi thêm p nhỏ nhất và lớn nhất khi so từng seed để thấy mức dao động.
- So sánh nhiều lần làm tăng nguy cơ dương tính giả: với m kiểm định ở mức 0,05 thì xác suất có ít nhất một "có ý nghĩa" do may mắn là 1 − 0,95^m. Holm–Bonferroni hiệu chỉnh: sắp p tăng dần, bác bỏ H_(k) nếu p_(j) ≤ α/(m − j + 1) với mọi j ≤ k.
- Quy ước của repo: họ Holm gồm các phép so sánh trong **cùng bộ dữ liệu và cùng metric**. Bảng baseline có ba cặp (BPR-MF với MostPop, BPR-MF với MostPop gần đây, MostPop với MostPop gần đây). Bảng RQ3 có ba phép so sánh cố định (T-KGAT với KGAT, T-KGAT thật với hoán vị, KGAT với KGAT không KG); một phép chưa chạy được tính p = 1 để p-value của các phép khác không đổi theo việc đã chạy gì.
- Báo cáo đúng cách: trung bình ± độ lệch chuẩn qua 5 seed, chênh lệch so với đối chứng, p Holm, n_pairs và n_nonzero, và ghi rõ cấu hình nào được chọn trên val (test chấm một lần). Không chọn seed đẹp nhất. Kết quả âm tính vẫn là kết quả và phải đi kèm giải thích.
- Khi so trên nhiều dataset, Demšar (2006) khuyên kiểm định trên các dataset hoặc Friedman + Nemenyi; nhóm chỉ có 2–3 bộ dữ liệu nên không đủ, báo cáo riêng từng bộ và nói rõ giới hạn.

## Công thức
Hiệu theo user: d_u = m_A(u) − m_B(u).

Paired t: t = d̄ / (s_d/√n), cần d_u xấp xỉ chuẩn.

Xác suất có ít nhất một dương tính giả: 1 − (1 − α)^m. Ví dụ m = 10, α = 0,05: 1 − 0,95¹⁰ ≈ 0,40.

Ví dụ Holm với p = {0,01; 0,02; 0,04}, α = 0,05, m = 3: ngưỡng lần lượt 0,05/3 ≈ 0,0167, 0,05/2 = 0,025, 0,05/1 = 0,05. 0,01 ≤ 0,0167 nên bác bỏ; 0,02 ≤ 0,025 nên bác bỏ; 0,04 ≤ 0,05 nên bác bỏ. Cả ba đều bị bác bỏ. Với Bonferroni thuần (ngưỡng 0,0167 cho tất cả) chỉ bác bỏ được kết quả đầu, nên Holm mạnh hơn.

## Dữ liệu của nhóm
- Mọi bảng so sánh được sinh từ `results/registry.csv` và các file điểm theo user `results/runs/<run_name>/<tag>_seed<seed>.npz`: `scripts/09_significance.py` cho baseline, `scripts/10_rq3_table.py` cho KGAT và T-KGAT. Không con số nào chép tay.
- Registry chỉ ghi thêm. `registry.load()` trả về dòng mới nhất của mỗi khóa (dataset, split, model, config, seed, phase); dòng mang nhãn `-dirty` bị bỏ khỏi bảng chính thức.
- ML-100K có 51 user test và ML-1M có 952 user test; kết luận mạnh dựa trên ML-1M. ml-1m-u10 (10% user của ML-1M) có 95 user test và luôn được gọi đúng tên, không gọi là ML-1M.
- Mọi dòng so sánh trên ML-100K và ml-1m-u10 đều có cảnh báo "dưới 100 user"; vì vậy một p không nhỏ ở đó chưa phải bằng chứng cho "không có khác biệt".

## Nằm ở đâu trong code
- Wilcoxon, paired t, bootstrap, Holm: `src/tkgat/eval/stats.py`
- Ngưỡng cảnh báo và bảng baseline: `src/tkgat/analysis/significance.py`, `scripts/09_significance.py`
- Bảng RQ3 và họ Holm cố định: `src/tkgat/analysis/rq3.py`, `scripts/10_rq3_table.py`
- Ghi từng lần chạy: `src/tkgat/registry.py`

## 3 câu tự kiểm tra
1. Vì sao dùng kiểm định cặp theo user thay vì so hai trung bình? Vì hai mô hình chấm trên cùng user, nên hiệu theo user loại bỏ độ khó riêng của từng user; kiểm định độc lập bỏ phí thông tin đó.
2. Std qua 5 seed tính bằng ddof nào, và vì sao? ddof = 1 (độ lệch chuẩn mẫu), vì 5 seed là một mẫu của các lần chạy có thể có; ddof = 0 của numpy đánh giá thấp độ dao động.
3. Một p Holm = 0,3 với 51 user nghĩa là gì? Không đủ bằng chứng để nói có khác biệt, và cũng chưa chứng minh là không có: với 51 user và dưới 30 chênh lệch khác 0 kiểm định có ít sức mạnh.

## Teach-back
Ngày: ..; 3 câu hỏi vặn: ..; trả lời thế nào: ..
