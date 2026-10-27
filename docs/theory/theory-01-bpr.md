# Chủ đề 1: Hệ gợi ý, phản hồi ngầm và BPR

Người trình bày: Triết · Người hỏi vặn: Phôn · Ngày teach-back: ../10/2026

## Ý chính
- Hệ gợi ý top-K trả cho mỗi user K item chưa tương tác mà user có khả năng thích nhất. Đây là bài toán xếp hạng, không phải dự đoán số sao: chỉ cần item user thích đứng trên item không thích.
- Phản hồi ngầm chỉ cho biết user có tương tác (xem, click, mua), không có thang điểm. Không tương tác không có nghĩa là không thích, vì user có thể chưa từng thấy item. Vì vậy không có nhãn âm thật, và không dùng được sai số bình phương (r - ŷ)².
- MovieLens là dữ liệu tường minh (rating 1-5). Nhóm chuyển sang dạng ngầm bằng quy tắc: rating >= 4 là tương tác dương (`min_rating: 4` trong `configs/data/*.yaml`); rating thấp hơn không được dùng làm dương.
- Matrix Factorization: mỗi user và item có một vector embedding dài `dim`; điểm ŷ_ui = p_u · q_i. Tích vô hướng càng lớn thì item càng đứng cao trong danh sách của user.
- BPR giả định user u thích item đã tương tác i hơn item chưa tương tác j. Mỗi mẫu huấn luyện là một bộ ba (u, i, j). Loss đẩy ŷ_ui - ŷ_uj lớn lên.
- Item âm j được lấy mẫu ngẫu nhiên trong các item user chưa tương tác trên train; không bao giờ dùng val hay test.
- Vòng huấn luyện: mỗi epoch đi qua mọi tương tác dương trong train, gom batch (u, i, j), tính loss, cập nhật bằng Adam. `dim` là độ dài embedding, `lr` là bước cập nhật, `l2` là hệ số phạt chống overfit.
- Overfit: loss train vẫn giảm nhưng Recall@20 trên val ngừng tăng hoặc giảm. Repo xử lý bằng dừng sớm theo val (đo mỗi 5 epoch, patience 5). Val dùng để chọn epoch và tham số; test chỉ chấm một lần, nếu dùng test để chọn thì có rò rỉ.
- MostPop gợi ý cùng một danh sách K item nhiều tương tác dương nhất trong train cho mọi user; bản recent chỉ đếm N ngày cuối. Là baseline bắt buộc vì trên MovieLens nó thường rất mạnh.

## Công thức
x̂_uij = ŷ_ui - ŷ_uj

L = Σ over (u,i,j) of -ln σ(x̂_uij) + λ‖Θ‖²

-ln σ(x) = -ln(1/(1 + e^(-x))) = ln(1 + e^(-x)) = softplus(-x), với softplus(z) = ln(1 + e^z). Code dùng `F.softplus(-(x_ui - x_uj))`.

## Số liệu (ML-1M, test, 952 user; từ `results/registry.csv`)
MostPop Recall@20 = 0,0902; MostPop 7 ngày = 0,1060; BPR-MF = 0,1194 ± 0,0031 (5 seed). Khoảng cách nhỏ và chưa có kiểm định thống kê. MostPop recent thắng MostPop cho thấy chỉ riêng thông tin gần đây đã có tín hiệu; điều đó chưa chứng minh mô hình có yếu tố thời gian sẽ tốt hơn (giả thuyết RQ3, cần kiểm bằng hoán vị timestamp). Mô hình có thời gian sau này phải thắng MostPop recent, không chỉ MostPop.

## Nằm ở đâu trong code
- Ngưỡng rating: `SplitConfig.min_rating` (`src/tkgat/data/split.py`), `configs/data/*.yaml`
- BPR-MF: `BPRMF.forward` (`src/tkgat/models/baselines.py`)
- Lấy mẫu âm: `NegativeSampler` (`src/tkgat/train/bpr.py`)
- Dừng sớm theo val: `fit_bpr` (`src/tkgat/train/bpr.py`)
- MostPop: `MostPop` (`src/tkgat/models/baselines.py`)
- Tham số huấn luyện: `configs/model/bprmf.yaml`

## 3 câu tự kiểm tra
1. Vì sao không dùng sai số bình phương trên rating cho phản hồi ngầm? Vì không có nhãn âm thật: ô trống có thể là chưa thấy chứ không phải không thích.
2. Chứng minh -ln σ(x) = softplus(-x). Xem phần Công thức.
3. Vì sao quyết định dừng phải dựa trên val, không phải test? Test chỉ chấm một lần cho cấu hình đã chọn; chọn trên test làm số bị thổi phồng.

## Teach-back với Phôn (điền sau buổi giảng)
Ngày: ..; 3 câu Phôn hỏi vặn: ..; trả lời thế nào: ..
