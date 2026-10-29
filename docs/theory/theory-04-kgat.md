# Chủ đề 4: KGAT chi tiết

Người trình bày: Triết · Người hỏi vặn: Phôn · Ngày teach-back: ../10/2026

Bản nháp soạn từ `docs/study/04-kgat.md`, `docs/design-kgat.md` và code `src/tkgat/models/kgat.py`; Triết đọc lại, sửa bằng lời của mình và điền phần teach-back.

## Ý chính
- KGAT = CKE (nhúng KG bằng TransR) cộng lan truyền nhiều bước trên đồ thị gộp (CKG) có attention. Ba khối: (1) embedding của user, item, entity, relation; (2) các lớp lan truyền có attention; (3) dự đoán bằng tích vô hướng của vector đã nối.
- Hàng xóm của một node h là mọi cạnh (h, r, t) trong CKG, kể cả cạnh ngược. Với user, đó là các item user đã tương tác; với item, đó là các user đã tương tác và các thuộc tính trong KG.
- Attention cho từng cạnh được tính trong không gian của quan hệ r (nhờ W_r) rồi chuẩn hóa softmax trên **toàn bộ** hàng xóm của h. Trực giác: hàng xóm nào "khớp" với h dưới quan hệ r thì được chú ý nhiều hơn.
- Aggregator mặc định là Bi-Interaction: một nhánh tổng (e_h + e_N) và một nhánh nhân từng phần tử (e_h ⊙ e_N). Nhánh nhân cho mô hình thấy mức độ giống nhau giữa node và hàng xóm, nhánh tổng thì không.
- Sau L lớp, node nhận thông tin từ các node cách nó tối đa L bước. Với L = 3 và MovieLens: user → phim đã xem → đạo diễn → phim khác của đạo diễn. Vector cuối của một node là phép **nối** embedding của mọi lớp (0..L), để giữ thông tin ở mọi bậc.
- Huấn luyện luân phiên: một bước cho loss KG (TransR), một bước cho loss CF (BPR). Attention được xem là hằng số trong bước CF và chỉ được cập nhật lại từ embedding hiện tại sau mỗi epoch.
- KGAT trong repo viết bằng PyTorch, không chạy code TensorFlow 1.x của tác giả; code gốc chỉ là tài liệu tham chiếu. Đây là lý do hai điều không giống paper: số của nhóm không so trực tiếp với số trong paper, và KGAT trên GPU không tái lập từng bit (xem mục "Dữ liệu của nhóm").

## Công thức
TransR (giống chủ đề 3): g(h, r, t) = ‖W_r e_h + e_r − W_r e_t‖², loss L_KG = Σ −ln σ(g(h, r, t') − g(h, r, t)).

Attention trước chuẩn hóa: π(h, r, t) = (W_r e_t)ᵀ tanh(W_r e_h + e_r). Sau đó π(h, r, t) ← exp π(h, r, t) / Σ_{(h, r', t') ∈ N_h} exp π(h, r', t').

Hàng xóm gộp: e_{N_h} = Σ_{(h, r, t) ∈ N_h} π(h, r, t) e_t.

Bi-Interaction: f(e_h, e_{N_h}) = LeakyReLU(W₁(e_h + e_{N_h})) + LeakyReLU(W₂(e_h ⊙ e_{N_h})).

Dự đoán: e*_u = e_u⁽⁰⁾ ‖ e_u⁽¹⁾ ‖ … ‖ e_u⁽ᴸ⁾, tương tự cho item, và ŷ(u, i) = e*_uᵀ e*_i.

Loss tổng: L = L_KG + L_CF + λ‖Θ‖². L_CF là BPR (chủ đề 1) trên ŷ. Ví dụ số cho softmax: một node có ba cạnh với π = (2, 1, 0) cho trọng số ≈ (0,665; 0,245; 0,090), cộng lại bằng 1.

## Dữ liệu của nhóm
- Cấu hình chính thức (`configs/model/kgat.yaml`): dim 64, kge_dim 64, ba lớp [64, 32, 16], mess_dropout 0,1, batch CF 1.024 và batch KG 2.048, tối đa 300 epoch, đánh giá mỗi 5 epoch, patience 10, chọn theo Recall@20 trên val. Lưới (lr, l2) có 4 điểm; điểm được chọn theo trung bình Recall@20 val qua 5 seed (2019–2023), test chấm một lần.
- CKG của nhóm: cạnh tương tác chỉ từ train; KG cho train/val dựng tại t1, KG cho test dựng tại t2 và bỏ các triple có đuôi chưa từng xuất hiện ở t1 (embedding của entity chưa được huấn luyện không được vào điểm test). Không huấn luyện lại trên train + val (quyết định 0002).
- KG hiện có hai nguồn: fact Wikidata (bản chụp 2026) và metadata MovieLens (thể loại, năm, thập niên; quyết định 0011). Kết quả KGAT trước 0011 chạy trên KG chỉ có Wikidata và nằm ở phụ lục của `docs/rq3-table.md`.
- KGAT và T-KGAT không tái lập từng bit trên GPU vì `torch.sparse.mm` không tất định; hai lần chạy cùng seed có thể khác nhau, nên coi chênh lệch đó là nhiễu seed. BPR-MF thì tái lập được.
- Điểm yếu của KGAT mà đề tài nhắm tới: không có thời gian; KG tĩnh có thể chứa tri thức tương lai; chia ngẫu nhiên của paper đo một bài toán dễ hơn (chủ đề 2 và 5).
- Số liệu mô hình lấy từ `results/registry.csv` qua `docs/rq3-table.md`, không chép vào ghi chú này.

## Nằm ở đâu trong code
- Đồ thị CKG và các kiểm tra: `src/tkgat/kg/ckg.py`
- Mô hình (TransR, attention theo nhóm quan hệ, ma trận thưa A, Bi-Interaction): `src/tkgat/models/kgat.py`
- Vòng huấn luyện luân phiên và dừng sớm theo val: `src/tkgat/train/kgat.py`
- Chạy lưới, chọn theo val, chấm test một lần: `scripts/run.py`
- Thiết kế và các quyết định kỹ thuật (Q1–Q9): `docs/design-kgat.md`

## 3 câu tự kiểm tra
1. Softmax của attention chuẩn hóa trên tập nào, và vì sao điều đó quan trọng cho T-KGAT? Trên toàn bộ cạnh đi ra của node h (kể cả cạnh ngược), nên thứ quyết định không phải π tuyệt đối mà là π so với các hàng xóm khác của cùng node; thêm một số hạng vào π của một nhóm cạnh sẽ dịch trọng số sang nhóm còn lại.
2. Với L = 2, user nhận thông tin từ những node nào? Từ các item user đã xem (bước 1) và từ các user khác đã xem cùng item đó cùng các thuộc tính KG của item (bước 2); ví dụ phim → đạo diễn.
3. Vì sao phải nối embedding của mọi lớp thay vì chỉ lấy lớp cuối? Để giữ tín hiệu ở mọi bậc: lớp 0 là đặc trưng riêng của node, các lớp sau là tín hiệu từ lân cận, và chỉ lấy lớp cuối sẽ làm mờ phần riêng.

## Teach-back với Phôn (điền sau buổi giảng)
Ngày: ..; 3 câu Phôn hỏi vặn: ..; trả lời thế nào: ..
