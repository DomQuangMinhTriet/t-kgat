# Chủ đề 5: Gợi ý có yếu tố thời gian

Người trình bày: Triết · Người hỏi vặn: Phôn · Ngày teach-back: ../10/2026

Bản nháp soạn từ `docs/study/05-thoi-gian.md`, quyết định 0004, 0005, 0008, 0009 và code `src/tkgat/models/kgat.py`; Triết đọc lại, sửa bằng lời của mình và điền phần teach-back.

## Ý chính
- Thời gian ảnh hưởng tới gợi ý theo ba cách: sở thích trôi (user đổi gu), độ phổ biến đổi (*Titanic* được xem nhiều hơn sau Oscar), và tri thức đổi (phần tiếp theo chưa ra mắt, giải thưởng chưa trao). Mô hình tĩnh (BPR-MF, KGAT) coi mọi tương tác như nhau.
- Có bốn họ cách đưa thời gian vào: tham số thay đổi theo thời gian (timeSVD++), trọng số suy giảm theo độ cũ, mô hình chuỗi (SASRec, TiSASRec dùng khoảng thời gian trong attention), và KG có thời gian hiệu lực. T-KGAT nằm giữa họ "suy giảm" và "attention có khoảng thời gian".
- Biến thể chính của nhóm (giai đoạn 1, quyết định 0005 và 0009): trừ γ·δ_r·Δt_norm khỏi π **trước** softmax. δ_r = 1 cho cạnh Interact và Interact⁻¹, bằng 0 cho cạnh KG. γ = 0 cho lại đúng KGAT, nên so với KGAT là so sánh lồng nhau. γ là siêu tham số chọn trên val (lưới {1, 3, 10}), không học bằng gradient.
- Δt_norm = max(0, t_ref − ts) / dt_den. t_ref là timestamp lớn nhất của train (dùng chung cho val và test, nên không dùng bất kỳ timestamp nào của val/test); dt_den là độ dài cửa sổ train. Δt âm được cắt về 0.
- Trừ trước softmax tương đương nhân trọng số cũ với e^{−γΔt} rồi chuẩn hóa lại: cạnh cũ hơn bị giảm so với các hàng xóm khác **của cùng node**. Nếu mọi cạnh của một node có cùng Δt thì tác dụng triệt tiêu ở node đó.
- Biến thể so sánh (giai đoạn 2, 0009 mục 7–10): thời gian trên triple KG. Mỗi cạnh KG mang ngày từ đó fact đúng (file `kg_t*_dates.tsv`); δ_r = 1 cho quan hệ KG và 0 cho Interact; fact không ngày có Δt = 0; Δt_norm của cạnh KG bị chặn ở 1 vì ngày của fact thường cách t_ref hàng chục năm, ở thang thời gian khác với tương tác.
- Đối chứng bắt buộc: hoán vị timestamp (giai đoạn 1: xáo `train_ts` trên toàn bộ cạnh train; giai đoạn 2: xáo ngày của fact trên các triple của KG t1), mỗi seed một hoán vị. Nếu bản hoán vị tốt bằng bản thật thì phần cải thiện, nếu có, không đến từ thông tin thời gian.
- Kết quả âm là kết quả hợp lệ nếu quy trình đúng (biên bản 07/08). Cần giải thích bằng đặc điểm dữ liệu chứ không giải thích cho qua.

## Công thức
π_time(h, r, t) = π(h, r, t) − γ · δ_r · Δt_norm, rồi softmax như KGAT.

Δt_norm = max(0, t_ref − ts) / dt_den, với t_ref = max(train_ts) và dt_den = max(train_ts) − min(train_ts).

Chứng minh tương đương với suy giảm mũ: softmax(π − γΔt)_t = e^{π_t} e^{−γΔt_t} / Σ_{t'} e^{π_{t'}} e^{−γΔt_{t'}}.

Ví dụ số: hai cạnh cùng π, một cạnh Δt_norm = 0 và một cạnh Δt_norm = 1, γ = 3. Trọng số tỷ lệ với (1, e^{−3} ≈ 0,050), nên sau chuẩn hóa là (≈ 0,953; ≈ 0,047).

## Dữ liệu của nhóm
- Timestamp của MovieLens được nhóm **giả định** là lúc nhập rating, không phải lúc xem phim (README của GroupLens chỉ ghi "số giây từ epoch"; quyết định 0008). Bằng chứng: 77,9% rating của ML-100K và 72,4% của ML-1M nằm trong 24 giờ đầu của user; 53,0% user ML-100K có toàn bộ cạnh train trong chưa đầy một giờ (`docs/data-report-numbers.md`).
- Hệ quả: Δt giữa các cạnh của một user thường gần 0, nên suy giảm ở phía user ít tác dụng. Phía item thì cạnh trải dài hơn (độ lệch chuẩn trung vị của timestamp cạnh train theo item là 45 ngày ở ML-100K và 67 ngày ở ML-1M), nên suy giảm vẫn có thể có tác dụng, miễn là cạnh Interact⁻¹ cũng bị suy giảm (nó bị, theo 0009).
- MostPop gần đây (cửa sổ 14 ngày ở ML-100K, 7 ngày ở ML-1M, chọn trên val) là baseline thời gian đơn giản nhất: đo xem "chỉ cần biết cái gì đang hot" đã đủ chưa. Trên ML-100K nó đang mạnh hơn mọi mô hình học, và đó là một phát hiện cần giải thích chứ không giấu.
- Chạy thực nghiệm trên ML-100K (51 user test) và ml-1m-u10 (10% user của ML-1M, không phải ML-1M đầy đủ). Số liệu từng mô hình ở `docs/rq3-table.md`, đọc từ registry.
- Chưa làm (phase 5): rating như trọng số hoặc relation, timestamp như một relation, LLM làm sạch metadata, theo ba hướng cô gợi ý 18/09.

## Nằm ở đâu trong code
- Δt trong attention: `src/tkgat/models/kgat.py` (`_time_penalty`, `delta_r`, tham số `time_on`)
- t_ref, dt_den, ngày trên cạnh KG, hoán vị: `src/tkgat/kg/ckg.py` (`build_ckg`, `permute_train_ts`, `permute_kg_ts`, `load_kg_dates`)
- Ngày của fact: `src/tkgat/kg/temporal.py` (`fact_valid_from`, `triple_dates`); script `scripts/06_build_kg.py`
- Cấu hình: `configs/model/tkgat.yaml`, `tkgat_perm.yaml`, `tkgat_kg.yaml`, `tkgat_kg_perm.yaml`
- Quyết định: `docs/decisions/0005-time-in-attention.md`, `0009-tkgat-delta-t.md`

## 3 câu tự kiểm tra
1. Vì sao trừ γΔt trước softmax là suy giảm mũ, và vì sao γ = 0 là KGAT? Vì softmax(π − γΔt) bằng trọng số cũ nhân e^{−γΔt} rồi chuẩn hóa; với γ = 0 thừa số này bằng 1.
2. Nếu mọi cạnh của một user có cùng timestamp thì suy giảm ảnh hưởng thế nào tới user đó? Không ảnh hưởng: Δt bằng nhau nên số hạng trừ giống nhau cho mọi cạnh và bị softmax triệt tiêu; tác dụng còn lại là ở phía item.
3. Thí nghiệm hoán vị cho biết điều gì? Nếu T-KGAT thật không tốt hơn bản hoán vị thì không có bằng chứng thời gian giúp; chỉ khi hoán vị làm mất phần cải thiện mới có bằng chứng cho RQ3.

## Teach-back với Phôn (điền sau buổi giảng)
Ngày: ..; 3 câu Phôn hỏi vặn: ..; trả lời thế nào: ..
