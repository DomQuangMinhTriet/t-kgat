# Kế hoạch làm lại KLTN T-KGAT

Oct 17, 2026 · @Dom

Đề xuất làm lại toàn bộ trên MovieLens-100K (Phôn) và MovieLens-1M (Triết) chạy song song, KG từ KB4Rec/Freebase, trong một repo mới, với dữ liệu chia train/val/test theo thời gian và Knowledge Graph chỉ chứa tri thức có trước mốc cắt. Dự kiến 13 tuần (19/10/2026 → 17/01/2027), 7 giai đoạn (0–6), mỗi giai đoạn có tiêu chí "xong" rõ ràng. Các quyết định đã chốt nằm ở mục 1. Việc phải làm đến buổi present lại với cô ngày 31/10 nằm ở mục 12.

## 1. Các quyết định đã chốt (17/10)

Các phương án ở cột thứ ba đã được chốt ngày 17/10. Cột "Phương án khác" được giữ lại để dự phòng, chỉ chuyển sang khi số liệu EDA hoặc ý kiến của cô buộc phải đổi.

| # | Quyết định | Đã chốt | Phương án khác | Lý do |
| --- | --- | --- | --- | --- |
| 1 | Bộ dữ liệu chính | MovieLens-100K (cô đã đồng ý; Phôn chạy) và MovieLens-1M chạy song song (Triết chạy, máy mạnh hơn); KG từ KB4Rec/Freebase, ghép qua tên phim + năm với MovieLens-20M; nếu Freebase thiếu ngày tháng của fact thì bổ sung từ Wikidata qua Freebase ID | Chỉ dùng 100K; hoặc MovieLens-25M / ml-latest-small + Wikidata (qua mã IMDb/TMDb) | Nhỏ, chạy nhanh trên CPU; đúng lời cô dặn 18/09 là thử trên dữ liệu dung lượng thấp trước; vẫn là MovieLens như cô gợi ý từ 04/06 |
| 2 | Cách chia dữ liệu | Chia theo mốc thời gian chung cho mọi user: train < T1 ≤ val < T2 ≤ test | Chia theo từng user (vài tương tác cuối của mỗi user làm val/test) | Chia theo từng user vẫn để tương tác "tương lai" của user khác lọt vào train |
| 3 | Knowledge Graph theo thời gian | KG "chụp tại mốc": chỉ giữ fact đã tồn tại trước mốc cắt | Dùng KG tĩnh như bản gốc của KGAT | Đây đúng là góp ý thứ 5 của cô; KG tĩnh là rò rỉ |
| 4 | Có train lại trên train+val trước khi chấm test không | Không: chọn epoch và tham số bằng val, chấm test một lần | Train lại trên train+val với số epoch đã chọn | Đơn giản, dễ kiểm chứng, ít chỗ sai |
| 5 | Cách chia việc | Chia theo tầng: Triết lo dữ liệu, KG và kiểm tra rò rỉ; Phôn lo mô hình, huấn luyện và đánh giá; review chéo mọi PR | Mỗi người tự làm trọn một thí nghiệm | Tránh hai bộ code lệch nhau như hai track trước |
| 6 | Công cụ | GitHub Issues + Projects + PR; cấu hình YAML; log JSON/CSV; GPU Kaggle hoặc Colab | Thêm MLflow hoặc Weights & Biases | Đủ dùng, không phải học thêm công cụ |
| 7 | Phần "Confidence" trong tên đề tài | Bỏ khỏi phạm vi; đề tài tập trung vào yếu tố thời gian | Giữ, làm sau phần thời gian | Chốt 17/10. Tên đề tài bỏ chữ "Confidence" |
| 8 | Repo cũ | Giữ nguyên làm lưu trữ chỉ đọc; không copy code sang repo mới, chỉ mang ý tưởng kèm ghi chú nguồn. KGAT dùng code tham chiếu chính thức của tác giả, thêm tập val và ghi chú mọi chỗ sửa | Copy các module dùng được | Code cũ chứa sẵn lỗi chọn epoch trên test |
| 9 | Mốc kết thúc | Present lại với cô ngày 31/10/2026; hạn nộp và lịch bảo vệ chưa cần xét lúc này | — | Lộ trình ở mục 9 lấy 31/10 làm cổng đầu tiên và giả định xong trước 17/01/2027 |
| 10 | Yếu tố thời gian gắn vào đâu trong mô hình | Giữ thiết kế 07/08: giảm attention theo Δt trên cạnh user–item (chốt 17/10) | Attention giữa các triple của KG theo ngày của fact (đúng mục tiêu 3 trên Notion), chạy kèm để so sánh | MovieLens-1M trải gần 3 năm nên có thể có tín hiệu thời gian mà 100K không có. Cần báo cáo: 78% rating của 100K và 72% của 1M nằm trong ngày đầu, nên thí nghiệm hoán vị timestamp là bắt buộc |
| 11 | Mức bám sát paper KGAT | Giữ Recall/NDCG@20 xếp hạng toàn bộ item; KG 2-hop; baseline dùng KG là CKE; thêm một lần chạy theo quy trình gốc của paper làm mốc. Khác paper có chủ đích: chia theo thời gian; 5-core làm chính, 10-core như paper chạy kèm để phân tích độ nhạy | Chỉ KG 1-hop và baseline không dùng KG | Hội đồng sẽ so với paper; mọi chỗ khác phải có lý do và được nêu rõ |
| 12 | Các góp ý ngày 18/09 chưa làm | Đưa vào giai đoạn 5: giá trị rating như relation hoặc trọng số, timestamp như relation, LLM làm sạch metadata phim | Bỏ qua | Cô đã giao; cần câu trả lời bằng thí nghiệm, kể cả khi kết quả âm |

## 2. Góp ý của cô Châu (17/10) → việc cụ thể

Cả năm góp ý quy về một ý: đề tài là bài toán dữ liệu, nên dữ liệu phải đúng và được hiểu kỹ trước khi nói đến mô hình. Cột cuối là bằng chứng cần mang đến buổi họp để cô thấy góp ý đã được xử lý.

| Góp ý | Hiểu là | Việc cụ thể | Bằng chứng đã làm đúng |
| --- | --- | --- | --- |
| Kiểm tra và khám phá dữ liệu kỹ hơn, "mình làm về data" | Dữ liệu là trọng tâm của đề tài (đúng tinh thần ngành Hệ thống thông tin) | Báo cáo khám phá dữ liệu (EDA) cho MovieLens và KG; từ điển dữ liệu; báo cáo chất lượng dữ liệu | Báo cáo EDA được cô duyệt trước khi train bất kỳ mô hình nào |
| Tìm kỹ hơn thông tin mở rộng (metadata, KG) xem có làm tốt hơn không | Đây là câu hỏi nghiên cứu, không phải chuyện thêm càng nhiều càng tốt | Thí nghiệm bóc tách: không KG → chỉ genre → KB4Rec 1-hop → thêm metadata mới, tất cả đều theo mốc thời gian | Bảng so sánh 5 seed, có kiểm định thống kê |
| Đảm bảo không có data leakage | Áp dụng cho toàn bộ quy trình, không chỉ bước chia dữ liệu | Danh sách kiểm tra rò rỉ + bộ test tự động (mục 5) | Test tự động chạy qua trên GitHub mỗi lần đẩy code |
| Phải có tập validation, nếu không là rò rỉ | Chọn epoch hoặc tham số dựa trên test là dùng test để huấn luyện | Chia train/val/test; dừng sớm và chọn tham số chỉ theo val; test chỉ chấm một lần cho cấu hình cuối | Log huấn luyện cho thấy epoch tốt nhất được chọn theo val |
| KG của MovieLens phải bám theo chiều thời gian, không dùng dữ liệu tương lai để dự đoán hiện tại | Một fact trong KG chỉ được dùng nếu nó đã tồn tại tại thời điểm dự đoán (ví dụ: giải thưởng nhận năm 2005 không được dùng cho rating năm 2000) | Xây KG theo mốc; loại fact có ngày sau mốc và thực thể ra đời sau mốc; fact không có ngày được gắn cờ và đo ảnh hưởng riêng | Báo cáo số fact bị loại theo từng quan hệ; test tự động kiểm tra không còn fact nào có ngày sau mốc |

## 3. Bài học từ giai đoạn trước

Các kết quả Section 05–08 không dùng được nữa vì cùng mắc 3 lỗi rò rỉ dưới đây. Tuy vậy, một số cách làm vẫn đáng giữ lại.

**Phải sửa, không lặp lại:**

- **Chọn epoch tốt nhất trên tập test.** Code cũ không có tập val. Mọi số Recall/NDCG đã báo đều bị thổi phồng ở mức chưa biết.
- **Lọc k-core trên toàn bộ dữ liệu rồi mới chia.** Quyết định giữ hay bỏ một user/item đã dùng tương tác của giai đoạn test. Phải lọc chỉ dựa trên phần train.
- **KG dựng từ metadata hiện tại.** Ví dụ: metadata Amazon thu năm 2023 nhưng tương tác trong train nằm trong khoảng 1998–2008. Đây chính là lỗi cô nêu ở góp ý thứ 5.
- **Chỉ chạy 1 seed, và chạy CPU với GPU cho kết quả lệch nhau khoảng 18%.** Những chênh lệch nhỏ hơn mức dao động đó đã bị báo cáo như một cải thiện.
- **Mỗi giai đoạn copy code sang một thư mục mới; hai người hai pipeline.** Khó truy vết số nào đến từ code nào, và tài liệu không đồng bộ với kết quả.

**Nên giữ:**

- Phát hiện rò rỉ do tương tác lặp lại, xử lý bằng cách gộp về cặp (user, item) duy nhất.
- Thói quen đo phân bố thật trên đúng tập dữ liệu sẽ dùng trước khi thiết kế.
- Thí nghiệm kiểm soát, ví dụ hoán vị timestamp để xem lợi ích có thật sự đến từ thời gian hay không.
- Chạy nhiều seed và báo cáo trung bình ± độ lệch chuẩn.
- Ý tưởng có thể kiểm tra lại: quan hệ tĩnh mới giúp nhiều khi KG còn nghèo. Giờ chỉ coi đây là giả thuyết, chưa phải kết luận.

## 4. Định hướng đề tài theo ngành Hệ thống thông tin

Đề xuất viết khóa luận như một bài toán tích hợp và quản lý dữ liệu cho hệ gợi ý, với T-KGAT là mô hình dùng để đánh giá giá trị của dữ liệu. Cách đặt vấn đề này khớp với câu "mình làm về data" của cô. Nó cũng giữ nguyên 6 mục tiêu cô đã ghi trên Notion.

Các khái niệm Hệ thống thông tin mà đề tài đụng tới:

- **Tích hợp dữ liệu và liên kết thực thể:** ghép phim trong MovieLens với thực thể trong Freebase hoặc Wikidata, đo độ phủ và độ chính xác của liên kết.
- **Chất lượng dữ liệu:** thiếu, trùng, sai và lệch phân bố.
- **Tính đúng theo thời điểm:** mỗi dự đoán chỉ được dùng dữ liệu có trước thời điểm đó, kể cả tri thức trong KG.
- **Quy trình dữ liệu tái lập được:** có phiên bản, có truy vết nguồn gốc, có kiểm thử.

| Câu hỏi nghiên cứu | Trả lời bằng | Khớp mục tiêu trên Notion |
| --- | --- | --- |
| RQ1. Dữ liệu MovieLens và KG liên kết có đặc điểm gì theo thời gian: độ phủ liên kết, tỷ lệ fact có ngày, tỷ lệ fact "tương lai" so với rating? | Báo cáo EDA và báo cáo chất lượng dữ liệu | Mục tiêu 2 (hạn chế của KGAT với yếu tố thời gian và độ tin cậy) |
| RQ2. Khi đã loại rò rỉ (có val, KG theo mốc), KG và metadata còn giúp KGAT hơn lọc cộng tác thuần hay không? | Thí nghiệm bóc tách các biến thể KG | Mục tiêu 1, 4, 5 |
| RQ3. Đưa thời gian vào attention (giảm theo Δt trên cạnh user–item như thiết kế 07/08, so sánh với thời gian trên các triple của KG) có cải thiện thêm không, và cải thiện có thật sự đến từ tín hiệu thời gian không? | T-KGAT so với KGAT, kèm thí nghiệm kiểm soát | Mục tiêu 3, 5, 6 |

Một kết quả âm của RQ2 hoặc RQ3 vẫn hợp lệ. Biên bản 07/08 đã ghi rõ điều này, miễn quy trình đánh giá đúng.

## 5. Quy trình dữ liệu chống rò rỉ

Quy tắc gốc: mọi thứ mô hình nhìn thấy khi dự đoán tại thời điểm T phải có trước T. Điều này áp dụng cho tương tác, thống kê và tri thức trong KG. Quy trình dưới đây viết một lần và dùng chung cho mọi thí nghiệm.

1. **Làm sạch:** gộp trùng về cặp (user, item) duy nhất, giữ lần sớm nhất. Chuẩn hóa timestamp. Chốt cách chuyển rating sang tín hiệu ngầm (ví dụ rating ≥ 4 là tích cực) và ghi lại trong tài liệu quyết định.
2. **Chia theo mốc thời gian chung:** T1, T2 lấy theo phân vị thời gian (ví dụ 80% và 90%), chỉ chốt sau khi xem EDA. Train = t < T1, val = T1 ≤ t < T2, test = t ≥ T2. User chưa có tương tác nào trong train bị loại khỏi val/test, và có ghi lại số lượng.
3. **Lọc k-core chỉ trên train**, sau bước chia.
4. **Mọi thống kê chỉ tính trên train:** độ phổ biến, rating trung bình, hệ số chuẩn hóa Δt, danh sách item dùng để lấy mẫu âm.
5. **KG theo mốc:** KG dùng cho train và val được dựng tại T1, KG dùng cho test được dựng tại T2. Quy tắc giữ một fact:
   - Fact có ngày (ngày phát hành, ngày nhận giải, ngày bắt đầu…) → giữ nếu ngày đó trước mốc.
   - Fact trỏ tới thực thể ra đời sau mốc (phần tiếp, bản làm lại, phim sau này của cùng diễn viên) → loại.
   - Fact không có ngày (ví dụ phim → thể loại, phim → đạo diễn): mặc định giữ nếu cả hai thực thể đã tồn tại trước mốc (phim đã phát hành; người và thể loại coi như luôn tồn tại), bỏ nếu thực thể đích là một tác phẩm ra đời sau mốc. Chạy thêm biến thể "bỏ toàn bộ fact không ngày" để đo ảnh hưởng, và báo cáo tỷ lệ theo từng quan hệ.
6. **Chọn mô hình chỉ bằng val:** dừng sớm, chọn γ, learning rate, số lớp… đều theo val. Test chỉ chạy một lần cho mỗi cấu hình cuối.
7. **Đánh giá:** xếp hạng trên toàn bộ item đã tồn tại trước mốc, bỏ item user đã tương tác trước đó. Metric: Recall@10/20, NDCG@10/20. Chạy 5 seed, báo cáo trung bình ± độ lệch chuẩn, kèm kiểm định cặp theo user (Wilcoxon).
8. **Baseline bắt buộc:** gợi ý theo độ phổ biến (toàn thời gian và gần đây), BPR-MF, KGAT không KG. Nếu mô hình không thắng gợi ý theo độ phổ biến thì phải nói rõ.

**Bộ test rò rỉ tự động (pytest, chạy mỗi lần đẩy code):**

- [ ] Timestamp lớn nhất của train < T1 ≤ timestamp nhỏ nhất của val; tương tự giữa val và test.
- [ ] Không có cặp (user, item) nào xuất hiện ở hai tập.
- [ ] k-core và các thống kê được tính từ file train, không phải từ file gốc.
- [ ] Không còn fact nào có ngày sau mốc trong KG tương ứng.
- [ ] Hàm huấn luyện không nhận test làm đầu vào; chỉ script đánh giá cuối được đọc test.
- [ ] Mỗi kết quả ghi kèm commit hash, file cấu hình, seed và checksum dữ liệu.

## 6. Bộ dữ liệu MovieLens và nguồn tri thức

Cập nhật 17/10: nhóm chọn MovieLens-100K (943 user, 1,682 phim, 100,000 rating, từ 19/09/1997 đến 22/04/1998). Bảng dưới giữ lại để so sánh, và MovieLens-1M chạy song song trên máy của Triết. Các phương án trong bảng là dự phòng, chỉ chuyển sang nếu EDA (xong 24/10) cho thấy A thiếu thông tin thời gian hoặc có quá ít user ở val/test.

Lưu ý quan trọng: [KB4Rec](https://github.com/RUCDM/KB4Rec) liên kết MovieLens-**20M** (cùng LFM-1b và Amazon Book) với Freebase, không liên kết trực tiếp MovieLens-1M. Việc "merge MovieLens-1M với KB4Rec" trong kế hoạch Notion vì vậy cần một bước ghép mã phim giữa hai bản MovieLens. Bước này phải đo độ phủ và kiểm tra tay một mẫu.

|  | A. MovieLens-1M + KB4Rec | B. MovieLens-20M (cắt giai đoạn) + KB4Rec | C. MovieLens-25M / ml-latest-small + Wikidata |
| --- | --- | --- | --- |
| Quy mô (xấp xỉ) | \~1 triệu rating, 6,040 user, \~3,900 phim | \~20 triệu rating, \~138 nghìn user; sẽ cắt còn một đoạn thời gian | 25 triệu rating, hoặc \~100 nghìn với bản small |
| Thời gian rating | 2000–2003, phần lớn dồn vào năm 2000 (cần đo lại) | 1995–2015 | 1995–2019 |
| Liên kết KG | Gián tiếp qua mã phim của bản 20M | Trực tiếp, có sẵn trong KB4Rec | Qua mã IMDb/TMDb trong `links.csv` |
| Thông tin ngày của fact | Freebase ít có ngày và đã ngừng cập nhật từ 2016 | Như A | Wikidata có nhiều thuộc tính kèm ngày (ngày phát hành, thời điểm nhận giải…) |
| Ưu điểm | Cô gợi ý từ đầu; nhỏ, chạy nhanh | Không phải tự ghép mã; trải dài nhiều năm | Dễ làm KG theo mốc thời gian nhất |
| Rủi ro | Giai đoạn val/test có thể rất ít user | Nặng, cần GPU | Tự xây KG, tốn công nhất |

Có thể kết hợp: dùng Freebase ID trong KB4Rec để tra Wikidata (Wikidata lưu Freebase ID cho nhiều thực thể), rồi lấy ngày tháng của fact từ đó. Quyết định cuối nằm ở cổng kiểm tra của giai đoạn 2 (mục 9), dựa trên số đo thật.

Phôn từng ghi là đã đổi sang "MovieLens300K" (biên bản 21/08 trên Notion). Cần xác nhận đó là bản nào để tận dụng hiểu biết cũ.

## 7. Repo GitHub mới

Một repo duy nhất, một pipeline duy nhất. Code nằm trong package `src/`, còn notebook chỉ dùng để khám phá và vẽ hình. Không copy thư mục theo từng giai đoạn như trước; giai đoạn được thể hiện bằng branch, Issue và milestone.

```
t-kgat/
├── README.md               # mục tiêu, cách cài, cách chạy lại toàn bộ
├── CONTRIBUTING.md         # quy ước branch, commit, review
├── pyproject.toml          # phiên bản thư viện cố định
├── configs/                # mỗi thí nghiệm = 1 file YAML
│   ├── data/ml1m_temporal.yaml
│   └── model/{mostpop,bprmf,kgat,tkgat}.yaml
├── data/                   # KHÔNG commit dữ liệu (gitignore)
│   └── README.md           # nguồn tải, checksum, phiên bản
├── src/tkgat/
│   ├── data/               # tải, làm sạch, chia theo thời gian, k-core
│   ├── kg/                 # liên kết thực thể, dựng KG theo mốc
│   ├── models/             # MostPop, BPR-MF, KGAT, T-KGAT
│   ├── train/              # vòng huấn luyện, dừng sớm theo val
│   └── eval/               # metric, kiểm định thống kê
├── scripts/                # 01_download.py, 02_build_splits.py, 03_build_kg.py, run.py
├── tests/                  # test rò rỉ (mục 5) + test đơn vị
├── notebooks/              # 01_eda_ratings_triet.ipynb, 02_eda_kg_phon.ipynb ...
├── results/
│   └── registry.csv        # mỗi lần chạy 1 dòng: commit, config, seed, val, test
└── docs/
    ├── decisions/          # mỗi quyết định 1 file ngắn (vì sao, thay thế gì)
    ├── theory/             # ghi chú ôn lý thuyết
    ├── data-dictionary.md  # ý nghĩa từng cột, từng quan hệ KG
    └── meetings/           # bản sao biên bản họp từ Notion
```

Quy trình làm việc:

- **Issue cho mọi việc**, gắn nhãn (`data`, `kg`, `model`, `eval`, `docs`, `theory`) và milestone theo giai đoạn. Bảng GitHub Projects có 4 cột: Todo → In progress → Review → Done.
- **Không commit thẳng lên `main`.** Mỗi Issue một branch (`feat/12-temporal-split`). Muốn merge Pull Request thì người còn lại phải duyệt, và test phải chạy qua.
- **GitHub Actions** chạy `pytest` (gồm test rò rỉ) trên một mẫu dữ liệu nhỏ mỗi lần mở PR.
- **Một thí nghiệm = một file cấu hình + một lệnh** `python scripts/run.py --config ...`. Kết quả tự ghi vào `results/registry.csv`, không chép số bằng tay.
- **Notebook** đặt tên theo số thứ tự và người làm, xóa output trước khi commit (dùng nbstripout).
- **Commit** theo mẫu `feat:`, `fix:`, `docs:`, `exp:`. Thêm pre-commit với ruff để code cùng một phong cách. Ngôn ngữ: code, comment, commit và README viết tiếng Anh; báo cáo dữ liệu, ghi chú lý thuyết và notebook present viết tiếng Việt.

## 8. Ôn lại lý thuyết

Cách học: dạy lại cho nhau (teach-back). Mỗi chủ đề có một người trình bày khoảng 30 phút, người kia hỏi vặn. Sau đó viết ghi chú vào `docs/theory/`, mỗi chủ đề không quá 2 trang và có 3 câu hỏi tự kiểm tra. Cả 7 chủ đề học trọn trong tuần 1 (18–24/10), mỗi ngày một chủ đề, và cả hai đều đọc toàn bộ tài liệu của mọi chủ đề. Ngày 24/10 có buổi kiểm tra chéo trên cả 7 chủ đề (lịch cụ thể ở mục 12). Chủ đề 2 học trước tiên vì nó trực tiếp gây ra kết quả tệ sáng nay.

| # | Chủ đề | Phải trả lời được | Tài liệu chính | Trình bày |
| --- | --- | --- | --- | --- |
| 1 | Hệ gợi ý và phản hồi ngầm | Vì sao dùng BPR thay cho hồi quy rating? Lấy mẫu âm sai thì hậu quả gì? | Rendle và cộng sự, *BPR: Bayesian Personalized Ranking from Implicit Feedback* (2009) | Triết |
| 2 | Đánh giá offline và rò rỉ dữ liệu | Các kiểu chia dữ liệu, vai trò của val, các nguồn rò rỉ thời gian, vì sao phải có baseline độ phổ biến | Ji và cộng sự, *A Critical Study on Data Leakage in Recommender System Offline Evaluation* (ACM TOIS); Meng và cộng sự, *Exploring Data Splitting Strategies for the Evaluation of Recommendation Models* (RecSys 2020); Dacrema và cộng sự, *Are We Really Making Much Progress?* (RecSys 2019) | Phôn |
| 3 | Knowledge Graph và KGE | Triple, TransE/TransR, hàm loss của KG, liên kết item với thực thể | Phần KGE trong paper KGAT; tài liệu ôn của nhóm ở Section04 | Phôn |
| 4 | KGAT chi tiết | Lan truyền trên đồ thị, attention tính cho triple nào, cách kết hợp hai loss, độ phức tạp tính toán, KGAT khác CKE và RippleNet ở đâu, quy trình thực nghiệm của paper (KG 2-hop, chia ngẫu nhiên 80/20) | [KGAT, KDD 2019](https://arxiv.org/abs/1905.07854) + code tham chiếu | Triết |
| 5 | Gợi ý có yếu tố thời gian | Trôi sở thích theo thời gian, hàm suy giảm, nhúng khoảng thời gian, KG có yếu tố thời gian (fact có hiệu lực theo thời gian) | Koren, *Collaborative Filtering with Temporal Dynamics* (KDD 2009); Li và cộng sự, *TiSASRec* (WSDM 2020) | Triết |
| 6 | Tích hợp dữ liệu và tính đúng theo thời điểm | Liên kết thực thể, đo chất lượng liên kết, fact có hiệu lực theo thời gian, truy vết nguồn gốc dữ liệu | Zhao và cộng sự, *KB4Rec* (Data Intelligence 2019); tài liệu Wikidata về qualifier thời gian | Phôn |
| 7 | Kiểm định thống kê cho thí nghiệm | Kiểm định cặp, nhiều seed, khoảng tin cậy | Giáo trình xác suất thống kê; tài liệu của scipy.stats | Cả hai |

Phân công trình bày ở đây cố ý đảo với phân công code (mục 10): người làm mô hình trình bày phần dữ liệu và ngược lại. Như vậy cả hai đều nắm toàn bộ nội dung khi cô hỏi.

## 9. Lộ trình và cổng kiểm tra

&#91;embedded content: lộ trình · 7 giai đoạn, 19/10/2026 – 17/01/2027\]

Giai đoạn 0–3 được dồn vào 2 tuần để kịp buổi present 31/10. Buổi đó chỉ trình bày dữ liệu, quy trình chống rò rỉ và baseline đơn giản (MostPop, BPR-MF). KGAT chỉ được chạy lại sau khi cô duyệt quy trình ở buổi này. Mốc 17/01/2027 vẫn là giả định (quyết định 9).

| Giai đoạn | Xong khi |
| --- | --- |
| 0. Chốt hướng, dựng repo (18–19/10) | Đã gửi câu hỏi cho cô; repo có CI chạy, nhánh `main` được bảo vệ |
| 1. Học lý thuyết (18–24/10) | Cả 7 chủ đề có ghi chú; ngày 24/10 mỗi người trả lời được câu hỏi tự kiểm tra của cả 7 chủ đề |
| 2. Khám phá dữ liệu và KG (25–28/10) | Báo cáo dữ liệu xong: EDA, chất lượng, độ phủ liên kết, tỷ lệ fact có ngày và fact "tương lai"; đã chốt hai mốc T1, T2 |
| 3. Pipeline chống rò rỉ, baseline (25–29/10) | Mọi test rò rỉ đều qua trong CI; MostPop và BPR-MF có số với 5 seed (chọn theo val, test chấm một lần) |
| Present 31/10 | Cô duyệt báo cáo dữ liệu và quy trình, hoặc chỉ rõ cần sửa gì |
| 4. KGAT, bóc tách KG (02–29/11) | Bảng RQ2: các biến thể KG so với BPR-MF; chọn cấu hình trên val, chấm test một lần; Triết chạy lại cùng pipeline trên MovieLens-1M (chỉ đổi file cấu hình); bám paper: KG mở rộng 2-hop, thêm baseline dùng KG là CKE (có sẵn trong code tham chiếu của tác giả), thêm một lần chạy theo quy trình gốc của paper (chia ngẫu nhiên) làm mốc tham chiếu |
| 5. Yếu tố thời gian (30/11–27/12) | Bảng RQ3 trên cả 100K và 1M: T-KGAT theo thiết kế 07/08 (giảm attention theo Δt trên cạnh user–item) so với biến thể thời gian trên các triple của KG; 3 hướng cô gợi ý 18/09 (giá trị rating như relation hoặc trọng số, timestamp như relation, LLM làm sạch metadata phim); kèm thí nghiệm kiểm soát hoán vị timestamp |
| 6. Phân tích, viết khóa luận (28/12–17/01) | Bản thảo các chương; mọi con số trong bài đều truy được về `results/registry.csv` |

Sau 31/10, họp với cô 2 tuần một lần. Mỗi buổi mang theo bằng chứng của cổng kiểm tra gần nhất.

## 10. Phân công Triết và Phôn

Cả hai làm chung một dataset và một pipeline. Việc được chia theo tầng của pipeline, không chia theo dataset như trước. Người còn lại luôn là người duyệt PR.

| Mảng | Triết (chính) | Phôn (chính) | Người kia làm gì |
| --- | --- | --- | --- |
| Repo, CI, quy ước | Dựng repo, cấu trúc, GitHub Actions | Viết CONTRIBUTING, mẫu Issue/PR | Duyệt |
| EDA tương tác (rating) | ✓ |  | Đọc và đặt câu hỏi |
| EDA và xây KG (liên kết, KG theo mốc) | ✓ |  | Làm phần liên kết thực thể và kiểm tra tay một mẫu (2 tuần đầu) |
| Chia dữ liệu và test rò rỉ | ✓ |  | Viết thêm test "cố tình phá" để thử |
| Baseline MostPop, BPR-MF |  | ✓ | Duyệt |
| Bộ đánh giá (metric, kiểm định thống kê) |  | ✓ | Đối chiếu metric với một thư viện có sẵn |
| KGAT (viết lại, dừng sớm theo val) |  | ✓ | Duyệt, chạy lại độc lập |
| Thí nghiệm bóc tách KG (RQ2) | ✓ (chuẩn bị các biến thể KG) | ✓ (chạy và đánh giá) | — |
| T-KGAT (RQ3) | ✓ (tín hiệu thời gian, thí nghiệm kiểm soát) | ✓ (sửa attention trong mô hình) | — |
| Biên bản họp và đồng bộ Notion | Luân phiên theo buổi | Luân phiên theo buổi | — |

Lý do Triết giữ phần dữ liệu: bạn đã có kinh nghiệm phát hiện rò rỉ và tự dựng KG ở các Section trước. Đây cũng là phần cô nhấn mạnh nhất. Nếu Phôn muốn đổi thì chỉ cần hoán đổi hai cột giữa.

## 11. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu sớm | Cách xử lý |
| --- | --- | --- |
| MovieLens-100K chỉ trải 7 tháng nên tín hiệu thời gian yếu, và val/test sau mốc cắt có thể có ít user | EDA cho thấy ít hơn vài trăm user có tương tác sau T2 | Đặt mốc cắt theo số lượng thay vì theo ngày, hoặc chuyển sang phương án B (MovieLens-20M cắt giai đoạn) |
| Fact trong Freebase gần như không có ngày, nên không làm được KG theo mốc | Tỷ lệ fact có ngày thấp trong báo cáo EDA KG | Bổ sung ngày từ Wikidata qua Freebase ID; hoặc chỉ giữ quan hệ ổn định theo thời gian (thể loại, đạo diễn) và loại quan hệ dễ "rò tương lai" (giải thưởng, phần tiếp) |
| KGAT vẫn không thắng BPR-MF hoặc gợi ý theo độ phổ biến | Số trên val của giai đoạn 4 | Chấp nhận và phân tích vì sao; đây vẫn là một kết quả hợp lệ cho RQ2. Báo cáo với cô sớm, không để đến cuối |
| Thiếu GPU, huấn luyện chậm | Một lần chạy KGAT mất hơn 1 giờ | Kaggle (có GPU miễn phí theo giờ mỗi tuần) hoặc Colab; thử trên bản nhỏ trước, đúng lời cô dặn 18/09 |
| Hai người lệch tiến độ, chờ nhau | PR mở quá 2 ngày chưa ai duyệt | Họp nhóm 15 phút 2 lần mỗi tuần, nhìn chung bảng GitHub Projects |
| Lặp lại lỗi cũ: đưa số cho cô trước khi kiểm tra quy trình | — | Quy tắc: không số nào được báo cáo nếu chưa qua test rò rỉ và chưa có trong `results/registry.csv` |
| Không tải được fact từ Wikidata (mạng bị chặn hoặc bị giới hạn số truy vấn) | Script tải báo lỗi khi chạy ngày 19/10 (việc #5) | Script tự thử lại và chạy tiếp từ batch đã lưu; nếu vẫn không được thì tải đồ thị 1-step Freebase của KB4Rec (Google Drive) thay thế |

## 12. Kế hoạch 2 tuần đến buổi present 31/10

Mục tiêu: ngày 31/10, hai bạn tự trình bày bằng notebook toàn bộ phần dữ liệu, KG theo thời gian, quy trình chống rò rỉ và baseline. Mỗi dòng code phải có một người hiểu và trả lời được khi cô hỏi. KGAT chỉ bắt đầu từ 02/11.

**Đã chuẩn bị sẵn ngày 17/10 trong `E:\KLTN HCMUS\t-kgat`.** Đây là bản nháp để hai bạn tiếp quản. Nó chỉ trở thành kết quả chính thức khi hai bạn đã tự đọc, chạy lại và duyệt:

- Dữ liệu: MovieLens-100K (MD5 khớp bản chính thức), MovieLens-1M, file liên kết KB4Rec, checksum SHA-256.
- Code: làm sạch, chia theo mốc, kiểm tra rò rỉ, đánh giá, MostPop, BPR-MF, liên kết thực thể, script tải Wikidata; 17 test đều qua.
- Tài liệu: README, CONTRIBUTING, CI, mẫu Issue/PR, 4 file quyết định, từ điển dữ liệu, bản sao biên bản họp.
- Phát hiện: 78% rating của 100K được nhập trong ngày đầu user đăng ký, nên test chỉ còn 51 user; KB4Rec phủ 97,2% số phim; Titanic đoạt Oscar ngày 23/03/1998, nằm giữa T1 và T2.

**Người chịu trách nhiệm chính** (hiểu sâu, sửa code, trả lời cô; người còn lại duyệt):

- Triết: `data/`, `kg/`, các script 01–05, báo cáo dữ liệu, chạy thực nghiệm trên MovieLens-1M (máy mạnh hơn).
- Phôn: `eval/`, `models/`, `train/`, `run.py`, registry, kiểm định thống kê, kiểm tra tay liên kết, chạy thực nghiệm trên MovieLens-100K.

**Tuần 1 (18–24/10): cả hai học trọn 7 chủ đề lý thuyết + việc nền tối thiểu**

Mỗi ngày một chủ đề. Cả hai đọc toàn bộ tài liệu của chủ đề đó trước buổi học. Người trình bày giảng khoảng 45 phút và viết ghi chú vào `docs/theory/` (kèm 3 câu hỏi tự kiểm tra). Người nghe chuẩn bị 3 câu hỏi vặn. Việc kỹ thuật trong tuần này giới hạn khoảng 1 giờ mỗi ngày.

| # | Việc | Người làm | Hạn | Sản phẩm / bằng chứng | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| 1 | Họp nhóm 60 phút: đọc doc này, chốt tài khoản GitHub và giờ học mỗi ngày | Cả hai | Oct 18, 2026 | Thống nhất lịch tuần 1 | Chưa làm |
| 2 | Tạo GitHub Organization và repo private `t-kgat`; `git init`, commit đầu tiên, push; bảo vệ `main`; bảng Projects, nhãn, milestone; mời Phôn | Triết | Oct 18, 2026 | Repo trên GitHub, CI chạy xanh | Chưa làm |
| 3 | **Chủ đề 2** — Đánh giá offline và rò rỉ dữ liệu; đọc kèm `split.py`, `leakage.py` | Phôn trình bày | Oct 18, 2026 | `docs/theory/02-danh-gia-va-ro-ri.md` | Chưa làm |
| 4 | **Chủ đề 1** — Hệ gợi ý, phản hồi ngầm, BPR; đọc kèm `models/baselines.py`, `train/bpr.py` | Triết trình bày | Oct 19, 2026 | `docs/theory/01-bpr.md` | Chưa làm |
| 5 | Cài môi trường trên máy mình, chạy `pytest` và các script 01→03 | Cả hai | Oct 19, 2026 | Lỗi gặp phải ghi vào Issue | Chưa làm |
| 6 | Nhắn cô 4 nội dung ở cuối mục này; chạy `python scripts/04_fetch_wikidata.py` (khoảng 10 phút) | Triết | Oct 19, 2026 | Tin nhắn đã gửi; `data/raw/wikidata/facts.tsv` | Chưa làm |
| 7 | **Chủ đề 3** — Knowledge Graph và KGE (triple, TransE/TransR, hàm loss của KG) | Phôn trình bày | Oct 20, 2026 | `docs/theory/03-kg-kge.md` | Chưa làm |
| 8 | **Chủ đề 4** — KGAT chi tiết: paper và code tham chiếu của tác giả (lan truyền, attention, hai hàm loss) | Triết trình bày | Oct 21, 2026 | `docs/theory/04-kgat.md` | Chưa làm |
| 9 | **Chủ đề 5** — Gợi ý có yếu tố thời gian; đối chiếu với phát hiện "78% rating nhập trong ngày đầu" của MovieLens | Triết trình bày | Oct 22, 2026 | `docs/theory/05-temporal.md` | Chưa làm |
| 10 | **Chủ đề 6** — Tích hợp dữ liệu, liên kết thực thể, tính đúng theo thời điểm; đọc kèm `kg/linking.py`, quyết định 0003–0004 | Phôn trình bày | Oct 23, 2026 | `docs/theory/06-data-integration.md` | Chưa làm |
| 11 | **Chủ đề 7** — Kiểm định thống kê cho thí nghiệm (kiểm định cặp, nhiều seed); đọc kèm `eval/stats.py` | Cả hai | Oct 24, 2026 | `docs/theory/07-statistics.md` | Chưa làm |
| 12 | Kiểm tra chéo cuối tuần: mỗi người trả lời ngẫu nhiên 10 câu tự kiểm tra của cả 7 chủ đề; sau đó họp chốt T1/T2 và quy tắc fact không ngày theo trả lời của cô | Cả hai | Oct 24, 2026 | Câu nào chưa trả lời được thì học lại trước 25/10; `docs/decisions/` cập nhật | Chưa làm |

**Tuần 2 (25–31/10): tiếp quản repo, dữ liệu, KG theo thời gian, baseline, present**

Tuần này dày việc. Nó khả thi vì code nháp đã có sẵn: phần lớn việc là đọc hiểu, chạy, kiểm tra và viết kết quả, không phải viết từ đầu.

| # | Việc | Người làm | Hạn | Sản phẩm / bằng chứng | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| 13 | Review chéo code: Triết đọc `eval/`, `models/`, `train/`; Phôn đọc `data/`, `kg/`; mỗi câu hỏi hoặc đề xuất sửa là một Issue | Cả hai | Oct 25, 2026 | Danh sách Issue | Chưa làm |
| 14 | Notebook EDA cho 100K và 1M: rating theo tháng, hành vi nhập rating theo phiên, phân bố rating, năm phát hành so với lúc rating, các cặp T1/T2 | Triết | Oct 26, 2026 | `notebooks/01_eda.ipynb` | Chưa làm |
| 15 | Kiểm tra tay 50 liên kết ngẫu nhiên MovieLens ↔ Freebase/Wikidata, ước lượng tỷ lệ lỗi | Phôn | Oct 26, 2026 | `notebooks/02_entity_linking.ipynb` | Chưa làm |
| 16 | Viết `scripts/05_build_kg.py` theo quyết định 0004 (KG tại T1 và T2) + test "không fact nào sau mốc" | Triết | Oct 27, 2026 | Script + test xanh | Chưa làm |
| 17 | Chạy chính thức MostPop, MostPop gần đây và BPR-MF (Phôn chạy 100K, Triết chạy 1M) (5 seed, chọn theo val, chấm test một lần) | Phôn + Triết | Oct 27, 2026 | Dòng mới trong `results/registry.csv` | Chưa làm |
| 18 | Notebook KG theo thời gian: fact theo quan hệ, tỷ lệ có ngày, fact "tương lai" bị loại tại T1/T2, ví dụ Titanic và Toy Story 2 | Triết | Oct 28, 2026 | `notebooks/03_kg_temporal.ipynb` | Chưa làm |
| 19 | Kiểm định Wilcoxon giữa các baseline; phân tích độ nhạy với 2–3 cặp T1/T2 khác và với 10-core như paper (\`configs/data/ml100k\_k10.yaml\`, chỉ xem val); vẽ learning curve (loss và val Recall@20 theo epoch) của BPR-MF, kèm bảng batch size, số epoch, learning rate (góp ý 03/07) | Phôn | Oct 28, 2026 | Bảng kết quả + bảng phụ | Chưa làm |
| 20 | Báo cáo dữ liệu tiếng Việt, 4–6 trang (Phôn duyệt) | Triết | Oct 29, 2026 | `docs/data-report.md` | Chưa làm |
| 21 | Đóng băng code (tag `v0.1-review`); clone repo sạch, chạy lại toàn bộ, số phải trùng registry | Phôn | Oct 29, 2026 | Ghi chú tái lập | Chưa làm |
| 22 | Notebook present: Triết phần dữ liệu và KG; Phôn phần quy trình, baseline và repo | Cả hai | Oct 30, 2026 | `notebooks/00_present_2026-10-31.ipynb` chạy không lỗi | Chưa làm |
| 23 | Tập dượt 2 lượt, lượt 2 đảo vai; chuẩn bị 10 câu cô có thể hỏi | Cả hai | Oct 30, 2026 | Mỗi người trình bày được mọi phần | Chưa làm |
| 24 | Present với cô; ghi biên bản lên Notion ngay sau buổi | Cả hai | Oct 31, 2026 | Biên bản 31/10 | Chưa làm |

**Dàn ý notebook present 31/10 (10 phần, khoảng 20 phút; tiếng Việt, code tiếng Anh):**

1. Góp ý ngày 17/10 và việc đã làm cho từng góp ý — Phôn
2. Vì sao làm lại: 3 lỗi rò rỉ của kết quả cũ — Triết
3. Định hướng theo ngành Hệ thống thông tin và 3 câu hỏi nghiên cứu — Phôn
4. MovieLens-100K theo thời gian: rating được nhập theo phiên, hệ quả cho cách chia và cho attention theo Δt — Triết
5. Liên kết KB4Rec: độ phủ, kết quả kiểm tra tay, lỗi phát hiện — Phôn
6. KG theo thời gian: fact có ngày, fact "tương lai", ví dụ Titanic — Triết
7. Quy trình chia train/val/test và bằng chứng không rò rỉ (test tự động, CI) — Triết
8. Baseline trên 100K và 1M: trung bình ± độ lệch chuẩn qua 5 seed, kiểm định, độ nhạy theo mốc cắt, learning curve và tham số huấn luyện — Phôn
9. Repo và cách làm việc mới — Phôn
10. Kế hoạch từ 02/11 (bám paper KGAT, 3 hướng cô gợi ý ngày 18/09) và các câu hỏi xin ý kiến cô, và kết quả trao đổi về 4 nội dung báo cô ngày 19/10 — Cả hai

**Nội dung báo cô ngày 19/10** (nhóm đã chốt nội bộ ngày 17/10, xin cô xác nhận):

1. Timestamp của MovieLens là lúc user nhập rating (78% rating của 100K nằm trong ngày đầu), nên khi chia theo mốc chung, test của 100K chỉ còn 51 user. Nhóm sẽ chạy song song: Phôn trên 100K, Triết trên MovieLens-1M (952 user test) để kết luận chắc hơn, cùng quy trình và cùng code.
2. Nhóm giữ thiết kế T-KGAT ngày 07/08 (giảm attention theo Δt trên cạnh user–item) làm mô hình chính, và thêm biến thể đưa thời gian vào các triple của KG (theo ngày của fact) để so sánh. Cả hai đều có thí nghiệm kiểm soát hoán vị timestamp.
3. Fact trong KG không có ngày: giữ nếu cả hai thực thể đã tồn tại trước mốc, bỏ nếu thực thể đích là tác phẩm ra đời sau mốc, và chạy thêm biến thể bỏ toàn bộ fact không ngày để đo ảnh hưởng.
4. Tên đề tài bỏ chữ "Confidence": T-KGAT — Temporal Knowledge Graph Attention Network for Recommendation.
