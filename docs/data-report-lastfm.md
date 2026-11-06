# Báo cáo dữ liệu: Last-FM, KG Freebase và KG theo thời gian

Dự án T-KGAT — Temporal Knowledge Graph Attention Network for Recommendation. Báo cáo dành cho cô Nguyễn Ngọc Minh Châu; nhóm thực hiện: Đởm Quang Minh Triết, Nguyễn Đăng Phôn.

Last-FM là bộ dữ liệu thứ ba của dự án, sau MovieLens-100K và MovieLens-1M (`docs/data-report.md`), theo định hướng của cô (xem quyết định 0012). **Protocol không đổi**: cùng mã chia dữ liệu, cùng mốc phân vị 0,8 / 0,9, k-core chỉ tính trên train, cùng `check_split`, cùng quy tắc KG theo thời gian của quyết định 0004 và 0006, cùng grid và cùng seed. Chỉ dataset là khác.

Hai quyết định nền của bộ dữ liệu này là quyết định 0012 (Last-FM là một dataset của dự án, kèm nguồn ngày tháng và các hạn chế của chính dữ liệu) và quyết định 0013 (các lần chạy chính thức nằm trên `lastfm-u25`). Báo cáo này không nhắc lại nội dung hai quyết định đó; đọc chúng trước khi đọc mục 5 và mục 6.

Mọi ngày tính theo UTC. Mọi con số do `scripts/07_report_numbers_lastfm.py` sinh từ các file của pipeline (bảng đầy đủ kèm nguồn: `docs/data-report-lastfm-numbers.md`); `tests/test_data_report_lastfm.py` kiểm tra điều đó. **Chưa có kết quả mô hình nào trên Last-FM**, nên báo cáo này chỉ nói về dữ liệu và KG.

## Tóm tắt

- **Dữ liệu.** 1.542.856 cặp (user, item) trong đúng tập user/item mà tác giả KGAT dùng cho Last-FM, mỗi cặp giữ lần nghe sớm nhất. Không có rating: mọi tương tác là positive (`min_rating: 1`). Mốc t1 = 28/04/2014, t2 = 06/06/2014.
- **Chia theo mốc t1/t2** giữ lại 92,1% positive val và 86,8% positive test, với 10.618 và 7.149 user — khác hẳn MovieLens, nơi cold-start ăn phần lớn val/test. Đây là điểm mạnh chính của Last-FM với dự án này.
- **KG** lấy từ KG công bố kèm KGAT, có 9 quan hệ; ngày hiệu lực lấy từ bản chụp Freebase ngày 09/08/2015. KG không có ngày của statement, nên một fact chỉ được định ngày qua ngày tồn tại của đích (quyết định 0006 D).
- **Bộ lọc thời gian gần như không cắt gì**: 15 triple trên 329.804 tại t1, và KG tại t1 giống KG tại t2. Khác MovieLens, Last-FM không có fact nào rơi vào giữa hai mốc.
- **Thang Delta-t của Stage 2 tốt hơn MovieLens rõ rệt**: 21,2% triple nằm trong cửa sổ chuẩn hóa (`dt_norm < 1`), so với 0,7% của ML-100K và 0,0% của `ml-1m-u10`. Stage 2 trên Last-FM thực sự có độ phân giải để so fact cũ với fact mới (mục 4).
- **Chưa chạy grid nào.** Trên máy hiện có, grid đầy đủ của `lastfm` cần hàng trăm giờ, nên các lần chạy chính thức sẽ nằm trên tập con `lastfm-u25` (mục 5).

## 1. Nguồn dữ liệu

Tương tác đọc từ `data-last-fm/lfm1b-tracks/kgat_author_earliest.tsv`: mỗi dòng là một cặp (user, item) của bộ Last-FM mà tác giả KGAT công bố, kèm timestamp nghe sớm nhất của cặp đó, theo id đã remap của tác giả. Dùng đúng tập user/item của tác giả để KG `kg_final.txt` khớp trực tiếp, không cần bước liên kết thực thể.

KG lấy từ `knowledge_graph_attention_network/Data/last-fm`: `kg_final.txt`, `entity_list.txt` (id → Freebase `m.xxxx`), `relation_list.txt`, `item_list.txt`. Ngày hiệu lực lấy từ `data-last-fm/freebase-subset.tsv`, bản chụp Freebase ngày 09/08/2015, có ba cột `subject`, `predicate`, `object`; `subject` cùng dạng với `org_id` của `entity_list.txt` nên thực thể khớp trực tiếp.

So với MovieLens, Last-FM **không có rating**: dữ liệu là sự kiện nghe nhạc. Vì thế `min_rating: 1`, tức mọi tương tác là positive. Giả định về timestamp cũng khác và nhẹ hơn: timestamp của LFM-1b là lúc sự kiện nghe xảy ra, không phải lúc người dùng nhập điểm, nên không có vấn đề "rating nhập theo phiên" của MovieLens (mục 3 của `docs/data-report.md`).

## 2. Chia theo mốc t1/t2 và bằng chứng không rò rỉ

`scripts/02_build_splits.py --data configs/data/lastfm.yaml` chạy đúng quy trình của quyết định 0002 và `check_split` không báo lỗi nào.

| | `lastfm` | `lastfm-u25` | `lastfm-u10` |
|---|---|---|---|
| Cặp (user, item) sau khử trùng | 1.542.856 | 370.979 | 148.084 |
| t1 (UTC) | 28/04/2014 | 29/04/2014 | 29/04/2014 |
| t2 (UTC) | 06/06/2014 | 07/06/2014 | 06/06/2014 |
| Train positive | 1.220.988 | 236.053 | 63.386 |
| User / item của train | 22.040 / 44.788 | 5.321 / 18.665 | 1.936 / 6.887 |
| Val positive / user | 142.023 / 10.618 | 23.708 / 2.319 | 5.337 / 770 |
| Test positive / user | 133.896 / 7.149 | 22.419 / 1.608 | 5.286 / 559 |
| Positive val còn lại sau lọc cold-start | 92,1% | 63,9% | 36,0% |
| Positive test còn lại sau lọc cold-start | 86,8% | 60,4% | 35,7% |
| Positive bị 5-core loại khỏi train | 13.295 | 60.730 | 55.081 |

*Bảng 1. Kết quả chia theo thời gian. `lastfm-u25` là 5.892 / 23.566 user và `lastfm-u10` là 2.357 / 23.566 user, lấy ngẫu nhiên (seed 2019) theo quyết định 0008; `lastfm-u25` là tập của các lần chạy chính thức (quyết định 0013), `lastfm-u10` là tập nhỏ để phát triển.*

**Last-FM giữ lại nhiều val/test hơn MovieLens rất nhiều.** Trên ML-100K chỉ 12,6% positive val và 13,3% test sống sót, còn 66 và 51 user; trên Last-FM là 92,1% và 86,8%, còn 10.618 và 7.149 user. Nguyên nhân là timestamp của Last-FM phân bố đều theo thời gian, trong khi 77,9% rating của ML-100K nằm trong 24 giờ đầu của user nên hầu hết tương tác muộn thuộc về user mới. Hệ quả cho dự án: val của Last-FM đủ lớn để chọn cấu hình một cách đáng tin, điều mà 66 user val của ML-100K không làm được.

**Tập con chỉ giữ được một phần tính chất đó.** Lấy một phần user làm rơi nhiều item khỏi train, nên một phần val/test bị loại vì *cold item* (trên bộ đầy đủ chỉ 3,0% positive val và 4,6% positive test):

- `lastfm-u25` mất 28,8% positive val và 28,1% positive test vì cold item, giữ lại 63,9% và 60,4%, tức 2.319 user val và 1.608 user test; 5-core loại 60.730 trong 296.783 positive của giai đoạn train. Val vẫn lớn hơn nhiều so với 66 user val của ML-100K, nhưng không còn là val gần như nguyên vẹn của `lastfm` đầy đủ.
- `lastfm-u10` mất 55,2% positive val và 50,5% positive test vì cold item, và 5-core loại 55.081 trong 118.467 positive của giai đoạn train.

Đây là khác biệt quan trọng so với `ml-1m-u10` và là hạn chế phải nói khi báo cáo số của `lastfm-u25` (mục 5).

## 3. KG theo thời gian

**Quy tắc** là quy tắc của quyết định 0004 và 0006, không viết lại: KG tại mốc t chỉ giữ fact đã đúng trước t. Mã dùng lại nguyên `cutoff_facts` / `drop_reason`, `assign_ids` và `triple_dates` của `tkgat.kg.temporal`; `src/tkgat/kg/lastfm.py` chỉ đọc nguồn Last-FM và kiểm tra lại.

**KG Last-FM không có ngày của statement.** Vì vậy bằng chứng duy nhất về thời gian của một fact là ngày tồn tại của đích (quyết định 0006 D: một fact không thể đúng trước khi đích của nó tồn tại). Ngày đó là ngày sớm nhất trong ba vị từ `music.release.release_date`, `music.artist.active_start`, `people.person.date_of_birth`. Ba dạng literal của bản chụp đọc theo quyết định 0006: `gYear` là ngày cuối cùng của năm đó, `gYearMonth` là ngày cuối cùng của tháng đó, `date` là đúng ngày đó — trừ ngày kết thúc bằng `-01-01`, đọc là chỉ-có-năm nên cũng thành ngày cuối năm (0006 A); năm trước Công nguyên và giá trị không đọc được coi như không có ngày (0006 C). Head là item của train nên ngày phát hành của chính nó không bao giờ loại fact của nó (0006 B).

| | `lastfm` | `lastfm-u25` | `lastfm-u10` |
|---|---|---|---|
| Triple của KG gốc | 464.567 | 464.567 | 464.567 |
| Sau bộ lọc theo item (item ở một trong hai đầu) | 331.097 | 154.076 | 62.554 |
| Chở được trong định dạng 1-hop | 329.804 | 152.979 | 61.929 |
| Bị loại vì head không phải item của train | 1.293 | 1.097 | 625 |
| Thực thể / thực thể đích | 84.585 / 39.797 | 44.392 / 25.727 | 21.123 / 14.236 |
| Quan hệ | 9 | 9 | 9 |

*Bảng 2. Từ KG gốc tới KG của pipeline.*

**KG của repo là 1-hop item → thuộc tính**, nên 1.293 triple (0,4%) có head không phải item của train không chở được; chúng bị loại với lý do `head_not_train_item`, ghi vào `dropped.csv` và đếm trong `kg_meta.json`. Toàn bộ số đó thuộc hai quan hệ: `music.recording.canonical_version` (1.180) và `music.single.versions` (113). Số thực thể của pipeline (84.585) cao hơn số thực thể của bộ lọc gốc, vì một thực thể đích trùng Freebase id với một item của train vẫn được cấp id riêng — đúng quy ước đã dùng cho KG MovieLens (id space của tác giả là một, của repo là hai).

| Biến thể | `lastfm` t1 | `lastfm` t2 | `lastfm-u25` t1 | `lastfm-u25` t2 | `lastfm-u10` t1 | `lastfm-u10` t2 |
|---|---|---|---|---|---|---|
| default | 329.789 | 329.789 | 152.976 | 152.976 | 61.928 | 61.928 |
| drop_undated | 259.859 | 259.859 | 122.653 | 122.653 | 49.924 | 49.924 |
| static (tham chiếu) | 329.804 | 329.804 | 152.979 | 152.979 | 61.929 | 61.929 |

*Bảng 3. Số triple của KG theo biến thể và mốc.*

- **Bộ lọc thời gian gần như không cắt gì:** static trừ default là 15 triple ở `lastfm`, 3 triple ở `lastfm-u25` và 1 triple ở `lastfm-u10`, tức 0,0%. Đó là các bản phát hành có ngày cuối năm 2014, sau cả hai mốc. Đối chiếu: trên MovieLens bộ lọc cắt 1,3% (ML-100K) và 1,5% (ML-1M) số fact.
- **KG tại t1 giống KG tại t2** ở cả ba bộ: không có fact nào mang ngày rơi vào khoảng giữa 28/04/2014 và 06/06/2014, và số thực thể chỉ xuất hiện trong KG t2 là 0. Vì vậy Last-FM **không đo được** hiện tượng "fact tương lai rơi vào giữa hai mốc" mà *Titanic* minh họa trên ML-100K.
- **Fact không có ngày nào** chiếm 69.930 triple (21,2%) ở `lastfm`, 30.323 (19,8%) ở `lastfm-u25` và 12.004 (19,4%) ở `lastfm-u10`; biến thể drop_undated bỏ đúng các fact này. Tỷ lệ gần với mức 18,1% của MovieLens.

**Fact không có ngày tập trung theo quan hệ.** 4 trong 9 quan hệ hoàn toàn không có ngày: `common.topic.notable_types`, `music.single.versions`, `music.recording.canonical_version`, `music.recording.song` (đều 100,0%). Kế đó là `music.recording.engineer` 74,5% và `music.recording.producer` 29,0%. Ngược lại hai quan hệ lớn nhất gần như luôn có ngày: `music.recording.releases` 2,1% không có ngày và `music.recording.artist` 1,6%. Nói cách khác, drop_undated trên Last-FM chủ yếu cắt quan hệ phân loại và quan hệ phiên bản, giữ gần nguyên quan hệ bản phát hành và nghệ sĩ.

## 4. Thang Delta-t của Stage 2

Stage 2 của quyết định 0009 đặt Delta-t lên cạnh KG, lấy ngày hiệu lực của fact, và chặn `dt_norm` ở 1 (0009, mục 10) vì ngày của fact nằm trên thang thời gian khác với tương tác. Cửa sổ chuẩn hóa `dt_den` ở cả ba bộ là 9,20 năm, mốc `t_ref` là 28/04/2014 (`lastfm`) và 29/04/2014 (`lastfm-u25`, `lastfm-u10`).

| | `lastfm` | `lastfm-u25` | `lastfm-u10` |
|---|---|---|---|
| Triple trong cửa sổ (`dt_norm < 1`) | 21,2% | 24,4% | 27,5% |
| Triple bị chặn ở 1 (cũ hơn cửa sổ) | 57,6% | 55,7% | 53,1% |
| Triple không có ngày (`Delta-t = 0`) | 21,2% | 19,8% | 19,4% |
| Trung vị `dt_norm` của triple có ngày | 1,59 | 1,45 | 1,34 |

*Bảng 4. Phân bố `dt_norm` trên KG default tại t1.*

Đây là khác biệt đáng chú ý nhất so với MovieLens. Trên ML-100K chỉ 0,7% triple có `dt_norm < 1` và trên `ml-1m-u10` là 0,0%, nên như CLAUDE.md đã ghi, Stage 2 ở đó chỉ hạ trọng số fact-có-ngày so với fact-không-ngày, chứ **không** đo được cũ so với mới. Trên Last-FM thì 21,2% triple nằm trong cửa sổ (24,4% trên `lastfm-u25`, tập của các lần chạy chính thức), nên Stage 2 có độ phân giải thật. Cảnh báo vẫn phải nêu: 57,6% triple (55,7% trên `lastfm-u25`) vẫn bị chặn ở 1, nên kết luận chỉ áp dụng cho phần fact đủ mới.

## 5. Chi phí chạy và lựa chọn tập dữ liệu

`scripts/measure_kgat.py` đo thời gian và bộ nhớ trên GPU của máy hiện tại (một seed, vài epoch, không ghi gì ra đĩa và không mở file test). Con số đo được nằm trong CLAUDE.md, mục "Known facts", và trong quyết định 0013; đó là phép đo thời gian của epoch chưa tinh chỉnh, không phải kết quả.

Kết luận thực tế: bộ nhớ GPU không phải vấn đề (đỉnh dùng chưa tới một nửa dung lượng), nhưng grid đầy đủ 5 seed × 4 điểm × 300 epoch của `lastfm` cần hàng trăm giờ, tức không chạy được trên máy hiện có. Vì vậy, theo đúng tinh thần quyết định 0008 cho ML-1M:

- Các lần chạy **chính thức** của Last-FM nằm trên `lastfm-u25` (25% user, seed 2019). Cổng đo thời gian, thứ tự cắt giảm khi hết thời gian và các phần không chạy lại nằm trong quyết định 0013.
- `lastfm-u10` giữ vai trò tập nhỏ để phát triển và kiểm thử mã; không báo số chính thức trên nó.
- `lastfm` đầy đủ được ghi là **không khả thi với phần cứng hiện có**, và mọi con số của `lastfm-u25` luôn báo kèm tên run.
- Khi báo cáo số của `lastfm-u25`, phải nêu hạn chế của mục 2: tập con làm mất 28,8% positive val và 28,1% positive test vì cold item, nên nó chỉ giữ được một phần ưu điểm "val lớn" của `lastfm` đầy đủ.

## 6. Hạn chế đã biết

1. Freebase là bản chụp ngày 09/08/2015, không phải tri thức có vào năm 2014. Phải nói rõ mỗi khi báo cáo kết quả KG.
2. KG không có ngày của statement, nên fact chỉ được định ngày qua ngày tồn tại của đích. Một fact mà đích đã tồn tại trước mốc vẫn được giữ, dù liên kết có thể được tạo muộn hơn.
3. Ngày kết thúc bằng `-01-01` đọc là chỉ-có-năm (0006 A), nên một fact thật sự có ngày 1 tháng 1 bị đặt muộn hơn thực tế. Luôn thiên về phía an toàn, không bao giờ gây rò rỉ.
4. 1.293 triple không chở được trong định dạng 1-hop (mục 3). Muốn giữ chúng phải cho phép head không phải item, tức đổi cấu trúc KG dùng chung với MovieLens; chưa làm.
5. Một số ít thực thể đích dùng id dạng `g.` thay vì `m.`; bản chụp Freebase không có subject nào dạng `g.`, nên các thực thể đó chắc chắn không có ngày.
6. KG tại t1 giống KG tại t2, nên Last-FM không kiểm được đường rò rỉ "fact tương lai giữa hai mốc".
7. Bộ lọc thời gian chỉ cắt 15 triple, nên trên Last-FM phép so default với static gần như không có hiệu lực. RQ về rò rỉ qua KG vẫn phải trả lời bằng MovieLens.
8. **Hạn chế của chính tập cặp (user, item)**, ghi trong quyết định 0012 và không có kiểm tra tự động nào thấy được: tập cặp của bản Last-FM mà tác giả KGAT công bố chỉ chứa các cặp còn được nghe lại ở giai đoạn cuối, nên bản thân tập cặp đã được chọn bằng thông tin sau mốc, có thể thiên vị các cặp "sống lâu". Xem quyết định 0012 để biết con số và cách kiểm. Phải nêu hạn chế này cùng mọi kết quả Last-FM.
9. Khoảng thời gian của train, val và test rất lệch nhau (xem quyết định 0012): một mô hình có thể được lợi từ tính mới theo cách MovieLens không cho phép. Đây là đặc điểm của dữ liệu, phải báo cáo, không phải lỗi.

## 7. Việc tiếp theo khi đã có kết quả

**Ghi chú để dùng sau.** Các con số của báo cáo này là nền để phân tích khi đã có kết quả mô hình trên Last-FM. Khi đó, đem chúng ra đối chiếu theo các hướng sau:

1. **Val lớn có làm selection ổn định hơn không.** `lastfm` có 10.618 user val, so với 66 của ML-100K và 480 của ML-1M. Nếu chạy được, so độ lệch chuẩn giữa các seed của Recall@20 val trên Last-FM với MovieLens: đây là phép kiểm trực tiếp cho nhận định "selection trên 66 user val là nhiễu" trong CLAUDE.md.
2. **Stage 2 với thang Delta-t có độ phân giải.** 21,2% triple nằm trong cửa sổ, nên nếu Stage 2 vẫn không khác tập đối chứng (ngày fact bị trộn), kết luận "thời gian trên KG không giúp" sẽ mạnh hơn nhiều so với kết luận hiện có trên MovieLens, nơi thang Delta-t gần như bão hòa. Đối chiếu trực tiếp với Bảng 4.
3. **Stage 1 trên dữ liệu có Δt thật.** Trên ML-100K 53,0% user có mọi cạnh train trong vòng một giờ, nên Δt giữa các cạnh của một user gần như bằng 0. Last-FM là sự kiện nghe nhạc trải theo thời gian, nên Stage 1 ở đây được đo trong điều kiện thuận lợi hơn. Cần đo và ghi phân bố Δt của cạnh user-item trên Last-FM trước khi đọc kết quả Stage 1 (chưa làm; song song với mục 3b của `notebooks/01_eda.ipynb`).
4. **RQ2 (KG có giúp không) trên domain khác.** KG Last-FM dày hơn tương đối: 39.797 thực thể đích cho 44.788 item, 9 quan hệ âm nhạc. So KGAT với KGAT-no-KG trên Last-FM và trên MovieLens để biết kết luận RQ2 có phụ thuộc domain hay không.
5. **drop_undated có ý nghĩa thật trên Last-FM.** Nó cắt 21,2% triple, tập trung vào quan hệ phân loại và phiên bản, chứ không cắt quan hệ bản phát hành và nghệ sĩ. Đây là phép kiểm "quan hệ nào thực sự mang thông tin" rõ hơn trên Last-FM so với MovieLens.
6. **Tập đối chứng vẫn bắt buộc.** `tkgat_perm` và `tkgat_kg_perm` phải chạy cùng lúc với `tkgat` và `tkgat_kg`, như trên MovieLens. Không đọc kết quả Stage 1 hay Stage 2 nào mà chưa có tập đối chứng của nó.
7. **Nếu sau này có máy mạnh hơn**, chạy lại grid trên `lastfm` đầy đủ và so với `lastfm-u25` để biết tập con lệch bao nhiêu — đặc biệt vì tập con mất một phần val/test vì cold item (mục 2).
