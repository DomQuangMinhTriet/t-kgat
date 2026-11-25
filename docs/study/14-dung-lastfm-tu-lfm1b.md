# Dựng Last-FM: interaction Drive, facts Freebase

Cập nhật 23/11/2026: Phôn đã đồng ý dùng dump Freebase đầy đủ ngoài Drive.
**Không đọc interaction hoặc `kg_final.txt` của paper.** Tham khảo format, 9 relation và cách trích KG.

**Cấu hình mới:** Phôn chọn **10-core trên train + khoảng 3 triệu interaction**.
`01b` vẫn dựng cache từ toàn bộ lịch sử gốc; `02b` mới chọn nhóm user bằng train.
Số user/item/facts không bị ép bằng paper, tổng interaction sau lọc cũng không cam kết đúng tuyệt đối.

## Ba nguồn dữ liệu chứa gì?

| Nguồn | File và dữ liệu liên quan | Dùng trong luồng hiện tại |
| --- | --- | --- |
| [Drive LFM-1b](https://drive.google.com/drive/folders/1NMkts7TVcFc9dDj1OsvsYXxK9Ms31Mfc) | `Bản sao của data-last-fm.zip` chứa `data-last-fm/lfm1b-tracks/lfm1b-tracks.inter`: `user_id`, `tracks_id`, `timestamp`. Đây là lịch sử nghe nhạc gốc. | Nguồn **interaction và timestamp** cho `01b`; gom mỗi cặp user–track theo lần nghe sớm nhất. |
| [Drive hỗ trợ](https://drive.google.com/drive/folders/12AUJ1uM9lXvm4jjp4BALLN1MLnTVEZDM) | `du-lieu-lastfm-hien-tai-09-11.zip` chứa `kgat-output/last-fm-temporal/item_list.txt` (mapping raw track → MID) và `freebase/freebase-subset.tsv` (Freebase đã trích một phần, có ngày/tên entity). Trong 9 relation đã chọn, subset chỉ có facts của `music.recording.releases`, thiếu 8 relation còn lại. | Dùng **mapping** `org_id → freebase_id`, bỏ ID remap cũ. Mapping có nguồn gốc paper và độ phủ một phần. Subset không dùng làm nguồn facts chính cho luồng 2-hop. |
| [Freebase trên Archive.org](https://archive.org/details/freebase-rdf-latest) | [`freebase-rdf-latest.gz`](https://archive.org/download/freebase-rdf-latest/freebase-rdf-latest.gz): dump RDF Freebase đầy đủ, chứa triples `(subject, predicate, object)`, các thuộc tính ngày và tên entity. | Nguồn **facts, bằng chứng ngày và tên entity** cho `01c`; trích 2-hop theo cả hai chiều, giữ chiều RDF gốc và chỉ chọn 9 relation đã chốt. |

[KGAT §4.1](https://hexiangnan.github.io/papers/kdd19-KGAT.pdf) và
[KB4Rec](https://github.com/RUCDM/KB4Rec#how-to-get-freebase-subgraph-with-our-linkage)
là tài liệu tham khảo phương pháp/schema; không lấy facts đã export của paper.

**Facts/relations cụ thể lấy trực tiếp từ Freebase:** `01c` đọc file
[`freebase-rdf-latest.gz`](https://archive.org/download/freebase-rdf-latest/freebase-rdf-latest.gz)
được lưu trên Archive.org. Mỗi fact là một triple RDF `(subject, predicate, object)`;
`predicate` chính là relation nối hai entity. Script stream và giải nén trong lúc đọc.

Danh sách **9 loại relation** chỉ là schema tham khảo từ paper, khai báo trong
`PAPER_RELATIONS`; script tìm và giữ các triples có thật trong Freebase khớp danh sách
này. Các liên kết cụ thể như `track A → artist → artist B` lấy từ dump, không sao chép
facts đã export của paper. Mapping track → MID vẫn dùng bảng Drive đã nêu ở trên.

Nguồn Google cũ trả 403 khi kiểm tra. Bản lưu trữ đọc được, có ngày metadata
**16/04/2015**; không gọi nó là dump 09/08/2015. Script pin kích thước/SHA-1 và
kiểm tra SHA-256 nguồn qua các lượt quét. `freebase-subset.tsv` cũ thiếu 8 relation,
nên chỉ còn là nguồn cho chế độ 1-hop cũ.

## Chạy tiếp sau 01b

Giữ nguyên `data/raw/lastfm-raw/pairs.tsv` và `pairs.tsv.meta.json` đã tạo.
Dùng các lệnh trong mục **Dựng lại bản khoảng 3 triệu / 10-core** bên dưới;
không cần quét lại nguồn đã có cache hợp lệ.

`01c` quét dump 3 lượt: lấy hop 1, hop 2, rồi ngày/tên của các entity đã tìm thấy.
Nguồn gzip khoảng 31,3 GB/lượt, tổng truyền khoảng 94 GB trước retry; cần nhiều thời
gian. **Không lưu toàn bộ dump nguồn**, chỉ lưu phần đã trích và database tạm.

## Rules dữ liệu

| Mục | Rule |
| --- | --- |
| Interaction | Cache toàn bộ lịch sử; mỗi cặp giữ lần nghe sớm nhất, `rating=1`. Bản nhỏ giữ lịch sử của nhóm user được chọn từ train. |
| Split | Phân vị 0,8/0,9: train `<t1`, val `[t1,t2)`, test `≥t2`. |
| 10-core / cold-start | 10-core lặp chỉ trên train; mỗi user có ≥10 track khác nhau, mỗi track có ≥10 user. Val/test chỉ giữ user/item có trong train. |
| Chọn quy mô | Chốt t1/t2 từ toàn bộ pairs gốc trước khi lấy mẫu. Xếp user đủ điều kiện trong train bằng SHA-256(`seed:user`), seed 2019; chọn prefix có core train gần `0,8 × 3.034.796` nhất, rồi tính lại 10-core. Không dùng số lượng/điểm val/test để chỉnh prefix. |
| Mask | Toàn bộ lịch sử của nhóm user đã chọn trước t1/t2, kể cả item bị core loại; khi encode/export giới hạn theo vocabulary train. |
| Entity ID | Bản lấy mẫu chỉ cấp ID cho mọi item train và entity xuất hiện trong KG t1/t2 sau lọc; bỏ entity không còn dùng, đồng bộ triples/timestamps/mapping. Facts bị loại vẫn có MID nguồn, ID số để trống nếu endpoint không còn trong namespace. |
| Mapping | Không gộp interaction của hai raw track khác nhau; thiếu MID vẫn giữ interaction nếu đạt rule split. Không tự đoán tên track. |
| Cache 01c | Seed bằng các MID trong mapping cung cấp; cache là tập dư. 02b chọn lại seed chỉ từ item train sau 10-core/lấy mẫu. |
| KG | Lấy cạnh có seed/lân cận ở head **hoặc tail**, tối đa 2-hop, chỉ qua 9 relation đã chốt; không tự tạo cạnh ngược hay đổi predicate. |
| Lọc chất lượng | Entity xuất hiện ≥10, relation ≥50 triples. Đếm từ graph hợp lệ tại t1; một lượt entity rồi một lượt relation, giữ cùng tiêu chí tại t2 để không chọn bằng facts tương lai. |
| Ngày | Năm → 31/12, tháng → cuối tháng; giữ quy ước `-01-01` cũ. Timestamp là ngày tồn tại muộn hơn trong các ngày biết của **hai endpoint**; không phải ngày tạo statement. |
| Thiếu ngày / cutoff | Không có ngày đọc được thì timestamp trống và giữ fact. Có ngày phải trước ngày UTC cutoff; bỏ cả cạnh có head tương lai. Tính lại reachability sau lọc, KG t1 ⊆ KG t2. |
| Tái lập / đối soát | Nguồn, số lượng, hash và lý do loại được ghi lại; dòng sai định dạng làm dừng xử lý. |

Ngưỡng 10/50 lấy từ paper; paper không nói đủ chi tiết thứ tự/lặp lọc. Cách lọc
một lượt trên t1 và giới hạn cố định 9 relation là lựa chọn được ghi rõ của bản này,
không cam kết tái tạo chính xác số facts của paper.

## File đầu ra

- `data/raw/lastfm-freebase2hop/freebase.tsv` + `.meta.json`: cache cạnh/bằng chứng từ dump, nguồn, ngày metadata, hash và thống kê 3 lượt quét.
- `data/processed/lastfm-3m-k10/`: train/val/test, history masks, KG t1/t2, báo cáo lấy mẫu và lý do loại.
- **`data/export/kgat/lastfm-3m-k10/`**: format KGAT, mapping, timestamp interaction/fact, `fact_sources.tsv`, `fact_date_evidence.tsv`, `dataset_meta.json`.

`02b` dùng đĩa cho interaction; `1GB` là bộ đệm DuckDB, còn mapping/KG vẫn cần RAM.
`--scratch-dir` chọn đĩa tạm. Mapping hiện vẫn một phần nên không bảo đảm mọi track có KG.
`01c` đã hoàn tất lúc **00:10 ngày 23/11/2026** (giờ Việt Nam); đã kiểm tra SHA-256
cache khớp manifest và đủ cả 3 lượt quét. `02b` đã hoàn tất lúc **09:25 cùng ngày**;
đã kiểm tra lại hash của đủ 24 file export và đối soát interaction không có cặp mất
không rõ lý do. Đây là **bản full 5-core cũ**; bản 3 triệu/10-core cần chạy lại **02b** theo lệnh dưới.
Xem [chi tiết](../lastfm-raw-rebuild.md) và [quyết định 0018](../decisions/0018-lastfm-freebase-two-hop.md).

## Dựng lại bản khoảng 3 triệu / 10-core

Chạy tại thư mục gốc bằng Conda `python`. Nếu còn cache đã hoàn tất, hai lệnh đầu
kiểm tra hash/quy tắc và dùng lại, không quét lại 1 tỷ lượt nghe hay 3 lượt dump.
Nếu cache chưa có thì trích từ nguồn gốc như trước; không tự ghi đè cache hỏng/khác rules.

```powershell
python scripts/01b_cut_lastfm.py --all-history --out data/raw/lastfm-raw/pairs.tsv --reuse-existing
python scripts/01c_extract_freebase.py --reuse-existing
python scripts/02b_build_lastfm_from_raw.py --pairs data/raw/lastfm-raw/pairs.tsv --kcore 10 --target-interactions 3034796 --user-seed 2019 --engine duckdb --memory-limit 1GB --out data/processed/lastfm-3m-k10 --export-dir data/export/kgat/lastfm-3m-k10
```

Mặc định mới của CLI `02b` là `--kcore 10 --target-interactions 3034796`;
`--target-interactions 0` giữ toàn bộ user. Không ghi đè thư mục output cũ nếu chưa có `--overwrite`.
Giữ nguyên API mặc định cũ cho các script gọi nội bộ; lệnh trên chỉ rõ profile mới.

`sampled_users.tsv` lưu danh sách user/prefix; `sampling_report.json` lưu seed, budget,
các lần kiểm tra core và số lượng thực tế. `validation_report.json` đối soát **toàn bộ**
pairs gốc, gồm lý do `outside_user_sample` theo từng split. Những dòng ngoài nhóm user
chỉ được đếm, không xuất hàng trăm triệu dòng vào `dropped_interactions.csv`; có thể
tái dựng chúng từ cache gốc và danh sách user. File CSV dropped chỉ chứa core/cold-start
của nhóm đã chọn. Tổng train/val/test có thể lệch mục tiêu vì không cắt lịch sử/ép test.

## Attention gộp interaction và facts (23/11/2026)

`configs/model/tkgat_joint.yaml` dùng `time_on: both`: mỗi model/seed học thời gian của
interaction và facts trong **cùng một lần train**, không cần chạy hai stage thời gian.
Chỉ mở rộng cách tính độ cũ; giữ nguyên điểm KGAT, một γ chung và softmax chung:

\[
\pi_{time}(h,r,t)=\pi(h,r,t)-\gamma\,\delta_r\,\Delta_{type}(ts),
\qquad
\alpha(h,r,t)=\frac{\exp\pi_{time}(h,r,t)}
{\sum_{(h,r',t')\in\mathcal N_h}\exp\pi_{time}(h,r',t')}.
\]

\[
\Delta_{type}(ts)=\operatorname{clip}
\left(\frac{b_{type}-ts}{b_{type}-a_{type}},0,1\right).
\]

- **Interaction:** `a_I = min(train_ts)`, `b_I = max(train_ts)` chỉ từ interaction train.
- **Fact:** `a_F`, `b_F` là min/max timestamp **có ngày** của facts trong KG t1 sau lọc.
- `δ_r = 1` cho cả hai loại và cạnh ngược; chọn cửa sổ bằng loại cạnh.
  Ví dụ interaction train `[100,500]`, fact train `[10,90]`: interaction `ts=300`
  và fact `ts=50` đều có `Δ=0,5`.
- Fact thiếu ngày có `Δ=0`; không có fact có ngày hoặc cửa sổ rộng 0 thì loại đó
  có `Δ=0`, không ảnh hưởng cửa sổ của loại còn lại.
- Chốt **cả hai cửa sổ tại train/t1**, giữ nguyên khi dùng KG t2 để test;
  không tính lại min/max khi chỉ đổi graph để đánh giá. Với **refit**, khởi tạo model
  mới trên train+val/KG t2 và tính lại hai cửa sổ từ chính phần refit đó; không dùng
  interaction test. Các rule cutoff của dữ liệu vẫn áp dụng.
- **Báo cáo cửa sổ:** `tkgat_joint_refit_summary.json` → `tkgat.time_windows`
  ghi `selection` và `refit`, mỗi phần có `interaction`/`fact` với `min_ts`, `max_ts`,
  `width_seconds`, `width_days`. `refit_vs_selection` ghi dịch min/max, chênh độ rộng,
  tỷ số độ rộng và phần trăm thay đổi; không tính tỷ số khi độ rộng cũ bằng 0.
  File riêng `tkgat_joint_window_summary.json` có cửa sổ selection ngay đầu lượt chạy,
  bổ sung so sánh refit sau khi chọn gamma, **trước khi train refit/chấm test**, kể cả
  khi tạm dừng theo ngân sách. `--val-only` chỉ ghi selection, không đọc KG t2/test.
  Chưa khóa cửa sổ: cùng γ có thể phạt khác khi min/max đổi. Ví dụ `[100,500]` thành
  `[100,1000]`, cạnh `ts=300` đổi tuổi chuẩn hóa từ `0,5` thành khoảng `0,778`.
  Cần xem số đo trước khi quyết định thay đổi chính sách; không chọn dựa vào điểm test.
- `γ=0` khớp KGAT. Chỉ đổi công thức không cần generate lại dữ liệu;
  riêng thay đổi quy mô/10-core thì chạy lại 02b. Không đổi loss, sampler,
  vòng tối ưu CF/KG; việc chọn γ trên validation giữ nguyên.

Khi train trên Kaggle, chọn `configs/model/tkgat_joint.yaml` qua wrapper bên dưới.
`tkgat.yaml` / `tkgat_perm.yaml` giữ mode Stage 1 cũ để đối chiếu lịch sử; mode gộp
và đối chứng tráo cả hai timestamp nằm ở `tkgat_both.yaml` / `tkgat_both_perm.yaml`.
Kết quả/checkpoint mới có `time_on: both`, không resume checkpoint hai stage cũ.

## Kaggle: chỉ chạy T-KGAT

1. Tạo gói code mới tại máy: `python scripts/kaggle_tkgat_joint.py --package-code`.
2. Upload hai Dataset Private: **`processed/lastfm-3m-k10/`** đã zip và
   **`data/export/kaggle/tkgat-joint-code.zip`**. Trainer chỉ đọc processed;
   ZIP thư mục export có thể giữ làm bản lưu, không cần upload. Không cần raw/Freebase.
3. Import `notebooks/10_kaggle_tkgat_joint.ipynb`, Add Input hai Dataset, chọn GPU.
   Đảm bảo các CSV trong Dataset processed đã được giải nén và nhìn thấy trong Input.
4. Chạy kiểm tra bộ nhớ rồi ô `RUN_BENCHMARK=True`, giữ `RUN_TRAINING=False`.
   Ô này đo 5 batch CF/KG, in ước lượng thời gian một epoch; chưa chọn model/chấm test.
   Dựa vào số đo mới bật train. `MAX_NEW_RUNS=2`, `EPOCHS_PER_FIT=None` hoàn tất
   tối đa 2 lượt gamma/seed hoặc refit/seed rồi dừng; vẫn lưu checkpoint mỗi epoch.
   Muốn dừng sớm theo epoch thì đặt `EPOCHS_PER_FIT` là số nguyên dương.
5. Lưu phiên bản **kèm output** `tkgat-joint/`. Phiên sau Add Input output cũ và
   đặt `RESTORE_FROM` tới thư mục đó; khôi phục cả registry/checkpoint/identity.
   Phiên đầu chưa có result nên để `RESTORE_FROM=None`. Các phiên sau giữ nguyên
   Input code/processed, chỉ thay result bằng output **mới nhất**; output này đã chứa
   tiến độ tích lũy. Có thể Add Input trực tiếp output notebook trước, không cần tải về
   rồi upload lại; nếu upload Dataset result riêng thì chỉ cần thư mục `tkgat-joint/`.
   Notebook chỉ hiện lượt đang chạy/tiếp tục, lượt đã có result và lượt tiếp theo;
   ẩn log batch/loss/metadata. Log chi tiết vẫn ở `tkgat-joint/training.log`, lỗi vẫn hiện.

Cấu hình riêng `configs/model/tkgat_joint.yaml` không phụ thuộc baseline:
`lr=0.001`, `l2=l2_kg=0.00001`, dropout `0.1` được cố định cho lần chạy này,
theo điểm Triết báo từ cổng tập nhỏ; chưa chọn lại bằng KGAT trên dữ liệu mới;
chọn gamma `{1,3,10}` trên validation trung bình 5 seed. Không chạy KGAT/BPR/MostPop
hay đối chứng trong notebook này. Sau đủ 15 lượt, chọn gamma rồi refit 5 seed trên
train+val với số epoch là median best epoch validation, không early stopping, rồi
chấm test bằng model refit. Mỗi lần chạy ô train hoàn tất tối đa **2 lượt selection/refit**;
selection và refit dùng chung bộ đếm 2 lượt. Lưu toàn bộ output, phiên sau Add Input
output đó và đặt `RESTORE_FROM` để tiếp tục các lượt còn lại.
Nếu đặt thêm `EPOCHS_PER_FIT`, một lượt đang dở có thể dừng trước khi đủ 2 lượt.
Checkpoint mỗi epoch lưu weights, cả hai Adam,
best validation, bộ đếm early stopping và RNG; tiếp tục từ epoch đã lưu, không chạy lại
cả lượt gamma/seed. Lượt đang dở chưa được đưa vào chọn gamma hay chấm test.

**Sửa sau phiên timeout 24/11/2026:** log Kaggle báo hết thời gian sau `43213.3s`
(khoảng 12 giờ), không đủ bằng chứng kết luận OOM từ mã 137. Trainer tính propagation
toàn CKG ở mỗi batch CF nên kiểm tra vừa bộ nhớ không chứng minh train kịp thời gian.
Code mới in tiến độ batch/epoch, lưu `*.epoch.pt` cùng history CSV sau mỗi epoch.
Không thay công thức, batch size hay tham số thí nghiệm; đây chưa phải tối ưu tốc độ.
Benchmark là ước lượng từ các batch đầu, chưa gồm validation/lưu checkpoint, không
cam kết cả lượt vừa một phiên Kaggle. Không tiếp tục được trọng số phiên cũ nếu chưa
có checkpoint `.pt`. Khi ngắt giữa epoch chỉ mất phần epoch chưa lưu; cần lưu output
và khôi phục `RESTORE_FROM` để tiếp tục ở phiên khác, cùng code/config và loại thiết bị.
Giới hạn 2 lượt không phải giới hạn thời gian: chưa bảo đảm 2 lượt vừa 12 giờ.
Nếu một epoch tự nó vượt 12 giờ thì cần tối ưu trainer. Tổng 20 lượt có thể chia thành
10 phiên nếu mỗi phiên hoàn tất đủ 2 lượt; timeout hoặc giới hạn epoch có thể làm tăng số phiên.

### Tham số Kaggle đã chuẩn bị theo tin Triết (23/11/2026)

| Tham số | Giá trị |
| --- | --- |
| Learning rate / schedule | `0.001`, cố định (`none`; trainer hiện không dùng scheduler) |
| `l2` / `l2_kg` | `0.00001` / `0.00001` |
| Message dropout | `0.1` |
| Embedding / layers | `dim=64`, `kge_dim=64`, layers `[64,32,16]` |
| Batch CF / KG | `1024` / `2048` |
| `max_epochs` | **150** theo tin đính chính cuối của Triết, thay cho 300 |
| Validation / patience | Mỗi 5 epoch; dừng sau 10 lần đánh giá không cải thiện |
| Gamma / seed | `{1,3,10}` × `{2019,2020,2021,2022,2023}`; đủ 15 lượt |
| Chọn cấu hình | Recall@20 validation trung bình 5 seed; từng lượt giữ best epoch trên validation |

**Phạm vi tích hợp #48/#50:** giữ `time_on: both`, công thức cửa sổ riêng của Phôn,
bản dữ liệu khoảng 3 triệu/10-core và cấu hình chạy riêng T-KGAT. Không dùng tau.
Đã thêm refit train+val và lưu checkpoint cho các lượt refit. Mặc định mọi relation
có ngày được decay; vẫn công bố đó là proxy endpoint, không phải ngày bắt đầu statement.
Các đối chứng/ablation và bộ công cụ tập nhỏ của Triết được giữ để chạy riêng khi cần.
Xem [quyết định tích hợp 0019](../decisions/0019-joint-windows-and-refit-integration.md).

ZIP code phải được tạo lại sau khi sửa YAML. Notebook in cấu hình thực tế trước khi
train; giữ output mới cho cấu hình 150 epoch có refit, không khôi phục checkpoint cũ
khác source/config/mode. Wrapper mặc định `--refit`; `--no-refit` dành cho cách đánh giá
checkpoint cũ. `--val-only` chỉ chọn trên validation, không refit/chấm test.

**Giới hạn:** script mới vẫn dùng trainer full-graph hiện tại. Với bộ full cũ đã dựng,
riêng graph và trạng thái embedding khi train có cận dưới khoảng **37 GiB VRAM**,
chưa tính propagation, attention, sampler và RAM để đọc/encode CSV. GPU 16 GB
không đáp ứng; preflight đọc metadata rồi dừng trước khi nạp dữ liệu lớn.
Không tự cắt dữ liệu hay giảm seed để làm vừa bộ nhớ. Đây chưa phải tối ưu streaming;
muốn train full trên Kaggle thông thường cần thay phần quản lý bộ nhớ của trainer.
Với bản 3 triệu/10-core, preflight tính lại theo metadata mới; chưa cam kết vừa RAM/VRAM
cho đến khi 02b hoàn tất và có số lượng entity thực tế. Run name tự lấy từ tên thư mục
processed (ví dụ `lastfm-3m-k10-joint`), không trộn kết quả bộ full cũ.
