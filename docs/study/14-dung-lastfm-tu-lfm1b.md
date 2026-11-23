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

`configs/model/tkgat.yaml` dùng `time_on: both`: mỗi model/seed học thời gian của
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
  không tính lại min/max từ val/test. Các rule cutoff của dữ liệu vẫn áp dụng.
- `γ=0` khớp KGAT. Chỉ đổi công thức không cần generate lại dữ liệu;
  riêng thay đổi quy mô/10-core thì chạy lại 02b. Không đổi loss, sampler,
  vòng tối ưu CF/KG; việc chọn γ trên validation giữ nguyên.

Khi train, chọn `--model configs/model/tkgat.yaml`; các mode `interact`/`kg` cũ còn
để đối chiếu kết quả trước đây. `tkgat_perm.yaml` cũng dùng `both`, nhưng chỉ tráo
timestamp interaction, giữ ngày facts: đây là control riêng cho thời gian interaction.
Kết quả/checkpoint mới có `time_on: both`, không resume checkpoint hai stage cũ.

## Kaggle: chỉ chạy T-KGAT

1. Tạo gói code mới tại máy: `python scripts/kaggle_tkgat_joint.py --package-code`.
2. Upload hai Dataset Private: **`processed/lastfm-3m-k10/`** đã zip và
   **`data/export/kaggle/tkgat-joint-code.zip`**. Trainer chỉ đọc processed;
   ZIP thư mục export có thể giữ làm bản lưu, không cần upload. Không cần raw/Freebase.
3. Import `notebooks/10_kaggle_tkgat_joint.ipynb`, Add Input hai Dataset, chọn GPU.
   Đảm bảo các CSV trong Dataset processed đã được giải nén và nhìn thấy trong Input.
4. Chạy ô chuẩn bị/kiểm tra bộ nhớ. Nếu đủ thì đặt `RUN_TRAINING=True`;
   mỗi lần chạy ô train hoàn tất thêm một lượt gamma/seed, chạy lại để tiếp tục.
5. Lưu phiên bản **kèm output** `tkgat-joint/`. Phiên sau Add Input output cũ và
   đặt `RESTORE_FROM` tới thư mục đó; khôi phục cả registry/checkpoint/identity.

Cấu hình riêng `configs/model/tkgat_joint.yaml` không phụ thuộc baseline:
`lr=0.001`, `l2=l2_kg=0.00001`, dropout `0.1` được cố định trước thí nghiệm;
chọn gamma `{1,3,10}` trên validation trung bình 5 seed. Không chạy KGAT/BPR/MostPop
hay đối chứng; test chỉ chấm gamma được chọn sau khi hoàn tất đủ 15 lượt.
Resume chỉ ở ranh giới lượt hoàn tất, chưa khôi phục giữa epoch.

**Giới hạn:** script mới vẫn dùng trainer full-graph hiện tại. Với bộ full cũ đã dựng,
riêng graph và trạng thái embedding khi train có cận dưới khoảng **37 GiB VRAM**,
chưa tính propagation, attention, sampler và RAM để đọc/encode CSV. GPU 16 GB
không đáp ứng; preflight đọc metadata rồi dừng trước khi nạp dữ liệu lớn.
Không tự cắt dữ liệu hay giảm seed để làm vừa bộ nhớ. Đây chưa phải tối ưu streaming;
muốn train full trên Kaggle thông thường cần thay phần quản lý bộ nhớ của trainer.
Với bản 3 triệu/10-core, preflight tính lại theo metadata mới; chưa cam kết vừa RAM/VRAM
cho đến khi 02b hoàn tất và có số lượng entity thực tế. Run name tự lấy từ tên thư mục
processed (ví dụ `lastfm-3m-k10-joint`), không trộn kết quả bộ full cũ.
