# Dựng Last-FM Temporal cho KGAT

Hướng dẫn dùng `scripts/generate_lastfm_drive.py` để tạo bộ **`lastfm-u25-latest`**:
lấy mẫu **25% user, seed 2019**, giữ **lần nghe muộn nhất** của mỗi cặp user–item,
chia theo thời gian và xuất định dạng KGAT kèm timestamp.

Cấu hình mặc định: [`configs/data/lastfm_u25_latest_drive.yaml`](../configs/data/lastfm_u25_latest_drive.yaml).
Dataset KGAT gốc được giữ nguyên. Chạy các lệnh từ thư mục gốc `t-kgat`.

## 1. Nguồn dữ liệu

Script đọc trực tiếp ZIP trên Drive; không cần tải và giải nén nguồn thủ công.

| Nguồn | Nội dung được sử dụng |
|---|---|
| [`data-last-fm.zip`](https://drive.google.com/file/d/1tXtohvT8RAFmz2vJo4vH90kkrpwD75jW/view), trong [thư mục Drive được repo chỉ định](https://drive.google.com/drive/folders/1NMkts7TVcFc9dDj1OsvsYXxK9Ms31Mfc) | File `lfm1b-tracks.inter`: lịch sử nghe gốc để tính timestamp lớn nhất của mỗi cặp |
| [`du-lieu-lastfm-hien-tai-09-11.zip`](https://drive.google.com/file/d/1C4dqi4zzv8KCUwq2fucR9stHqYG-85R2/view) | File `freebase/freebase-subset.tsv`: ngày tồn tại của entity phía đuôi trong KG; ZIP được kiểm tra SHA-256 trước khi dùng |
| `knowledge_graph_attention_network/Data/last-fm/` trên máy | Tập cặp, KG và ánh xạ ID gốc của tác giả KGAT |

File nghe gốc chứa user, track và timestamp Unix giây. Với mỗi cặp thuộc tập cặp KGAT,
script giữ `max(timestamp)` trên toàn bộ lịch sử, không phụ thuộc thứ tự dòng.
Sau đó mới lấy mẫu user và chia dữ liệu.

`kgat_author_earliest.tsv` chỉ lưu lần nghe **sớm nhất**. File này không được dùng
để dựng timestamp muộn nhất vì không còn các lượt nghe sau của cùng cặp.

Thư mục KGAT gốc trên máy cần đủ bảy file: `train.txt`, `test.txt`, `user_list.txt`,
`item_list.txt`, `entity_list.txt`, `relation_list.txt`, `kg_final.txt`.
Nếu chưa có, lấy repo gốc:

```powershell
git clone https://github.com/xiangwang1223/knowledge_graph_attention_network.git
```

## 2. Cài và chạy

Yêu cầu Python 3.10 trở lên. Nếu đã activate Conda, dùng `python` của môi trường đó:

```powershell
python -m pip install -e ".[drive]"
python scripts/generate_lastfm_drive.py --variant default
```

Lệnh trên dựng split, dựng KG `default` theo hai mốc thời gian và xuất dữ liệu KGAT
kèm timestamp. Script không chạy huấn luyện mô hình.

| Tham số | Ý nghĩa |
|---|---|
| `--data <file.yaml>` | Dùng cấu hình khác |
| `--author-data <thư-mục>` | Chỉ định vị trí dữ liệu KGAT gốc trên máy |
| `--variant default` | Lọc fact theo ngày và giữ fact không có ngày theo chính sách hiện tại |
| `--variant drop_undated` | Thêm điều kiện bỏ fact không có ngày sử dụng được |
| `--variant static` | Giữ KG tĩnh làm bản tham chiếu, không lọc theo ngày |
| `--variant all` | Dựng cả ba KG và xuất `default`; là mặc định nếu không truyền `--variant` |
| `--export-dir <thư-mục>` | Đổi thư mục xuất KGAT; đổi thư mục split bằng `processed_dir` trong bản sao cấu hình |

Script từ chối thư mục đầu ra đã có file. Chỉ dùng `--overwrite` khi chủ động muốn
thay thế kết quả. Không cần activate thêm `.venv` nếu đang dùng Conda.

## 3. Bản temporal khác bản gốc ở đâu?

| Bản KGAT gốc | Bản `lastfm-u25-latest` |
|---|---|
| Chỉ có train/test | Có train/val/test theo thời gian toàn cục, phân vị 0,8 và 0,9 |
| Không lưu timestamp lượt nghe trong file train/test | Có timestamp Unix giây cho từng cặp được giữ lại |
| Có cặp user–item lặp | Giữ **lần nghe muộn nhất**, tức `max(timestamp)` của mỗi cặp |
| Dùng tập user của bản phát hành | Lấy mẫu 25% user, seed 2019, trước bước chia và 5-core |
| Không có validation/mask trong định dạng gốc | Có validation và mask dựa trên các bản ghi đã rút gọn trước mốc cắt |
| Không có bước kiểm tra của pipeline này | Kiểm tra thời gian, giao tập, cold-start và train-only k-core trước khi lưu split |

Mỗi interaction là phản hồi ngầm định dương (`rating = 1`). 5-core chỉ tính trên train.
Val/test chỉ giữ user và item có trong train. Tỷ lệ cuối có thể khác 8/1/1 do lọc
k-core, cold-start và timestamp trùng tại mốc cắt.

## 4. Kết quả đầu ra

Với cấu hình mặc định và `--variant default`, dữ liệu processed nằm tại:

```text
data/processed/lastfm-u25-latest/
```

Gồm `train.csv`, `val.csv`, `test.csv` với cột `user, item, rating, ts`, các file
`seen_before_val.csv`, `seen_before_test.csv`, `meta.json` và thư mục `kg/default/`.
Thư mục KG chứa `kg_t1.tsv`, `kg_t2.tsv`, `kg_t1_dates.tsv`, `kg_t2_dates.tsv`,
`entities.csv`, `relations.csv`, `dropped.csv`, `kg_meta.json`.

Bản xuất để dùng với KGAT nằm tại:

```text
data/export/kgat/lastfm-u25-latest-default-timestamp/
```

| File | Ý nghĩa |
|---|---|
| `train.txt`, `val.txt`, `test.txt` | Một dòng `user item item ...`; tên validation trong bản xuất này là `val.txt` |
| `train_timestamp.txt`, `val_timestamp.txt`, `test_timestamp.txt` | Một dòng `user item:timestamp ...`; dùng cho thời gian trên cạnh user–item |
| `train_rating.txt`, `val_rating.txt`, `test_rating.txt` | Nhãn phản hồi của các cặp, trong Last-FM là 1 |
| `val_mask.txt`, `test_mask.txt` | Item cần mask theo các bản ghi được giữ trước mốc tương ứng |
| `kg_final.txt`, `kg_test.txt` | KG tại t1 cho train/val và tại t2 cho test |
| `user_list.txt`, `item_list.txt`, `entity_list.txt`, `relation_list.txt` | Ánh xạ ID của bản xuất |
| `entity_metadata.tsv` | Loại và nhãn entity |
| `interactions_chronological.tsv` | Các interaction theo thời gian, kèm split và timestamp |
| `dataset_meta.json` | Mốc cắt, số lượng và thông tin bản xuất |

Timestamp interaction và ngày KG là hai loại riêng. Các file `kg_t1_dates.tsv`,
`kg_t2_dates.tsv` trong thư mục processed chứa triple `h, r, t` kèm `ts` suy ra từ
Freebase; `ts` trống khi không có ngày sử dụng được.

## 5. Kiểm tra kết quả

Trong `data/processed/lastfm-u25-latest/meta.json`, kiểm tra:

- `clean.lastfm_timestamps.timestamp_policy` là `latest`.
- `clean.lastfm_timestamps.raw_interactions` trỏ tới nguồn Drive đã cấu hình.
- `clean.user_subsample.user_fraction` là `0.25`, `user_seed` là `2019`.
- `t1 < t2`; số cặp, user, item và SHA-256 các split được ghi từ lần chạy thực tế.

Trong bản xuất KGAT, `dataset_meta.json` có `timestamp_variant: true`. Trong
`kg/default/kg_meta.json`, xem số fact có ngày, các fact bị bỏ và `known_limitations`.

Không dùng số lượng hay SHA-256 của bộ `lastfm-u25` sớm nhất để đối chiếu bản này:
chuyển sang lần nghe muộn nhất có thể đổi mốc cắt, split và user/item còn lại.

## 6. Đọc Drive và lỗi kết nối

ZIP dữ liệu nghe gốc khoảng 7,4 GB được đọc và giải nén theo luồng; không ghi ZIP
hay file nghe thô xuống đĩa. ZIP hỗ trợ khoảng 181 MB được giữ trong RAM. Cần thêm
RAM cho tập cặp, pandas và KG; quá trình quét toàn bộ lịch sử có thể mất nhiều thời gian.

Khi mất kết nối, phần đọc dữ liệu thô thử nối lại tối đa 5 lần từ byte đã nhận,
giữ trạng thái giải nén và timestamp đã quét trong cùng tiến trình. Nếu server
không trả đúng đoạn byte để tiếp tục an toàn, script dừng. Nếu Python đã thoát
hoặc vượt số lần thử, lần chạy mới vẫn quét từ đầu vì không có checkpoint trên đĩa.

Quyền đọc của Python tách biệt với kết nối Google Drive trong chat. Với file riêng tư,
xem cách cấu hình quyền đọc trong [lastfm-drive.md](lastfm-drive.md#drive-access).

## 7. Giới hạn cần biết

Timestamp interaction là **lần nghe cuối cùng được quan sát**. Ngày của fact KG được
suy ra từ ngày tồn tại của entity phía đuôi trong Freebase, không phải ngày tạo
statement hay thời điểm user nghe nhạc. KGAT gốc không có ngày statement; Freebase
là bản chụp 2015-08-09, nên các giới hạn trong `kg_meta.json` vẫn áp dụng.

Giữ lần nghe cuối làm mất các lượt nghe trước của cùng cặp. Nếu cặp được nghe cả
trước và sau mốc cắt, nó có thể chỉ còn ở giai đoạn sau; mask không đại diện cho
toàn bộ lịch sử nghe trước mốc. Các kiểm tra ranh giới vẫn chạy, nhưng không loại
bỏ hạn chế do chọn lần nghe cuối trên toàn bộ lịch sử. Xem
[decision 0015](decisions/0015-lastfm-latest-timestamps.md).

Tập cặp vẫn là tập KGAT đã phát hành, nên giữ hạn chế chọn cặp theo cửa sổ tháng 04/2014
được mô tả trong [decision 0012](decisions/0012-lastfm-dataset.md). Đây chưa phải bộ cặp
không giới hạn cửa sổ thời gian.

KGAT gốc chỉ đọc `train.txt`, `test.txt`, `kg_final.txt`. Loader và bước đánh giá cần
được điều chỉnh để dùng validation, mask, timestamp và `kg_test.txt` theo giao thức này.
