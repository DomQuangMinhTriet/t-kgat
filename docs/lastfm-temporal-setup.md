# Dựng Last-FM Temporal cho KGAT

Hướng dẫn này tạo dataset mới tại `knowledge_graph_attention_network/Data/last-fm-temporal/`.
Dataset Last-FM gốc của tác giả không bị sửa.

## 1. Tải và giải nén

Tải `data-last-fm.zip`:

> [Google Drive — data-last-fm](https://drive.google.com/drive/folders/1NMkts7TVcFc9dDj1OsvsYXxK9Ms31Mfc?usp=sharing)

Giải nén ZIP vào thư mục gốc `t-kgat`.

Windows PowerShell:

```powershell
cd C:\path\to\t-kgat
Expand-Archive "$env:USERPROFILE\Downloads\data-last-fm.zip" -DestinationPath .
```

Linux/macOS:

```bash
cd /path/to/t-kgat
unzip ~/Downloads/data-last-fm.zip -d .
```

Kiểm tra có đủ hai file:

```text
data-last-fm/lfm1b-tracks/lfm1b-tracks.inter
data-last-fm/lfm1b-tracks/kgat_author_earliest.tsv
```

`kgat_author_earliest.tsv` là cache đã quét sẵn hơn một tỷ event. Không xóa file này, nếu không script phải quét
lại file 29 GB.

Nếu chưa có source và dataset Last-FM gốc của tác giả KGAT, chạy:

```bash
git clone https://github.com/xiangwang1223/knowledge_graph_attention_network.git
```

## 2. Cài và chạy

Yêu cầu Python 3.10 trở lên:

```bash
python -m pip install -e .
python scripts/generate_lastfm_temporal_kgat.py --kg-policy static
```

Kết quả nằm tại:

```text
knowledge_graph_attention_network/Data/last-fm-temporal/
```

Script không ghi đè thư mục đã tồn tại. Muốn tạo bản khác, truyền `--output <thư-mục-mới>`.

## 3. Kiểm tra kết quả

Mở `last-fm-temporal/dataset_meta.json` và kiểm tra:

```text
interaction_leakage_problems: []
timestamp_cache_hit: true
n_users: 22040
n_items: 44788
train: 1220988
valid: 142023
test: 133896
```

## 4. Bản temporal khác bản gốc ở đâu?

| Bản gốc | Bản temporal |
|---|---|
| Chỉ có train/test | Có train/valid/test theo thời gian, tỷ lệ ban đầu 8/1/1 |
| Không lưu timestamp | Có timestamp cho từng interaction |
| Có cặp user-item lặp | Giữ lần nghe sớm nhất |
| Không có validation/mask | Có validation và mask item đã nghe |
| Không kiểm tra leakage | Có kiểm tra thời gian, giao tập, cold-start và k-core |

Script khai thác dữ liệu từ các file sau:

| File nguồn | Dùng để làm gì? |
|---|---|
| `data-last-fm/lfm1b-tracks/lfm1b-tracks.inter` | Lấy timestamp gốc của interaction |
| `data-last-fm/lfm1b-tracks/kgat_author_earliest.tsv` | Cache timestamp sớm nhất của mỗi cặp user-item; có file này thì không cần quét lại 29 GB |
| `Data/last-fm/train.txt`, `test.txt` | Xác định đúng tập cặp user-item mà tác giả KGAT đã sử dụng |
| `Data/last-fm/user_list.txt`, `item_list.txt` | Đổi ID gốc sang ID của KGAT và giữ Freebase ID của item |
| `Data/last-fm/kg_final.txt` | Lấy các KG fact gốc, sau đó chỉ giữ fact nối với item còn lại trong train |
| `Data/last-fm/entity_list.txt`, `relation_list.txt` | Giữ ánh xạ entity/relation khi tạo KG mới |

Các file kết quả chính:

- `train.txt`, `valid.txt`, `test.txt`: dùng cho train, chọn mô hình và test.
- `*_timestamp.txt`: T-KGAT dùng để tính `Δt` hoặc time decay.
- `valid_mask.txt`, `test_mask.txt`: loại item user đã nghe khi xếp hạng.
- `*_events.tsv`: interaction từng dòng, đã sắp tăng dần theo thời gian.
- `kg_final.txt`, `kg_valid.txt`, `kg_test.txt`: KG cho từng giai đoạn.
- `dataset_meta.json`: cut-off, thống kê và kết quả kiểm tra leakage.

## Lưu ý

KGAT gốc chỉ đọc `train.txt`, `test.txt`, `kg_final.txt`; nó chưa tự dùng timestamp, validation hay mask. Muốn khai
thác temporal data phải mở rộng loader/model để đọc các file mới.

Hiện lệnh `--kg-policy static` chỉ temporal hóa interaction. KG gốc không có ngày của từng fact nên ba KG snapshot
đang giống nhau và chưa thể xem là KG temporal leakage-free.

Chỉ chia sẻ archive theo đúng điều khoản sử dụng của LFM-1b; ưu tiên Drive riêng cho thành viên dự án.
