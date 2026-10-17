# Chủ đề 6 — Tích hợp dữ liệu, liên kết thực thể và tính đúng theo thời điểm

> Người trình bày: **Phôn** · Ngày: T6 23/10 · Đọc kèm code: `src/tkgat/data/clean.py`, `src/tkgat/kg/linking.py`, `scripts/03_link_entities.py`, `scripts/04_fetch_wikidata.py`

Đây là phần đề tài gắn với chuyên ngành **Hệ thống thông tin**: ghép dữ liệu từ nhiều nguồn, kiểm soát chất lượng, và bảo đảm dữ liệu "đúng tại thời điểm" (point-in-time correctness), một yêu cầu quen thuộc trong data warehouse.

## 1. Bức tranh tích hợp của đề tài

```
MovieLens (u.item: id, tên, năm)
    │  KB4Rec ml2fb (MovieLens id → Freebase mid)
    ▼
Freebase mid (vd. m.0dr_4 = Titanic)
    │  Wikidata P646 "Freebase ID"
    ▼
Wikidata item (vd. Q44578 = Titanic)
    │  18 quan hệ + ngày tháng (P577, P585, P580, P569, P571)
    ▼
facts.tsv → KG theo mốc t1 / t2
```

Ba nguồn, hai bước nối. Mỗi bước đều có thể sai, nên mỗi bước phải được **đo** (độ phủ, tỉ lệ lỗi) và **ghi lại** (checksum, ngày tải, phiên bản).

## 2. Làm sạch dữ liệu nguồn
Những gì nhóm đã gặp ở ML-100K (xem `src/tkgat/data/clean.py`, `docs/data-dictionary.md`):
- **18 tiêu đề trùng:** cùng một phim có 2 id khác nhau, nên gộp về một id.
- **Item 267 "unknown":** không có tên, không liên kết được, nên loại.
- **Trùng cặp (user, item):** gộp lại, giữ lần đầu (chủ đề 2).
- **Kiểm chứng file tải về:** ML-100K lấy từ bản sao GitHub vì grouplens bị chặn, nên đối chiếu MD5 `0e33842e24a9c977be4e0107933c0723` với một bản sao thứ hai.

## 3. Liên kết thực thể (entity linking)
**Định nghĩa:** xác định một bản ghi ở nguồn A và một bản ghi ở nguồn B cùng chỉ một đối tượng thực tế.

| Cách | Mô tả | Trong đề tài |
|---|---|---|
| Dùng khóa có sẵn | Bảng ánh xạ đã được người khác xây | KB4Rec (MovieLens → Freebase); Wikidata P646 |
| So khớp chuỗi | So tên + năm, có chuẩn hóa ("Godfather, The (1972)" → "the godfather") | Dùng để **kiểm tra** KB4Rec |
| Học máy | Huấn luyện mô hình phân loại cặp | Không dùng |

**Kiểm tra liên kết là bắt buộc**, không được tin hoàn toàn bảng có sẵn:
- Ví dụ lỗi tìm được: "Clubland" (1998) bị KB4Rec nối với *Cloudland Canyon State Park*.
- Số liệu: KB4Rec phủ 97,2% item và 99,5% rating của ML-100K. ML-100K → ML-1M khớp 1.647/1.663 item (99%). ML-1M id = ML-20M id (92,7% khớp tên YAGO).
- Đo **precision** bằng cách lấy mẫu ngẫu nhiên khoảng 50 liên kết và kiểm tay; báo cáo tỉ lệ đúng.

## 4. Wikidata: mô hình dữ liệu cần biết
- Mỗi **item** có id `Q...`, mỗi **property** có id `P...`.
- Một **statement** = (item, property, value) + **qualifiers** + references.
- Qualifier về thời gian: **P585** point in time (vd. ngày nhận giải), **P580** start time, **P582** end time.
- Thuộc tính ngày của chính entity: **P577** publication date (phim), **P571** inception (công ty, giải thưởng), **P569** date of birth (người).
- **Độ chính xác (precision) của ngày:** 9 = năm, 10 = tháng, 11 = ngày. Ngày "1998" với precision 9 được lưu thành `1998-01-01`, **không phải** ngày 1/1 thật.

**Quy tắc của nhóm:** ngày chỉ có năm thì coi là **cuối năm** (31/12). Ngày chỉ có tháng thì coi là cuối tháng. Lý do: đây là lựa chọn **bảo thủ**, vì một fact chỉ được dùng khi chắc chắn nó đã xảy ra trước mốc. Coi là đầu năm sẽ có nguy cơ đưa tri thức tương lai vào.

## 5. Tính đúng theo thời điểm (point-in-time correctness)
Khái niệm từ data warehouse và feature store: **khi tái tạo trạng thái tại thời điểm $t$, chỉ được dùng dữ liệu đã biết tại $t$.** Áp dụng cho KG (quyết định 0004):

| Loại fact | Quy tắc với mốc $t$ | Ví dụ |
|---|---|---|
| Có ngày | Giữ nếu ngày < $t$ | *Titanic* award_received Best Picture (23/03/1998): **không** có trong KG tại $t_1$ = 07/03/1998, **có** trong KG tại $t_2$ = 31/03/1998 |
| Trỏ tới entity chưa tồn tại | Bỏ | Phim A followed_by phim B ra năm 2000 |
| Không có ngày, hai đầu đều đã tồn tại | Giữ (mặc định); chạy thêm biến thể "bỏ hết fact không ngày" | *Titanic* genre romance film |

- KG cho train/val dựng tại $t_1$, KG cho test dựng tại $t_2$.
- Phải báo cáo tỉ lệ fact có ngày / không ngày theo **từng quan hệ**.

**Lưu ý quan trọng:** Wikidata là ảnh chụp **hôm nay** (2026). Thứ ta lọc được là *ngày sự kiện xảy ra*, không phải *ngày sự kiện được ghi vào KG*. Đó là giới hạn cần nói rõ trong báo cáo.

## 6. Dòng dõi dữ liệu (data lineage) và tái lập
- `data/checksums.json`: SHA-256 của mọi file thô (kiểm tra bằng `scripts/01_verify_data.py`).
- `data/raw/kb4rec/SOURCE_COMMIT.txt`: commit của KB4Rec đã dùng.
- `data/raw/wikidata/`: `facts.tsv` kèm ngày tải và checksum.
- `results/registry.csv`: mỗi kết quả gắn với commit, config, seed.
- Dữ liệu thô **không đưa lên GitHub** vì giấy phép MovieLens và KB4Rec không cho phân phối lại; ai clone repo thì chạy `scripts/01_verify_data.py` để kiểm tra bản của mình.

## 7. Liên hệ với repo
| Phần | Vị trí |
|---|---|
| Làm sạch tiêu đề trùng, item unknown | `src/tkgat/data/clean.py` |
| Đọc KB4Rec, kiểm tra tên + năm | `src/tkgat/kg/linking.py`, `scripts/03_link_entities.py` |
| Truy vấn SPARQL, 18 quan hệ, precision | `scripts/04_fetch_wikidata.py` |
| Từ điển dữ liệu | `docs/data-dictionary.md` |
| Quy tắc KG theo mốc | `docs/decisions/0003-entity-linking.md`, `0004-kg-temporal-rules.md` |

## 8. Tự kiểm tra
1. Vẽ chuỗi nối MovieLens → Freebase → Wikidata và nêu một lỗi có thể xảy ra ở mỗi bước.
2. Vì sao phải kiểm tra lại bảng KB4Rec dù đó là dữ liệu đã được công bố?
3. Precision 9, 10, 11 của ngày Wikidata nghĩa là gì?
4. Vì sao ngày chỉ có năm được coi là cuối năm? Nếu coi là đầu năm thì rủi ro gì?
5. Phát biểu nguyên tắc point-in-time correctness bằng ví dụ *Titanic*.
6. KG cho tập test được dựng tại mốc nào? Vì sao?
7. Wikidata hôm nay có thể chứa thông tin mà năm 1998 chưa ai biết. Quy tắc 0004 giải quyết được bao nhiêu phần vấn đề đó?
8. Vì sao không push dữ liệu thô lên GitHub? Người khác tái lập bằng cách nào?
9. Qualifier P585 khác thuộc tính P577 thế nào?

## 9. Tài liệu đọc
- Christen. *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection.* Springer 2012 — chương 1–2.
- Vrandečić, Krötzsch. *Wikidata: A Free Collaborative Knowledgebase.* CACM 2014.
- Wikidata Help: Dates — trang hướng dẫn về `timePrecision` (https://www.wikidata.org/wiki/Help:Dates).
- Zhao et al. *KB4Rec.* Data Intelligence 2019.
- Kimball, Ross. *The Data Warehouse Toolkit* (3rd ed.) — chương về slowly changing dimensions, để hiểu tư duy "đúng tại thời điểm".
