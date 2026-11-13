# Chạy Last-FM trên Kaggle

Bộ chạy dùng dữ liệu đã processed **`lastfm-u25-latest`**, timestamp lần nghe muộn
nhất, và bản PyTorch trong `src/tkgat/`. Không đọc lại Drive và không dùng source
TensorFlow trong `knowledge_graph_attention_network/` để train.

## 1. Hai file cần tải lên Kaggle

- Notebook: `notebooks/04_kaggle_lastfm_latest.ipynb`.
- Dataset riêng tư: `data/export/kaggle/tkgat-lastfm-latest-kaggle.zip`.

Để tạo lại gói từ code và dữ liệu hiện có trên máy:

```powershell
python scripts/package_lastfm_kaggle.py
```

Lệnh đóng gói không train và không đổi nhánh/commit của repo đang làm việc. Nó tạo
một repo xuất tạm thời chỉ chứa code/cấu hình/tài liệu, đóng băng thành commit riêng
và lưu `source.bundle`. Dữ liệu processed được đóng gói bên ngoài Git. File
`SOURCE_SNAPSHOT.json` ghi commit gốc, trạng thái chỉnh sửa gốc và hash các file
source; đây không phải tuyên bố rằng checkout gốc đang sạch hoặc trùng nguyên vẹn
với commit của Triết. Manifest gói ghi commit xuất và SHA-256 từng file dữ liệu.

Trên Kaggle, tạo Dataset từ ZIP và giữ **Private**. Import notebook, thêm Dataset
bằng **Add Input**, chọn GPU, rồi chạy các ô chuẩn bị. Nếu Kaggle giữ nguyên ZIP
thay vì giải nén, notebook tự nhận và giải nén gói nhỏ vào `/kaggle/working/`.
Không cần API token, quyền Drive hay tải lại file nghe gốc.

## 2. Kiểm tra trước khi train

Notebook clone code từ Git bundle, kiểm tra hash và chép dữ liệu vào
`/kaggle/working/t-kgat/data/processed/lastfm-u25-latest/`. Không thay thế PyTorch
CUDA có sẵn trên Kaggle. Môi trường cần PyTorch, NumPy, pandas, SciPy, PyYAML và
matplotlib; phần code được import từ `src/` của gói.

Mặc định `RUN_MEASUREMENT = False` và `RUN_TRAINING = False`. Chạy ô chuẩn bị chưa
khởi chạy huấn luyện. Kiểm tra CUDA và tên GPU, sau đó đặt `RUN_MEASUREMENT = True`
để đo vài epoch trên train/val. Đo thời gian không mở file test hoặc ghi điểm vào
registry; notebook vẫn lưu log và thông tin môi trường.

Theo quy trình team, chốt thí nghiệm cho bộ `latest` trước khi chạy lưới chính thức.
Bộ này khác bản `lastfm-u25` lấy lần nghe sớm nhất trong báo cáo cũ; không kỳ vọng
tái lập đúng các số cũ. Kết quả vẫn được xem là sơ bộ khi chưa đóng băng/phê duyệt.

## 3. Chạy đủ bộ mô hình và chọn gamma

Khi sẵn sàng, đặt `RUN_TRAINING = True`. Notebook lần lượt chọn mô hình chưa hoàn
thành theo thứ tự:

| Mô hình | Lưới và seed |
|---|---|
| MostPop | Một cấu hình xác định |
| BPR-MF | 3 cấu hình × 5 seed |
| KGAT | 4 cấu hình × 5 seed |
| KGAT không KG | 4 cấu hình × 5 seed |
| T-KGAT Stage 1 | Gamma `{1, 3, 10}` × 5 seed |
| Đối chứng Stage 1 | Gamma đã chọn của Stage 1 × 5 seed |
| T-KGAT Stage 2 | Gamma `{1, 3, 10}` × 5 seed |
| Đối chứng Stage 2 | Gamma đã chọn của Stage 2 × 5 seed |

Mỗi mô hình học dùng các seed 2019–2023. Gamma được chọn **riêng cho từng stage**
bằng Recall@20 trung bình trên validation của cả 5 seed. Test chỉ chấm cấu hình đã
chọn sau khi đủ lưới; cả Recall và NDCG báo cáo từ cùng cấu hình đó. Không chọn
gamma theo test hoặc lấy seed tốt nhất. Các đối chứng giữ cấu hình/gamma đã chọn,
chỉ thay phép hoán vị tương ứng.

`STEPS_THIS_SESSION` mặc định là 1: một lần gọi mô hình chưa xong. Với KGAT/T-KGAT,
`NEW_KG_FITS_PER_STEP` mặc định là 2: tối đa hai lượt `(cấu hình, seed)` mới mỗi lần
gọi. Lượt hoàn tất có checkpoint và val row; lần gọi sau dùng `--resume`. Lưới và
5 seed không bị cắt giảm. Chỉ khi mọi lượt hoàn tất mới chọn cấu hình và chấm test.
Tăng số bước dựa vào thời gian đo được và thời lượng còn lại của phiên Kaggle.

MostPop chạy trọn một lần. BPR-MF hiện chạy trọn 15 lượt trong một bước, chưa có
resume giữa các seed. Vì vậy nên dành riêng một phiên cho bước BPR-MF. KGAT/T-KGAT
chỉ resume các lượt đã hoàn tất; nếu phiên bị ngắt giữa một lượt, lượt đó chạy lại
từ đầu. Không resume optimizer giữa epoch. GPU thứ hai không tự được sử dụng:
pipeline hiện dùng một GPU qua `--device cuda`.

## 4. Lưu kết quả và tiếp tục phiên sau

Sau mỗi bước, kể cả khi tiến trình con báo lỗi, wrapper cố lưu:

```text
/kaggle/working/tkgat-lastfm-results.zip
```

Gói chứa `results/registry.csv`, checkpoint, history, summary, kết quả từng user,
log, thông tin môi trường và báo cáo nếu đã tạo. Không chỉ giữ file `.pt`:
registry và kết quả từng user cũng cần thiết để tránh chấm test lại và làm Wilcoxon.

Lưu phiên bản notebook cùng output bằng chức năng lưu của Kaggle; có thể dùng
**Save Version → Save & Run All** cho phiên chạy đã cấu hình. Các giới hạn thời
lượng/quota thay đổi; kiểm tra trực tiếp trên tài khoản và chia lượt để phiên kết
thúc bình thường. Không coi file trong phiên tạm là đã được lưu vĩnh viễn.

Phiên tiếp theo:

1. Giữ Dataset code/dữ liệu ban đầu làm Input.
2. Thêm output phiên đã lưu trước đó làm Input để có `tkgat-lastfm-results.zip`.
3. Notebook tự tìm một gói kết quả; nếu có nhiều gói, đặt `RESTORE_ZIP` thành đường
   dẫn gói muốn tiếp tục. Nó kiểm tra commit source và hash meta trước khi khôi phục.
4. Chạy lại phần chuẩn bị và train. Code được clone mới từ bundle, không phụ thuộc
   vào `.git` của output phiên trước. Dữ liệu không thay đổi, lượt hoàn tất được giữ.

Trong cùng phiên, chạy lại ô train sẽ tiếp tục tiến độ đang có. Không khôi phục
gói cũ lên tiến độ mới. Không đổi code/cấu hình rồi trộn với checkpoint của gói cũ;
nếu cần đổi, tạo gói và tên thí nghiệm riêng.

## 5. Chạy nền riêng KGAT sau MostPop/BPR-MF

Dùng `notebooks/05_kaggle_kgat_resume.ipynb` khi hai baseline đã hoàn tất. Notebook này
chỉ gọi KGAT; thiếu Recovery hoặc thiếu kết quả baseline thì dừng báo lỗi, không
quay về chạy MostPop/BPR-MF. Nó nhận cả ZIP có `results/kaggle-run.json` và thư mục
`results/` đã giải nén trong Input, kiểm tra đúng commit/split rồi mới khôi phục.

Giữ nguyên Dataset code/data và `source.bundle` đã dùng cho baseline; không đóng gói
lại source khi chỉ đổi notebook. Upload thêm gói kết quả hiện tại làm Dataset Private
(có thể dùng `data/export/kaggle/tkgat-lastfm-kgat-recovery.zip`). Nếu có nhiều Recovery,
đặt `RESTORE_SOURCE` thành đường dẫn ZIP hoặc thư mục `results/` muốn dùng.

Notebook mặc định `RUN_TRAINING = True`, `NEW_KG_FITS_THIS_SESSION = 2`. Để kiểm tra
khôi phục trước mà chưa train, đặt `RUN_TRAINING = False`. Log phải có
`mostpop: complete (verified)` và `bprmf: complete (verified)`.

Để chạy khi tắt máy: bật GPU T4 rồi **Save Version → Save & Run All**. Kiểm tra log
của Version có `Running kgat only`; chạy cell trong Draft không phải chạy nền Version.
Sau mỗi lượt cấu hình–seed hoàn thành, notebook cập nhật ZIP. Lượt bị ngắt giữa chừng
vẫn phải chạy lại từ đầu. Giới hạn thời gian/quota Kaggle vẫn áp dụng; mức mặc định
chỉ chạy 2 lượt trong tổng 20 lượt KGAT. Muốn chạy nhiều hơn, tăng số lượt theo thời
gian đã đo được để phiên có thể kết thúc và lưu Output bình thường.

Sau khi Version hoàn tất, tải ZIP kết quả trong Output. Phiên sau thay Recovery cũ
bằng gói mới nhất hoặc gắn Output Version trước làm Input; tránh gắn đồng thời nhiều
bản kết quả khi chưa chỉ định `RESTORE_SOURCE`.

## 6. Bảng kết quả

Đặt `RUN_REPORT = True` khi có kết quả, hoặc dùng ô báo cáo sau khi hoàn tất bộ chạy.
Wrapper gọi `scripts/10_rq3_table.py`, đọc registry và kết quả từng user; không chép
số bằng tay. File nằm trong `reports/` và được đưa vào ZIP kết quả:

- `lastfm-latest.md`: bảng kết quả và so sánh.
- `models.csv`: trung bình ± độ lệch chuẩn mẫu trên seed.
- `pairs.csv`: Wilcoxon theo user và p-value hiệu chỉnh Holm.
- `figures/`: các đường học validation có sẵn.

Giữ cách nhóm Holm hiện có trong repo (riêng metric, riêng nhóm Stage 1/Stage 2).
Mô hình chưa hoàn thành hiện là chưa chạy, không có số giả. Dữ liệu, code, checkpoint
và kết quả nên giữ Private theo quyền sử dụng của nhóm.

Tài liệu Kaggle về [notebook, input và output](https://www.kaggle.com/docs/notebooks)
và [sử dụng GPU hiệu quả](https://www.kaggle.com/docs/efficient-gpu-usage).
