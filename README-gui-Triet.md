# Nhờ Triết chạy T-KGAT Stage 2 trên Kaggle

Stage 2 có **15 lượt train** gồm gamma `{1, 3, 10}` × 5 seed. Mặc định chạy thêm
**2 lượt mỗi phiên**, tối đa 8 phiên để hoàn tất.

## Setup lần đầu

Phôn gửi 3 file sau.

- [08_kaggle_tkgat_stage2_resume.ipynb](notebooks/08_kaggle_tkgat_stage2_resume.ipynb).
- `tkgat-lastfm-latest-kaggle.zip` chứa code/data. Khi upload, đặt tên Dataset **LastFM-Code-Data**.
- `tkgat-lastfm-results.zip` chứa KGAT hoàn chỉnh. Đặt tên Dataset **LastFM-Stage2-Recovery**.

Giữ nguyên tên file ZIP. Các phiên sau cập nhật cùng Dataset Recovery bằng **New Version**.

1. Tạo notebook Kaggle riêng, import file `08` qua **File → Import Notebook**.
2. Upload hai ZIP thành Dataset **Private** và gắn vào **Input**.
3. Bật **Settings → Accelerator → GPU T4**, giữ cấu hình mặc định trong notebook.
4. Bấm **Save Version → Save & Run All**.
5. Mở **Logs của Version**, chờ `kgat: complete (verified)` và
   `Running tkgat Stage 2 only; fit budget 1/2`. Khi đó có thể đóng tab, tắt máy.

## Sau mỗi phiên

1. Khi Version chạy thành công, vào **Output**, tải đúng file
   **`tkgat-lastfm-stage2-results.zip`** chứa tiến độ vừa train.
2. Mở Dataset Recovery → **New Version**, gỡ các file/thư mục cũ trong bản mới rồi
   upload ZIP vừa tải để chỉ còn một bộ kết quả.
3. Trong notebook, cập nhật Input sang **phiên bản Recovery mới nhất**. Chỉ gắn một
   bản Recovery để notebook tự tìm được.
4. Giữ notebook `08`, **Save Version → Save & Run All** để chạy tiếp 2 lượt mới.

Các dòng seed cũ xuất hiện ngay đầu log là đọc checkpoint. `pending` nghĩa là còn
lượt cần chạy. Lượt hoàn tất được giữ; lượt bị ngắt giữa chừng sẽ train lại từ đầu.

Khi log báo **`tkgat_kg: complete`**, tải ZIP cuối và gửi lại Phôn. Notebook đã tự
chọn gamma theo validation và chấm test đủ 5 seed.

Nếu gặp `Recovery path does not exist`, đặt `RESTORE_SOURCE = None` khi Input chỉ
có một Recovery. Với lỗi khác, gửi Phôn phần traceback cùng ZIP của phiên
thành công gần nhất.
