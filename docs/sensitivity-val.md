# Độ nhạy của thứ tự baseline theo cách chia (chỉ val; tự sinh bởi `scripts/10_sensitivity.py`)

Nguồn: `results/registry.csv` (dòng val của điểm lưới được chọn trên val) và `meta.json` của từng split. Không đọc dòng test hay file test nào. Recall@20 trên val, trung bình ± std mẫu (`ddof=1`) qua seed; MostPop và MostPop recent là tất định nên chỉ có một giá trị.

| Cấu hình (run_name) | k-core | q_val / q_test | t1 / t2 (UTC) | user val | positive val | MostPop | MostPop recent | BPR-MF | Thứ tự (cao đến thấp) |
|---|---|---|---|---|---|---|---|---|---|
| ml-100k | 5 | 0,8 / 0,9 | 1998-03-07 / 1998-03-31 | 66 | 712 | 0,1700 | 0,2828 | 0,1739 ± 0,0133 | MostPop recent > BPR-MF > MostPop |
| ml-100k-k10 | 10 | 0,8 / 0,9 | 1998-03-07 / 1998-03-31 | 62 | 604 | 0,1743 | 0,2920 | 0,2003 ± 0,0073 | MostPop recent > BPR-MF > MostPop |
| ml-100k-q70-85 | 5 | 0,7 / 0,85 | 1998-02-09 / 1998-03-27 | 99 | 1.252 | 0,1251 | 0,2967 | 0,2125 ± 0,0078 | MostPop recent > BPR-MF > MostPop |
| ml-100k-q85-95 | 5 | 0,85 / 0,95 | 1998-03-27 / 1998-04-04 | 34 | 272 | 0,1455 | 0,2031 | 0,2809 ± 0,0207 | BPR-MF > MostPop recent > MostPop |

**Cách đọc.** Mỗi split có tập val khác nhau (khác user, khác positive), nên điểm val tuyệt đối **không so sánh được giữa các cấu hình** và không được dùng để xếp hạng hay chọn cấu hình. Chỉ so thứ tự ba mô hình trong từng dòng.

**Nhận xét (tự sinh).**

- `ml-100k` là cấu hình gốc, thứ tự: MostPop recent > BPR-MF > MostPop. Cặp có chênh lệch nhỏ hơn std của BPR-MF qua seed: BPR-MF và MostPop.
- `ml-100k-k10`: thứ tự giữ nguyên so với `ml-100k` (MostPop recent > BPR-MF > MostPop).
- `ml-100k-q70-85`: thứ tự giữ nguyên so với `ml-100k` (MostPop recent > BPR-MF > MostPop).
- `ml-100k-q85-95`: thứ tự đổi so với `ml-100k` (BPR-MF > MostPop recent > MostPop). Val chỉ có 34 user (< 50), nên "giữ nguyên" hay "đổi" đều là bằng chứng yếu.
- Cấu hình gốc cũng chỉ có 66 user val: chọn điểm lưới trên val còn nhiễu (xem `docs/study/` và CLAUDE.md).
