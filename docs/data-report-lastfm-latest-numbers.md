# Số liệu của bản Last-FM `lastfm-u25-latest` (của Phôn)

Sinh tự động bởi `scripts/11_numbers_latest_lastfm.py` từ các file do pipeline ghi ra. **Không sửa tay.**
Mọi con số trong `notebooks/00_present_lastfm.ipynb` phải có trong file này (kiểm tra bởi `tests/test_latest_numbers.py`).
Giá trị "chưa có" nghĩa là nguồn chưa tồn tại; script điền số khi nguồn xuất hiện.
Định dạng: dấu chấm ngăn hàng nghìn, dấu phẩy thập phân; phần trăm có ký hiệu %.

## Quy ước cố định

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `conv.freebase_snapshot` | 09/08/2015 | tkgat.kg.lastfm.FREEBASE_SNAPSHOT |

## 1. Dữ liệu và chia theo thời gian

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm-u25-latest.pairs` | 370.979 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.t1` | 23/06/2014 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.t2` | 19/07/2014 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.train_pos` | 236.339 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.users` | 5.367 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.items` | 19.165 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.val_pos` | 25.860 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.val_users` | 1.516 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.val_kept_share` | 69,7% | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.val_dropped_cold_user_share` | 6,1% | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.val_dropped_cold_item_share` | 24,2% | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.test_pos` | 24.244 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.test_users` | 1.341 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.test_kept_share` | 65,4% | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.test_dropped_cold_user_share` | 12,8% | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.test_dropped_cold_item_share` | 21,9% | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.train_pos_before_kcore` | 296.783 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.kcore_removed` | 60.444 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.users_sampled` | 5.892 / 23.566 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.user_fraction` | 25% | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.user_seed` | 2019 | data/processed/lastfm-u25-latest/meta.json |
| `lastfm-u25-latest.last_ts` | 02/09/2014 | data/processed/lastfm-u25-latest/train.csv, val.csv, test.csv, cột ts lớn nhất |

## 2. KG theo thời gian

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm-u25-latest.kg.entities` | 45.346 | data/processed/lastfm-u25-latest/kg/default/kg_meta.json |
| `lastfm-u25-latest.kg.relations` | 9 | data/processed/lastfm-u25-latest/kg/default/kg_meta.json |
| `lastfm-u25-latest.kg.default.t1` | 156.826 | data/processed/lastfm-u25-latest/kg/default/kg_meta.json |
| `lastfm-u25-latest.kg.default.dated_t1` | 125.640 | data/processed/lastfm-u25-latest/kg/default/kg_meta.json |
| `lastfm-u25-latest.kg.undated_t1_share` | 19,9% | data/processed/lastfm-u25-latest/kg/default/kg_meta.json |

## 3. Thang Delta-t của Stage 2

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm-u25-latest.dt.window_years` | 0,23 | data/processed/lastfm-u25-latest/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u25-latest.dt.t_ref` | 23/06/2014 | data/processed/lastfm-u25-latest/train.csv |
| `lastfm-u25-latest.dt.undated_share` | 19,9% | data/processed/lastfm-u25-latest/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u25-latest.dt.inside_window_share` | 0,0% | data/processed/lastfm-u25-latest/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u25-latest.dt.capped_share` | 80,1% | data/processed/lastfm-u25-latest/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u25-latest.dt.median_dated` | 59,53 | data/processed/lastfm-u25-latest/train.csv + kg/default/kg_t1_dates.tsv |

## 4. Thời gian của cạnh train (Stage 1) của bản latest

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm-u25-latest.edge.window_days` | 83 | data/processed/lastfm-u25-latest/train.csv, max - min của ts |
| `lastfm-u25-latest.edge.first_day` | 01/04/2014 | data/processed/lastfm-u25-latest/train.csv, ts nhỏ nhất |
| `lastfm-u25-latest.edge.dt_norm_median` | 0,73 | data/processed/lastfm-u25-latest/train.csv, dt_norm = (t_ref - ts) / dt_den |
| `lastfm-u25-latest.edge.dt_norm_below_0_25` | 10,2% | data/processed/lastfm-u25-latest/train.csv, dt_norm = (t_ref - ts) / dt_den |
| `lastfm-u25-latest.edge.last_7d_share` | 3,3% | data/processed/lastfm-u25-latest/train.csv |
| `lastfm-u25-latest.edge.last_30d_share` | 16,2% | data/processed/lastfm-u25-latest/train.csv |
| `lastfm-u25-latest.week.first_full` | 39.197 | data/processed/lastfm-u25-latest/train.csv qua eda.weekly_counts, bỏ tuần đầu và tuần cuối (không đủ) |
| `lastfm-u25-latest.week.last_full` | 7.717 | data/processed/lastfm-u25-latest/train.csv qua eda.weekly_counts, bỏ tuần đầu và tuần cuối (không đủ) |
| `lastfm-u25-latest.spread.user_under_1h` | 0,93% | data/processed/lastfm-u25-latest/train.csv qua eda.node_time_spread, trải < 1 giờ |
| `lastfm-u25-latest.spread.user_std_days_median` | 9,2 | data/processed/lastfm-u25-latest/train.csv |

## 6. Thời gian chạy và epoch tốt nhất (registry, chỉ dòng val)

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm-u25-latest.hours.device.cpu` | 0,0 | registry gộp của các zip kết quả của Phôn (phase=val), tổng cột seconds |
| `lastfm-u25-latest.hours.device.Tesla_T4` | 55,1 | registry gộp của các zip kết quả của Phôn (phase=val), tổng cột seconds |
| `lastfm-u25-latest.hours.cpu.mostpop` | 0,0 | registry gộp của các zip kết quả của Phôn (phase=val), tổng cột seconds |
| `lastfm-u25-latest.hours.Tesla_T4.bprmf` | 0,2 | registry gộp của các zip kết quả của Phôn (phase=val), tổng cột seconds |
| `lastfm-u25-latest.hours.Tesla_T4.kgat` | 19,3 | registry gộp của các zip kết quả của Phôn (phase=val), tổng cột seconds |
| `lastfm-u25-latest.hours.Tesla_T4.kgat_nokg` | 8,6 | registry gộp của các zip kết quả của Phôn (phase=val), tổng cột seconds |
| `lastfm-u25-latest.hours.Tesla_T4.tkgat` | 9,7 | registry gộp của các zip kết quả của Phôn (phase=val), tổng cột seconds |
| `lastfm-u25-latest.hours.Tesla_T4.tkgat_kg` | 10,2 | registry gộp của các zip kết quả của Phôn (phase=val), tổng cột seconds |
| `lastfm-u25-latest.hours.Tesla_T4.tkgat_kg_perm` | 4,2 | registry gộp của các zip kết quả của Phôn (phase=val), tổng cột seconds |
| `lastfm-u25-latest.hours.Tesla_T4.tkgat_perm` | 2,9 | registry gộp của các zip kết quả của Phôn (phase=val), tổng cột seconds |
| `lastfm-u25-latest.best_epoch.bprmf.min` | 55 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.bprmf.max` | 300 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.kgat.min` | 70 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.kgat.max` | 300 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.kgat_nokg.min` | 55 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.kgat_nokg.max` | 300 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.mostpop.min` | 0 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.mostpop.max` | 0 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.tkgat.min` | 45 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.tkgat.max` | 295 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.tkgat_kg.min` | 65 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.tkgat_kg.max` | 290 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.tkgat_kg_perm.min` | 125 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.tkgat_kg_perm.max` | 190 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.tkgat_perm.min` | 45 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.tkgat_perm.max` | 130 | registry gộp của các zip kết quả của Phôn (phase=val), mọi điểm lưới và seed |
| `lastfm-u25-latest.best_epoch.bprmf.chosen.min` | 100 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.bprmf.chosen.max` | 300 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.kgat.chosen.min` | 70 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.kgat.chosen.max` | 165 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.kgat_nokg.chosen.min` | 55 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.kgat_nokg.chosen.max` | 100 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.mostpop.chosen.min` | 0 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.mostpop.chosen.max` | 0 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.tkgat.chosen.min` | 45 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.tkgat.chosen.max` | 205 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.tkgat_kg.chosen.min` | 80 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.tkgat_kg.chosen.max` | 255 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.tkgat_kg_perm.chosen.min` | 125 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.tkgat_kg_perm.chosen.max` | 190 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.tkgat_perm.chosen.min` | 45 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25-latest.best_epoch.tkgat_perm.chosen.max` | 130 | registry gộp của các zip kết quả của Phôn (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
