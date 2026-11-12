# Số liệu của báo cáo dữ liệu Last-FM

Sinh tự động bởi `scripts/07_report_numbers_lastfm.py` từ các file do pipeline ghi ra. **Không sửa tay.**
Mọi con số trong `docs/data-report-lastfm.md` phải có trong file này (kiểm tra bởi `tests/test_data_report_lastfm.py`).
Giá trị "chưa có" nghĩa là nguồn chưa tồn tại; script điền số khi nguồn xuất hiện.
Định dạng: dấu chấm ngăn hàng nghìn, dấu phẩy thập phân; phần trăm có ký hiệu %.

## Quy ước cố định

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `conv.freebase_snapshot` | 09/08/2015 | tkgat.kg.lastfm.FREEBASE_SNAPSHOT |
| `conv.date_predicates` | 3 | tkgat.kg.lastfm.DATE_PREDICATES |
| `conv.quantiles` | 0,8 / 0,9 | configs/data/lastfm.yaml |
| `conv.kcore` | 5 | configs/data/lastfm.yaml |
| `conv.min_rating` | 1 | configs/data/lastfm.yaml |
| `conv.seeds` | 5 | configs/model/kgat.yaml |
| `conv.grid_points` | 4 | configs/model/kgat.yaml |
| `conv.max_epochs` | 300 | configs/model/kgat.yaml |
| `conv.max_epochs_bprmf` | 300 | configs/model/bprmf.yaml |

## 1. Dữ liệu và chia theo thời gian

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm.pairs` | 1.542.856 | data/processed/lastfm/meta.json |
| `lastfm.t1` | 28/04/2014 | data/processed/lastfm/meta.json |
| `lastfm.t2` | 06/06/2014 | data/processed/lastfm/meta.json |
| `lastfm.train_pos` | 1.220.988 | data/processed/lastfm/meta.json |
| `lastfm.users` | 22.040 | data/processed/lastfm/meta.json |
| `lastfm.items` | 44.788 | data/processed/lastfm/meta.json |
| `lastfm.val_pos` | 142.023 | data/processed/lastfm/meta.json |
| `lastfm.val_users` | 10.618 | data/processed/lastfm/meta.json |
| `lastfm.val_kept_share` | 92,1% | data/processed/lastfm/meta.json |
| `lastfm.val_dropped_cold_user_share` | 5,0% | data/processed/lastfm/meta.json |
| `lastfm.val_dropped_cold_item_share` | 3,0% | data/processed/lastfm/meta.json |
| `lastfm.test_pos` | 133.896 | data/processed/lastfm/meta.json |
| `lastfm.test_users` | 7.149 | data/processed/lastfm/meta.json |
| `lastfm.test_kept_share` | 86,8% | data/processed/lastfm/meta.json |
| `lastfm.test_dropped_cold_user_share` | 8,6% | data/processed/lastfm/meta.json |
| `lastfm.test_dropped_cold_item_share` | 4,6% | data/processed/lastfm/meta.json |
| `lastfm.train_pos_before_kcore` | 1.234.283 | data/processed/lastfm/meta.json |
| `lastfm.kcore_removed` | 13.295 | data/processed/lastfm/meta.json |
| `lastfm.last_ts` | 02/09/2014 | data/processed/lastfm/train.csv, val.csv, test.csv, cột ts lớn nhất |
| `lastfm-u25.pairs` | 370.979 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.t1` | 29/04/2014 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.t2` | 07/06/2014 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.train_pos` | 236.053 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.users` | 5.321 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.items` | 18.665 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.val_pos` | 23.708 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.val_users` | 2.319 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.val_kept_share` | 63,9% | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.val_dropped_cold_user_share` | 7,3% | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.val_dropped_cold_item_share` | 28,8% | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.test_pos` | 22.419 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.test_users` | 1.608 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.test_kept_share` | 60,4% | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.test_dropped_cold_user_share` | 11,5% | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.test_dropped_cold_item_share` | 28,1% | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.train_pos_before_kcore` | 296.783 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.kcore_removed` | 60.730 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.users_sampled` | 5.892 / 23.566 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.user_fraction` | 25% | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.user_seed` | 2019 | data/processed/lastfm-u25/meta.json |
| `lastfm-u25.last_ts` | 01/09/2014 | data/processed/lastfm-u25/train.csv, val.csv, test.csv, cột ts lớn nhất |
| `lastfm-u10.pairs` | 148.084 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.t1` | 29/04/2014 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.t2` | 06/06/2014 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.train_pos` | 63.386 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.users` | 1.936 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.items` | 6.887 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.val_pos` | 5.337 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.val_users` | 770 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.val_kept_share` | 36,0% | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.val_dropped_cold_user_share` | 8,7% | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.val_dropped_cold_item_share` | 55,2% | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.test_pos` | 5.286 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.test_users` | 559 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.test_kept_share` | 35,7% | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.test_dropped_cold_user_share` | 13,8% | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.test_dropped_cold_item_share` | 50,5% | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.train_pos_before_kcore` | 118.467 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.kcore_removed` | 55.081 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.users_sampled` | 2.357 / 23.566 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.user_fraction` | 10% | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.user_seed` | 2019 | data/processed/lastfm-u10/meta.json |
| `lastfm-u10.last_ts` | 01/09/2014 | data/processed/lastfm-u10/train.csv, val.csv, test.csv, cột ts lớn nhất |

## 2. KG theo thời gian

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm.kg.author_triples` | 464.567 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.item_seeded` | 331.097 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.head_seeded` | 329.804 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.dropped_not_1hop` | 1.293 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.dropped_not_1hop_share` | 0,4% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.entities` | 84.585 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.tail_entities` | 39.797 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.relations` | 9 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.default.t1` | 329.789 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.default.t2` | 329.789 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.default.dated_t1` | 259.859 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.drop_undated.t1` | 259.859 | data/processed/lastfm/kg/drop_undated/kg_meta.json |
| `lastfm.kg.drop_undated.t2` | 259.859 | data/processed/lastfm/kg/drop_undated/kg_meta.json |
| `lastfm.kg.drop_undated.dated_t1` | 259.859 | data/processed/lastfm/kg/drop_undated/kg_meta.json |
| `lastfm.kg.static.t1` | 329.804 | data/processed/lastfm/kg/static/kg_meta.json |
| `lastfm.kg.static.t2` | 329.804 | data/processed/lastfm/kg/static/kg_meta.json |
| `lastfm.kg.static.dated_t1` | 259.874 | data/processed/lastfm/kg/static/kg_meta.json |
| `lastfm.kg.filtered_t1` | 15 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.filtered_t1_share` | 0,0% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.undated_t1` | 69.930 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.undated_t1_share` | 21,2% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.items_with_facts` | 44.788 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.entities_only_in_t2` | 0 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.dropped_not_1hop.music.recording.canonical_version` | 1.180 | data/processed/lastfm/kg/default/dropped.csv |
| `lastfm.kg.dropped_not_1hop.music.single.versions` | 113 | data/processed/lastfm/kg/default/dropped.csv |
| `lastfm.kg.no_date.music.recording.releases` | 2,1% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.no_date.music.recording.artist` | 1,6% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.no_date.common.topic.notable_types` | 100,0% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.no_date.music.recording.producer` | 29,0% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.no_date.music.recording.engineer` | 74,5% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.no_date.music.single.versions` | 100,0% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.no_date.music.recording.canonical_version` | 100,0% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.no_date.music.recording.featured_artists` | 9,0% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.no_date.music.recording.song` | 100,0% | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm.kg.relations_fully_undated` | 4 | data/processed/lastfm/kg/default/kg_meta.json |
| `lastfm-u25.kg.author_triples` | 464.567 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.item_seeded` | 154.076 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.head_seeded` | 152.979 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.dropped_not_1hop` | 1.097 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.dropped_not_1hop_share` | 0,7% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.entities` | 44.392 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.tail_entities` | 25.727 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.relations` | 9 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.default.t1` | 152.976 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.default.t2` | 152.976 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.default.dated_t1` | 122.653 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.drop_undated.t1` | 122.653 | data/processed/lastfm-u25/kg/drop_undated/kg_meta.json |
| `lastfm-u25.kg.drop_undated.t2` | 122.653 | data/processed/lastfm-u25/kg/drop_undated/kg_meta.json |
| `lastfm-u25.kg.drop_undated.dated_t1` | 122.653 | data/processed/lastfm-u25/kg/drop_undated/kg_meta.json |
| `lastfm-u25.kg.static.t1` | 152.979 | data/processed/lastfm-u25/kg/static/kg_meta.json |
| `lastfm-u25.kg.static.t2` | 152.979 | data/processed/lastfm-u25/kg/static/kg_meta.json |
| `lastfm-u25.kg.static.dated_t1` | 122.656 | data/processed/lastfm-u25/kg/static/kg_meta.json |
| `lastfm-u25.kg.filtered_t1` | 3 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.filtered_t1_share` | 0,0% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.undated_t1` | 30.323 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.undated_t1_share` | 19,8% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.items_with_facts` | 18.665 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.entities_only_in_t2` | 0 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.dropped_not_1hop.music.recording.canonical_version` | 1.023 | data/processed/lastfm-u25/kg/default/dropped.csv |
| `lastfm-u25.kg.dropped_not_1hop.music.single.versions` | 74 | data/processed/lastfm-u25/kg/default/dropped.csv |
| `lastfm-u25.kg.no_date.music.recording.releases` | 1,9% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.no_date.music.recording.artist` | 1,1% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.no_date.common.topic.notable_types` | 100,0% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.no_date.music.recording.producer` | 27,5% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.no_date.music.recording.engineer` | 74,8% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.no_date.music.single.versions` | 100,0% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.no_date.music.recording.canonical_version` | 100,0% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.no_date.music.recording.featured_artists` | 5,1% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.no_date.music.recording.song` | 100,0% | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u25.kg.relations_fully_undated` | 4 | data/processed/lastfm-u25/kg/default/kg_meta.json |
| `lastfm-u10.kg.author_triples` | 464.567 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.item_seeded` | 62.554 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.head_seeded` | 61.929 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.dropped_not_1hop` | 625 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.dropped_not_1hop_share` | 1,0% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.entities` | 21.123 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.tail_entities` | 14.236 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.relations` | 9 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.default.t1` | 61.928 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.default.t2` | 61.928 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.default.dated_t1` | 49.924 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.drop_undated.t1` | 49.924 | data/processed/lastfm-u10/kg/drop_undated/kg_meta.json |
| `lastfm-u10.kg.drop_undated.t2` | 49.924 | data/processed/lastfm-u10/kg/drop_undated/kg_meta.json |
| `lastfm-u10.kg.drop_undated.dated_t1` | 49.924 | data/processed/lastfm-u10/kg/drop_undated/kg_meta.json |
| `lastfm-u10.kg.static.t1` | 61.929 | data/processed/lastfm-u10/kg/static/kg_meta.json |
| `lastfm-u10.kg.static.t2` | 61.929 | data/processed/lastfm-u10/kg/static/kg_meta.json |
| `lastfm-u10.kg.static.dated_t1` | 49.925 | data/processed/lastfm-u10/kg/static/kg_meta.json |
| `lastfm-u10.kg.filtered_t1` | 1 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.filtered_t1_share` | 0,0% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.undated_t1` | 12.004 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.undated_t1_share` | 19,4% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.items_with_facts` | 6.887 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.entities_only_in_t2` | 0 | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.dropped_not_1hop.music.recording.canonical_version` | 587 | data/processed/lastfm-u10/kg/default/dropped.csv |
| `lastfm-u10.kg.dropped_not_1hop.music.single.versions` | 38 | data/processed/lastfm-u10/kg/default/dropped.csv |
| `lastfm-u10.kg.no_date.music.recording.releases` | 1,8% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.no_date.music.recording.artist` | 1,2% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.no_date.common.topic.notable_types` | 100,0% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.no_date.music.recording.producer` | 27,6% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.no_date.music.recording.engineer` | 76,3% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.no_date.music.single.versions` | 100,0% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.no_date.music.recording.featured_artists` | 6,0% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.no_date.music.recording.canonical_version` | 100,0% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.no_date.music.recording.song` | 100,0% | data/processed/lastfm-u10/kg/default/kg_meta.json |
| `lastfm-u10.kg.relations_fully_undated` | 4 | data/processed/lastfm-u10/kg/default/kg_meta.json |

## 3. Thang Delta-t của Stage 2

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm.dt.window_years` | 9,20 | data/processed/lastfm/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm.dt.t_ref` | 28/04/2014 | data/processed/lastfm/train.csv |
| `lastfm.dt.undated_share` | 21,2% | data/processed/lastfm/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm.dt.inside_window_share` | 21,2% | data/processed/lastfm/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm.dt.capped_share` | 57,6% | data/processed/lastfm/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm.dt.median_dated` | 1,59 | data/processed/lastfm/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u25.dt.window_years` | 9,20 | data/processed/lastfm-u25/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u25.dt.t_ref` | 29/04/2014 | data/processed/lastfm-u25/train.csv |
| `lastfm-u25.dt.undated_share` | 19,8% | data/processed/lastfm-u25/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u25.dt.inside_window_share` | 24,4% | data/processed/lastfm-u25/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u25.dt.capped_share` | 55,7% | data/processed/lastfm-u25/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u25.dt.median_dated` | 1,45 | data/processed/lastfm-u25/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u10.dt.window_years` | 9,20 | data/processed/lastfm-u10/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u10.dt.t_ref` | 29/04/2014 | data/processed/lastfm-u10/train.csv |
| `lastfm-u10.dt.undated_share` | 19,4% | data/processed/lastfm-u10/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u10.dt.inside_window_share` | 27,5% | data/processed/lastfm-u10/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u10.dt.capped_share` | 53,1% | data/processed/lastfm-u10/train.csv + kg/default/kg_t1_dates.tsv |
| `lastfm-u10.dt.median_dated` | 1,34 | data/processed/lastfm-u10/train.csv + kg/default/kg_t1_dates.tsv |

## 4. Thời gian của cạnh train (Stage 1)

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm.spread.user_under_1h` | 0,50% | data/processed/lastfm/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 giờ |
| `lastfm.spread.user_std_days_median` | 247,7 | data/processed/lastfm/train.csv qua eda.node_time_spread (mốc t1) |
| `lastfm.spread.item_std_days_median` | 596,4 | data/processed/lastfm/train.csv qua eda.node_time_spread (mốc t1) |
| `lastfm-u25.spread.user_under_1h` | 0,53% | data/processed/lastfm-u25/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 giờ |
| `lastfm-u25.spread.user_std_days_median` | 249,6 | data/processed/lastfm-u25/train.csv qua eda.node_time_spread (mốc t1) |
| `lastfm-u25.spread.item_std_days_median` | 552,6 | data/processed/lastfm-u25/train.csv qua eda.node_time_spread (mốc t1) |
| `lastfm-u10.spread.user_under_1h` | 0,46% | data/processed/lastfm-u10/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 giờ |
| `lastfm-u10.spread.user_std_days_median` | 243,1 | data/processed/lastfm-u10/train.csv qua eda.node_time_spread (mốc t1) |
| `lastfm-u10.spread.item_std_days_median` | 511,7 | data/processed/lastfm-u10/train.csv qua eda.node_time_spread (mốc t1) |

## 5. Số cạnh train theo tuần quanh 01/04/2014

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm.week.before_mean` | 12.334 | data/processed/lastfm/train.csv qua eda.weekly_counts, 4 tuần trước 31/03/2014 |
| `lastfm.week.after_mean` | 68.738 | data/processed/lastfm/train.csv qua eda.weekly_counts, 4 tuần từ 31/03/2014 |
| `lastfm-u25.week.before_mean` | 2.494 | data/processed/lastfm-u25/train.csv qua eda.weekly_counts, 4 tuần trước 31/03/2014 |
| `lastfm-u25.week.after_mean` | 12.416 | data/processed/lastfm-u25/train.csv qua eda.weekly_counts, 4 tuần từ 31/03/2014 |
| `lastfm-u10.week.before_mean` | 652 | data/processed/lastfm-u10/train.csv qua eda.weekly_counts, 4 tuần trước 31/03/2014 |
| `lastfm-u10.week.after_mean` | 3.035 | data/processed/lastfm-u10/train.csv qua eda.weekly_counts, 4 tuần từ 31/03/2014 |

## 6. Thời gian chạy và epoch tốt nhất (registry, chỉ dòng val)

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `lastfm-u25.hours.device.cpu` | 0,0 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.device.NVIDIA_GeForce_RTX_3050_Laptop_GPU` | 59,1 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.device.Tesla_T4` | 16,2 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.cpu.mostpop` | 0,0 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.cpu.mostpop_recent` | 0,0 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.NVIDIA_GeForce_RTX_3050_Laptop_GPU.bprmf` | 0,4 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.NVIDIA_GeForce_RTX_3050_Laptop_GPU.kgat` | 23,2 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.NVIDIA_GeForce_RTX_3050_Laptop_GPU.kgat_nokg` | 14,5 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.NVIDIA_GeForce_RTX_3050_Laptop_GPU.tkgat` | 14,8 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.NVIDIA_GeForce_RTX_3050_Laptop_GPU.tkgat_perm` | 6,3 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.Tesla_T4.tkgat_kg` | 12,3 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.hours.Tesla_T4.tkgat_kg_perm` | 3,9 | results/registry.csv (phase=val), tổng cột seconds |
| `lastfm-u25.best_epoch.bprmf.min` | 45 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.bprmf.max` | 210 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.kgat.min` | 80 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.kgat.max` | 300 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.kgat_nokg.min` | 95 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.kgat_nokg.max` | 300 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.mostpop.min` | 0 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.mostpop.max` | 0 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.mostpop_recent.min` | 0 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.mostpop_recent.max` | 0 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.tkgat.min` | 85 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.tkgat.max` | 285 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.tkgat_kg.min` | 70 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.tkgat_kg.max` | 250 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.tkgat_kg_perm.min` | 85 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.tkgat_kg_perm.max` | 240 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.tkgat_perm.min` | 80 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.tkgat_perm.max` | 300 | results/registry.csv (phase=val), mọi điểm lưới và seed |
| `lastfm-u25.best_epoch.bprmf.chosen.min` | 120 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.bprmf.chosen.max` | 210 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.kgat.chosen.min` | 125 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.kgat.chosen.max` | 240 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.kgat_nokg.chosen.min` | 100 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.kgat_nokg.chosen.max` | 275 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.mostpop.chosen.min` | 0 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.mostpop.chosen.max` | 0 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.mostpop_recent.chosen.min` | 0 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.mostpop_recent.chosen.max` | 0 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.tkgat.chosen.min` | 135 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.tkgat.chosen.max` | 200 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.tkgat_kg.chosen.min` | 95 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.tkgat_kg.chosen.max` | 210 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.tkgat_kg_perm.chosen.min` | 85 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.tkgat_kg_perm.chosen.max` | 240 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.tkgat_perm.chosen.min` | 80 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |
| `lastfm-u25.best_epoch.tkgat_perm.chosen.max` | 300 | results/registry.csv (phase=val), điểm lưới có trung bình Recall@20 val cao nhất |

## 7. Đối chiếu với MovieLens

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `cmp.conv.first_window` | 24 giờ kể từ rating đầu tiên của user | docs/data-report-numbers.md (`conv.first_window`) |
| `cmp.ml100k.val_kept_share` | 12,6% | docs/data-report-numbers.md (`ml100k.val_kept_share`) |
| `cmp.ml100k.test_kept_share` | 13,3% | docs/data-report-numbers.md (`ml100k.test_kept_share`) |
| `cmp.ml100k.val_users` | 66 | docs/data-report-numbers.md (`ml100k.val_users`) |
| `cmp.ml100k.test_users` | 51 | docs/data-report-numbers.md (`ml100k.test_users`) |
| `cmp.ml100k.first_24h_share` | 77,9% | docs/data-report-numbers.md (`ml100k.first_24h_share`) |
| `cmp.ml100k.user_spread_under_1h` | 53,0% | docs/data-report-numbers.md (`ml100k.user_spread_under_1h`) |
| `cmp.ml100k.no_date_share` | 18,1% | docs/data-report-numbers.md (`ml100k.no_date_share`) |
| `cmp.ml100k.filtered_share_t1` | 1,3% | docs/data-report-numbers.md (`ml100k.filtered_share_t1`) |
| `cmp.ml1m.val_kept_share` | 21,0% | docs/data-report-numbers.md (`ml1m.val_kept_share`) |
| `cmp.ml1m.test_kept_share` | 75,7% | docs/data-report-numbers.md (`ml1m.test_kept_share`) |
| `cmp.ml1m.val_users` | 480 | docs/data-report-numbers.md (`ml1m.val_users`) |
| `cmp.ml1m.test_users` | 952 | docs/data-report-numbers.md (`ml1m.test_users`) |
| `cmp.ml1m.filtered_share_t1` | 1,5% | docs/data-report-numbers.md (`ml1m.filtered_share_t1`) |
| `ml-100k.dt.window_years` | 0,46 | data/processed/ml-100k/train.csv + kg/default/kg_t1_dates.tsv |
| `ml-100k.dt.t_ref` | 07/03/1998 | data/processed/ml-100k/train.csv |
| `ml-100k.dt.undated_share` | 16,7% | data/processed/ml-100k/train.csv + kg/default/kg_t1_dates.tsv |
| `ml-100k.dt.inside_window_share` | 0,7% | data/processed/ml-100k/train.csv + kg/default/kg_t1_dates.tsv |
| `ml-100k.dt.capped_share` | 82,6% | data/processed/ml-100k/train.csv + kg/default/kg_t1_dates.tsv |
| `ml-100k.dt.median_dated` | 97,94 | data/processed/ml-100k/train.csv + kg/default/kg_t1_dates.tsv |
| `ml-1m-u10.dt.window_years` | 0,60 | data/processed/ml-1m-u10/train.csv + kg/default/kg_t1_dates.tsv |
| `ml-1m-u10.dt.t_ref` | 30/11/2000 | data/processed/ml-1m-u10/train.csv |
| `ml-1m-u10.dt.undated_share` | 16,7% | data/processed/ml-1m-u10/train.csv + kg/default/kg_t1_dates.tsv |
| `ml-1m-u10.dt.inside_window_share` | 0,0% | data/processed/ml-1m-u10/train.csv + kg/default/kg_t1_dates.tsv |
| `ml-1m-u10.dt.capped_share` | 83,3% | data/processed/ml-1m-u10/train.csv + kg/default/kg_t1_dates.tsv |
| `ml-1m-u10.dt.median_dated` | 84,77 | data/processed/ml-1m-u10/train.csv + kg/default/kg_t1_dates.tsv |
