# Số liệu của báo cáo dữ liệu

Sinh tự động bởi `scripts/07_report_numbers.py` từ các file do pipeline ghi ra. **Không sửa tay.**
Mọi con số trong `docs/data-report.md` phải có trong file này (kiểm tra bởi `tests/test_data_report.py`).
Giá trị "chưa có" nghĩa là nguồn chưa tồn tại; script điền số khi nguồn xuất hiện.
Định dạng: dấu chấm ngăn hàng nghìn, dấu phẩy thập phân; phần trăm có ký hiệu %.

## Quy ước cố định

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `conv.first_window` | 24 giờ kể từ rating đầu tiên của user | hằng số trong scripts/07_report_numbers.py (dùng chính chúng khi tính) |
| `conv.session_gap` | phiên kết thúc khi nghỉ quá 60 phút | hằng số trong scripts/07_report_numbers.py (dùng chính chúng khi tính) |
| `conv.session_gap_alt` | ngưỡng nhạy: 30 phút | hằng số trong scripts/07_report_numbers.py (dùng chính chúng khi tính) |
| `conv.spread_thresholds` | 1 giờ, 1 ngày, 30 ngày | hằng số trong scripts/07_report_numbers.py (dùng chính chúng khi tính) |
| `conv.top_item_fractions` | 10% và 20% | hằng số trong scripts/07_report_numbers.py (dùng chính chúng khi tính) |
| `conv.positive_rating` | rating ≥ 4 | hằng số trong scripts/07_report_numbers.py (dùng chính chúng khi tính) |
| `conv.seeds` | 5 seed (2019–2023) | hằng số trong scripts/07_report_numbers.py (dùng chính chúng khi tính) |
| `conv.k` | 10 và 20 | hằng số trong scripts/07_report_numbers.py (dùng chính chúng khi tính) |

## 1. Bộ dữ liệu và nguồn

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `checksum.files` | 9 | data/checksums.json và data/raw/ (SHA-256) |
| `checksum.ok` | 9 | data/checksums.json và data/raw/ (SHA-256) |
| `checksum.missing` | 0 | data/checksums.json và data/raw/ (SHA-256) |
| `checksum.mismatch` | 0 | data/checksums.json và data/raw/ (SHA-256) |
| `ml100k.users` | 943 | data/raw/ml-100k qua eda.overview |
| `ml100k.items` | 1.682 | data/raw/ml-100k qua eda.overview |
| `ml100k.ratings` | 100.000 | data/raw/ml-100k qua eda.overview |
| `ml100k.density` | 6,3% | data/raw/ml-100k qua eda.overview |
| `ml100k.first_date` | 20/09/1997 | data/raw/ml-100k qua eda.overview |
| `ml100k.last_date` | 22/04/1998 | data/raw/ml-100k qua eda.overview |
| `ml100k.span_days` | 214 | data/raw/ml-100k qua eda.overview |
| `ml100k.median_ratings_per_user` | 65 | data/raw/ml-100k qua eda.overview |
| `ml100k.peak_year` | 1997 | data/raw/ml-100k qua eda.overview qua eda.year_shares |
| `ml100k.peak_year_share` | 52,9% | data/raw/ml-100k qua eda.overview qua eda.year_shares |
| `ml100k.last_year` | 1998 | data/raw/ml-100k qua eda.overview qua eda.year_shares |
| `ml1m.users` | 6.040 | data/raw/ml-1m qua eda.overview |
| `ml1m.items` | 3.706 | data/raw/ml-1m qua eda.overview |
| `ml1m.ratings` | 1.000.209 | data/raw/ml-1m qua eda.overview |
| `ml1m.density` | 4,5% | data/raw/ml-1m qua eda.overview |
| `ml1m.first_date` | 25/04/2000 | data/raw/ml-1m qua eda.overview |
| `ml1m.last_date` | 28/02/2003 | data/raw/ml-1m qua eda.overview |
| `ml1m.span_days` | 1.038 | data/raw/ml-1m qua eda.overview |
| `ml1m.median_ratings_per_user` | 96 | data/raw/ml-1m qua eda.overview |
| `ml1m.peak_year` | 2000 | data/raw/ml-1m qua eda.overview qua eda.year_shares |
| `ml1m.peak_year_share` | 90,5% | data/raw/ml-1m qua eda.overview qua eda.year_shares |
| `ml1m.last_year` | 2003 | data/raw/ml-1m qua eda.overview qua eda.year_shares |

## 2. Làm sạch

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `ml100k.untitled_items` | 267 | data/processed/ml-100k/meta.json |
| `ml100k.untitled_ratings` | 9 | data/processed/ml-100k/meta.json |
| `ml100k.duplicate_title_items` | 18 | data/processed/ml-100k/meta.json |
| `ml100k.remapped_ratings` | 608 | data/processed/ml-100k/meta.json |
| `ml100k.ratings_after_clean` | 99.991 | data/processed/ml-100k/meta.json |
| `ml100k.duplicate_pairs_removed` | 307 | data/processed/ml-100k/meta.json |
| `ml100k.ratings_after_dedupe` | 99.684 | data/processed/ml-100k/meta.json |
| `ml1m.cleaning` | không làm sạch (config clean: false) | data/processed/ml-1m/meta.json |
| `ml1m.ratings_after_clean` | 1.000.209 | data/processed/ml-1m/meta.json |
| `ml1m.duplicate_pairs_removed` | 0 | data/processed/ml-1m/meta.json |
| `ml1m.ratings_after_dedupe` | 1.000.209 | data/processed/ml-1m/meta.json |

## 3. Rating theo phiên và Δt

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `ml100k.first_24h_share` | 77,9% | data/raw qua eda.first_window_share / eda.sessions |
| `ml100k.first_calendar_day_share` | 73,5% | data/raw qua eda.first_window_share / eda.sessions |
| `ml100k.one_day_users` | 63,5% | data/raw qua eda.first_window_share / eda.sessions (eda.active_days) |
| `ml100k.sessions_60` | 2.692 | data/raw qua eda.first_window_share / eda.sessions |
| `ml100k.sessions_30` | 2.793 | data/raw qua eda.first_window_share / eda.sessions |
| `ml100k.sessions_per_user_median` | 1 | data/raw qua eda.first_window_share / eda.sessions |
| `ml100k.session_median_minutes` | 13,4 | data/raw qua eda.first_window_share / eda.sessions |
| `ml100k.gap_under_1min` | 76,8% | data/processed/ml-100k/train.csv qua eda.node_time_spread (mốc t1), eda.consecutive_gaps |
| `ml100k.gap_under_1h` | 97,5% | data/processed/ml-100k/train.csv qua eda.node_time_spread (mốc t1), eda.consecutive_gaps |
| `ml100k.user_spread_under_1h` | 53,0% | data/processed/ml-100k/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 giờ |
| `ml100k.user_spread_under_1d` | 68,1% | data/processed/ml-100k/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 ngày |
| `ml100k.user_spread_under_30d` | 80,9% | data/processed/ml-100k/train.csv qua eda.node_time_spread (mốc t1), độ trải < 30 ngày |
| `ml100k.user_std_minutes_median` | 14 | data/processed/ml-100k/train.csv qua eda.node_time_spread (mốc t1) |
| `ml100k.item_spread_under_1h` | 0,0% | data/processed/ml-100k/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 giờ |
| `ml100k.item_spread_under_1d` | 0,0% | data/processed/ml-100k/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 ngày |
| `ml100k.item_spread_under_30d` | 0,3% | data/processed/ml-100k/train.csv qua eda.node_time_spread (mốc t1), độ trải < 30 ngày |
| `ml100k.item_std_days_median` | 45 | data/processed/ml-100k/train.csv qua eda.node_time_spread (mốc t1) |
| `ml1m.first_24h_share` | 72,4% | data/raw qua eda.first_window_share / eda.sessions |
| `ml1m.first_calendar_day_share` | 66,0% | data/raw qua eda.first_window_share / eda.sessions |
| `ml1m.one_day_users` | 58,7% | data/raw qua eda.first_window_share / eda.sessions (eda.active_days) |
| `ml1m.sessions_60` | 24.413 | data/raw qua eda.first_window_share / eda.sessions |
| `ml1m.sessions_30` | 25.163 | data/raw qua eda.first_window_share / eda.sessions |
| `ml1m.sessions_per_user_median` | 1 | data/raw qua eda.first_window_share / eda.sessions |
| `ml1m.session_median_minutes` | 13,1 | data/raw qua eda.first_window_share / eda.sessions |
| `ml1m.gap_under_1min` | 82,9% | data/processed/ml-1m/train.csv qua eda.node_time_spread (mốc t1), eda.consecutive_gaps |
| `ml1m.gap_under_1h` | 98,8% | data/processed/ml-1m/train.csv qua eda.node_time_spread (mốc t1), eda.consecutive_gaps |
| `ml1m.user_spread_under_1h` | 53,4% | data/processed/ml-1m/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 giờ |
| `ml1m.user_spread_under_1d` | 71,1% | data/processed/ml-1m/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 ngày |
| `ml1m.user_spread_under_30d` | 90,1% | data/processed/ml-1m/train.csv qua eda.node_time_spread (mốc t1), độ trải < 30 ngày |
| `ml1m.user_std_minutes_median` | 14 | data/processed/ml-1m/train.csv qua eda.node_time_spread (mốc t1) |
| `ml1m.item_spread_under_1h` | 0,0% | data/processed/ml-1m/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 giờ |
| `ml1m.item_spread_under_1d` | 0,0% | data/processed/ml-1m/train.csv qua eda.node_time_spread (mốc t1), độ trải < 1 ngày |
| `ml1m.item_spread_under_30d` | 0,4% | data/processed/ml-1m/train.csv qua eda.node_time_spread (mốc t1), độ trải < 30 ngày |
| `ml1m.item_std_days_median` | 67 | data/processed/ml-1m/train.csv qua eda.node_time_spread (mốc t1) |

## 4. Chia theo mốc t1/t2

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `ml100k.quantiles` | 80% và 90% thời gian | data/processed/ml-100k/meta.json |
| `ml100k.kcore` | 5-core, tính trên train | data/processed/ml-100k/meta.json |
| `ml100k.t1` | 07/03/1998 | data/processed/ml-100k/meta.json |
| `ml100k.t2` | 31/03/1998 | data/processed/ml-100k/meta.json |
| `ml100k.val_window_days` | 25 | data/processed/ml-100k/meta.json |
| `ml100k.test_window_days` | 22 | data/processed/ml-100k/meta.json và ngày cuối của data/raw |
| `ml100k.train_positives_before_kcore` | 43.929 | data/processed/ml-100k/meta.json |
| `ml100k.train_positives` | 43.018 | data/processed/ml-100k/meta.json |
| `ml100k.train_users` | 745 | data/processed/ml-100k/meta.json |
| `ml100k.train_items` | 932 | data/processed/ml-100k/meta.json |
| `ml100k.val_positives_before_filter` | 5.648 | data/processed/ml-100k/meta.json |
| `ml100k.val_dropped_cold_user` | 4.886 | data/processed/ml-100k/meta.json |
| `ml100k.val_dropped_cold_item` | 50 | data/processed/ml-100k/meta.json |
| `ml100k.val_cold_items` | 140 | data/processed/ml-100k/meta.json |
| `ml100k.val_positives` | 712 | data/processed/ml-100k/meta.json |
| `ml100k.val_users` | 66 | data/processed/ml-100k/meta.json |
| `ml100k.val_kept_share` | 12,6% | data/processed/ml-100k/meta.json |
| `ml100k.test_positives_before_filter` | 5.612 | data/processed/ml-100k/meta.json |
| `ml100k.test_dropped_cold_user` | 4.805 | data/processed/ml-100k/meta.json |
| `ml100k.test_dropped_cold_item` | 60 | data/processed/ml-100k/meta.json |
| `ml100k.test_cold_items` | 183 | data/processed/ml-100k/meta.json |
| `ml100k.test_positives` | 747 | data/processed/ml-100k/meta.json |
| `ml100k.test_users` | 51 | data/processed/ml-100k/meta.json |
| `ml100k.test_kept_share` | 13,3% | data/processed/ml-100k/meta.json |
| `ml100k.check_split_problems` | 0 | tkgat.data.leakage.check_split chạy trên split đã lưu |
| `ml1m.quantiles` | 80% và 90% thời gian | data/processed/ml-1m/meta.json |
| `ml1m.kcore` | 5-core, tính trên train | data/processed/ml-1m/meta.json |
| `ml1m.t1` | 02/12/2000 | data/processed/ml-1m/meta.json |
| `ml1m.t2` | 29/12/2000 | data/processed/ml-1m/meta.json |
| `ml1m.val_window_days` | 27 | data/processed/ml-1m/meta.json |
| `ml1m.test_window_days` | 791 | data/processed/ml-1m/meta.json và ngày cuối của data/raw |
| `ml1m.train_positives_before_kcore` | 463.017 | data/processed/ml-1m/meta.json |
| `ml1m.train_positives` | 461.858 | data/processed/ml-1m/meta.json |
| `ml1m.train_users` | 5.350 | data/processed/ml-1m/meta.json |
| `ml1m.train_items` | 3.009 | data/processed/ml-1m/meta.json |
| `ml1m.val_positives_before_filter` | 57.812 | data/processed/ml-1m/meta.json |
| `ml1m.val_dropped_cold_user` | 45.605 | data/processed/ml-1m/meta.json |
| `ml1m.val_dropped_cold_item` | 74 | data/processed/ml-1m/meta.json |
| `ml1m.val_cold_items` | 157 | data/processed/ml-1m/meta.json |
| `ml1m.val_positives` | 12.133 | data/processed/ml-1m/meta.json |
| `ml1m.val_users` | 480 | data/processed/ml-1m/meta.json |
| `ml1m.val_kept_share` | 21,0% | data/processed/ml-1m/meta.json |
| `ml1m.test_positives_before_filter` | 54.452 | data/processed/ml-1m/meta.json |
| `ml1m.test_dropped_cold_user` | 12.911 | data/processed/ml-1m/meta.json |
| `ml1m.test_dropped_cold_item` | 343 | data/processed/ml-1m/meta.json |
| `ml1m.test_cold_items` | 221 | data/processed/ml-1m/meta.json |
| `ml1m.test_positives` | 41.198 | data/processed/ml-1m/meta.json |
| `ml1m.test_users` | 952 | data/processed/ml-1m/meta.json |
| `ml1m.test_kept_share` | 75,7% | data/processed/ml-1m/meta.json |
| `ml1m.check_split_problems` | 0 | tkgat.data.leakage.check_split chạy trên split đã lưu |

## 5. Liên kết thực thể

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `ml100k.link_items` | 1.663 | data/processed/links/linking_report.json |
| `ml100k.link_freebase_items` | 1.617 | data/processed/links/linking_report.json |
| `ml100k.link_freebase_item_share` | 97,2% | data/processed/links/linking_report.json |
| `ml100k.link_freebase_rating_share` | 99,5% | data/processed/links/linking_report.json |
| `ml100k.link_yago_item_share` | 83,1% | data/processed/links/linking_report.json |
| `ml1m.link_items` | 3.883 | data/processed/links/linking_report.json |
| `ml1m.link_freebase_items` | 3.811 | data/processed/links/linking_report.json |
| `ml1m.link_freebase_item_share` | 98,2% | data/processed/links/linking_report.json |
| `ml1m.link_freebase_rating_share` | 99,7% | data/processed/links/linking_report.json |
| `ml1m.link_yago_item_share` | 84,1% | data/processed/links/linking_report.json |
| `ml100k.link_rule.title_year` | 1.628 | data/processed/links/linking_report.json (match_rules) |
| `ml100k.link_rule.unmatched` | 16 | data/processed/links/linking_report.json (match_rules) |
| `ml100k.link_rule.aka_title_year` | 13 | data/processed/links/linking_report.json (match_rules) |
| `ml100k.link_rule.contains_year` | 4 | data/processed/links/linking_report.json (match_rules) |
| `ml100k.link_rule.title_year_pm1` | 2 | data/processed/links/linking_report.json (match_rules) |
| `link_check.n_checked` | chưa có | results/linking_manual_check.json (việc 15) |
| `link_check.n_wrong` | chưa có | results/linking_manual_check.json (việc 15) |
| `link_check.error_rate` | chưa có | results/linking_manual_check.json (việc 15) |

## 6. KG theo thời gian

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `kg.snapshot` | 17/10/2026 | kg_meta.json (wikidata_snapshot) |
| `kg.facts_total` | 119.748 | data/raw/wikidata/facts.tsv |
| `kg.relations` | 18 | data/raw/wikidata/facts.tsv |
| `kg.films_with_facts` | 3.725 | data/raw/wikidata/facts.tsv |
| `kg.point_in_time_count` | 6.295 | data/raw/wikidata/facts.tsv qua kg.date_precision |
| `kg.point_in_time_year_only_share` | 44,4% | data/raw/wikidata/facts.tsv |
| `kg.film_publication_jan1_share` | 58,3% | data/raw/wikidata/facts.tsv (đuôi -01-01) |
| `kg.tail_inception_jan1_share` | 45,1% | data/raw/wikidata/facts.tsv (đuôi -01-01) |
| `kg.tail_birth_jan1_share` | 4,8% | data/raw/wikidata/facts.tsv (đuôi -01-01) |
| `titanic.title` | Titanic (1997) | ml100k_links.csv |
| `titanic.facts` | 136 | kg.film_fact_status (facts.tsv, dropped.csv của ML-100K) |
| `titanic.missing_t1` | 28 | kg.film_fact_status (facts.tsv, dropped.csv của ML-100K) |
| `titanic.awards_added_t2` | 11 | kg.film_fact_status (facts.tsv, dropped.csv của ML-100K) |
| `titanic.oscars_added_t2` | 11 | kg.film_fact_status (facts.tsv, dropped.csv của ML-100K) |
| `titanic.awards_date` | 23/03/1998 | kg.film_fact_status (facts.tsv, dropped.csv của ML-100K) |
| `titanic.dropped_both` | 15 | kg.film_fact_status (facts.tsv, dropped.csv của ML-100K) |
| `titanic.dropped_both_year_only` | 13 | kg.film_fact_status (facts.tsv, dropped.csv của ML-100K) |
| `toystory.title` | Toy Story (1995) | ml100k_links.csv |
| `toystory.facts_lost_t1` | 2 | kg.film_fact_status (facts.tsv, dropped.csv của ML-100K) |
| `toystory.sequel` | Toy Story 2 | kg.film_fact_status (facts.tsv, dropped.csv của ML-100K) |
| `toystory.sequel_date` | 24/11/1999 | kg.film_fact_status (facts.tsv, dropped.csv của ML-100K) |
| `ml100k.kg_items` | 932 | data/processed/ml-100k/kg/*/kg_meta.json |
| `ml100k.kg_items_linked` | 927 | data/processed/ml-100k/kg/*/kg_meta.json |
| `ml100k.kg_items_with_facts` | 927 | data/processed/ml-100k/kg/*/kg_meta.json |
| `ml100k.kg_facts` | 37.103 | data/processed/ml-100k/kg/*/kg_meta.json qua kg.relation_date_kinds |
| `ml100k.statement_dated_share` | 7,6% | data/processed/ml-100k/kg/*/kg_meta.json qua kg.relation_date_kinds |
| `ml100k.award_share_of_statement_dated` | 97,1% | data/processed/ml-100k/kg/*/kg_meta.json qua kg.relation_date_kinds |
| `ml100k.no_date_share` | 18,1% | data/processed/ml-100k/kg/*/kg_meta.json qua kg.relation_date_kinds |
| `ml100k.triples_default_t1` | 36.560 | data/processed/ml-100k/kg/*/kg_meta.json qua kg.variant_triples |
| `ml100k.triples_default_t2` | 36.580 | data/processed/ml-100k/kg/*/kg_meta.json qua kg.variant_triples |
| `ml100k.triples_drop_undated_t1` | 29.840 | data/processed/ml-100k/kg/*/kg_meta.json qua kg.variant_triples |
| `ml100k.triples_drop_undated_t2` | 29.860 | data/processed/ml-100k/kg/*/kg_meta.json qua kg.variant_triples |
| `ml100k.triples_static_t1` | 37.103 | data/processed/ml-100k/kg/*/kg_meta.json qua kg.variant_triples |
| `ml100k.triples_static_t2` | 37.103 | data/processed/ml-100k/kg/*/kg_meta.json qua kg.variant_triples |
| `ml100k.filtered_share_t1` | 1,5% | 1 − default/static tại t1 |
| `ml100k.filtered_facts_t1` | 543 | static − default tại t1 |
| `ml100k.dropped_future_tail_t1` | 433 | data/processed/ml-100k/kg/*/kg_meta.json |
| `ml100k.dropped_future_statement_t1` | 110 | data/processed/ml-100k/kg/*/kg_meta.json |
| `ml100k.dropped_undated_t1` | 6.720 | data/processed/ml-100k/kg/*/kg_meta.json |
| `ml100k.entities_only_in_t2` | 1 | data/processed/ml-100k/kg/*/kg_meta.json |
| `ml100k.award_future_t1` | 52 | dropped.csv (award_received, future_statement) |
| `ml100k.award_future_t2` | 33 | dropped.csv (award_received, future_statement) |
| `ml100k.award_between_t1_t2` | 19 | dropped.csv: có ở t1, không có ở t2 |
| `ml100k.top10_items` | 93 | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml100k.top10_share_all_facts` | 15,8% | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml100k.top10_share_future_award_facts` | 44,3% | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml100k.future_award_facts_t1` | 97 | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml100k.future_award_films_t1` | 25 | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml100k.top20_items` | 186 | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml100k.top20_share_all_facts` | 27,8% | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml100k.top20_share_future_award_facts` | 64,9% | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml100k.affected_items` | 344 | kg.items_with_filtered_facts (static so với default cùng mốc), kg.positive_share |
| `ml100k.affected_item_share` | 36,9% | kg.items_with_filtered_facts (static so với default cùng mốc), kg.positive_share |
| `ml100k.affected_val_positive_share` | 53,1% | kg.items_with_filtered_facts (static so với default cùng mốc), kg.positive_share |
| `ml100k.affected_test_positive_share` | 48,1% | kg.items_with_filtered_facts (static so với default cùng mốc), kg.positive_share |
| `ml1m.kg_items` | 3.009 | data/processed/ml-1m/kg/*/kg_meta.json |
| `ml1m.kg_items_linked` | 2.988 | data/processed/ml-1m/kg/*/kg_meta.json |
| `ml1m.kg_items_with_facts` | 2.975 | data/processed/ml-1m/kg/*/kg_meta.json |
| `ml1m.kg_facts` | 104.782 | data/processed/ml-1m/kg/*/kg_meta.json qua kg.relation_date_kinds |
| `ml1m.statement_dated_share` | 5,8% | data/processed/ml-1m/kg/*/kg_meta.json qua kg.relation_date_kinds |
| `ml1m.award_share_of_statement_dated` | 97,8% | data/processed/ml-1m/kg/*/kg_meta.json qua kg.relation_date_kinds |
| `ml1m.no_date_share` | 18,1% | data/processed/ml-1m/kg/*/kg_meta.json qua kg.relation_date_kinds |
| `ml1m.triples_default_t1` | 103.443 | data/processed/ml-1m/kg/*/kg_meta.json qua kg.variant_triples |
| `ml1m.triples_default_t2` | 103.443 | data/processed/ml-1m/kg/*/kg_meta.json qua kg.variant_triples |
| `ml1m.triples_drop_undated_t1` | 84.522 | data/processed/ml-1m/kg/*/kg_meta.json qua kg.variant_triples |
| `ml1m.triples_drop_undated_t2` | 84.522 | data/processed/ml-1m/kg/*/kg_meta.json qua kg.variant_triples |
| `ml1m.triples_static_t1` | 104.782 | data/processed/ml-1m/kg/*/kg_meta.json qua kg.variant_triples |
| `ml1m.triples_static_t2` | 104.782 | data/processed/ml-1m/kg/*/kg_meta.json qua kg.variant_triples |
| `ml1m.filtered_share_t1` | 1,3% | 1 − default/static tại t1 |
| `ml1m.filtered_facts_t1` | 1.339 | static − default tại t1 |
| `ml1m.dropped_future_tail_t1` | 1.069 | data/processed/ml-1m/kg/*/kg_meta.json |
| `ml1m.dropped_future_statement_t1` | 270 | data/processed/ml-1m/kg/*/kg_meta.json |
| `ml1m.dropped_undated_t1` | 18.921 | data/processed/ml-1m/kg/*/kg_meta.json |
| `ml1m.entities_only_in_t2` | 0 | data/processed/ml-1m/kg/*/kg_meta.json |
| `ml1m.award_future_t1` | 84 | dropped.csv (award_received, future_statement) |
| `ml1m.award_future_t2` | 84 | dropped.csv (award_received, future_statement) |
| `ml1m.award_between_t1_t2` | 0 | dropped.csv: có ở t1, không có ở t2 |
| `ml1m.top10_items` | 301 | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml1m.top10_share_all_facts` | 15,0% | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml1m.top10_share_future_award_facts` | 35,8% | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml1m.future_award_facts_t1` | 260 | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml1m.future_award_films_t1` | 74 | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml1m.top20_items` | 602 | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml1m.top20_share_all_facts` | 27,6% | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml1m.top20_share_future_award_facts` | 44,6% | kg.popular_concentration (static t1, dropped.csv, train.csv) |
| `ml1m.affected_items` | 899 | kg.items_with_filtered_facts (static so với default cùng mốc), kg.positive_share |
| `ml1m.affected_item_share` | 29,9% | kg.items_with_filtered_facts (static so với default cùng mốc), kg.positive_share |
| `ml1m.affected_val_positive_share` | 40,6% | kg.items_with_filtered_facts (static so với default cùng mốc), kg.positive_share |
| `ml1m.affected_test_positive_share` | 42,6% | kg.items_with_filtered_facts (static so với default cùng mốc), kg.positive_share |

## 7. Baseline

| Khóa | Giá trị | Nguồn |
|---|---|---|
| `ml100k.mostpop.selected` | chưa có | results/registry.csv qua registry.load(), mostpop, dataset = ml-100k |
| `ml100k.mostpop.test_recall20` | chưa có | results/registry.csv qua registry.load(), mostpop, dataset = ml-100k |
| `ml100k.mostpop.test_ndcg20` | chưa có | results/registry.csv qua registry.load(), mostpop, dataset = ml-100k |
| `ml100k.mostpop.n_seeds` | chưa có | results/registry.csv qua registry.load(), mostpop, dataset = ml-100k |
| `ml100k.mostpop.val_users` | chưa có | results/registry.csv qua registry.load(), mostpop, dataset = ml-100k |
| `ml100k.mostpop.test_users` | chưa có | results/registry.csv qua registry.load(), mostpop, dataset = ml-100k |
| `ml100k.mostpop_recent.selected` | chưa có | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-100k |
| `ml100k.mostpop_recent.test_recall20` | chưa có | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-100k |
| `ml100k.mostpop_recent.test_ndcg20` | chưa có | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-100k |
| `ml100k.mostpop_recent.n_seeds` | chưa có | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-100k |
| `ml100k.mostpop_recent.val_users` | chưa có | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-100k |
| `ml100k.mostpop_recent.test_users` | chưa có | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-100k |
| `ml100k.bprmf.selected` | chưa có | results/registry.csv qua registry.load(), bprmf, dataset = ml-100k |
| `ml100k.bprmf.test_recall20` | chưa có | results/registry.csv qua registry.load(), bprmf, dataset = ml-100k |
| `ml100k.bprmf.test_ndcg20` | chưa có | results/registry.csv qua registry.load(), bprmf, dataset = ml-100k |
| `ml100k.bprmf.n_seeds` | chưa có | results/registry.csv qua registry.load(), bprmf, dataset = ml-100k |
| `ml100k.bprmf.val_users` | chưa có | results/registry.csv qua registry.load(), bprmf, dataset = ml-100k |
| `ml100k.bprmf.test_users` | chưa có | results/registry.csv qua registry.load(), bprmf, dataset = ml-100k |
| `ml1m.mostpop.selected` | toàn bộ train (không giới hạn cửa sổ) | results/registry.csv qua registry.load(), mostpop, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.mostpop.test_recall20` | 0,0902 | results/registry.csv qua registry.load(), mostpop, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) (sample std, ddof=1) |
| `ml1m.mostpop.test_ndcg20` | 0,1942 | results/registry.csv qua registry.load(), mostpop, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) (sample std, ddof=1) |
| `ml1m.mostpop.n_seeds` | 1 | results/registry.csv qua registry.load(), mostpop, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.mostpop.val_users` | 480 | results/registry.csv qua registry.load(), mostpop, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.mostpop.test_users` | 952 | results/registry.csv qua registry.load(), mostpop, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.mostpop_recent.selected` | window_days = 7 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.mostpop_recent.test_recall20` | 0,1060 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) (sample std, ddof=1) |
| `ml1m.mostpop_recent.test_ndcg20` | 0,2034 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) (sample std, ddof=1) |
| `ml1m.mostpop_recent.n_seeds` | 1 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.mostpop_recent.val_users` | 480 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.mostpop_recent.test_users` | 952 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.mostpop_recent.grid` | 1, 3, 5, 7, 14, 30 ngày | configs/model/mostpop_recent.yaml |
| `ml1m.mostpop_recent.val_recall20.w1` | 0,0748 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m (phase val) |
| `ml1m.mostpop_recent.val_recall20.w3` | 0,1130 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m (phase val) |
| `ml1m.mostpop_recent.val_recall20.w5` | 0,1289 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m (phase val) |
| `ml1m.mostpop_recent.val_recall20.w7` | 0,1336 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m (phase val) |
| `ml1m.mostpop_recent.val_recall20.w14` | 0,1082 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m (phase val) |
| `ml1m.mostpop_recent.val_recall20.w30` | 0,0986 | results/registry.csv qua registry.load(), mostpop_recent, dataset = ml-1m (phase val) |
| `ml1m.mostpop_recent.on_grid_edge` | không | so selected với configs/model/mostpop_recent.yaml |
| `ml1m.bprmf.selected` | dim = 64, l2 = 0,0001, lr = 0,001 | results/registry.csv qua registry.load(), bprmf, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.bprmf.test_recall20` | 0,1194 ± 0,0031 | results/registry.csv qua registry.load(), bprmf, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) (sample std, ddof=1) |
| `ml1m.bprmf.test_ndcg20` | 0,2208 ± 0,0069 | results/registry.csv qua registry.load(), bprmf, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) (sample std, ddof=1) |
| `ml1m.bprmf.n_seeds` | 5 | results/registry.csv qua registry.load(), bprmf, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.bprmf.val_users` | 480 | results/registry.csv qua registry.load(), bprmf, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `ml1m.bprmf.test_users` | 952 | results/registry.csv qua registry.load(), bprmf, dataset = ml-1m; cấu hình chọn = trường selected của summary (decision 0007) |
| `registry.rows_all` | 53 | results/registry.csv qua registry.load()(latest=False) |
| `registry.rows_latest` | 29 | results/registry.csv qua registry.load() |
| `registry.rows_dirty_label` | 19 | results/registry.csv qua registry.load() |
| `registry.key` | dataset, split_sha256, model, config, seed, phase | registry.RUN_KEY |
| `registry.splits` | 1 | results/registry.csv qua registry.load() |
