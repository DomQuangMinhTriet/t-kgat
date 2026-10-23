# Data dictionary

## Processed split (`data/processed/<dataset>/`)
| File | Columns | Meaning |
|---|---|---|
| `train.csv` | user, item, rating, ts | Positive interactions (rating ≥ 4) with `ts < t1`, after 5-core on train |
| `val.csv` | user, item, rating, ts | Positives with `t1 ≤ ts < t2`, users and items known in train |
| `test.csv` | user, item, rating, ts | Positives with `ts ≥ t2`, users and items known in train |
| `seen_before_val.csv` | user, item, ts | Every interaction (any rating) with `ts < t1`; masked when ranking for val |
| `seen_before_test.csv` | user, item, ts | Every interaction (any rating) with `ts < t2`; masked when ranking for test |
| `meta.json` | — | Cut-offs, protocol, cleaning report, counts, SHA-256 of each file |

Ids are the raw MovieLens ids (after cleaning). `ts` is Unix seconds (UTC).

## Links (`data/processed/links/`)
| File | Columns |
|---|---|
| `ml100k_links.csv` | item_100k, title, year, item_ml1m, rule, freebase_id, yago_id |
| `ml1m_links.csv` | item (ML-1M = ML-20M id), title, year, freebase_id, yago_id |
| `linking_report.json` | coverage counts by rule, item- and rating-weighted coverage |

## Wikidata facts (`data/raw/wikidata/facts.tsv`)
One row per (film, relation, target). Dates are ISO `YYYY-MM-DD`; empty when Wikidata has none.

| Column | Meaning |
|---|---|
| freebase_id | Film, as KB4Rec's Freebase mid (`m.0dr_4` = Titanic) |
| relation | One of 18 short names (`director`, `cast_member`, `award_received`, …), see `RELATIONS` in `scripts/04_fetch_wikidata.py` |
| tail, tail_label | Wikidata id and English label of the target |
| point_in_time, point_in_time_precision | P585 qualifier of the statement; precision 9 = year, 10 = month, 11 = day |
| start_time | P580 qualifier of the statement |
| tail_publication, tail_inception, tail_birth | P577 / P571 / P569 of the target: when the target came into existence |
| film_publication | Earliest P577 of the film |

## Results
`results/registry.csv`: run_at, git_commit, dataset, split_sha256, model, config, seed, phase (val/test),
best_epoch, recall@10, recall@20, ndcg@10, ndcg@20, n_users, seconds, device (`cpu` or `cuda:<GPU name>`).
