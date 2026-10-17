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

## Results
`results/registry.csv`: run_at, git_commit, dataset, split_sha256, model, config, seed, phase (val/test),
best_epoch, recall@10, recall@20, ndcg@10, ndcg@20, n_users, seconds.
