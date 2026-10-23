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

## Time-consistent KG (`data/processed/<dataset>/kg/<variant>/`)
Built by `scripts/06_build_kg.py` (decisions 0004, 0006). KG at `t1` is used for train/val, KG at `t2`
for test. A fact is kept at cut-off `t` iff its statement date and its target's existence date, when
known, fall before the UTC day of `t`.

| Variant | Content |
|---|---|
| `default` | Time-filtered; facts with no usable date at all are kept (main setting) |
| `drop_undated` | Time-filtered; facts with no usable date at all are dropped |
| `static` | Every fact of the train items, no time filter, `kg_t1` = `kg_t2` (as in the KGAT paper). Reference only, to measure leakage through the KG; never a time-consistent result |

All three variants share one entity and relation vocabulary (that of the static KG), so ids are
comparable across variants; in `default` and `drop_undated` some entities have no edge.

| File | Columns | Meaning |
|---|---|---|
| `kg_t1.tsv`, `kg_t2.tsv` | h, r, t | Integer triples, sorted, no duplicates, no inverse relations (the model adds them) |
| `entities.csv` | entity, kind, key, label | `0..n_items-1` are the train items in `tkgat.data.encode` order (`kind=item`, `key` = raw item id, `label` = Freebase mid); then Wikidata entities (`kind=wikidata`, `key` = Wikidata id, `label` = English label), sorted by id |
| `relations.csv` | relation, name | The 18 relation names, sorted |
| `dropped.csv` | freebase_id, item, relation, tail, tail_label, point_in_time, start_time, tail_publication, tail_inception, tail_birth, reason, cutoff | Facts removed at each cut-off; `reason` = `future_statement`, `future_tail` or `undated` (empty for `static`) |
| `kg_meta.json` | — | Cut-offs, `time_consistent`, known limitations, md5 of inputs, counts per relation and reason, date kind per relation, entities only in the t2 KG |

## Results
`results/registry.csv`: run_at, git_commit, dataset, split_sha256, model, config, seed, phase (val/test),
best_epoch, recall@10, recall@20, ndcg@10, ndcg@20, n_users, seconds, device (`cpu` or `cuda:<GPU name>`).
