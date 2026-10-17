# t-kgat

Leakage-safe, time-aware knowledge-graph recommendation on MovieLens.
Graduation thesis, HCMUS (advisor: Dr. Nguyễn Ngọc Minh Châu; students: Đởm Quang Minh Triết, Nguyễn Đăng Phôn).

## Research questions
- **RQ1** What do MovieLens interactions and the linked knowledge graph look like over time
  (link coverage, dated facts, facts from the future)?
- **RQ2** Once leakage is removed (validation set, time-consistent KG), does the KG still help KGAT beat
  pure collaborative filtering?
- **RQ3** Does time-aware attention improve further (the 2026-08-07 design: Δt decay on user–item edges, compared with time on KG triples), and does the gain really come from time?

## Reproduce
```bash
pip install -e ".[dev]"
python scripts/01_verify_data.py                                    # raw data present and unchanged
python scripts/02_build_splits.py --data configs/data/ml100k.yaml   # refuses to save a leaky split
python scripts/03_link_entities.py                                  # ML-100K/1M -> ML-20M -> Freebase/YAGO
python scripts/run.py --data configs/data/ml100k.yaml --model configs/model/mostpop.yaml
python scripts/run.py --data configs/data/ml100k.yaml --model configs/model/bprmf.yaml
pytest
```

## Layout
```
configs/      one YAML per data protocol and per model (with its search grid)
data/         raw + processed data (not committed) and checksums
src/tkgat/   data (load, clean, split, leakage checks), kg (linking), models, train, eval
scripts/      numbered pipeline steps + run.py
tests/        leakage, split, metric and linking tests
notebooks/    01_eda, 02_entity_linking, 00_present_<date>
results/      registry.csv (every val/test score) and per-user run files
docs/         decisions/, data-report.md (week 2), data-dictionary.md, meetings/, theory/
```

## Protocol in one paragraph
Pairs are deduplicated; two global cut-offs (80th and 90th percentile of time) give train / val / test;
ratings ≥ 4 are positives; the 5-core is computed on train only; val/test keep only users and items known
in train; items a user rated before a cut-off are masked from the ranking. Selection uses validation
Recall@20 averaged over seeds; only the selected configuration is scored on test.
See `docs/decisions/0002-data-split.md`.

## Planning
- `docs/plan.md` — the 13-week plan exported from the team doc (Vietnamese).
- `docs/study/` — study guides for the 7 theory topics (Vietnamese).
