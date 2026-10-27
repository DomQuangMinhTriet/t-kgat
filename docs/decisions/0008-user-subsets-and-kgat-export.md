# 0008 — Random user subsets of ML-1M and the KGAT export of the official split

- Status: accepted (2026-10-27); Phôn xác nhận qua chat
- Clarifies decision 0005 on one point (timestamps, below); supersedes nothing. Implemented in
  `src/tkgat/data/subsample.py`, `scripts/08_export_kgat.py` and `src/tkgat/export/kgat.py`.
- Context: Phôn's machine cannot run the full datasets with the KGAT code. PR #20 made a 10,000-interaction set
  (the 10,000 earliest ML-100K ratings, split by row position 8,000/1,000/1,000). Measured ad hoc on 2026-10-27
  (not a pipeline output): it spans 14 days, so no item has a time spread (median std of an item's train edges
  3.8 days, against 45 days in ML-100K and 67 days in ML-1M), and 75% of the positives of its test part belong to
  users unknown to train (24% in ML-1M). A small set that does not behave like ML-1M cannot stand in for it.
  The same script also treats every rating as a training positive, accepts equal timestamps at the boundaries and
  copies the date and linking logic of `tkgat.kg.temporal` and `tkgat.kg.linking`.
- Decision:
  1. **Small datasets are random user subsets of ML-1M.** The data config takes `user_fraction` and `user_seed`;
     the subset keeps every rating of the sampled users and is taken before cleaning, the split and the 5-core.
     Everything else is the project protocol (decision 0002): cut-offs are the 0.8 / 0.9 quantiles of the subset's
     own timestamps, rating >= 4, 5-core on train, no unknown users or items in val/test, `check_split` must pass.
     Two configs: `ml1m_u10.yaml` (10%) and `ml1m_u05.yaml` (5%), seed 2019, with `run_name` `ml-1m-u10` and
     `ml-1m-u05` so the registry never mixes them with `ml-1m`.
  2. **Subsets are for development and quick iteration.** Conclusions and the reported baselines stay on the full
     ML-1M (and ML-100K); a number computed on a subset is always reported with its run name.
  3. **The KGAT file format is produced by `scripts/08_export_kgat.py` from the official split and KG**, never by a
     separate split. Variants: KG `default`, `drop_undated`, `static`, each with or without `--timestamps`. Besides
     the usual files it writes `val_mask.txt` and `test_mask.txt` (every earlier interaction, any rating, to exclude
     when ranking) and `kg_test.txt` (KG at t2); `kg_final.txt` is the KG at t1.
  4. **The scripts of PR #20 are kept as `experiments/legacy_kgat_data/`**, for reference only, with a banner; no
     number computed on their output is reported.
  5. **Runs on the original KGAT code are exploratory** until the same formula is reproduced in `src/tkgat/models/kgat.py`
     through `scripts/run.py` (5 seeds, registry). A loader for the exported folder must choose epochs on `val.txt`,
     apply the mask files, and read `kg_test.txt` when it scores test.
  6. **Timestamps.** The GroupLens READMEs only say that a timestamp is seconds since the epoch. That it is the time
     the user entered the rating is the team's assumption, consistent with the sessions in the data (most ratings of a
     user fall within 24 h of the user's first rating) but not proven. Decision 0005 and CLAUDE.md state it as an
     assumption from now on; 0005 itself is not edited.
- Consequences (from `data/processed/<run>/meta.json`, seed 2019):

  | | ML-1M | ML-1M, 10% users | ML-1M, 5% users |
  |---|---|---|---|
  | Users sampled / ratings kept | 6,040 / 1,000,209 | 604 / 98,148 | 302 / 50,699 |
  | t1 / t2 (UTC) | 2000-12-02 / 2000-12-29 | 2000-11-30 / 2000-12-21 | 2000-12-08 / 2001-01-19 |
  | Train users / items / positives | 5,350 / 3,009 / 461,858 | 537 / 1,639 / 42,554 | 276 / 1,234 / 21,233 |
  | Val positives / users | 12,133 / 480 | 969 / 41 | 376 / 24 |
  | Test positives / users | 41,198 / 952 | 3,450 / 95 | 1,678 / 48 |
  | Positives dropped for unknown users, val / test | 78.9% / 23.7% | 78.3% / 36.8% | 88.0% / 28.8% |

  - Val and test of a subset have few users (24 to 95), so scores on them are noisy: use them to debug and to compare
    ideas roughly, not to choose between close configurations.
  - The cut-offs of a subset differ from those of the full ML-1M, so its KG at t1 and t2 can differ even though the
    full ML-1M has the same KG at both (5%: 51,540 and 51,815 triples; 10%: 65,344 and 65,344).
  - Entity ids are shared by the KG variants of a run as before; the exported folders of two runs are not comparable.
  - ML-100K items no longer need the title-and-year mapping to ML-1M for the KG of a subset: ML-1M ids are already
    linked by KB4Rec.
