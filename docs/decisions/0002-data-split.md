# 0002 — Leakage-safe temporal split

- Status: accepted (2026-10-17); cut-offs to be confirmed with the advisor on 2026-10-31
- Decision:
  1. Deduplicate (user, item) pairs, keep the earliest.
  2. Global cut-offs: `t1` = 80th, `t2` = 90th percentile of all timestamps. train `< t1`, val `[t1, t2)`,
     test `>= t2`. Global (not per-user) so that no user's future reaches another user's past.
  3. Positive = rating ≥ 4 (implicit feedback).
  4. 5-core, iterative, computed on train positives only.
  5. Val/test keep only users and items present in train; dropped cold users/items are counted.
  6. Ranking masks contain every interaction (any rating) before the cut-off.
  7. Selection: mean validation Recall@20 over seeds. Test: once, selected configuration only.
     No retraining on train+val.
- Evidence (MovieLens-100K, `data/processed/ml-100k/meta.json`): t1 = 1998-03-07, t2 = 1998-03-31;
  745 users, 932 items, 43,018 train positives; val 712 positives / 66 users; test 747 positives / 51 users.
- Known limitation: 78% of MovieLens-100K ratings are entered on the user's first day, so most late
  interactions come from new users and are dropped as cold-start. The test set is small (51 users);
  report mean ± std over seeds and paired tests, and compare with MovieLens-1M (952 test users).

## Update 2026-10-17 — k-core
5-core stays the main setting. A 10-core variant (`configs/data/ml100k_k10.yaml`), as in the KGAT
paper, is run as a sensitivity check (validation only) and reported next to the main results.
