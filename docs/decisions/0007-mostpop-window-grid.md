# 0007 — Wider window grid for MostPop recent

- Status: accepted (2026-10-26 by Triết); to be reported to the advisor
- Context: `configs/model/mostpop_recent.yaml` searched the window over {7, 14, 30} days. On ML-1M the
  validation Recall@20 of the three points falls steadily as the window grows (7 days 0.1336, 14 days
  0.1082, 30 days 0.0986), so the selected value (7 days) was the smallest of the grid: the optimum may
  lie below it. MostPop recent is the simplest time-aware baseline that T-KGAT must beat, so its window
  should not be limited by the edge of the grid.
- Decision: the grid becomes {1, 3, 5, 7, 14, 30} days. The window is still chosen on validation only
  (mean validation Recall@20 over seeds; MostPop has no randomness, so one seed); the selected point is
  scored on test once, by `scripts/run.py`, as for every other model. The window is still anchored at the
  last train timestamp (decision 0002 is unchanged).
- Selection rule: the choice between the old and the new grid is made by this decision and by validation,
  never by comparing test scores. The earlier test score (window 7) stays in the registry; it is not an
  official result unless the selected window is still 7.
- Consequences:
  - Old registry rows are not edited (append-only). The run adds six validation rows and one test row;
    the rows of windows 7, 14 and 30 are deterministic and repeat the earlier values, so
    `registry.load(latest=True)` returns the same numbers for them.
  - `load(latest=True)` only replaces rows with the same key (dataset, split, model, config, seed, phase).
    A test row of a different window is a different key and is not replaced, so the official MostPop recent
    result is the test row of the window that the latest run selected (`selected` in
    `results/runs/<run_name>/mostpop_recent_summary.json`), not the newest or the best test row.
  - Reports and notebooks name the selected window and say whether it still lies on the edge of the grid.
  - The same check applies to any other grid whose selected value is its smallest or largest.
