# 0010 — gamma at the edge of the grid: a validation-only sensitivity analysis, not a new grid

- Status: proposed (2026-10-28 by Triết); becomes accepted when Phôn approves the pull request
- Builds on 0009 (T-KGAT Stage 1, the gamma grid {1, 3, 10}) and on 0007 (a selected value at the edge of its grid is
  reported, and the grid is not widened after the fact). Changes nothing in 0009.
- Context: decision 0009, item 4, registered the grid `gamma in {1, 3, 10}` before any official T-KGAT run, with
  `gamma = 0` being the KGAT run that already exists. The official run selected **gamma = 1, the smallest point of the grid**.
  On validation, Recall@20 falls as gamma grows (the three points are ordered 1 > 3 > 10). On `ml-1m-u10`, the validation
  Recall@20 of gamma = 1 is **lower** than that of KGAT (gamma = 0). Taken together this suggests that the best gamma may lie
  below the grid, possibly at 0, where T-KGAT is KGAT: the time term may not help on these data. The test score of
  gamma = 1 has already been seen. This draft quotes no number: they belong to `results/registry.csv` (read with
  `registry.load`) and are not in `main` yet; they are to be taken from there, and the datasets on which gamma = 1 was
  selected confirmed, before the draft is approved.
- Decision:
  1. **The official grid does not change.** It was registered before the runs, and the test score of gamma = 1 was looked at.
     The official T-KGAT result stays the test row of the gamma that the latest official run selected (`selected` in
     `results/runs/<run_name>/tkgat_summary.json`), with its permutation control.
  2. **A sensitivity analysis on validation only.** Two more points, `gamma in {0.1, 0.3}`, in a new config
     `configs/model/tkgat_sens.yaml` (the same keys as `tkgat.yaml`: `from_kgat: kgat`, the same `fixed`, `kg` and `seeds`,
     a grid that only holds gamma), run with `scripts/run.py --val-only`: no test file is opened and no test row reaches the
     registry. Its model tag is `tkgat_sens`, so its rows never mix with those of `tkgat`; its summary is
     `tkgat_sens_val_summary.json`. No code changes: `scripts/run.py` already accepts any `gamma > 0` in a T-KGAT grid.
  3. **How it is reported.** As an appendix of the report, a table of validation Recall@20 (mean ± sample std over the five
     seeds) for gamma in {0.1, 0.3, 1, 3, 10}, plus gamma = 0 (KGAT), read from the registry. The appendix states that
     gamma = 1 lies at the lower edge of the official grid and that the points 0.1 and 0.3 were added after the official run.
  4. **If validation prefers a smaller gamma**, that is written in the limitations of the report ("the selected gamma is at the
     edge of the pre-registered grid, and a smaller gamma is better on validation"), together with the reading that the time
     signal may be weak. The official result and the claims about it are not changed.
  5. **Any test run for another gamma is a separate decision.** Scoring gamma = 0.1 or 0.3 on test needs its own record in
     `docs/decisions/` and Phôn's approval; this decision does not authorise it. The risk to state in that record: the choice
     would be made after the first test score and after the sensitivity result were known, so it is post hoc. It adds a second
     look at test and a degree of freedom the pre-registered grid did not have; if the new value then looks better, the gain
     cannot be separated from the selection. The report would have to show both test scores, say which gamma was registered, and
     not present the later one as the pre-registered result.
- Consequences:
  - The official table (`scripts/10_rq3_table.py`) is unchanged and reads `tkgat` and `tkgat_perm` only; `tkgat_sens` rows are
    validation rows and never appear in the test comparison.
  - The registry is append-only: the sensitivity run only adds validation rows under the tag `tkgat_sens`.
  - The run costs GPU time of the same order as the official T-KGAT grid, with two points instead of three. It is done after the
    official runs and does not delay them.
  - If the sensitivity analysis is not run, the report still says that gamma = 1 is at the lower edge of the grid and that
    validation Recall@20 falls as gamma grows (rule of 0007: a selected value at the edge is named).
  - A result of "smaller gamma is better on validation" weakens the claim that the time term helps, and is reported as such;
    a result of "gamma = 1 is the best of the five points" strengthens the choice of the grid and is reported as well.
  - Open points for Phôn: (a) the datasets on which gamma = 1 was selected; (b) whether the appendix also covers `ml-100k`
    or only `ml-1m-u10`; (c) the GPU budget for the extra points.
