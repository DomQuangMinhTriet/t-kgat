# CLAUDE.md — working agreement for AI assistants on t-kgat

Read this file first, then `README.md`, `CONTRIBUTING.md` and the decision records in `docs/decisions/`.
The full plan (Vietnamese) is `docs/plan.md`; section 12 is the current two-week schedule.

## Project in five lines
- HCMUS graduation thesis (KLTN). Students: Đởm Quang Minh Triết (Triết), Nguyễn Đăng Phôn (Phôn).
  Advisor: Dr. Nguyễn Ngọc Minh Châu ("cô Châu"). Major: Information Systems.
- Title: **T-KGAT — Temporal Knowledge Graph Attention Network for Recommendation**
  (the word "Confidence" was dropped on 2026-10-17; never reintroduce it).
- Base paper: Wang et al., *KGAT*, KDD 2019. Our model adds time: attention on user–item edges is
  decayed by Δt (`π − γ·δ_r·Δt` before the softmax; γ = 0 gives KGAT back), plus a comparison variant
  that puts time on KG triples through each fact's validity date (decisions 0004, 0005).
- The project was restarted on 2026-10-17 because the earlier results leaked test information
  (best epoch chosen on test, no validation set, preprocessing on the full data). Everything here exists
  to make that impossible. See `docs/meetings/Meeting_Notes_17102026.md` and decision 0001.
- Data: MovieLens-100K (main; Phôn runs it) and MovieLens-1M (run in parallel by Triết),
  KB4Rec links to Freebase, Wikidata facts with dates (fetched 2026-10-17).

## Non-negotiable rules (refuse or ask before breaking any of them)
1. **Test is scored once.** Epochs, hyper-parameters and variants are chosen on validation only
   (mean val Recall@20 over seeds). Only `scripts/run.py` touches test, after selection. Never add code
   that reads test scores inside training, early stopping, grid search or plotting of candidates.
2. **Global temporal split.** Two cut-offs t1 < t2 (quantiles 0.8 / 0.9 of all timestamps).
   Rating ≥ 4 is a positive. k-core is computed on train only. Val/test keep only users and items known
   in train. Masks hold only interactions before the cut-off. `check_split` must pass; never weaken a check
   in `src/tkgat/data/leakage.py` or `tests/test_leakage.py` to make something run.
3. **Time-consistent KG.** A KG used at cut-off t keeps a fact only if it was true before t
   (year-only dates = 31 Dec, month-only = last day of month). KG for train/val is built at t1, for test
   at t2. Wikidata is a 2026 snapshot: say so whenever KG results are reported.
4. **No hand-copied numbers.** Every number in a notebook or report is read from `results/registry.csv`
   or from files written by the scripts. 5 seeds (2019–2023), report mean ± std, paired Wilcoxon per user
   with Holm correction (`src/tkgat/eval/stats.py`). "std" is always the **sample** standard deviation
   over seeds (`ddof=1`; numpy's default `ddof=0` is wrong here, pandas' `.std()` is right). Read the
   registry through `tkgat.registry.load()` (newest run per dataset/split_sha256/model/config/seed/phase;
   "dataset" is the data config's `run_name`, unique per split); the file is append-only, so never edit or
   delete old rows.
5. **Data never enters git.** `data/raw/` and `data/processed/` are ignored (MovieLens/KB4Rec licences).
   Do not print large data dumps into the chat either.
6. **Protocol changes need a decision record.** New file in `docs/decisions/NNNN-*.md` (context,
   decision, consequences). Accepted records are never edited; supersede them. Ask the user first.

## Language
- English: code, comments, docstrings, commit messages, README/CONTRIBUTING, decision records.
- Vietnamese: explanations to the user, `docs/plan.md`, `docs/study/`, `docs/theory/`,
  `docs/data-report.md`, and the presentation notebook text (code cells stay English).

## Repo map
| Path | What |
|---|---|
| `src/tkgat/data/` | `loaders`, `clean` (duplicate titles, item 267), `split` (temporal split, k-core), `leakage` (`check_split`), `encode` |
| `src/tkgat/kg/linking.py` | ML-100K → ML-1M → ML-20M → Freebase/YAGO links, name + year checks |
| `src/tkgat/kg/temporal.py` | time-consistent KG: date parsing (decision 0006), `build_kg` (variants `default`, `drop_undated`, `static`), `check_kg`, shared entity ids |
| `src/tkgat/models/baselines.py` | `MostPop` (optionally recent window), `BPRMF` |
| `src/tkgat/device.py` | `resolve_device("auto"/"cuda"/"cpu")`, `seed_all` (deterministic kernels), device label for the registry |
| `src/tkgat/train/bpr.py` | BPR training on any device; negatives sampled on the device from train positives only; early stop on val |
| `src/tkgat/eval/` | full-ranking Recall/NDCG@K per user; paired Wilcoxon, paired t, bootstrap CI, Holm |
| `src/tkgat/registry.py` | appends every val/test score with git commit, config, seed |
| `scripts/01…08` | verify data → build splits → link entities → fetch Wikidata → Wikidata coverage report → build KG at t1/t2 → report numbers → KGAT export |
| `scripts/run.py` | train every grid point × seed, select on val, score test once |
| `configs/data/` | `ml100k.yaml` (5-core, main), `ml100k_k10.yaml` (10-core sensitivity), `ml1m.yaml`, `ml1m_u10.yaml` / `ml1m_u05.yaml` (random 10% / 5% of the ML-1M users, decision 0008) |
| `src/tkgat/data/subsample.py` | `subsample_users`: all ratings of a seeded random sample of users (`user_fraction`, `user_seed` in a data config) |
| `src/tkgat/export/kgat.py` | writes the official split and one KG variant in the KGAT file format (`scripts/08_export_kgat.py`, output in `data/export/`, ignored by git) |
| `experiments/legacy_kgat_data/` | the 10k-interaction scripts of PR #20, reference only: not the project protocol, never report numbers from them |
| `configs/model/` | `mostpop.yaml`, `mostpop_recent.yaml`, `bprmf.yaml` |
| `docs/` | `plan.md`, `decisions/`, `data-dictionary.md`, `meetings/`, `study/` (guides), `theory/` (team notes) |

Planned (not written yet): `src/tkgat/models/kgat.py` (+ T-KGAT),
`notebooks/02_entity_linking.ipynb`,
`00_present_2026-10-31.ipynb`, `docs/data-report.md`.

## Commands (Windows PowerShell, repo root)
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU build first (RTX 3050, 4 GB)
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available())"             # must print True
python scripts/01_verify_data.py
python scripts/02_build_splits.py --data configs/data/ml100k.yaml   # also ml100k_k10, ml1m
python scripts/03_link_entities.py
python scripts/05_wikidata_report.py
python scripts/06_build_kg.py --data configs/data/ml100k.yaml      # also ml1m; all 3 KG variants
python scripts/run.py --data configs/data/ml1m.yaml --model configs/model/bprmf.yaml   # --device auto|cuda|cpu
python scripts/02_build_splits.py --data configs/data/ml1m_u10.yaml    # small dev set; then 06, then 08_export_kgat.py
ruff check .; pytest                # both must pass before every commit (CI runs them)
```
`data/processed/` is rebuilt from `data/raw/` by scripts 02–03; run them on a fresh machine.

## Known facts (verified 2026-10-17; re-check before quoting in a report)
- ML-100K cut-offs t1 = 1998-03-07, t2 = 1998-03-31; train 43,018 positives, 745 users, 932 items;
  val 66 users, test 51 users. ML-1M test has 952 users.
- MovieLens timestamps are *assumed* to be rating-entry times (the GroupLens READMEs only say "seconds since the epoch"; the
  assumption is the team's, decision 0008): 78% of ML-100K ratings (72% of ML-1M) fall within 24 h
  of the user's first rating (calendar-day UTC: 73.5% / 66.0%). Δt between one user's edges is often ~0,
  so the timestamp-permutation control is mandatory for every time-aware variant
  (`notebooks/01_eda.ipynb`, section 3b: 53% of users have all train edges within one hour, while an
  item's train edges have a median std of 45 / 67 days on ML-100K / ML-1M).
- Smoke baselines on ML-100K (not official): MostPop test Recall@20 = 0.1555. BPR-MF val Recall@20 of the
  three grid points are within one std of each other (0.168–0.181 over 5 seeds), so the selected point and
  its test score (0.156 ± 0.026 with the old CPU sampler, 0.189 ± 0.019 with the device sampler) change
  with the random stream. Selection on 66 val users is noisy: report it, and prefer ML-1M for conclusions.
- Official ML-1M baselines (verified 2026-10-26 with `registry.load()`), test Recall@20: MostPop 0.0902,
  MostPop with a 7-day window 0.1060, BPR-MF 0.1194 ± 0.0031 (sample std over 5 seeds; grid point
  dim 64, lr 1e-3, l2 1e-4 selected on val). BPR-MF is reproducible on the GPU: the re-run at 08f5e63
  gave the same test score for every seed as the run at 4759618.
- MostPop recent window (decision 0007): the grid was widened from {7, 14, 30} to {1, 3, 5, 7, 14, 30} days
  because 7 was the edge of the old grid. Val Recall@20 on ML-1M: 1 d 0.0748, 3 d 0.1130, 5 d 0.1289,
  7 d 0.1336, 14 d 0.1082, 30 d 0.0986. 7 days is still selected, now in the interior of the grid, so the
  edge is no longer an issue; the test score is unchanged (0.1060). The window is anchored at the last
  train timestamp, not at t1 (they differ by 0.33 h on ML-100K and 0.01 h on ML-1M).
- KB4Rec covers 97.2% of ML-100K items. Known wrong link: "Clubland" → Cloudland Canyon State Park.
- Wikidata: 119,748 facts, 18 relations, 3,725 of 3,811 linked films. Statement dates exist almost
  only for `award_received` (80%) and `nominated_for` (97%). Titanic's Oscars (1998-03-23) fall between
  t1 and t2 of ML-100K.
- KG triples at t1 (verified 2026-10-23, `kg_meta.json`): ML-100K default 36,560 / drop_undated 29,840 /
  static 37,103; ML-1M default 103,443 / drop_undated 84,522 / static 104,782.
  ML-1M has the same KG at t1 and t2 (no dated fact between 2000-12-02 and 2000-12-29).
- The time filter drops only ~1.5% of facts (ML-100K: 543 of 37,103 at t1), but they fall on popular
  films: the 10% most-rated train items hold 16% of all facts yet 44% of the future award/nomination facts
  removed at t1 (ML-1M: 15% vs 36%; `notebooks/03_kg_temporal.ipynb`, section 5). Titanic alone loses
  28 facts at t1; its 11 Oscars (1998-03-23) are only in the test KG (t2).
- Only ~1.5% of facts are time-filtered, but the affected items carry ~48-53% (ML-100K) / ~41-43% (ML-1M)
  of val/test positives (`notebooks/03_kg_temporal.ipynb`, section 6: 344 / 899 items that lose at least
  one fact; val 53.1% / 40.6% at t1, test 48.1% / 42.6% at t2).

## GPU rules
- Every model, trainer and script takes a `device` (default `"auto"`) and keeps tensors on it: no
  per-sample Python loops or `.cpu()` calls inside the training loop; sample negatives with a
  `torch.Generator` on the device (see `NegativeSampler`). Score functions may return torch tensors;
  `evaluate` masks and takes top-K on their device.
- Call `seed_all(seed)` at the start of each run. Same seed + same device = same numbers; CPU and GPU
  streams differ, which is why the registry has a `device` column.
- The laptop GPU has 4 GB: keep full-graph tensors for KGAT in float32 and use sparse adjacency
  (`torch.sparse`) for propagation; check `torch.cuda.max_memory_allocated()` on ML-1M before long runs.
- Tests must pass on CPU (CI has no GPU); GPU-only checks use `pytest.mark.skipif(not torch.cuda.is_available())`.

## How to work with the user
- Before starting a task, say which item of `docs/plan.md` §12 it belongs to and what will change.
- Small, reviewable steps; one branch and one PR per task (`feat/16-build-kg`); run `ruff` and
  `pytest` before proposing a commit. Do not push or merge unless asked.
- If a request conflicts with the rules above, explain the conflict and ask instead of complying.
- When a result is surprising (too good, or a baseline beating the model), suspect leakage first and
  check before reporting.
- At the end of a session, list in Vietnamese: đã làm / chưa làm / cần làm.
