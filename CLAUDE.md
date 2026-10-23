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
   with Holm correction (`src/tkgat/eval/stats.py`).
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
| `src/tkgat/models/baselines.py` | `MostPop` (optionally recent window), `BPRMF` |
| `src/tkgat/train/bpr.py` | BPR training, negatives sampled from train only, early stop on val |
| `src/tkgat/eval/` | full-ranking Recall/NDCG@K per user; paired Wilcoxon, paired t, bootstrap CI, Holm |
| `src/tkgat/registry.py` | appends every val/test score with git commit, config, seed |
| `scripts/01…05` | verify data → build splits → link entities → fetch Wikidata → Wikidata coverage report |
| `scripts/run.py` | train every grid point × seed, select on val, score test once |
| `configs/data/` | `ml100k.yaml` (5-core, main), `ml100k_k10.yaml` (10-core sensitivity), `ml1m.yaml` |
| `configs/model/` | `mostpop.yaml`, `mostpop_recent.yaml`, `bprmf.yaml` |
| `docs/` | `plan.md`, `decisions/`, `data-dictionary.md`, `meetings/`, `study/` (guides), `theory/` (team notes) |

Planned (not written yet): `scripts/06_build_kg.py`, `src/tkgat/models/kgat.py` (+ T-KGAT),
`notebooks/01_eda.ipynb`, `02_entity_linking.ipynb`, `03_kg_temporal.ipynb`,
`00_present_2026-10-31.ipynb`, `docs/data-report.md`.

## Commands (Windows PowerShell, repo root)
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"            # CPU torch is fine for 100K; install CUDA torch first if a GPU is used
python scripts/01_verify_data.py
python scripts/02_build_splits.py --data configs/data/ml100k.yaml   # also ml100k_k10, ml1m
python scripts/03_link_entities.py
python scripts/05_wikidata_report.py
python scripts/run.py --data configs/data/ml1m.yaml --model configs/model/bprmf.yaml
ruff check .; pytest                # both must pass before every commit (CI runs them)
```
`data/processed/` is rebuilt from `data/raw/` by scripts 02–03; run them on a fresh machine.

## Known facts (verified 2026-10-17; re-check before quoting in a report)
- ML-100K cut-offs t1 = 1998-03-07, t2 = 1998-03-31; train 43,018 positives, 745 users, 932 items;
  val 66 users, test 51 users. ML-1M test has 952 users.
- MovieLens timestamps are rating-entry times: 78% of ML-100K ratings (72% of ML-1M) fall on the
  user's first day. Δt between one user's edges is often ~0, so the timestamp-permutation control is
  mandatory for every time-aware variant.
- Smoke baselines on ML-100K (not official): MostPop Recall@20 = 0.1555; BPR-MF 0.156 ± 0.026.
- KB4Rec covers 97.2% of ML-100K items. Known wrong link: "Clubland" → Cloudland Canyon State Park.
- Wikidata: 119,748 facts, 18 relations, 3,725 of 3,811 linked films. Statement dates exist almost
  only for `award_received` (80%) and `nominated_for` (97%). Titanic's Oscars (1998-03-23) fall between
  t1 and t2 of ML-100K.

## How to work with the user
- Before starting a task, say which item of `docs/plan.md` §12 it belongs to and what will change.
- Small, reviewable steps; one branch and one PR per task (`feat/16-build-kg`); run `ruff` and
  `pytest` before proposing a commit. Do not push or merge unless asked.
- If a request conflicts with the rules above, explain the conflict and ask instead of complying.
- When a result is surprising (too good, or a baseline beating the model), suspect leakage first and
  check before reporting.
- At the end of a session, list in Vietnamese: đã làm / chưa làm / cần làm.
