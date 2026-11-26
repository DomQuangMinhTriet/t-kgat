# 0021 - Protocol v2: one protocol for the restart after the 15/11 meeting

- Status: **proposed** (2026-11-26, Triet). Not accepted until the other author has reviewed it in a pull request
  (rule 1 of section 8). Sections 3 and 4 are owned by Phon, sections 5 to 7 by Triet; each owner writes the numbers of
  their own sections, the other author reviews them.
- Builds on 0002 (leakage-safe temporal split), 0009 (delta-t in the attention) and the 12 remarks of the advisor in
  `docs/meetings/Meeting_Notes_15112026.md`. It replaces everything decided after 15/11: 0015 (`latest`), 0016, 0017, 0018,
  0019 and 0020 are **not binding**. They stay in the archive tags (section 2) and are used only as reference.
- Why a new file: after 15/11 two incompatible delta-t formulas were both marked accepted (0016 and 0019), `main` still
  carries the `latest` timestamp policy (PR #47), four datasets were built in ten days with different parameters
  (`max_epochs` 100/150/300, k-core 5/10, grid 12/3 points, 5/3 seeds), and two people edited `kgat.py` and `ckg.py`
  without a review. From this record on, a number is valid only if it is in the parameter table of this file.

## 1. Decisions of the meeting of 2026-11-26 (Triet and Phon)
D1 base A; D2 one protocol file with cross review; D3 formula of 0019; D4 `earliest`; D5 14-day window, 10-core,
Phon builds the data; D6 report everything, 3 Kaggle accounts, fixed training parameters, 3 seeds; D7 `v02/clean` is the
default branch; D8 one pull request per topic, review within 24 hours; D9 the advisor was told about the problems and the
request to move the report. The details are the sections below.

## 2. Base of the code
- `v02/clean` = `v0.1b-u25` (`9c646a4`) + `c3412fc` (fixes who was busy in the meeting notes). Code from later work is ported
  from the archive tags by `git checkout <tag> -- <path>`, reviewed, one topic per pull request:
  `archive/pre-restart-2026-11-25` (`9315227`), `archive/origin-main-2026-11-26` (`f9e0d83`), `archive/pr49-head-2026-11-26`
  (`b1ea496`; resume per epoch and the selection/refit window report are the pieces that are needed).
- Not ported: PR #47 and 0015 (`latest`); the default `latest` still on `main` is removed by the default branch change.
- Official runs start from a clean commit (no `-dirty`); the commit hash is stored in every registry row.

## 3. Data (owner: Phon)
1. **Source.** Built directly from `lfm1b-tracks.inter` (advisor remarks 1 and 12). Cutting and processing are two separate
   scripts (remark 2); every intermediate file has a SHA-256.
2. **Window.** 14 days, 2014-03-18 to 2014-04-01 (end exclusive). Events are sorted by time and then cut.
3. **Timestamp of a pair.** Earliest listen of the pair **inside the window** (`timestamp_policy: earliest`, remark 7).
   The cache of script 01b keeps the earliest listen over the whole history, so the 14-day window needs a new scan of the
   raw `.inter` file (a window built from that cache would lose repeat listens inside the window).
4. **Split.** t1 and t2 are the 0.8 and 0.9 quantiles of the pair timestamps inside the window (0002 item 2). train `< t1`,
   validation `[t1, t2)`, test `>= t2`. Global, not per user.
5. **Filters.** 10-core, iterative, on train only; no user sampling (`--target-interactions 0`); validation and test keep
   only users and items seen in train, and the dropped counts are reported. `min_rating = 1` (implicit feedback).
6. **Reference numbers.** From the simulation `window_sizes.csv` (not from the real build): train 1,749,616 pairs, 24,476
   users, 71,541 items, validation kept 31.5%, test kept 31.2%, t1 = 2014-03-28T15:36:46, t2 = 2014-03-30T09:31:12;
   validation about 42 hours and test about 38 hours. **These are replaced by the real build.**
7. **Knowledge graph.** Freebase dump from Archive.org, snapshot of 2015-04-16 (not the 2015-08-09 mirror used in 0012),
   9 relations extracted by script 01c (including `music.recording.releases`). The track to MID mapping is the new one
   from RecBole/RecSysDatasets with **1,254,923 tracks**; the earlier cache of `freebase.tsv` used a mapping of 44,788 tracks
   and is not reused. Scripts 01c and 02b must receive the new `--item-links` (their default still points to the old
   mapping), and `item_list.txt` and `freebase.tsv` are rebuilt for the 14-day item set.
   The KG of the 15/11 report (152,976 triples, 9 relations) is a different KG and its numbers do not carry over.
8. **Provenance.** Mapping file: `item_links.tsv`, 1,254,923 mapping lines plus one header line (checked by Phon on 2026-11-26);
   copy shared at https://drive.google.com/file/d/1SqoD1_3VgqjrsOo4TJfoR_AeKe_jHl6x/view?usp=sharing, derived from RecBole/RecSysDatasets
   (to be confirmed by Phon: the exact upstream file and path). **Still open (Phon), before the first run:** SHA-256 of
   `item_links.tsv`; dump file name and SHA-256; SHA-256 of train, validation, test, `freebase.tsv`, `item_list.txt`.
9. **Check.** `scripts/16_check_clean_dataset.py` (to be added by a pull request) passes before the data are uploaded
   to the shared private Kaggle dataset; its output goes to the registry meta file.
10. **KG coverage.** Reported as two separate rates: share of items linked to a MID, and share of items that still have facts
    after filtering. No minimum is fixed before the build, and none is chosen after looking at a test score. A low
    coverage is written in the limitations; it does not stop the runs.

## 4. Time formula (owner: Phon, reviewer: Triet)
1. **Formula of 0019.** For each edge type the age of an edge is normalised to the train window of that type:
   `delta = clip((max_train - ts) / (max_train - min_train), 0, 1)`; no absolute scale `tau`; the attention logit is
   reduced by `gamma_type * delta` before the per-head softmax, as in 0009 item 1. Facts without a date have `delta = 0`.
   `time_on: both` (interaction edges and KG triples in one model, remark 11). *Phon restates here the exact definition of the
   window for KG edges, with a worked example, so that the code and this text can be compared.*
2. **Grid.** `gamma in {1, 3, 10}`, with `gamma_ui = gamma_kg` during selection (3 fits per model and seed). `gamma = 0` is KGAT.
3. **Refit.** The normalisation window is recomputed on train + validation at refit. Triet accepted this on 2026-11-26. The
   consequence is a small change in the meaning of gamma between selection and refit (estimated at about 1.17 times for the
   14-day window, against a grid spacing of 3 times). This is stated in the report.
4. **Which relations take part.** Default of 0019: every relation that has a date. For track-to-artist the "date" is
   the start of the artist's activity or the birth date, not the time the fact became true or recent, so a long-active artist
   is treated as "old". This is a known limitation and is stated in the report. The ablation `gamma_kg = 0` shows whether the
   time term on the KG carries anything.
5. **Anchor and leakage rules of 0009** that are not changed by this section: anchors come from train timestamps only (at
   refit from train + validation), and `gamma` and `delta` are hyper-parameters, not learned.

## 5. Training (owner: Triet)
1. **Models.** BPR-MF, KGAT, KGAT without KG, T-KGAT (both edge types), and the controls of section 6.
2. **Parameters.** `max_epochs` 150, `patience` 10, `eval_every` 5. Learning rate fixed at 1e-3 (no schedule), `l2 = l2_kg = 1e-5`,
   dropout 0.1 for every model. The fixed learning rate deviates from advisor remark 5 (a decreasing rate): on the earlier
   gate a cosine schedule was not better than the seed noise (`docs/lr-schedule.md`, in the archive). The deviation is
   written in the report.
3. **Selection.** Mean validation Recall@20 over seeds. The grid for T-KGAT is the gamma grid of section 4 (selection
   already uses the (lr, l2) above, which are fixed and not tuned per model).
4. **Refit (remark 6).** After selection: train again on train + validation for the number of epochs found in selection, with the
   graph made of train + validation interactions and the KG at t2; score test **once**, for every model. This replaces item 7
   of 0002 ("no retraining on train + validation"). *The rule that turns the per-seed `best_epoch` values into one epoch count
   (mean or median) is fixed here before any run; proposal: per-seed `best_epoch` of the selected configuration, each seed
   refit with its own count. To be confirmed against the trainer code in the port.*
5. **Seeds.** 3 seeds from the start (2019, 2020, 2021). More seeds only if GPU hours are left after every item of section 7.
6. **Statistics.** Unchanged from 0013: paired Wilcoxon with Holm correction, plus bootstrap confidence intervals
   (`stats.py`). With 3 seeds these intervals are wide, and the report says so.
7. **Reproducibility.** Append-only registry; every row records the commit, the data hashes, the device, `gamma`, the
   normalisation constants and the permutation seed. No `--resume` after any change of data, KG or code that changes a number;
   epoch resume (from `b1ea496`) is only for continuing an interrupted session of the same run.

## 6. Controls and ablations (owner: Phon writes the specification, Triet adds the support, Phon reviews)
- **Permutation control.** Re-run the selected gamma with the train timestamps permuted globally over all train edges, one
  permutation per seed (0009 item 5). If the permuted run is as good as the real one, the gain is not from time.
- **Ablation.** With the selected gamma, one run with `gamma_ui = 0` (time only on the KG) and one with `gamma_kg = 0` (time only on
  the interaction edges).
- The standalone mode of the current code refuses `control` and `ablate`; they are added in the port, with tests
  (`gamma = 0` equals KGAT; permutation keeps the multiset of timestamps).

## 7. Compute (owner: Triet)
1. **Accounts.** Triet 1 Kaggle account, Phon 2. Triet's account runs BPR-MF and KGAT-noKG; Phon's accounts run KGAT, T-KGAT and the
   controls. The 14-day data are shared through one private Kaggle dataset uploaded by Phon after the check of section 3.
   The split of accounts is reviewed after the quotas reset (see item 3).
2. **Order of priority.** BPR-MF, KGAT-noKG, KGAT, T-KGAT, permutation control, ablations. Benchmark of seconds per epoch first.
   Before the code is ported, the benchmark and the first seed run on the code of `b1ea496` (Phon agreed, 2026-11-26), on the quota that
   would otherwise be lost at the reset; those runs are never official results. **Every run before the cut decision of item 5 uses
   `--val-only`** (`scripts/run.py` and `scripts/kaggle_tkgat_joint.py` of `b1ea496` both have it: test files are not opened, no test row is
   written, refit is refused); the batch timing uses `--benchmark-batches` with `--val-only`, which never selects or scores a model. The first seed of KGAT and KGAT-noKG is also the check that the run is correct and the clock; it is **not** used to draw a conclusion about
   the KG (agreed with Phon, 2026-11-26).
3. **Quota on 2026-11-26.** Triet 29 h 41 m of 30 h, reset at about 2026-11-28 07:00 (UTC+7, that is Saturday 00:00 UTC); Phon one
   account with 30 h and one with 10 h (40 h). All accounts reset on the same day (Phon, 2026-11-26). The quota is not carried over, so hours that are not
   used before the reset are lost. Kaggle limits one session to 12 hours; session splitting and epoch resume deal with that limit
   and do not add hours to the weekly quota.
4. **Estimate (rough, not verified).** About 5.4 minutes per epoch at 14 days (a quadratic extrapolation of 10.4 minutes per epoch measured
   on 2.43 million train pairs), about 10 hours for one selection fit and about 6 hours for one refit. The full plan (3 seeds of four main
   models, the permutation control and two ablations) is then about 360 hours; the hours from 28/11 07:00 to 5/12 07:00 are about 90
   (three accounts of 30 hours). The plan does not fit in one week.
5. **What is cut, and when.** *Open.* Phon has no further opinion on the order (2026-11-26) and asked that every cut (fewer seeds of the controls or of the main models down to 3, a smaller gamma
   grid) be written here before it is applied, and has not agreed to a fixed order. The decision is taken together after the benchmark
   (morning of 2026-11-28), written in this file in a commit of its own, **before anyone opens a test score**. Proposal of Triet for the
   order: one seed of the four main models, then one seed of the controls and the ablations, then more seeds.

## 8. Rules of work
1. One protocol file (this one). Nobody marks a record accepted alone; the other author approves it in the pull request.
2. One parameter table: this file. Configs and notebooks read it, and a test compares them with the values here.
3. Areas: Phon owns the data scripts (`01b`, `01c`, `02b`, the loader) and the formula text; Triet owns the trainer, refit, configs and gates. A
   change in the other person's area goes through a pull request and waits for approval.
4. One pull request per topic; no pull request is merged into another; the reviewer answers within 24 hours.
5. Order of freezing: protocol, then data (with hashes), then the benchmark, then the official runs. Changing a frozen item needs its own
   pull request, both approvals and a reason.
6. Clean-up work comes with a manifest and an undo script; nothing is deleted before both agree.
7. `v02/clean` is the default branch. `main` and the archive tags stay. PR #48, #49 and #50 are closed with a comment that points to the tag after the
   code has been ported. No branch is deleted.

## 9. To be disclosed in the report
- Changes made after test scores of earlier datasets were seen (the learning-rate, epoch and refit changes follow the advisor's remarks of
  15/11 and not a score; 0016 listed the earlier test scores).
- Fixed learning rate against remark 5; one window of 14 days with a short validation (about 42 h) and test (about 38 h), so a weak time signal is possible.
- KG source and mapping changed after 15/11; coverage rates of section 3 item 10.
- Meaning of gamma changes slightly at refit (section 4 item 3); date of the artist relation (section 4 item 4).
- Any cut of runs (section 7 item 5), and that estimates of hours were made before the benchmark.

## 10. Open items
| # | Item | Owner | Due |
|---|---|---|---|
| 1 | Exact window definition for KG edges (section 4 item 1) | Phon | 27/11 |
| 2 | SHA-256 of the mapping, dump and data files; upstream path of the mapping (section 3 item 8) | Phon | 27/11 |
| 3 | Real build: window numbers, KG coverage (two rates) | Phon | 27/11 |
| 4 | Time to scan the raw `.inter` for the 14-day window (Phon: "probably in time" for 27/11, not measured) | Phon | 26/11 evening |
| 6 | Refit epoch rule (section 5 item 4) | Triet | 29/11 |
| 7 | What is cut and in which order (section 7 item 5) | both | 28/11, after the benchmark |
| 8 | Specification of `control` and `ablate` | Phon | 28/11 |

## 11. Schedule (proposed; fixed only if the advisor agrees to move the report to 5-6/12)
27/11 data built and checked, `b1ea496` reviewed, benchmark on the quota that would otherwise be lost; 28-29/11 port (data pipeline by Phon;
trainer, refit, formula, configs by Triet), benchmark read, cut decided, `pytest` and `ruff`; 30/11 official runs start; 1-3/12 T-KGAT,
controls, ablations; 4/12 results frozen, table with confidence intervals; 5-6/12 report.
