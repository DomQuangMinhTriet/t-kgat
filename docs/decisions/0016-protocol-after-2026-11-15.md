# 0016 — Protocol changes after the advisor meeting of 2026-11-15

- Status: accepted (2026-11-23: Phôn approved the pull request; proposed 2026-11-15 by Triết, draft written the night of
  the meeting). Follows the meeting notes `docs/meetings/Meeting_Notes_15112026.md` (12 points from the advisor).
- Supersedes in part: 0002 item 7 (no retraining on train+val); 0013 (user subset as the way to get a small dataset);
  the pair-set part of 0012 (the KGAT pair set); 0009 items 7 to 10 as far as the combined model is concerned (Stage 1
  and Stage 2 stay as ablations, see item 7). Supersedes the draft 0015 (`lastfm-u25-latest`), which is dropped.
  Everything not listed here (leakage checks, 5-core on train only, ranking masks, five seeds 2019 to 2023, Recall@20 and
  NDCG@20, paired Wilcoxon with Holm, append-only registry, the permutation controls) is unchanged.
- Context: the results on `lastfm-u25` (tag `v0.1b-u25`) are all not significant after Holm; three problems were found
  by the group itself (pair set selected with information after the cut-off; 25% user sample loses 28% of the positives
  to cold items; Stage 2 `dt_norm` capped at 1 for 55.7% of the triples). The advisor asked for a different way to get
  the data and a different way to train.
- Disclosure (applies to every item): these changes are made **after** the test scores of `ml-100k`, `ml-1m-u10` and
  `lastfm-u25` were seen. Those results stay in the repository and the thesis as the first protocol (appendix); the
  numbers of the new protocol are a new run, not a correction. Nothing below is tuned on any test score.

## Decision

### Data
1. **Time-window subset instead of a user sample** (advisor points 1 and 10). All events of the raw source are sorted by
   time and only the last `window_months` months are kept. `window_months` is chosen **before** looking at any
   score, from two numbers only: the number of events and the measured seconds per epoch (gate of item 9). The window is
   cut on the raw events, before the k-core, t1/t2 and everything else. t1 and t2 stay the 0.8 and 0.9 quantiles, now
   computed inside the window. The rule `user_fraction` of 0013 is not used.
2. **Pairs from the raw LFM-1b** (advisor point 12). The pair set is built from `lfm1b-tracks.inter`, not from the KGAT
   pair set. A pair (user, item) has the timestamp of its **earliest** listen in the window (advisor point 7: no new-item
   task; same convention as KGAT). Users are kept if they have at least `min_events_user` events in the window (fixed
   together with `window_months`, same rule as item 1). KG: the KGAT knowledge graph restricted to the items of the new
   pair set, with the dates of 0012; the 0013 rule on triples whose head is not a train item stays.
3. **Cutting and processing are two scripts** (advisor point 2). `scripts/01b_cut_lastfm.py` only sorts, cuts, deduplicates
   and writes `pairs_w<N>.tsv` and a `.meta.json` (parameters, row counts, min and max timestamp, sha256 of the output). The
   split script reads that file and nothing else from the raw data. The sha256 of the cut file goes into the split
   `meta.json` and the registry config. Running the cut twice gives the same sha256 (tested).

### Training
4. **Refit on train and validation** (advisor point 6). Hyper-parameters and the number of epochs are selected on
   validation as before (mean validation Recall@20 over the 5 seeds). Then, for the selected grid point and each seed,
   the model is trained again from scratch on train ∪ validation for `E` epochs, with no early stopping, where `E` is the
   median over the 5 seeds of the best epoch of the selected point. The graph has the train and validation edges, the KG at
   t2, and the ranking masks of test are unchanged (every interaction before t2). `T_ref` and any scale that
   depends on the train window are recomputed on train ∪ validation (item 7 removes the scale dependence of gamma). The
   refit model is scored on test once. This applies to **every** trained model (BPR-MF, KGAT, KGAT without KG, T-KGAT and its
   controls); MostPop counts on train ∪ validation. The registry gets `phase = test` rows with `refit: true` and `epochs: E`
   in the config. A test must show that the refit code path never reads the test split and that `E` comes from validation
   rows only. The old path (`--no-refit`) stays and reproduces the old numbers.
5. **Learning rate schedule** (advisor point 5). One schedule for all models: cosine decay from the grid learning rate to
   1/100 of it over `max_epochs`, against the constant rate of 0013. The schedule (`none` or `cosine`) is chosen on the small
   set by validation (item 9) and then fixed for every model and dataset; its parameters are written in the config and so in
   the registry key. The learning rate stays in the grid.
6. **Fewer epochs** (advisor points 3 and 4). The evidence is `docs/epoch-budget.md` (script `13_epoch_budget.py`, validation
   histories of the 365 runs already in the registry; no test score). `max_epochs` is the smallest value in {100, 150, 200}
   at which every model, at the learning rate of its grid, has a median validation Recall@20 of at least 98% of its best;
   the same value for all models; `patience` stays 10 evaluations (every 5 epochs). A "90th percentile of the best epoch plus
   one patience window" rule was considered and dropped: on `lastfm-u25` it gives about 300 epochs, which is what the runs
   already used. The learning rate schedule of item 5 changes the curves, so the value is checked again on the small set
   (item 9) before it is written into the Update section. Seeds and grid points are not reduced for time (0013 item 6
   stays); the grid itself may be made smaller by item 9, never larger.

### Model
7. **One T-KGAT with time on both kinds of edges** (advisor point 11).
   `pi_time = pi - gamma_ui * delta_ui * dt_ui / tau_ui - gamma_kg * delta_kg * dt_kg / tau_kg`, subtracted before the
   per-head softmax as in 0009. `dt_ui = max(0, T_ref - ts)` on Interact edges; `dt_kg = max(0, T_ref - fact date)` on
   dated KG triples, `0` for undated ones (0009 item 8). `tau_ui` and `tau_kg` are **absolute time scales** (days and years),
   not a window length, so that `gamma` means the same on the train window and on train ∪ validation and the
   cap of 0009 item 10 is not needed: there is **no cap** at 1. `gamma_ui = gamma_kg = 0` is exactly KGAT (test as in 0009).
   `delta_kg(r) = 1` only for a relation whose date is the start of the fact itself; for a relation whose date is an attribute
   of the tail entity (for example the age of an artist for track-to-artist) `delta_kg(r) = 0`. The list is read from the
   per-relation report of item 8 and written in an update section of this file before any official run. If no relation
   qualifies, the model is Stage 1 with the new scale and this is reported as such.
   Search on validation, all with 5 seeds: `gamma_ui = gamma_kg = gamma` in {1, 3, 10}, `tau_ui` in {90, 365} days, `tau_kg` in
   {5, 20} years (12 points); `(lr, l2, l2_kg, mess_dropout)` come from KGAT as in 0009 item 4. The small set (item 9) may remove
   values from this grid by validation; it never adds any. At the selected point two ablations are run (`gamma_kg = 0`,
   Stage 1 only; `gamma_ui = 0`, KG time only). Controls: one global permutation of `train_ts` **and** of the fact dates, same
   seed, run at the selected point, as in 0005 and 0009 item 5.
8. **Relation pruning** (advisor point 8). `scripts/14_relation_report.py` writes, for every relation of the KG on the small
   set: number of triples, share of items with at least one triple, share of dated triples, number of distinct tails.
   Candidate variants are fixed here: `default` (all relations) and `pruned` (drop every relation that covers fewer than 5% of
   the items **or** has fewer than 1,000 triples). `pruned` is adopted for the official runs only if its validation
   Recall@20 on the small set is not lower than `default` (mean over 3 seeds) with fewer triples; otherwise `default`
   stays. The relations kept are written to `kg_meta.json` and into the thesis.
9. **Small set and gate** (advisor points 9 and 10). A small window (`lastfm_w3`: the last 3 months of the available
   events, 2 months if KGAT takes more than 60 s per epoch) is built with items 1 to 3 and used only on validation to
   decide items 5, 7 (grid values), 8 and the epoch budget of item 6. It runs BPR-MF, KGAT and KGAT without KG, 3 seeds.
   If KGAT is not above KGAT without KG on validation there, the official runs still go ahead, but the advisor is told
   before they start. No number of the small set is reported as a result. It also tests dropping the learning rate 1e-4 grid
   points: in `docs/epoch-budget.md` they were cut at 300 epochs in 19 of 20 KGAT and KGAT-no-KG runs on `lastfm-u25`
   (about 22 of the 38 hours of those two models) and none of them was the selected point.

### Official run
10. Order: MostPop and MostPop-recent, BPR-MF, KGAT without KG, KGAT, T-KGAT (item 7), its ablations and its permutation
   control. Selection on validation, refit, test scored once, all statistics as in 0002 to 0009. Heavy runs may go to
   Kaggle under the rules of 0014 (same commit, same split hash, rows appended unchanged in a commit of their own).
   Tables get a 95% bootstrap confidence interval of the relative difference to KGAT next to each paired test, so that a
   non-significant result can be read as "no effect larger than x%" (script 10). The tag `v0.2-kgat` is placed on the
   commit of these runs, after Phôn reproduces BPR-MF seed by seed and one KGAT seed within noise from a clean clone.

## Consequences
- All Last-FM numbers of the first protocol become the "before 2026-11-15" appendix (`docs/rq3-table.md` is kept under a new
  name when the new table is made). ML-100K and ML-1M-u10 are not re-run unless time allows; if they are, the same items
  apply and the same disclosure.
- Refit gives the models more data at test but also removes the validation-based early stopping at test time; a model
  with a noisy validation curve can lose from it. This is a property of the protocol, reported, not tuned away.
- The new dataset has a different set of users and items, so numbers cannot be compared with `lastfm-u25` row by row.
  The 25% user sample and the KGAT pair set are no longer a source of the Last-FM results.
- The time scales of item 7 are in absolute units; the old `dt_norm` (window-relative) is kept only inside the old
  protocol's code path, so that the old rows can still be reproduced.
- Risks: the window may still be shifted by the same dating problem of Freebase (0012); a 3-month small set may show no KG
  effect for reasons of size and not of method; refit doubles the number of runs of the official stage (selection runs plus
  refit runs). The gate of item 9 and the hour measurement of 0013 are the controls.

## Update sections (to be added before the official run, from the small set and validation only)
- Values of `window_months` and `min_events_user`, with the event and seconds-per-epoch numbers they came from. *(pending: official window and 5-core are fixed in the data PR)*
- **Update 2026-11-23, items 5, 6, 9** (`docs/lr-schedule.md`, from `docs/small-set-gate.md`): learning-rate schedule `none`
  for all models (cosine is not better than seed noise on KGAT and KGAT without KG and is worse on BPR-MF and pruned KGAT).
  `max_epochs = 150`, `patience = 10` evaluations, `eval_every = 5`, same for all models. The pooled rule of the gate script gives
  100, but BPR-MF at its selected point reaches only 94% (median) of its best by epoch 100 and 100% by 150, so the rule read
  per grid point gives 150. Official files: `configs/model/v02/`.
- **Update 2026-11-23, item 8**: `default` relations stay. `pruned` was not better on validation (Recall@20 0.0629 against
  0.0630 with schedule none, 0.0618 against 0.0638 with cosine) and removes only 1,105 of 61,002 triples. *(`delta_kg = 1`
  list: pending Phôn, from `docs/relation-report-lastfm-w3.md`.)*
- **Update 2026-11-23, reduced grid** (items 7, 9): lr 1e-4 points dropped from every grid. KGAT and KGAT without KG have 2 points
  (lr 1e-3, l2 in {1e-5, 1e-4}); BPR-MF has 3. The T-KGAT grid of item 7 (12 points) is unchanged. *(Whether `tau` stays in the
  grid or is fixed a priori: pending Phôn.)*
