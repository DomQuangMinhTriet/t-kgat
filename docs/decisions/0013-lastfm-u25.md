# 0013 — Last-FM: official run on a user subset (lastfm-u25)

- Status: proposed (2026-11-05, Triết; subset changed from `lastfm-u10` to `lastfm-u25` on 2026-11-06). Follows 0008
  (subsets) and 0012 (Last-FM dataset).
- Context: `scripts/measure_kgat.py`, one seed, RTX 3050 with 4 GB, no test scores. Full Last-FM, KG default graph: 106,625
  nodes, 3,101,554 edges, 214.0 s per epoch, peak 1,303 MB; without KG: 153.3 s per epoch. The KGAT grid alone (4 grid points,
  5 seeds, 300 epochs) would take about 357 hours (119 hours at 100 epochs). A 10% subset (`lastfm-u10`, 3.79 s per epoch)
  is cheap but loses 55.2% of the validation and 50.5% of the test positives to cold items, so it no longer has the large
  validation set that motivates Last-FM. The 25% subset `lastfm-u25` (`user_seed` 2019, as `ml1m_u10`), measured on
  2026-11-06 (3 steady epochs after one warm-up epoch; `s/epoch` = CF + KG + attention phases plus one fifth of a
  validation pass):

  | graph | nodes | edges | s/epoch | peak MB |
  |---|---|---|---|---|
  | KGAT, KG default | 49,713 | 778,058 | 20.13 and 20.52 (two runs) | 532 |
  | T-KGAT Stage 1 (gamma 3), KG default | 49,713 | 778,058 | 20.29 | 532 |
  | no KG | 23,986 | 472,106 | 12.22 and 11.96 (two runs) | 518 |

  T-KGAT costs the same as KGAT per epoch (the Delta-t term only touches the attention, 0.05 to 0.07 s per epoch). Two
  `measure_kgat.py` processes on the same GPU took 41.41 and 41.46 s per KGAT epoch each, so running in parallel gives no
  throughput gain; the runs are sequential. Stage 2 (`time_on: kg`) was not measured; its epoch time is assumed equal to
  Stage 1.
- Decision:
  1. The official Last-FM experiments run on `lastfm-u25` (`configs/data/lastfm_u25.yaml`: `user_fraction` 0.25,
     `user_seed` 2019; 5,892 of 23,566 users sampled): train 236,053 pairs, 5,321 users, 18,665 items; validation 23,708
     pairs, 2,319 users; test 22,419 pairs, 1,608 users; t1 = 1398783377 (2014-04-29 14:56:17 UTC), t2 = 1402150940
     (2014-06-07 14:22:20 UTC). `lastfm-u10` stays as a development set; no official number is reported on it.
  2. Full Last-FM is not run; it is reported as infeasible on the available hardware, as 0008 did for ML-1M.
  3. Grids, five seeds (2019 to 2023), selection on validation, test scored once and all statistics are unchanged.
  4. In the one-hop KG, triples whose head is not a train item are dropped (1,293 of 331,097 on full Last-FM, 1,097 of
     154,076 on `lastfm-u25`, all undated), because keeping them would change the KG structure shared with MovieLens
     (`ckg.py`).
  5. Measurement gate: `lastfm-u25` is run in full if KGAT on the KG default graph takes no more than 20 s per epoch
     (`s/epoch` of `scripts/measure_kgat.py`, as defined above). The measurement of 2026-11-06 gave 20.13 and 20.52 s,
     1% to 3% above the gate (the training phases alone, without the validation share, took 19.98 and 20.34 s). Whether
     this meets the gate is settled by the team before this record is accepted.
  6. If the time runs out, Stage 2 (`tkgat_kg` and `tkgat_kg_perm`) is dropped first and reported as not run for lack of
     time. Seeds, grids and epochs are never reduced to save time.
  7. Option A (a pair set not restricted by the April 2014 window of 0012) is not re-run on `lastfm-u25` for lack of time.
     The jump of new pairs at 2014-04-01 stays a limitation of the dataset (0012) and is reported with every Last-FM
     result.
- Consequences:
  - The test set has 1,608 users and validation 2,319 (MovieLens test: 51 and 95), so paired tests keep much more power
    than on MovieLens, and selection on validation rests on far more users.
  - The subset still loses part of validation and test to cold items (28.8% and 28.1% of the positives, against 3.0% and
    4.6% on full Last-FM); report it with every `lastfm-u25` number. Results hold for a 25% user sample of the KGAT Last-FM
    pair set; they are not claimed for the full dataset.
  - Estimated hours on `lastfm-u25`, upper bound without early stopping (runs x epochs x s/epoch; KG models 20.3 s, no KG
    12.1 s per epoch):

    | model config | runs | 100 epochs | 300 epochs |
    |---|---|---|---|
    | `kgat` | 20 | 11.3 | 33.9 |
    | `kgat_nokg` | 20 | 6.7 | 20.1 |
    | `tkgat` | 15 | 8.5 | 25.4 |
    | `tkgat_perm` | 5 | 2.8 | 8.5 |
    | `tkgat_kg` (Stage 2, not measured) | 15 | 8.5 | 25.4 |
    | `tkgat_kg_perm` (Stage 2, not measured) | 5 | 2.8 | 8.5 |
    | total | 80 | 40.6 | 121.7 |
    | total without Stage 2 | 60 | 29.3 | 87.8 |

    Early stopping on validation (patience 10 evaluations, every 5 epochs) can end a run before 300 epochs; how early
    on Last-FM is not known yet, so 300 epochs is the bound to plan with. MostPop and BPR-MF are not counted (minutes, not hours).
