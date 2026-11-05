# 0013 — Last-FM: official run on a user subset (lastfm-u10)

- Status: proposed (2026-11-05, Triết). Follows 0008 (subsets) and 0012 (Last-FM dataset).
- Context: `scripts/measure_kgat.py`, one seed, RTX 3050 with 4 GB, no test scores. Full Last-FM, KG default graph: 106,625
  nodes, 3,101,554 edges, 214.0 s per epoch, peak 1,303 MB; without KG: 153.3 s per epoch. The KGAT grid alone (4 grid points,
  5 seeds, 300 epochs) would take about 357 hours (119 hours at 100 epochs). The subset `lastfm-u10` (10% of the users,
  `user_seed` 2019, as `ml1m_u10`): KG default graph 23,059 nodes, 250,628 edges, 3.79 s per epoch, peak 238 MB; without KG
  2.51 s per epoch. The KGAT grid on it takes about 6.3 hours.
- Decision:
  1. The official Last-FM experiments run on `lastfm-u10` (`configs/data/lastfm_u10.yaml`: `user_fraction` 0.10, `user_seed`
     2019): train 63,386 pairs, 1,936 users, 6,887 items; test 5,286 pairs, 559 users; t1 = 1398735201, t2 = 1402046786.
  2. Full Last-FM is not run; it is reported as infeasible on the available hardware, as 0008 did for ML-1M.
  3. Grids, five seeds (2019 to 2023), selection on validation, test scored once and all statistics are unchanged.
  4. In the one-hop KG, triples whose head is not a train item (1,293 of 331,097, all undated) are dropped, because keeping
     them would change the KG structure shared with MovieLens (`ckg.py`).
- Consequences:
  - The test set has 559 users (MovieLens: 51 and 95), so paired tests keep much more power than on MovieLens.
  - Results hold for a 10% user sample of the KGAT Last-FM pair set; they are not claimed for the full dataset.
  - The estimated total for the official runs on `lastfm-u10` is about 30 hours (KGAT grid 6.3, no-KG 4.2, Stage 1 and its
    permuted control about 4.7 each, Stage 2 about 9.5; T-KGAT epoch time assumed equal to KGAT's, not yet measured).
