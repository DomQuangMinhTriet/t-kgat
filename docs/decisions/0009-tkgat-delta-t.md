# 0009 — Delta-t in the attention (T-KGAT) and the KG-time variant

- Status: accepted (2026-10-28). Proposed by Triet on 2026-10-28; reviewed and approved by Phon on 2026-10-28 (chat). Builds on 0005 (where time enters) and 0004/0006 (KG dates).
- Owner of the code: Triet (with Claude Code); review: Phon.

## Decision

### Stage 1 (core, mandatory): Delta-t on Interact edges
1. **Formula.** `pi_time(h, r, t) = pi(h, r, t) - gamma * delta_r * dt_norm(h, t)`, subtracted **before** the per-head softmax.
   `delta_r = 1` for `Interact` and `Interact^-1` (edges that carry a timestamp), `0` for every KG relation. `gamma = 0` is exactly KGAT
   (same `A` and `e*`, element by element). `gamma` and `delta_r` are hyper-parameters, not learned (design-kgat Q4: no BPR gradient into attention).
2. **Anchor.** `T_ref` = largest timestamp in the **train** split, the same value for validation and test (test edges are train edges).
   `dt = max(0, T_ref - ts)`. For a user node a shift of `T_ref` cancels in the softmax; for an item node it sets how strongly interactions are
   penalised against KG edges (`delta = 0`).
3. **Normalisation.** `dt_norm = dt / (T_ref - min train ts)`, in [0, 1] for Interact edges. Both numbers come from `train_ts` only (leakage point L7).
4. **gamma protocol (two stages).** (lr, l2) are the point KGAT selected on validation for the same dataset; only `gamma in {1, 3, 10}` is searched,
   5 seeds each, selected by mean validation Recall@20; `gamma = 0` is the KGAT run already done. Caveat to report: T-KGAT gets an extra search over gamma
   after (lr, l2) were tuned for KGAT, so the comparison is slightly tilted toward KGAT on (lr, l2) and toward T-KGAT on gamma.
5. **Control (mandatory, 0005).** Re-run the selected gamma with `train_ts` permuted **globally** over all train edges (one permutation per seed, drawn
   from a generator seeded by the run seed; only `train_ts` is shuffled; the multiset of timestamps is unchanged). Report real vs permuted. If permuted is
   as good as real, the gain is not from time.
6. **Tests.** `test_gamma_zero_is_kgat`, `test_zero_dt_is_kgat` (design-kgat appendix), plus: at equal `pi` an older edge never gets more attention than a
   newer one; `edge_ts` is built from `train_ts` only; the permutation keeps the multiset; NaN `edge_ts` on KG edges never reaches the output when `delta_r = 0`.

### Stage 2 (conditional): time on KG triples
7. Only if Stage 1 official runs have started by **Thu 2026-10-29 14:00**; otherwise it moves after 31/10 and is announced to the advisor as the next step.
8. KG edges get `ts` = validity start date of the fact (0004/0006); `delta_r = 1` for KG relations in this variant. Undated facts (18.1%) keep `Delta-t = 0`
   in `default`; `drop_undated` removes them.
9. **Data.** Existing `kg_t1.tsv` / `kg_t2.tsv` stay untouched (their hashes are checked by `load_kg`). Dates go to new files `kg_t1_dates.tsv` / `kg_t2_dates.tsv`
   in the same triple order, checked against the same split hash. Script 06 gets an extra output, no change to the old ones.
10. **Two problems to settle before coding Stage 2:**
    - At test the KG is the t2 graph, so some facts are dated after `T_ref` (the largest train timestamp): `dt < 0`. They are clipped to 0 (rule in item 2). Settled as proposed.
    - Fact dates (film or award years, often decades before `T_ref`) live on another time scale than interactions: `dt / window` can be 10-100, so
      `gamma * dt_norm` would switch off old facts completely. Decision: **cap `dt_norm` at 1 for KG edges**. Settled as proposed (approved by Phon together with 0009).

### Out of scope until after 2026-10-31
Rating value as weight or relation, timestamp as a relation, LLM-assisted cleaning of film metadata (0005 "also in phase 5").

## Schedule and gates
- KGAT and KGAT-no-KG grids on ml-100k and ml-1m-u10 run first (overnight 28 -> 29/10); they provide the (lr, l2) for item 4.
- Tags (agreed 2026-10-28): `v0.1-review` (Thu 2026-10-29 18:00) freezes data, KG, baselines and the report. KGAT and T-KGAT get their own tag `v0.2-kgat`, placed on the
  main commit that holds the KGAT and T-KGAT Stage 1 code once the official runs are done. Official runs start from a clean commit (no `-dirty` label); Phon re-runs from a clean clone.
  Any KGAT/T-KGAT number shown before that tag is labelled "not frozen" and "preliminary: the advisor has not yet approved the protocol (0002 t1/t2)".
- Stage 1 code and tests merged by Thu 29/10 noon; official Stage 1 runs start the same afternoon. Full ML-1M is not run (estimated 29 h for one KGAT config).
- If Stage 1 cannot be run for both datasets by the evening of 30/10, the presentation reports KGAT vs KGAT-no-KG (RQ2) and presents T-KGAT as design + partial results.

## Consequences
- The attention tensors depend on `edge_ts`, so the registry row of a T-KGAT run must record gamma, anchor, normalisation constant and the permutation seed.
- No gradient reaches gamma; a learned gamma is future work.
