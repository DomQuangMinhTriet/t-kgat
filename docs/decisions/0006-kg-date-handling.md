# 0006 — Reading Wikidata dates when building the time-consistent KG

- Status: accepted (2026-10-23 by Triết); to be reported to the advisor
- Clarifies decision 0004 (which stays unchanged), in particular its note "a film's own facts are valid
  from its release" and its example of an undated fact (film → director). Implemented in `src/tkgat/kg/temporal.py` and `scripts/06_build_kg.py`.
- Context: `scripts/04_fetch_wikidata.py` keeps the precision only for `point_in_time`. Every other date
  (`start_time`, `tail_publication`, `tail_inception`, `tail_birth`, `film_publication`) is cut to
  `YYYY-MM-DD`, and Wikidata writes a year-only date as `YYYY-01-01`. In the 2026-10-17 snapshot 58% of
  `film_publication`, 60% of `tail_publication` and 45% of `tail_inception` values end in `-01-01`.
  Read as day dates, they would place many facts too early. Read as 31 December, 1998 films would lose
  their own facts at t1 although users rated them before t1.

## Decision
- A. **Dates without precision.** A date without a precision field that ends in `-01-01` is read as
  year-only, that is 31 December of that year. Other dates without precision are read as day dates.
  This is a known limitation: a fact that really happened on 1 January is placed at 31 December, so it
  may be dropped at a cut-off where it was already true (conservative, never leaks). The fetch script is
  not changed now; re-fetching with precision would remove the limitation.
- B. **The film's own release date never drops its facts.** Every item in the KG is an item of the train
  split, so it was rated before t1 and existed at both cut-offs. `film_publication` is not used to filter.
  The target entity's existence date still is (a sequel, a remake, a later work released after `t`).
- C. **Precision outside 8–11.** `point_in_time` is read with its precision: 11 = day, 10 = month (last day
  of the month), 9 = year (31 December), 8 = decade (31 December of the decade's last year). Any other
  precision (e.g. 3, millennia) is treated as no date, so the fact follows the rule for undated facts.
  Dates that cannot be parsed (e.g. BC years) are also treated as no date.
- D. **What "undated" means.** A fact is *undated* when it has no usable date at all: neither a
  statement date (`point_in_time`, else `start_time`) nor an existence date of its target
  (`tail_publication`, `tail_inception`, `tail_birth`). The target's existence date is also evidence in
  time, because a fact cannot hold before its target exists. This clarifies the example
  "film → director" in 0004: a director with a known birth date makes that fact dated, not undated.
  Reading "undated" as "no statement date" would leave the drop-undated variant with about 7% of the
  triples, close to the no-KG baseline. Undated facts are mostly `genre`, `original_language` and
  `main_subject`.
- E. **Static reference variant.** A third KG, `static`, keeps every fact of the train items with no
  time filter, as in the KGAT paper; its KG at t1 equals its KG at t2. It is used only as a reference to
  measure the effect of leakage through the KG, is never reported as a time-consistent result, and is
  labelled so in `kg_meta.json`. The leakage check does not test its dates. All three variants share one
  entity vocabulary (the static KG's) so that their results can be compared directly.
- Rule applied at cut-off `t` (UTC calendar day of the timestamp): keep a fact iff its statement date,
  if any, is before that day **and** its target's earliest existence date, if any, is before that day.

## Consequences
- `kg_meta.json` records limitation A, the number of statement- and target-dated facts per relation, and
  the number of entities that appear only in the KG at t2 (not handled yet; decided in the KGAT phase).
- The share of undated facts per relation is reported next to the main results.
- Entities without edges in a given variant keep their id (shared vocabulary); the model must handle
  entities that have no KG edge.
