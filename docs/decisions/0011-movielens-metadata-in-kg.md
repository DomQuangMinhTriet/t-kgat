# 0011 — MovieLens metadata in the KG, and a re-run of every KG-dependent model

- Status: accepted (2026-10-29 by Triết). The KG change itself is Phôn's PR #34 ("Add movielens relations", merged
  2026-10-29). Builds on 0004 and 0006 (time-consistent KG, year-only dates = 31 Dec); changes the content of the KG
  variants of 0004, not their rules.
- Context: PR #34 adds three relations built from the MovieLens item catalogue to every KG variant (`default`,
  `drop_undated`, `static`): `movielens.genre`, `movielens.release_year`, `movielens.release_decade`. Each fact is dated at
  the film's release (exact date on ML-100K, year only on ML-1M) and passes through the same temporal guards as the
  Wikidata facts. The PR was merged without a decision record. The official KGAT and T-KGAT runs of 2026-10-28
  (`kgat`, `tkgat`, `tkgat_perm`, and the validation-only `tkgat_sens` of 0010) used the Wikidata-only KG, and their test
  scores have been seen. Code on `main` now builds a KG that differs from the one behind those registry rows.
- Decision:
  1. **The KG with MovieLens metadata is the KG of the project from now on**, for all three variants. Wikidata facts and
     their rules are unchanged.
  2. **Every model that reads the KG is re-run** on `ml-100k` and `ml-1m-u10` with the unchanged grids and seeds, from a
     clean commit: `kgat` (grid, 5 seeds, test once), then `tkgat` and `tkgat_perm` (their (lr, l2) come from the new `kgat`
     selection), then `tkgat_sens` (`--val-only`). `kgat_nokg`, MostPop, MostPop recent and BPR-MF read no KG and are not
     re-run.
  3. The registry is append-only. The new rows have the same key (dataset, split_sha256, model, config, seed, phase) as the
     old ones, so `registry.load()` returns the new ones; the old rows stay in the file. `--resume` must **not** be used for
     these re-runs, because it would treat the old-KG rows as already done.
  4. **The new results are the official ones whatever they show.** The Wikidata-only results are reported in an appendix as
     "Wikidata-only KG, superseded by 0011", with their commit, so a reader can see both.
- Consequences:
  - Risk to state in the report: the KG was changed after the test scores of the old KG were known. The change adds
    information that the KGAT paper and the advisor's direction of 2026-09-18 (item metadata) call for. It was not chosen to
    improve a score. Even so, it is a second look at test for `kgat`, `tkgat` and `tkgat_perm`. Point 4 is the mitigation.
  - Year-only release dates follow 0006 (31 Dec). On `ml-1m-u10` (cut-offs in December 2000), the 58 train films of the
    year 2000 have no MovieLens fact in either KG (99 genre, 58 year and 58 decade facts dropped at t1 and at t2). On
    `ml-100k` 4 facts are dropped. This is conservative and is reported as a limitation.
  - KG sizes change (for example `ml-100k` default at t1: 40,210 triples, of which 3,650 are MovieLens facts;
    `ml-1m-u10` default: 71,553, of which 6,209). Every document that quotes KG sizes is regenerated after the re-run:
    `scripts/07_report_numbers.py` and `docs/data-report.md`, `docs/design-kgat.md`, `notebooks/03_kg_temporal.ipynb`,
    `docs/rq3-table.md`, and the known facts in `CLAUDE.md`. The temporal analysis of the Wikidata facts (dated facts,
    Titanic) is unchanged.
  - About 5.5 hours of GPU time, based on the 2026-10-28 runs. The present of 2026-10-31 uses the new numbers if the re-run
    is finished and checked, and the old ones, labelled as such, otherwise.
