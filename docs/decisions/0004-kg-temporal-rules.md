# 0004 — Time-consistent knowledge graph

- Status: accepted (2026-10-17), including the default rule for undated facts
- Rule: a KG used with cut-off `t` keeps a fact only if it was true before `t`.
  - Dated fact (release date, award date, start date …): keep iff `date < t`.
  - Fact pointing to an entity that did not exist yet (a sequel, a remake, a later film of the same
    actor): drop.
  - Undated fact between entities that both existed before `t` (film → genre, film → director): keep by
    default; also run the variant "drop every undated fact" and report the share per relation.
- KG for train/val is built at `t1`, KG for test at `t2`.
- Blocker: the KB4Rec Freebase subgraph is on Google Drive and Wikidata is not reachable from the build
  environment; facts must be fetched from a normal internet connection (see `scripts/04_fetch_wikidata.py`).
