# 0004 — Time-consistent knowledge graph

- Status: accepted (2026-10-17), including the default rule for undated facts
- Rule: a KG used with cut-off `t` keeps a fact only if it was true before `t`.
  - Dated fact (release date, award date, start date …): keep iff `date < t`.
  - Fact pointing to an entity that did not exist yet (a sequel, a remake, a later film of the same
    actor): drop.
  - Undated fact between entities that both existed before `t` (film → genre, film → director): keep by
    default; also run the variant "drop every undated fact" and report the share per relation.
- KG for train/val is built at `t1`, KG for test at `t2`.
- Data (fetched 2026-10-17, `data/raw/wikidata/facts.tsv`, report from `scripts/05_wikidata_report.py`):
  119,748 facts over 18 relations for 3,725 of the 3,811 linked films (the other 86 have no Wikidata item
  with that Freebase id). ML-100K: 1,582 of 1,617 linked items have facts.
- What the dates actually cover:
  - Statement dates (P585 point in time, P580 start time) exist almost only for `award_received` (80%) and
    `nominated_for` (97%). These are the relations that can leak the future (Titanic, Best Sound Editing,
    1998-03-23, lies between t1 = 1998-03-07 and t2 = 1998-03-31).
  - Other relations carry the target entity's own date (a person's birth, a company's inception, a
    sequel's publication). It tells whether the target existed before `t`, not when the link became true;
    `follows` / `followed_by` use it to drop sequels released after `t`.
  - Film release date (P577) is known for 99.97% of films; a film's own facts are valid from its release.
  - `genre`, `original_language`, `main_subject` are undated and fall under the default rule above.
  - Year-only dates (precision 9; 2,793 of the 6,295 point-in-time dates) are read as 31 December of that
    year; month-only as the last day of the month.
