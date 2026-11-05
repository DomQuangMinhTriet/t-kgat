# 0012 — Last-FM as a second dataset

- Status: proposed (2026-11-05, Triết). The dataset builder is Phôn's PR #41 ("lastfm: building script for tkgat").
  Builds on 0002 (time split), 0004 and 0006 (dating rules) and 0008 (subsets and KGAT export).
- Context: On 2026-11-01 the advisor said the MovieLens results point to a problem in the input data and asked the group
  to switch to Last-FM and repeat the MovieLens experiments. The KGAT release of Last-FM has no timestamps and no
  validation split. Timestamps come from LFM-1b (`lfm1b-tracks.inter`, cached as `kgat_author_earliest.tsv`). The
  decision to move to Last-FM was taken after the MovieLens test scores were seen (0009, 0010, 0011); it follows the
  advisor's direction and was not chosen to improve a score.
- Decision:
  1. **Dataset.** The user-item pairs of the KGAT Last-FM release (3,034,763 listening tokens, 1,542,856 unique pairs),
     with the earliest listen of each pair as its timestamp. Global time split at the 0.8 and 0.9 quantiles
     (t1 = 1398720396 = 2014-04-28 21:26:36 UTC, t2 = 1402083109 = 2014-06-06 19:31:49 UTC), 5-core on train only,
     `min_rating = 1` (implicit feedback), validation and test restricted to users and items seen in train.
     Result: 22,040 users, 44,788 items; train 1,220,988, valid 142,023, test 133,896;
     `interaction_leakage_problems: []`.
  2. **KG.** The author's KG (82,769 entities, 9 relations, 331,097 of 464,567 triples attached to train items) has no
     fact dates. Dates are taken from the Freebase RDF dump of 2015-08-09 (the KG itself comes from Freebase; a
     community mirror is used because Google's download no longer works, SHA-256 verified against the mirror's
     description): the existence date of the tail entity is the earliest of `music.release.release_date`,
     `music.artist.active_start` and `people.person.date_of_birth`, read as in 0004 and 0006 (a year means
     31 December, a `-01-01` date means 31 December, BC years are undated). 78.5% of the triples have a dated tail
     (releases 97.9%, artists 98.4%, producers 71.0%, engineers 25.5%; type, canonical-version, single-versions and
     song facts are undated). The Wikidata route was tried first and dated only about 21% of the triples. The KG is
     built with the default time-consistent policy used for MovieLens (undated facts kept); only 15 triples have a
     tail date on or after t1, so the default and the static variants are almost identical. The static variant stays a
     reference. T-KGAT Stage 2 (delta-t on KG triples) can be run with the dated triples.
     In the repo's one-hop KG, 1,293 of the 331,097 triples (canonical-version and single-version links, all undated)
     have a head that is not a train item and are dropped (0.4%); 329,789 triples over 84,585 entities remain in the
     default variant (an entity that shares its Freebase id with a train item gets its own id, as on MovieLens).
  3. **Protocol** as in 0002 and 0008: select on validation, test scored once, append-only registry, five seeds unless a
     later decision (0013) reduces them for compute reasons. Stage 1 (delta-t on user-item edges) with the permuted
     control. No `--resume` after any change of data or KG.
  4. **Loader and KG builder.** The generator writes the original KGAT folder format. A loader that produces
     `EncodedSplit` and a KG builder that reads the Freebase dates and writes the KG for the PyTorch pipeline are added
     in a separate PR before any training run.
- Consequences:
  - delta-t means "time since the first listen of the pair", not the time of an explicit rating as on MovieLens. Results
    are interpreted accordingly.
  - Validation spans about 39 days and test about 88 days (2014-06-06 to 2014-09-02), while train spans 2005-02-14 to
    2014-04-28. A model can benefit from recency in a way MovieLens did not allow; reported as a feature of the data.
  - The interaction cold-start filter drops 12,263 validation (7.9%) and 20,391 test (13.2%) interactions.
  - Time consistency of the KG is almost vacuous here (15 triples dropped at t1), so leakage through the KG is
    much less of a concern than on MovieLens. 21.5% of the triples have no date (mostly entity types, engineers), so
    Stage 2 has no time signal on them. Dates are release dates, not the time the fact entered Freebase, and the dump is
    a community mirror of Google's last release.
  - Pair selection: the KGAT pair set only contains pairs with at least one listen on or after 2014-04-01 (checked on a
    sample of 2,475 users: 99.97% of the pairs have their latest listen on or after that date, 99.95% even among pairs
    whose first listen is before April). Hence the jump in new pairs per day at 2014-04-01 (about 2,800 to 11,900) and the
    decay afterwards, and about 23% of the train pairs fall in the month before t1. Older pairs are in the data because
    they were listened to again later, so the pair universe is selected with information from after the cut-off, which may
    favour long-lived pairs. The raw LFM-1b events have no such jump. Reported as a limitation of the dataset.
  - The test set has 7,149 users (MovieLens: 51 and 95), so paired tests have much more power.
  - Note: the released KGAT files (train.txt + test.txt) hold 3,034,763 tokens, i.e. 33 fewer than the 3,034,796 interactions published for Last-FM. The upstream GitHub files give the same count and, after removing Windows line endings, the same SHA-256 as the local copy, so the gap is in the release itself, not in this pipeline.
  - Compute: size and time to be measured with `scripts/measure_kgat.py` on the local machine (RTX 3050, 4 GB) before the
    grid is run; a user subset and fewer seeds (0013) may be needed.
