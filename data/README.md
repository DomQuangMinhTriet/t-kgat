# Data

Nothing in `data/raw/` or `data/processed/` is committed. The MovieLens and KB4Rec licences do not
allow redistribution.

| Path under `data/raw/` | Source | Notes |
|---|---|---|
| `ml-100k.zip`, `ml-100k/` | <https://files.grouplens.org/datasets/movielens/ml-100k.zip> | MD5 `0e33842e24a9c977be4e0107933c0723`. 100,000 ratings, 943 users, 1,682 movies, 1997-09-20 → 1998-04-22 (UTC). **Do not use the bundled `u1`–`u5`, `ua`, `ub` splits: they are random.** |
| `ml-1m/` | <https://files.grouplens.org/datasets/movielens/ml-1m.zip> | 1,000,209 ratings, 6,040 users, 3,883 movies, 2000-04-25 → 2003-02-28 (UTC). Same movie id space as MovieLens-20M. |
| `kb4rec/ml2fb.txt`, `kb4rec/ml2yago.txt` | <https://github.com/RUCDM/KB4Rec/tree/master/Linkage> | MovieLens-20M id → Freebase mid / YAGO entity. Commit recorded in `kb4rec/SOURCE_COMMIT.txt`. Cite Zhao et al., *Data Intelligence* 2019. |
| `wikidata/facts.tsv` (+ `.gz`) | <https://query.wikidata.org/> (SPARQL, films found by Freebase id P646) | Fetched 2026-10-17: 119,748 facts, 18 relations, 3,725 films. SHA-256 `a81031563700d08318567da3867272470889644903fa2e726d930f69811da677`. Wikidata is CC0; it is still kept out of git because it is a dated snapshot. Re-create with `scripts/04_fetch_wikidata.py`; the counts will drift as Wikidata is edited. |

`data/checksums.json` holds the SHA-256 of every raw file; run `python scripts/01_verify_data.py`.

## How the files were obtained for the first run (2026-10-17)
GroupLens was not reachable from the build environment, so `ml-100k.zip` was taken from a public GitHub
copy and checked: its MD5 equals the official value above and `u.data` is byte-identical to a second,
independent copy. `ml-1m/` comes from the team's earlier download. Re-download from GroupLens when
possible and re-run `01_verify_data.py`; the checksums must not change.
