# Raw Last-FM interactions with Freebase two-hop facts

Vietnamese commands/source table: [quick start](study/14-dung-lastfm-tu-lfm1b.md).
Current fact preparation is [decision 0018](decisions/0018-lastfm-freebase-two-hop.md),
explicitly requested by Phôn on 2026-11-22. It supersedes the new preparation's
Drive-only, outgoing-one-hop fact source. Historical official datasets remain separate.

## Sources and lineage

| Input | Role |
| --- | --- |
| `data-last-fm/lfm1b-tracks/lfm1b-tracks.inter` in [Drive copy](https://drive.google.com/file/d/14cMEh6U9zxu8dv8lfnwEmdI8ldT_Zvff/view) | Complete raw interactions; `tracks_id` is the raw item id |
| `kgat-output/last-fm-temporal/item_list.txt` in [support ZIP](https://drive.google.com/file/d/1C4dqi4zzv8KCUwq2fucR9stHqYG-85R2/view) | Explicit raw track-to-MID mapping; old remap ids ignored |
| [Freebase mirror](https://archive.org/details/freebase-rdf-latest) | Original RDF facts, names and date evidence; streamed gzip, not a paper KG |

The mapping has paper lineage and partial coverage. We do not reconstruct title
matching for all raw tracks because the supplied interactions contain ids, not
track titles. An unmapped track keeps its interactions if it meets split rules.
No published pairs, user map, KG triples, entity map or relation map are read.

The Google dump URL returned 403 during verification. The readable archive's
metadata date is **2015-04-16**, not 2015-08-09. This is a metadata date, not a
byte-verified claim of KGAT's dump version. Pin compressed size 31,305,093,084 and
SHA-1 `d1950f3c966932358a79a02cd1d56ae3165b26f4`; compute compressed SHA-256 across
all three passes. The extracted cache gets its own SHA-256. All source identity
and date information is embedded in output metadata.

## Workflow

01b has already completed on Phôn's machine. Preserve pairs.tsv and its manifest.

```powershell
conda activate python
python scripts/01c_extract_freebase.py
python scripts/02b_build_lastfm_from_raw.py --pairs data/raw/lastfm-raw/pairs.tsv --kg-mode two-hop --engine duckdb --memory-limit 1GB
```

For a fresh interaction cut only:

```powershell
python scripts/01b_cut_lastfm.py --all-history --out data/raw/lastfm-raw/pairs.tsv
```

01c accepts an alternative HTTPS/local RDF or gzip via `--dump`, an explicit
`--item-links`, and `--snapshot-date` for custom source metadata. Do not supply
a prebuilt paper KG as a dump. It caches `data/raw/lastfm-freebase2hop/freebase.tsv`
plus a manifest. 02b verifies this manifest and hash before importing interactions.
Changing the mapping MID set requires another extraction. 02b defaults to two-hop
and fails clearly if 01c has not completed; it never silently substitutes the old subset.

## Extraction and relation schema

The extraction follows [KB4Rec's neighborhood procedure](https://github.com/RUCDM/KB4Rec#how-to-get-freebase-subgraph-with-our-linkage):
pass 1 includes original edges touching any supplied MID at either endpoint;
pass 2 includes edges touching the frozen first-hop entity set; pass 3 collects
names/date evidence for the resulting nodes. A pass never expands its frontier
while scanning. Keep original RDF directions; model inverse edges are separate.
Only extracted edges/evidence and temporary SQLite data are written locally.
Each pass validates gzip CRC, source identity and row accounting. HTTP retries
resume compressed byte ranges without resetting the decoder; unsafe ranges fail.

The compressed source is about 31.3GB per scan, roughly 94GB total before retries.
This is a sustained network/CPU job, not a small API request. A failed process
needs to restart 01c; persistent extraction checkpoints are not provided.

The previously requested relation schema is fixed in original id order:

| ID | Predicate under `http://rdf.freebase.com/ns/` |
| --- | --- |
| 0 | `common.topic.notable_types` |
| 1 | `music.recording.releases` |
| 2 | `music.recording.artist` |
| 3 | `music.recording.engineer` |
| 4 | `music.recording.producer` |
| 5 | `music.recording.canonical_version` |
| 6 | `music.recording.song` |
| 7 | `music.single.versions` |
| 8 | `music.recording.featured_artists` |

This is an explicit restriction on traversal, not a claim that KGAT began with
a nine-predicate whitelist. Resource edges must have original Freebase endpoints.
Literal names/dates provide evidence; they do not add graph relation types.
A relation without surviving facts retains its schema id with zero triples.

## Temporal construction and quality filtering

The cache is a superset over supplied links. 02b re-seeds from train items after
iterative train-only 5-core. It applies the cutoff to edges before selecting
bidirectional <=2-hop neighborhoods, then rechecks reachability after quality
filtering so an excluded bridge cannot introduce disconnected facts.

[KGAT section 4.1](https://hexiangnan.github.io/papers/kdd19-KGAT.pdf) gives entity
occurrence >=10 and relation occurrence >=50, but does not fully specify the
filtering order/iteration. Our explicit interpretation counts distinct source
edges at both endpoints, applies one entity pass, then one relation pass. Learn
eligibility from the valid train-seeded t1 graph only and reuse it at t2. This
temporal adaptation prevents future facts from selecting entities or relations;
it does not promise the paper's exact graph size.

Use the earliest usable release date, artist active-start or birth date of each
endpoint, with year/month precision moved to period-end and the inherited
-01-01 heuristic. A fact's timestamp is the later available endpoint date.
This covers incoming edges with future heads. It is an existence proxy, not a
statement creation/validity date. Both endpoints without usable dates leave a
blank timestamp and retain the fact. Dated edges must precede the cutoff's UTC
day. The t1 KG must be contained in t2. At test, non-item nodes unseen at either
endpoint in t1 cannot enter the model graph.

## Interaction rules and scale

Interactions retain earliest raw-pair timestamps and global 0.8/0.9 quantiles.
Train is before t1, validation is [t1,t2), test is at/after t2. Core computation
uses train only; evaluation requires train users/items. Masks contain all source
pairs before the corresponding cutoff; exported masks use the train vocabulary.
Every split exclusion has a core/cold-user/cold-item reason. Missing KG links
are never interaction exclusions.

The CLI's `--engine auto` uses DuckDB for at least 2 million pairs. Imports use
250k-row batches, exports 50k-row batches, two DuckDB threads, external SQL sorts
and aggregation. `--memory-limit` is the DB buffer budget; item vocabularies/KG
metadata need additional RAM. `--scratch-dir` selects temporary disk storage.
The training loader still loads full interaction tables; export does not establish
that full-data training fits a laptop.

## Outputs, compatibility and verification

- Raw cache: `data/raw/lastfm-freebase2hop/freebase.tsv` and `.meta.json`.
- Processed: `data/processed/lastfm-raw/`, split/history CSVs and hashes,
  `validation_report.json`, `dropped_interactions.csv`, source reports, and
  `kg/default/` triples/dates/entities/relations/quality exclusions.
- Export: `data/export/kgat/lastfm-raw/`, KGAT layouts, aligned interaction and
  KG timestamps, `fact_sources.tsv`, `fact_date_evidence.tsv` and `dataset_meta.json`.

Metadata records graph scope, extraction source/date/hash, partial mapping lineage,
frequency policy and cutoff rules. Processed two-hop graphs use an explicit scope
flag allowing either endpoint in the shared entity namespace. Legacy one-hop
guards remain intact. Duplicate MID aliases keep separate raw-track interactions.
Data files use LF and deterministic ids/order; original inputs are protected.

Legacy inspection is available through `--kg-mode one-hop`, which defaults to
the original Drive subset. It is not the new full-dump workflow.

```powershell
python -m pytest tests/test_lastfm_twohop.py tests/test_lastfm_raw.py tests/test_drive_resume.py tests/test_ckg.py
```

Tests cover incoming/second-hop edges, exclusion of third-hop edges, early date
evidence, source reordering/hashes, truncated or changed dumps, compressed HTTP
resume, future heads, t1-only quality selection, aliases, loader/CKG compatibility,
and byte-identical pandas/DuckDB exports. Full dump extraction completed at 00:10
Asia/Saigon on 2026-11-23. The extracted SHA-256 matches its manifest, and all
three pass fingerprints/accounting were verified after the laptop lost power.
Full two-hop 02b completed at 09:25 Asia/Saigon on 2026-11-23. All 24 exported
file hashes were independently rechecked, and interaction accounting balances
with zero unexplained pairs. Data generation/export is complete; official model
training on this new dataset has not been run.
