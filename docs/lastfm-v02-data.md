# Build Last-FM data for proposed protocol 0021

This is a data-only port of PR #49, commit
`b1ea496bed33786a448b6e38efc30a32adbf1884`. No merge, training code, refit code,
model configuration or old results are included. The source scripts and helpers
are `01b_cut_lastfm.py`, `01c_extract_freebase.py`, `02b_build_lastfm_from_raw.py`,
`data/{drive,lastfm_raw,lastfm_disk,lastfm_audit}.py`,
`kg/{freebase_dump,lastfm_freebase,lastfm_twohop}.py` and
`pipeline/lastfm_raw.py`. The exporter adds namespace options and split persistence
allows deterministic line endings. `data/protocol_v2.py`, `data/lastfm_check.py`
and script `16` implement the new protocol gates and pre-upload verification.

The protocol remains **proposed**. Running these scripts does not approve it.

## Inputs

- Raw `lfm1b-tracks.inter`, with raw numeric user and track IDs. A local text file,
  gzip, `archive.zip::member`, or the raw Drive ZIP stream can be read. The default
  is `data-last-fm.zip` (file ID `1uJv74KwX3e0w7azRlfW80kfwbHGt1ZXh`) in
  <https://drive.google.com/drive/folders/1NMkts7TVcFc9dDj1OsvsYXxK9Ms31Mfc>.
- New RecBole/RecSysDatasets mapping: **1,254,923 distinct linked tracks**.
  The shared ZIP `lastfm-links-recbole.zip` is
  <https://drive.google.com/file/d/1SqoD1_3VgqjrsOo4TJfoR_AeKe_jHl6x/view>.
  Both 01c and 02b default to reading its member
  `lastfm-links-recbole/item_links.tsv` directly from Drive. This is the new
  item mapping the user calls `item_list`; its actual source filename is
  `item_links.tsv`, with header `org_id\tfreebase_id`. The generated KGAT export
  still has its own `item_list.txt` with new remapped IDs.
  The exact upstream path is still pending Phon's confirmation; no path or hash
  is invented by the scripts. Mapping and dump SHA-256 are measured on the inputs.
  The mapping member hash verified on 2026-11-26 is
  `ec6783bfae5c17b1495fcd244eabc58cf69af2ac9067bbab352ef948bb46705f`.
  Code measures the current member bytes rather than hard-coding that hash, so
  replacing a source is detected when checking an existing extraction cache.
- Original Freebase RDF dump: Archive.org snapshot **2015-04-16**,
  <https://archive.org/download/freebase-rdf-latest/freebase-rdf-latest.gz>.
  01c defaults to this stream, validates the pinned compressed size/SHA-1 and
  records SHA-256. A local copy needs `--snapshot-date 2015-04-16` explicitly.
  This date describes the dump, not when individual facts became known.

Do not reuse the old 44,788-track mapping, whole-history earliest-pair cache,
old `freebase.tsv`, or old exported `item_list.txt`. Existing nonempty outputs are refused
unless `--overwrite` is explicitly passed. Use new output paths for a new source.

## Commands (from the repository root)

Install the project with the data dependencies:

```powershell
python -m pip install -e ".[drive]"
```

Cut raw events separately from splitting and KG construction:

```powershell
python scripts/01b_cut_lastfm.py --out data/raw/lastfm-v02-14d/pairs.tsv
```

These are the default bounds. Filtering raw events before taking a pair minimum
preserves repeated listens inside the window, even if that pair was first heard
before 18 March. The implementation accepts unsorted raw events and writes
chronological earliest pairs; the resulting pairs equal sorting raw events,
cutting `[start, end)` and then taking the earliest listen per pair. No whole-history
pair cache is used. Its manifest records all exclusions and the output SHA-256.

Extract the nine relations (including `music.recording.releases`) and date evidence:

```powershell
python scripts/01c_extract_freebase.py --pairs data/raw/lastfm-v02-14d/pairs.tsv --resume-dir data/raw/lastfm-v02-freebase/resume
```

01c makes **three full passes** through the dump: two hops and date/name evidence.
With `--resume-dir`, pass 1 saves compressed bytes while reading, and later passes
read those bytes locally. Without this option, it streams the dump again on each
pass. Alternatively, supply
an existing local dump with `--dump path/to/freebase-rdf-latest.gz
--snapshot-date 2015-04-16`. Run 01b first. 01c derives the train-only 10-core
from its cut on disk and **only surviving train item MIDs seed extraction**.
02b verifies this seed set again when constructing both cutoff snapshots.
`--reuse-existing` verifies source identity, output hash and the mapping hash
and pair-cut hash before avoiding a dump scan.

### Stop and continue 01c

Use the command above with `--resume-dir` **from the start of a new run**. Ctrl+C
stops the process while keeping the gzip byte cache and committed SQLite progress.
To continue, activate the same Python environment and **rerun exactly that command**.
No separate `--resume` flag is needed. A completed extraction can be verified with
`--reuse-existing` as before.

- `resume/dump.gz` stores the compressed source bytes as they arrive; the full
  official cache is 31,305,093,084 bytes. Reserve that space plus the extraction DB
  and final outputs. No full uncompressed dump is saved.
- `resume/download.json` records source URL, expected size and HTTP validator.
  Missing bytes are fetched by Range/If-Range; an ignored range or changed source
  is refused. Bytes already on disk are not downloaded again.
- `resume/facts.sqlite` commits facts, counters, physical line cursor and pass
  state together every 100,000 RDF rows (`--checkpoint-every` can change this).
  An interrupted uncommitted batch is rolled back and processed again.
- Restarting mid-pass replays/decompresses the cached gzip prefix locally to
  rebuild the decoder, skipping lines already committed. This takes some time;
  it does not seek instantly into compressed data. Completed passes are skipped.
- The frontier of an active pass is frozen in the checkpoint, so resuming cannot
  accidentally extend the two-hop neighborhood. Mapping, pair cut, seed set and
  extraction rules must match the saved checkpoint.
- The OS writer lock is released when the process exits. Only one process can
  write a resume directory at a time. Keep the whole directory until done.

The existing running legacy process cannot acquire this feature after startup;
it has no durable cursor/cache to import. Enabling this mode requires a new run.
Do not assume the progress of a run started without `--resume-dir` can be recovered.

Build the processed split and KGAT exports:

```powershell
python scripts/02b_build_lastfm_from_raw.py --pairs data/raw/lastfm-v02-14d/pairs.tsv --engine duckdb
```

Defaults are global quantiles 0.8/0.9, train-only iterative 10-core,
`min_rating=1`, no user sampling (`--target-interactions 0`), and no extra KG
frequency pruning. Phon confirmed on 2026-11-26 that the old PR's entity/relation
thresholds 10/50 are removed. All nine relation IDs remain present even if a relation has no
facts. Low KG coverage is reported and does not reject the build. The command
refuses changed split parameters, a different window, an old mapping size or a
different snapshot. The Python helpers also support synthetic/nonofficial data;
the CLI always enables the protocol-v2 gates. Override `--interactions` or
`--item-links` when replacing sources. Both 01c and 02b must receive the same
mapping URI. Local files and `archive.zip::member` remain supported for offline
use. The mapping SHA-256 always refers to member contents, not the ZIP container.

The large interaction ZIP is streamed without saving it locally; the small
mapping ZIP is cached in memory. Public Drive files are read directly. For a
private file, the standalone scripts need `TKGAT_DRIVE_ACCESS_TOKEN`; the ChatGPT
Drive connection is not automatically available to Python. No legacy support ZIP
or pre-extracted Freebase subset is used by the default pipeline.

`--engine duckdb --memory-limit 1GB --scratch-dir path/to/scratch` bounds the
interaction split's buffer usage. KG construction still uses Python/Pandas RAM.
`--engine auto` chooses DuckDB at two million pairs; `pandas` is useful on fixtures.

## KG snapshots and outputs

Both snapshots start at train item MIDs, traverse two undirected steps and preserve
original RDF directions. A fact's timestamp is the later available endpoint
existence proxy; facts without usable endpoint dates remain undated. Conservative
date precision follows the existing parser: a date must be strictly before the
cutoff's UTC calendar day. Traversal is repeated after the
date cutoff to prevent reaching an entity through a future fact.

- `kg_t1.tsv` / `kg_final.txt`: KG before t1, for selection.
- `kg_t2.tsv` / `kg_test.txt`: KG before t2, for refit/test.
- Timestamp sidecars preserve undated values as blanks; normalization and
  permutation are the trainer's responsibility and are outside this port.
- Interaction IDs come from train only; items without a MID or retained KG facts
  stay in the interaction splits. Validation/test cold-start exclusions are counted.
- Coverage includes two separate rates: `item_mid_link_rate` and
  `item_fact_rate` at each cutoff, plus validation/test retained/dropped counts.

01c logs cumulative triples, entities with edges and maximum degree after each
hop, plus elapsed time per dump pass and total extraction time. These measurements
are saved in `freebase.tsv.meta.json`. 02b records the train-seeded hop statistics
before time filtering and final graph statistics at each cutoff. Degree means
in-degree plus out-degree over triples; parallel relations contribute separately
and a self-loop contributes twice. Isolated items do not enter this degree count.
`kg_meta.json` and `dataset_meta.json` include `timing_seconds` and a structured
`date_semantics` object listing predicates, earliest-date selection per endpoint,
max across endpoints, missing-date handling and conservative date precision.
The total build timer ends before final metadata/hash-manifest finalization.

Processed files default to `data/processed/lastfm-v02-14d`; exports default to
`data/export/kgat/lastfm-v02-14d`. `meta.json` records hashes of all processed
artifacts and `dataset_meta.json` records hashes of every exported data file,
including train/val/test, `item_list.txt` and both KG snapshots. The original
dump and full mapping hashes are retained in provenance. These data folders
are ignored by Git.

Before uploading, independently rebuild the split checks from the cut source:

```powershell
python scripts/16_check_clean_dataset.py --report data/processed/lastfm-v02-14d-check.json
```

The report must be outside the frozen processed/export folders. Script 16 verifies
hashes, raw-source quantiles, core, masks, grouped exports, KG IDs, dates and snapshot
containment. Attach its JSON to the run registry metadata. No scores are computed.

Data-only regression checks (the test groups selected out below exercise the trainer):

```powershell
python -m pytest tests/test_protocol_v2_data.py tests/test_split.py tests/test_leakage.py tests/test_loaders_lastfm.py tests/test_kg_temporal.py tests/test_kg_lastfm.py tests/test_export_kgat.py -k "not run_py_resolves and not a_lastfm_run_needs"
ruff check .
```

The current branch's old training KG loader supports one-hop graphs. Triet's
separate trainer/loader port is required to consume the two-hop artifacts, as
specified in the restart plan. A 14-day **training data config** is also a separate
deliverable; this port only contains data-construction scripts and their checks.
