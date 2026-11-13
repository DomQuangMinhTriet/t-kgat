# Rebuild Last-FM directly from Google Drive

Run from the repository root (Python 3.10+):

```powershell
python -m pip install -e ".[drive]"
python scripts/generate_lastfm_drive.py
```

The default config is `configs/data/lastfm_u25_latest_drive.yaml`: it streams raw
listening events from [data-last-fm.zip](https://drive.google.com/file/d/1tXtohvT8RAFmz2vJo4vH90kkrpwD75jW/view)
(about 7.4 GB compressed), the source linked in [lastfm-temporal-setup.md](lastfm-temporal-setup.md)
under [this Drive folder](https://drive.google.com/drive/folders/1NMkts7TVcFc9dDj1OsvsYXxK9Ms31Mfc).
It decompresses the `lfm1b-tracks.inter` member sequentially,
keeping the **maximum timestamp** of each published KGAT pair. Event order does
not matter. The raw ZIP and raw listening file are not saved to disk or loaded
whole into RAM. The small support ZIP (about 181 MB) is held in RAM for Freebase.
Allow additional memory for the pair dictionary, pandas and the graph. Internet
access is required; scanning the full raw history can take a long time.

The raw reader retries interrupted HTTP transfers up to five times and requests
the remaining compressed bytes with `Range`, keeping the ZIP decoder and scanned
pair timestamps alive in memory. It validates `206`/`Content-Range`, file size
and available file validators before continuing. If Drive ignores the requested
range or changes the file, it stops rather than mixing duplicate or changed bytes.
This resumes interruptions **within the same running process**; it is not a disk
checkpoint. If Python exits or the retry limit is reached, a new invocation still
starts from the beginning. No raw ZIP or listening file is saved locally.

The command verifies the support ZIP SHA-256, rebuilds the 25% user sample (seed
2019), builds the three KG variants and exports the default KG with timestamps.
The raw member's CRC is checked during streaming. Sampling, quantile cut-offs,
train-only 5-core and boundary checks are retained; timestamp selection changes
under decision 0015. The earliest reference hashes do not apply. It does not
train a model.

The author's KGAT files must already exist in
`knowledge_graph_attention_network/Data/last-fm/`. To use another copy, pass
`--author-data <directory>`. The raw listening history is required for latest
timestamps: the support ZIP's earliest-only cache cannot supply them. The raw
history is read directly from Drive. This rebuilds the existing pair universe,
including its April 2014 selection bias; it does not implement the proposed
window-free dataset. Last-event selection also loses earlier listening history;
generated masks do not represent every pre-cut-off listen (decision 0015).

Generated outputs are local:

- `data/processed/lastfm-u25-latest/`: `train.csv`, `val.csv`, `test.csv` with `ts`;
  histories, metadata, and KG variants under `kg/`.
- `data/export/kgat/lastfm-u25-latest-default-timestamp/`: KGAT text files, masks,
  `train_timestamp.txt`, `val_timestamp.txt`, `test_timestamp.txt`, and
  `interactions_chronological.tsv` with timestamps.
- KG dates: `kg/<variant>/kg_t1_dates.tsv` and `kg_t2_dates.tsv`. These contain
  inferred tail existence dates, not original statement timestamps. Undated
  facts have an empty `ts` field. See decision 0012 for the limitations.

Existing nonempty output directories are refused by default. Choose a different
`processed_dir` in a copied config and pass `--export-dir <new-directory>`, or
explicitly pass `--overwrite` to replace generated files. `--variant default`,
`drop_undated`, or `static` builds/exports only that variant; `all` (default)
builds all three and exports `default`.

## Drive access

Public links use [gdown](https://github.com/wkentaro/gdown) for the small support
ZIP and its confirmation parser for large raw downloads. Raw data uses a requests
stream with byte-range retries. Cookies stay in memory. The ZIP file must be readable by that
Python process. Access through the ChatGPT Drive connection does not give local
Python the same credentials.

For a private file, supply a short-lived OAuth access token through the environment
variable `TKGAT_DRIVE_ACCESS_TOKEN` using your existing authentication workflow.
The reader then uses the authenticated Google Drive
[`files.get?alt=media` API](https://developers.google.com/workspace/drive/api/guides/manage-downloads).
Do not put tokens in YAML, source code, command arguments, or git.

The Drive source notation is `gdrive://<ZIP-file-id>/<path-inside-ZIP>`.
For another support ZIP, copy the config and update `drive_zip_id`, `drive_zip_sha256`,
`raw_dir`, `freebase`, and `reference_meta`. A folder URL is not a file source.
Remove `reference_meta` only when the new source has no matching reference split.
The small in-memory ZIP reader has a 512 MiB compressed-size limit. Large raw
sources use `gdrive-stream://<ZIP-file-id>/<member-path-or-basename>` and bounded
buffering through [stream-unzip](https://stream-unzip.docs.trade.gov.uk/get-started/).

To reproduce the older earliest-timestamp dataset, explicitly use
`--data configs/data/lastfm_u25_drive.yaml`. Only that config uses the cached
earliest timestamps and compares the split hashes to the original reference.
The local generator `generate_lastfm_temporal_kgat.py` also defaults to latest;
use `--timestamp-policy earliest` for its original behavior.
