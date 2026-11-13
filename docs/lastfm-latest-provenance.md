# `lastfm-u25-latest`: provenance and how to regenerate

`lastfm-u25-latest` is Phôn's dataset. It has the same users (seed 2019, 25%), the same t1/t2 quantiles (0.8 / 0.9), the same 5-core (train only)
and the same KG pipeline as `lastfm-u25`. The only change: the timestamp of a (user, item) pair is the **latest** listen, not the earliest
(`timestamp_policy: latest` in its `meta.json`). Decision 0015 (Phôn's draft) describes this; it is **not accepted** and **not in this repository**
(its own status line, in the export, reads "implementation requested by Phôn on 2026-11-10; not run on real data").

The data, the training code and the registry rows of `latest` are **not in this repository**. The repository holds only the summary tables
regenerated from Phôn's result zips, with the provenance below. No row is added to `results/registry.csv` (append-only; adding rows needs its own decision).

## Source of the numbers

| Item | Value |
|---|---|
| Export commit (separate, unrelated history) | `873ca7f` ("Freeze Kaggle source snapshot", 2026-11-10, 100 files; `source.bundle` of the payload) |
| Base of that export | upstream commit `c1763f9` plus modified and untracked files (`SOURCE_SNAPSHOT.json` in the export) |
| Difference with the head of PR #42 (`fffd2cf`) | 9 new and 12 modified files of the export (including `SOURCE_SNAPSHOT.json`, `.gitignore`, `.gitattributes`, `pyproject.toml` and the draft decision 0015); compared on 2026-11-13 |
| Split (`split_sha256`) | `9e095ba0acb2` (the start of the SHA-256 of `meta.json`) |
| Registry rows | 323 rows over the five zips, **132 distinct**, 0 `-dirty`, 0 conflicting values (`scripts/12_merge_latest_zips.py` stops on a conflict) |
| Devices | `cpu` (MostPop) and `Tesla T4` (all neural models) |

The five result zips (SHA-256 in `results/lastfm-latest-provenance.json`, written by `scripts/12_merge_latest_zips.py`):

| Zip | SHA-256 |
|---|---|
| `tkgat-lastfm-results.zip` | `241ec7d38ad36da18921bf81c64d26b10cf64d5f88892d403b8210c08af8632d` |
| `tkgat-lastfm-nokg-results.zip` | `f5430456aeb890c3f0c06230aacbfd554eae4e89e0422153596cd1f216f551dd` |
| `tkgat-lastfm-stage1-results.zip` | `d6ec4957239d2ecbdfd700ed8b62e73c66c5956a99f221ae95f31999c9d09938` |
| `tkgat-lastfm-stage1-control-results.zip` | `a03a20aa2d3d54701a4bee85e4424cddddfdc60cbdd5b5dd0ab1f0c8b87ae0d1` |
| `tkgat-lastfm-stage2-perm-results.zip` | `389ad585f0e16cc3be3d51547f8cc060f0bfad636f47f35e3081dd2c0c892518` |

The processed data (`data/processed/lastfm-u25-latest/`, not committed) comes from the Kaggle payload `tkgat-lastfm-latest-kaggle.zip` that Phôn sent
(10,858,625 bytes, SHA-256 `d96e5c0fb891e91c42041c8d35647ea60513f5a60843b267e173c12c05654d0e`). The payload only holds the `default` KG variant.
The 14 data files under `data/processed/lastfm-u25-latest/` of a local copy match the SHA-256 of `payload-manifest.json` of the payload (checked 2026-11-13).

## Regenerate (from the repository root)

```powershell
# 1. unpack the zips (without the *.pt model states) and merge their registries; the repo registry is not touched
python scripts/12_merge_latest_zips.py <folder with the five zips> <staging folder>

# 2. unpack data/processed/lastfm-u25-latest/ from the payload zip into the repository root

# 3. the RQ3 tables and the paired tests (the repo's own script, other input and output paths).
#    On Windows set PYTHONIOENCODING=utf-8 (or run python -X utf8): the script prints Vietnamese text.
python scripts/10_rq3_table.py --registry <staging>/results/registry.csv --runs-dir <staging>/results/runs `
  --datasets lastfm-u25-latest --out-md docs/rq3-table-lastfm-latest.md `
  --out-models results/rq3_models_lastfm-latest.csv --out-pairs results/rq3_pairs_lastfm-latest.csv

# 4. the numbers of the dataset
python scripts/11_numbers_latest_lastfm.py --registry <staging>/results/registry.csv

# 5. check
$env:LATEST_REGISTRY = "<staging>\results\registry.csv"; python -m pytest tests/test_latest_numbers.py
```

Regenerated on 2026-11-13 in the project `.venv`, from the five zips: `docs/data-report-lastfm-latest-numbers.md` is byte-identical;
`results/rq3_pairs_lastfm-latest.csv`, `results/lastfm-latest-provenance.json` and the text of `docs/rq3-table-lastfm-latest.md` are identical
(the committed files use LF; Windows writes CRLF; the table differs only in the figure path when `--figures-dir` is changed);
the numeric columns of `results/rq3_models_lastfm-latest.csv` differ by at most 9.0e-17 (floating-point noise, as for the other datasets);
the PNG `docs/figures/rq3_curves_lastfm-u25-latest.png` is not reproduced byte for byte (matplotlib output differs between environments).
The four tests of `tests/test_latest_numbers.py` pass with `LATEST_REGISTRY` set.

## What is not claimed

- These results are **preliminary** and cannot be reproduced from this repository alone (the code of the export is not merged).
- `docs/rq3-table-lastfm-latest.md` is written by the repo script, whose text says the source is `results/registry.csv`; for `latest` the source is the merged registry
  of the five zips above.
- MostPop recent was not run on `latest`; the BPR-MF versus MostPop test (`scripts/09_significance.py`) needs it, so it is not produced.
- The numbers file holds `val_kept_share` (the share of validation positives kept after the cold-user and cold-item filters). It is not the share
  "pairs of val/test first heard before the window" of the 2026-11-12 log (69.7% / 64.1%): that measurement was not reproduced and is not used anywhere.
- A later PR must bring Phôn's code (the 9 new and 12 modified files above) and decision 0015 under review, and decide how to append his rows to the registry
  (they carry the commit `873ca7f`, which is not in this history).
