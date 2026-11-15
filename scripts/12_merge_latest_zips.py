"""Unpack Phon's result zips of ``lastfm-u25-latest`` into one staging folder and merge their registries.

    python scripts/12_merge_latest_zips.py <folder with the zips> <staging folder>

The five Kaggle result zips of ``lastfm-u25-latest`` (commit ``873ca7f`` of a separate export, not in this repository)
overlap: the same run files and registry rows appear in several of them. This script

  * unpacks every file except the model states (``*.pt``) into ``<staging>/results/``;
  * stops if a file name occurs in two zips with different bytes (it has not happened: 0 conflicts);
  * writes ``<staging>/results/registry.csv`` with the rows of all zips, each distinct row once, and
  * writes ``<staging>/PROVENANCE.json`` with the SHA-256 of every zip and the row counts.

The repository's own ``results/registry.csv`` is never read or written. Then run, from the repository root:

    python scripts/10_rq3_table.py --registry <staging>/results/registry.csv --runs-dir <staging>/results/runs \\
        --datasets lastfm-u25-latest --out-md docs/rq3-table-lastfm-latest.md \\
        --out-models results/rq3_models_lastfm-latest.csv --out-pairs results/rq3_pairs_lastfm-latest.csv
    python scripts/11_numbers_latest_lastfm.py --registry <staging>/results/registry.csv
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

# oldest first; the name of a zip says which runs it adds
ZIPS = (
    "tkgat-lastfm-results.zip",
    "tkgat-lastfm-nokg-results.zip",
    "tkgat-lastfm-stage1-results.zip",
    "tkgat-lastfm-stage1-control-results.zip",
    "tkgat-lastfm-stage2-perm-results.zip",
)
REGISTRY = "results/registry.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(source: Path, staging: Path) -> dict:
    frames, seen, info = [], {}, {}
    for name in ZIPS:
        path = source / name
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if entry.filename.endswith("/") or entry.filename.endswith(".pt"):
                    continue
                data = archive.read(entry)
                if entry.filename == REGISTRY:
                    frames.append(pd.read_csv(io.BytesIO(data)))
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if seen.setdefault(entry.filename, digest) != digest:
                    raise SystemExit(f"{entry.filename} differs between zips (seen again in {name})")
                target = staging / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        info[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "registry_rows": len(frames[-1])}
    merged = pd.concat(frames, ignore_index=True).drop_duplicates()
    (staging / "results").mkdir(parents=True, exist_ok=True)
    merged.to_csv(staging / REGISTRY, index=False)
    return {
        "zips": info,
        "registry_rows_all_zips": int(sum(len(f) for f in frames)),
        "registry_rows_distinct": len(merged),
        "git_commits": sorted(merged["git_commit"].astype(str).unique()),
        "datasets": sorted(merged["dataset"].astype(str).unique()),
        "dirty_rows": int(merged["git_commit"].astype(str).str.contains("dirty").sum()),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="folder with the five result zips")
    ap.add_argument("staging", help="empty folder to unpack into")
    args = ap.parse_args(argv)
    provenance = merge(Path(args.source), Path(args.staging))
    text = json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    Path(args.staging, "PROVENANCE.json").write_text(text, encoding="utf-8")
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
