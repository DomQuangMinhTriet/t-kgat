"""Step 02b: split raw LFM-1b pairs and export KGAT files with timestamps."""

from __future__ import annotations

import argparse
import json

from tkgat.kg.freebase_dump import DEFAULT_EXTRACT, validate_extract
from tkgat.pipeline.lastfm_raw import ITEM_LINKS_DRIVE, build_raw_dataset

FREEBASE_DRIVE = "gdrive://1C4dqi4zzv8KCUwq2fucR9stHqYG-85R2/freebase/freebase-subset.tsv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="TSV and manifest produced by 01b_cut_lastfm.py")
    parser.add_argument("--out", default="data/processed/lastfm-raw")
    parser.add_argument("--export-dir", default="data/export/kgat/lastfm-raw")
    parser.add_argument("--freebase", help="verified 01c extraction; legacy one-hop defaults to Drive subset")
    parser.add_argument("--kg-mode", choices=["two-hop", "one-hop"], default="two-hop",
                        help="two-hop uses original Freebase dump facts; one-hop is the legacy Drive subset")
    parser.add_argument(
        "--item-links", default=ITEM_LINKS_DRIVE,
        help="track-to-Freebase mapping; default: mapping in the supplied Drive ZIP",
    )
    parser.add_argument("--q-val", type=float, default=0.8)
    parser.add_argument("--q-test", type=float, default=0.9)
    parser.add_argument("--kcore", type=int, default=5)
    parser.add_argument("--engine", choices=["auto", "pandas", "duckdb"], default="auto",
                        help="auto uses disk-backed processing for at least 2 million pairs")
    parser.add_argument("--memory-limit", default="1GB", help="DuckDB buffer budget; Python/KG use additional RAM")
    parser.add_argument("--scratch-dir", help="disk space for the temporary split database and external sorting")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.kcore < 1:
        parser.error("--kcore must be at least 1")
    source = args.freebase or (DEFAULT_EXTRACT if args.kg_mode == "two-hop" else FREEBASE_DRIVE)
    if args.kg_mode == "two-hop":
        try:
            validate_extract(source)
        except (OSError, ValueError) as exc:
            parser.error(f"Run python scripts/01c_extract_freebase.py first; invalid extraction: {exc}")
    meta = build_raw_dataset(
        args.pairs, args.out, args.export_dir, source, kg_mode=args.kg_mode,
        item_links=args.item_links, q_val=args.q_val,
        q_test=args.q_test, kcore=args.kcore, overwrite=args.overwrite,
        engine=args.engine, memory_limit=args.memory_limit, scratch_dir=args.scratch_dir,
    )
    print(json.dumps({k: meta[k] for k in ("n_users", "n_items", "interactions", "coverage")}, indent=2))
    print(f"Processed: {args.out}\nKGAT export: {args.export_dir}")
    print(f"Fact sources (before cutoff filtering): {meta['coverage']['fact_sources']}")


if __name__ == "__main__":
    main()
