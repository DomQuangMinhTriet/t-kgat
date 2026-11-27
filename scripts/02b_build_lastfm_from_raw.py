"""Step 02b: split raw LFM-1b pairs and export KGAT files with timestamps."""

from __future__ import annotations

import argparse
import json

from tkgat.data.protocol_v2 import ITEM_LINKS_DRIVE, KCORE, Q_TEST, Q_VAL
from tkgat.kg.freebase_dump import DEFAULT_EXTRACT, validate_extract
from tkgat.pipeline.lastfm_raw import build_raw_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="TSV and manifest produced by 01b_cut_lastfm.py")
    parser.add_argument("--out", default="data/processed/lastfm-v02-14d")
    parser.add_argument("--export-dir", default="data/export/kgat/lastfm-v02-14d")
    parser.add_argument("--freebase", default=DEFAULT_EXTRACT, help="verified 01c extraction from the 2015-04-16 dump")
    parser.add_argument(
        "--item-links", default=ITEM_LINKS_DRIVE,
        help="new mapping in the supplied Drive ZIP; use the same source as 01c",
    )
    parser.add_argument("--q-val", type=float, default=Q_VAL)
    parser.add_argument("--q-test", type=float, default=Q_TEST)
    parser.add_argument("--kcore", type=int, default=KCORE)
    parser.add_argument("--target-interactions", type=int, default=0,
                        help="0021 requires 0: keep all users, no sampling")
    parser.add_argument("--user-seed", type=int, default=2019)
    parser.add_argument("--compact-entities", action="store_true", help="omit unused KG entity IDs")
    parser.add_argument("--engine", choices=["auto", "pandas", "duckdb"], default="auto",
                        help="auto uses disk-backed processing for at least 2 million pairs")
    parser.add_argument("--memory-limit", default="1GB", help="DuckDB buffer budget; Python/KG use additional RAM")
    parser.add_argument("--scratch-dir", help="disk space for the temporary split database and external sorting")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.kcore < 1:
        parser.error("--kcore must be at least 1")
    if args.target_interactions < 0 or args.user_seed < 0:
        parser.error("--target-interactions and --user-seed must be nonnegative")
    source = args.freebase
    try:
        validate_extract(source)
    except (OSError, ValueError) as exc:
        parser.error(f"Run python scripts/01c_extract_freebase.py first; invalid extraction: {exc}")
    meta = build_raw_dataset(
        args.pairs, args.out, args.export_dir, source, kg_mode="two-hop", protocol_v2=True,
        item_links=args.item_links, q_val=args.q_val,
        q_test=args.q_test, kcore=args.kcore, overwrite=args.overwrite,
        engine=args.engine, memory_limit=args.memory_limit, scratch_dir=args.scratch_dir,
        target_interactions=args.target_interactions or None, user_seed=args.user_seed,
        compact_entities=args.compact_entities,
    )
    print(json.dumps({k: meta[k] for k in ("n_users", "n_items", "interactions", "coverage")}, indent=2))
    print(f"Processed: {args.out}\nKGAT export: {args.export_dir}")
    print(f"Fact sources (before cutoff filtering): {meta['coverage']['fact_sources']}")


if __name__ == "__main__":
    main()
