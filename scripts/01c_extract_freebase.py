"""Extract original Freebase two-hop facts; never read published KGAT triples."""

import argparse
import hashlib
import json
from pathlib import Path

from tkgat.data.lastfm_raw import source_sha256
from tkgat.data.protocol_v2 import ITEM_LINKS_DRIVE, MAPPING_TRACKS, SNAPSHOT
from tkgat.kg.freebase_dump import DEFAULT_DUMP, DEFAULT_EXTRACT, extract_dump, train_seed_mids, validate_extract
from tkgat.pipeline.lastfm_raw import read_item_links


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default=DEFAULT_DUMP, help="HTTPS gzip stream, or an existing RDF/gzip file")
    parser.add_argument("--pairs", required=True, help="14-day 01b cut; only items surviving train 10-core seed the KG")
    parser.add_argument("--item-links", default=ITEM_LINKS_DRIVE,
                        help="new mapping: reads the supplied Drive ZIP; a local file or ZIP::member also works")
    parser.add_argument("--out", default=DEFAULT_EXTRACT)
    parser.add_argument("--snapshot-date", choices=[SNAPSHOT], help="required for a local copy of the 2015-04-16 dump")
    parser.add_argument("--scratch-dir", help="temporary disk space for deriving train items")
    parser.add_argument("--memory-limit", default="1GB", help="DuckDB buffer budget for deriving train items")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--overwrite", action="store_true")
    reuse.add_argument("--reuse-existing", action="store_true", help="verify and reuse the completed two-hop cache")
    args = parser.parse_args(argv)
    links = read_item_links(args.item_links)
    if len(links) != MAPPING_TRACKS:
        parser.error(f"0021 requires the new {MAPPING_TRACKS:,}-track mapping; got {len(links):,}")
    if args.dump != DEFAULT_DUMP and args.snapshot_date is None:
        parser.error("A local/custom dump requires --snapshot-date 2015-04-16")
    if args.reuse_existing and Path(args.out).is_file():
        print("[01c] Verifying extraction cache; no full-dump passes", flush=True)
        result = validate_extract(args.out)
        if result["source"] != args.dump or (args.snapshot_date and result["source_snapshot"] != args.snapshot_date):
            parser.error("Existing extraction uses another dump/snapshot; choose a new --out or rebuild explicitly")
        seeds, pairs_hash = train_seed_mids(args.pairs, links, scratch_dir=args.scratch_dir,
                                            memory_limit=args.memory_limit)
        if result.get("seed_pairs_sha256") != pairs_hash or result.get("seed_scope") != "protocol-v2 train items":
            parser.error("Extraction uses another pair cut or seed scope; rebuild 01c explicitly")
        if source_sha256(args.item_links) != result.get("item_links_sha256"):
            parser.error("Mapping SHA-256 changed or is missing; rebuild 01c explicitly")
        if hashlib.sha256("\n".join(sorted(seeds)).encode()).hexdigest() != result["seed_mids_sha256"]:
            parser.error("Mapping MID set changed; rebuild 01c explicitly")
        print(json.dumps(result, indent=2))
        return
    del links
    result = extract_dump(args.dump, args.item_links, args.out, train_pairs=args.pairs,
                          snapshot_date=args.snapshot_date, overwrite=args.overwrite,
                          scratch_dir=args.scratch_dir, memory_limit=args.memory_limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
