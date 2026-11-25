"""Extract original Freebase two-hop facts; never read published KGAT triples."""

import argparse
import hashlib
import json
from pathlib import Path

from tkgat.kg.freebase_dump import DEFAULT_DUMP, DEFAULT_EXTRACT, extract_dump, validate_extract
from tkgat.pipeline.lastfm_raw import ITEM_LINKS_DRIVE, read_item_links


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default=DEFAULT_DUMP, help="HTTPS gzip stream, or an existing RDF/gzip file")
    parser.add_argument("--item-links", default=ITEM_LINKS_DRIVE)
    parser.add_argument("--out", default=DEFAULT_EXTRACT)
    parser.add_argument("--snapshot-date", help="date of a custom dump; default mirror metadata says 2015-04-16")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--overwrite", action="store_true")
    reuse.add_argument("--reuse-existing", action="store_true", help="verify and reuse the completed two-hop cache")
    args = parser.parse_args(argv)
    if args.reuse_existing and Path(args.out).is_file():
        print("[01c] Verifying extraction cache; no full-dump passes", flush=True)
        result = validate_extract(args.out)
        if result["source"] != args.dump or (args.snapshot_date and result["source_snapshot"] != args.snapshot_date):
            parser.error("Existing extraction uses another dump/snapshot; choose a new --out or rebuild explicitly")
        seeds = set(read_item_links(args.item_links).values())
        if hashlib.sha256("\n".join(sorted(seeds)).encode()).hexdigest() != result["seed_mids_sha256"]:
            parser.error("Mapping MID set changed; rebuild 01c explicitly")
        print(json.dumps(result, indent=2))
        return
    result = extract_dump(args.dump, args.item_links, args.out,
                          snapshot_date=args.snapshot_date, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
