"""Extract original Freebase two-hop facts; never read published KGAT triples."""

import argparse
import json

from tkgat.kg.freebase_dump import DEFAULT_DUMP, DEFAULT_EXTRACT, extract_dump
from tkgat.pipeline.lastfm_raw import ITEM_LINKS_DRIVE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default=DEFAULT_DUMP, help="HTTPS gzip stream, or an existing RDF/gzip file")
    parser.add_argument("--item-links", default=ITEM_LINKS_DRIVE)
    parser.add_argument("--out", default=DEFAULT_EXTRACT)
    parser.add_argument("--snapshot-date", help="date of a custom dump; default mirror metadata says 2015-04-16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = extract_dump(args.dump, args.item_links, args.out,
                          snapshot_date=args.snapshot_date, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
