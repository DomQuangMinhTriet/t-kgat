"""Check protocol-v2 artifacts before upload; write a report for registry metadata."""

import argparse
import json
from pathlib import Path

from tkgat.data.lastfm_check import check_clean_dataset
from tkgat.data.lastfm_raw import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed/lastfm-v02-14d")
    parser.add_argument("--export-dir", default="data/export/kgat/lastfm-v02-14d")
    parser.add_argument("--report", required=True, help="JSON report outside the frozen data folders")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--memory-limit", default="1GB")
    args = parser.parse_args(argv)
    report = Path(args.report).resolve()
    if any(report.is_relative_to(Path(folder).resolve()) for folder in (args.processed_dir, args.export_dir)):
        parser.error("--report must be outside the frozen processed/export directories")
    try:
        result = check_clean_dataset(args.processed_dir, args.export_dir,
                                     scratch_dir=args.scratch_dir, memory_limit=args.memory_limit)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Dataset check failed: {exc}\n")
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
