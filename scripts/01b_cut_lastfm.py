"""Cut raw LFM-1b events and write earliest pairs, without any KGAT user/item filter."""

from __future__ import annotations

import argparse
import json

from tkgat.data.drive import DriveQuotaError
from tkgat.data.lastfm_raw import cut_pairs, utc_seconds

RAW_DRIVE = "gdrive-stream://14cMEh6U9zxu8dv8lfnwEmdI8ldT_Zvff/lfm1b-tracks.inter"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", default=RAW_DRIVE, help="text, gzip, local ZIP::member, or Drive stream URI")
    parser.add_argument("--out", required=True, help="output TSV; a .meta.json is written beside it")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all-history", action="store_true", help="keep the full available history (default)")
    mode.add_argument("--window-months", type=int, help="calendar months before an explicit --end")
    parser.add_argument("--start", type=utc_seconds, help="inclusive ISO date/time, UTC by default")
    parser.add_argument("--end", type=utc_seconds, help="exclusive ISO date/time; also records out-of-range events")
    parser.add_argument("--min-events-user", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--scratch-dir", help="disk space for SQLite and sorting")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.all_history and args.start is not None:
        parser.error("--all-history cannot be combined with --start")
    try:
        meta = cut_pairs(
            args.interactions, args.out, start=args.start, end=args.end, window_months=args.window_months,
            min_events_user=args.min_events_user, chunksize=args.chunksize, scratch_dir=args.scratch_dir,
            overwrite=args.overwrite,
        )
    except DriveQuotaError as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
