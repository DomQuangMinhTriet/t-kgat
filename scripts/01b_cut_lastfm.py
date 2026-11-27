"""Cut raw LFM-1b events and write earliest pairs, without any KGAT user/item filter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tkgat.data.drive import DriveQuotaError
from tkgat.data.lastfm_raw import cut_pairs, utc_seconds, validate_pair_cache
from tkgat.data.protocol_v2 import WINDOW_END, WINDOW_START

RAW_DRIVE = "gdrive-stream://1uJv74KwX3e0w7azRlfW80kfwbHGt1ZXh/lfm1b-tracks.inter"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", default=RAW_DRIVE, help="text, gzip, local ZIP::member, or Drive stream URI")
    parser.add_argument("--out", required=True, help="output TSV; a .meta.json is written beside it")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all-history", action="store_true", help="legacy exploration only; refused by protocol-v2 02b")
    mode.add_argument("--window-months", type=int, help="calendar months before an explicit --end")
    parser.add_argument("--start", type=utc_seconds, help="inclusive ISO date/time, UTC by default")
    parser.add_argument("--end", type=utc_seconds, help="exclusive ISO date/time; also records out-of-range events")
    parser.add_argument("--min-events-user", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--scratch-dir", help="disk space for SQLite and sorting")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--overwrite", action="store_true")
    reuse.add_argument("--reuse-existing", action="store_true", help="verify and reuse an identical pair cache")
    args = parser.parse_args(argv)
    if args.all_history and args.start is not None:
        parser.error("--all-history cannot be combined with --start")
    if not args.all_history and args.window_months is None:
        args.start = WINDOW_START if args.start is None else args.start
        args.end = WINDOW_END if args.end is None else args.end
    try:
        if args.reuse_existing and Path(args.out).is_file():
            print("[01b] Verifying existing pair cache; no Drive scan", flush=True)
            meta = validate_pair_cache(
                args.out, args.interactions, start=args.start, end=args.end, window_months=args.window_months,
                min_events_user=args.min_events_user,
            )
            print(json.dumps(meta, indent=2))
            return
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
