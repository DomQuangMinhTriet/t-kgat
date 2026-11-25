"""Build LFM-1b pairs from raw events, independently of the published KGAT pairs."""

from __future__ import annotations

import calendar
import csv
import datetime as dt
import gzip
import io
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from tkgat.data.drive import open_text_source
from tkgat.registry import sha256_file


def local_source_path(source: str | Path) -> Path | None:
    """The backing local file (including ZIP/gzip); Drive URIs have no local path."""
    value = str(source)
    return None if value.startswith("gdrive") else Path(value.split("::", 1)[0]).resolve()


@contextmanager
def open_raw_text(source: str | Path):
    """Read local text/gzip, ``archive.zip::member``, or the existing Drive URIs."""
    value = str(source)
    if "::" in value and not value.startswith("gdrive"):
        archive, member = value.split("::", 1)
        with ZipFile(archive) as zipped, zipped.open(member) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8") as handle:
                yield handle
    elif value.endswith(".gz") and not value.startswith("gdrive"):
        with gzip.open(value, "rt", encoding="utf-8") as handle:
            yield handle
    else:
        with open_text_source(source) as handle:
            yield handle


def utc_seconds(value: str) -> int:
    """Parse an ISO date/time; a timezone-free input means UTC."""
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp())


def window_start(end: int, months: int) -> int:
    """Subtract calendar months, clamping the day to the destination month's end."""
    if months <= 0:
        raise ValueError("window_months must be positive")
    value = dt.datetime.fromtimestamp(end, dt.timezone.utc)
    year, month = divmod(value.year * 12 + value.month - 1 - months, 12)
    month += 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return int(value.replace(year=year, month=month, day=day).timestamp())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_pair_cache(path, source, *, start=None, end=None, window_months=None, min_events_user=0):
    """Reuse only a complete, unchanged cache of exactly the requested extraction."""
    path = Path(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    if window_months is not None:
        if start is not None or end is None:
            raise ValueError("window_months requires an explicit end and no start")
        start = window_start(end, window_months)
    expected = {"source": str(source), "start_inclusive": start, "end_exclusive": end,
                "window_months": window_months, "min_events_user": min_events_user,
                "pair_universe": "raw_events", "timestamp_policy": "earliest_in_window"}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing pair cache has other source/window rules; choose a new --out or rebuild explicitly")
    if sha256_file(path) != meta["sha256"]:
        raise ValueError("Existing pair cache hash does not match its manifest")
    return meta


def cut_pairs(
    source: str | Path,
    output: str | Path,
    *,
    start: int | None = None,
    end: int | None = None,
    window_months: int | None = None,
    min_events_user: int = 0,
    chunksize: int = 250_000,
    scratch_dir: str | Path | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    """Stream events into a disk-backed pair table, then write chronological earliest pairs.

    Bounds are [start, end). No user/item mapping, random sample, KG coverage filter,
    k-core, split or model score is consulted here. RAM is bounded by ``chunksize``;
    scratch storage grows with the number of distinct pairs in the selected window.
    """
    if chunksize <= 0 or min_events_user < 0:
        raise ValueError("chunksize must be positive and min_events_user nonnegative")
    if window_months is not None:
        if start is not None or end is None:
            raise ValueError("window_months requires an explicit end and no start")
        start = window_start(end, window_months)
    if start is not None and end is not None and start >= end:
        raise ValueError("need start < end")
    output = Path(output)
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if not overwrite and (output.exists() or meta_path.exists()):
        raise FileExistsError(f"Output exists: {output}; choose another path or use --overwrite")
    source_path = local_source_path(source)
    if source_path in {output.resolve(), meta_path.resolve(), output.with_name(output.name + ".partial").resolve()}:
        raise ValueError("source and output must differ (including archive and temporary output paths)")
    output.parent.mkdir(parents=True, exist_ok=True)
    if scratch_dir is not None:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    counts = {"rows_scanned": 0, "events_before_start": 0, "events_at_or_after_end": 0, "events_in_window": 0}
    source_min = source_max = None
    with tempfile.TemporaryDirectory(prefix="lfm1b-", dir=scratch_dir or output.parent) as temporary:
        with closing(sqlite3.connect(str(Path(temporary) / "pairs.sqlite"))) as db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("PRAGMA cache_size=-65536")
            db.execute("PRAGMA temp_store=FILE")
            db.execute(
                "CREATE TABLE pairs (user INTEGER, item INTEGER, ts INTEGER, events INTEGER, "
                "PRIMARY KEY(user, item)) WITHOUT ROWID"
            )
            with open_raw_text(source) as handle:
                header = handle.readline().strip().split("\t")
                names = [name.split(":", 1)[0] for name in header]
                # The supplied Drive archive calls the raw track column tracks_id.
                names = ["item_id" if name == "tracks_id" else name for name in names]
                if names not in (["user_id", "item_id", "timestamp"], ["user", "item", "ts"]):
                    raise ValueError(f"Expected LFM-1b user_id/(item_id or tracks_id)/timestamp header, got {header}")
                for chunk in pd.read_csv(
                    handle, sep="\t", names=["user", "item", "ts"], header=None,
                    dtype="int64", chunksize=chunksize, on_bad_lines="error", skip_blank_lines=False,
                ):
                    if not isinstance(chunk.index, pd.RangeIndex):
                        raise ValueError("Raw event rows must have exactly three columns")
                    if chunk.isna().any().any() or (chunk < 0).any().any():
                        raise ValueError("Raw events must contain nonnegative integer user/item/timestamp values")
                    counts["rows_scanned"] += len(chunk)
                    low, high = int(chunk.ts.min()), int(chunk.ts.max())
                    source_min = low if source_min is None else min(source_min, low)
                    source_max = high if source_max is None else max(source_max, high)
                    before = chunk.ts < start if start is not None else pd.Series(False, index=chunk.index)
                    after = chunk.ts >= end if end is not None else pd.Series(False, index=chunk.index)
                    counts["events_before_start"] += int(before.sum())
                    counts["events_at_or_after_end"] += int(after.sum())
                    selected = chunk[~before & ~after]
                    counts["events_in_window"] += len(selected)
                    grouped = selected.groupby(["user", "item"], sort=False).ts.agg(["min", "size"]).reset_index()
                    db.executemany(
                        "INSERT INTO pairs VALUES (?, ?, ?, ?) ON CONFLICT(user, item) DO UPDATE SET "
                        "ts=MIN(pairs.ts, excluded.ts), events=pairs.events+excluded.events",
                        grouped.itertuples(index=False, name=None),
                    )
                    db.commit()
                    if progress:
                        print(
                            f"scanned={counts['rows_scanned']:,} kept_events={counts['events_in_window']:,}",
                            flush=True,
                        )
            pairs_before_user_filter = db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
            if min_events_user:
                db.execute(
                    "DELETE FROM pairs WHERE user IN "
                    "(SELECT user FROM pairs GROUP BY user HAVING SUM(events) < ?)", (min_events_user,),
                )
            n_pairs, n_users, n_items, minimum, maximum, retained_events = db.execute(
                "SELECT COUNT(*), COUNT(DISTINCT user), COUNT(DISTINCT item), MIN(ts), MAX(ts), SUM(events) FROM pairs"
            ).fetchone()
            if not n_pairs:
                raise ValueError("No pairs remain; check the event bounds and min_events_user")
            accounting = {
                "before_start": counts["events_before_start"],
                "at_or_after_end": counts["events_at_or_after_end"],
                "below_user_event_threshold": counts["events_in_window"] - retained_events,
                "repeated_listens_collapsed": retained_events - n_pairs,
                "output_pairs": n_pairs,
            }
            if sum(accounting.values()) != counts["rows_scanned"]:
                raise RuntimeError("Raw event accounting does not balance")
            pending = Path(temporary) / "pairs.tsv"
            with pending.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["user", "item", "ts"])
                writer.writerows(db.execute("SELECT user, item, ts FROM pairs ORDER BY ts, user, item"))
            meta = {
                "source": str(source), "source_id_space": "raw LFM-1b user/track ids",
                "timestamp_policy": "earliest_in_window", "pair_universe": "raw_events",
                "start_inclusive": start, "end_exclusive": end, "window_months": window_months,
                "min_events_user": min_events_user, **counts,
                "accounting": accounting,
                "pairs_dropped_user_threshold": pairs_before_user_filter - n_pairs,
                "events_dropped_user_threshold": counts["events_in_window"] - retained_events,
                "pairs": n_pairs, "users": n_users, "items": n_items,
                "min_ts": minimum, "max_ts": maximum, "raw_min_ts": source_min, "raw_max_ts": source_max,
                "sha256": sha256_file(pending),
                "selection_note": "min_events_user > 0 selects users by their whole-window event counts",
            }
            # Copy before replace: scratch_dir can be on another drive.
            staged = output.with_name(output.name + ".partial")
            try:
                shutil.copyfile(pending, staged)
                os.replace(staged, output)
            finally:
                staged.unlink(missing_ok=True)
            write_json(meta_path, meta)
    return meta


def load_raw_pairs(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read a completed cut file and verify its recorded hash before splitting."""
    path = Path(path)
    meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text(encoding="utf-8"))
    if meta.get("pair_universe") != "raw_events" or meta.get("timestamp_policy") != "earliest_in_window":
        raise ValueError("Expected a raw-event earliest-pair cut manifest")
    if sha256_file(path) != meta["sha256"]:
        raise ValueError("Cut file SHA-256 does not match its manifest; rebuild the cut")
    pairs = pd.read_csv(path, sep="\t", dtype="int64")
    if list(pairs.columns) != ["user", "item", "ts"] or pairs.duplicated(["user", "item"]).any():
        raise ValueError("Expected unique raw user,item,ts pairs")
    if pairs.isna().any().any() or (pairs < 0).any().any() or len(pairs) != meta["pairs"]:
        raise ValueError("Cut rows do not match the manifest")
    pairs["rating"] = np.int64(1)
    return pairs[["user", "item", "rating", "ts"]], meta
