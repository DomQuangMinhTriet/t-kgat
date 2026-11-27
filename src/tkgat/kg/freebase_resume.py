"""Durable byte cache and transactional extraction checkpoints for Freebase."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".partial")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


@contextmanager
def extraction_workspace(parent, resume_dir=None):
    """Keep an opt-in workspace across stops, with an OS-released writer lock."""
    if resume_dir is None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="freebase-extract-", dir=parent) as directory:
            yield Path(directory)
        return
    directory = Path(resume_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "extract.lock").open("a+b") as handle:
        if not handle.seek(0, os.SEEK_END):
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            def lock():
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def unlock():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def lock():
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock():
                fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise ValueError("Another extraction is using this --resume-dir") from exc
        try:
            yield directory
        finally:
            unlock()


def cached_http_chunks(source, directory):
    """Replay cached compressed bytes, then append a validated HTTP Range tail.

    A fresh gzip decoder always starts at byte zero. No gzip decoder state is
    assumed to survive process exit. The SQLite row cursor skips parsed work.
    """
    from tkgat.kg.freebase_dump import DEFAULT_DUMP, http_chunks

    directory = Path(directory)
    path, meta_path = directory / "dump.gz", directory / "download.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "version": 1, "source": source, "total": None, "validator": None, "complete": False,
    }
    if meta.get("version") != 1 or meta.get("source") != source:
        raise ValueError("Dump cache belongs to another source; choose another --resume-dir")
    size = path.stat().st_size if path.exists() else 0
    if size and not meta_path.exists():
        raise ValueError("Nonempty dump cache lacks its source metadata; choose another --resume-dir")
    if meta["total"] is not None and size > meta["total"]:
        raise ValueError("Dump cache exceeds the source size")
    if meta["complete"] and size != meta["total"]:
        raise ValueError("Completed dump cache size changed")
    if size:
        print(f"[01c] Reading {size:,} cached compressed bytes from disk", flush=True)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                yield chunk
    if meta["complete"]:
        return
    if meta["total"] is not None and size == meta["total"]:
        meta["complete"] = True
        atomic_json(meta_path, meta)
        return
    if size and not meta["validator"] and source != DEFAULT_DUMP:
        raise ValueError("Custom HTTP source has no stable validator for byte-range resume")

    def received_headers(total, validator):
        meta.update(total=total, validator=validator)
        atomic_json(meta_path, meta)

    print(f"[01c] Downloading missing dump bytes from offset {size:,}", flush=True)
    last_log = size
    with path.open("ab", buffering=0) as handle:
        chunks = http_chunks(source, offset=size, total=meta["total"], validator=meta["validator"],
                             on_response=received_headers)
        try:
            for chunk in chunks:
                if handle.write(chunk) != len(chunk):
                    raise OSError("Short write while saving the dump cache")
                size += len(chunk)
                if size - last_log >= 64 * 1024 * 1024:
                    suffix = f"/{meta['total']:,} ({100 * size / meta['total']:.1f}%)" if meta["total"] else ""
                    print(f"[01c] Dump cached: {size:,}{suffix} compressed bytes", flush=True)
                    last_log = size
                yield chunk
        finally:
            chunks.close()
        os.fsync(handle.fileno())
    if meta["total"] is not None and size != meta["total"]:
        raise ValueError("Downloaded dump size does not match HTTP metadata")
    meta.update(total=size, complete=True)
    atomic_json(meta_path, meta)


class ExtractionCheckpoint:
    """Commit facts and their row cursor together; never grow an active frontier."""

    def __init__(self, db, spec):
        self.db = db
        db.execute("CREATE TABLE IF NOT EXISTS checkpoint(id INTEGER PRIMARY KEY,value TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS frontier(node TEXT PRIMARY KEY) WITHOUT ROWID")
        row = db.execute("SELECT value FROM checkpoint WHERE id=1").fetchone()
        if row:
            self.state = json.loads(row[0])
            if self.state["spec"] != spec:
                raise ValueError("Checkpoint inputs/rules changed; use the original inputs or a new --resume-dir")
        else:
            if db.execute("SELECT count(*) FROM edges").fetchone()[0]:
                raise ValueError("Extraction database has facts but no checkpoint; use another --resume-dir")
            self.state = {
                "spec": spec, "step": 0, "line": 0, "report": None, "reports": [],
                "first_fingerprint": None,
            }
            self.save()

    def save(self):
        self.db.execute("INSERT OR REPLACE INTO checkpoint VALUES(1,?)", (json.dumps(self.state),))
        self.db.commit()

    def begin(self, step, seeds):
        if self.state["step"] != step:
            self.db.execute("DELETE FROM frontier")
            self.db.executemany("INSERT OR IGNORE INTO frontier VALUES(?)", ((seed,) for seed in seeds))
            if step > 1:
                self.db.execute("INSERT OR IGNORE INTO frontier SELECT h FROM edges")
                self.db.execute("INSERT OR IGNORE INTO frontier SELECT t FROM edges")
            self.state.update(step=step, line=0, report=None)
            self.save()
        return {row[0] for row in self.db.execute("SELECT node FROM frontier")}
