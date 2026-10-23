"""Append-only experiment registry (``results/registry.csv``).

Every number that ends up in a report must come from a row of this file.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

FIELDS = [
    "run_at", "git_commit", "dataset", "split_sha256", "model", "config", "seed", "phase",
    "best_epoch", "recall@10", "recall@20", "ndcg@10", "ndcg@20", "n_users", "seconds", "device",
]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - outside a git checkout
        return "no-git"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    row = {
        "run_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        **row,
    }
    if isinstance(row.get("config"), dict):
        row["config"] = json.dumps(row["config"], sort_keys=True)
    if not new:
        with path.open(encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != FIELDS:
            raise ValueError(f"{path} has columns {header}, expected {FIELDS}; "
                             "move the old file aside instead of mixing formats")
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
