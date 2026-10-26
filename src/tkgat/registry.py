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

import pandas as pd

FIELDS = [
    "run_at",
    "git_commit",
    "dataset",
    "split_sha256",
    "model",
    "config",
    "seed",
    "phase",
    "best_epoch",
    "recall@10",
    "recall@20",
    "ndcg@10",
    "ndcg@20",
    "n_users",
    "seconds",
    "device",
]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        # tracked files only, and nothing under results/: run outputs (registry.csv, histories) are
        # committed, so every run after the first would otherwise mark its own commit dirty
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ":(top,exclude)results"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - outside a git checkout
        return "no-git"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


HISTORY_FIELDS = ["epoch", "loss", "recall@10", "recall@20", "ndcg@10", "ndcg@20"]


def save_history(path: str | Path, history: list[dict], meta: dict) -> Path:
    """Write one training run's per-epoch history as CSV, one row per epoch.

    ``history`` is ``FitResult.history``: ``epoch`` and ``loss`` for every epoch, validation metrics
    only on the epochs where validation ran (empty elsewhere). ``meta`` (dataset, model, grid index,
    seed, config, device, ...) is repeated on every row so each file stands alone. Test scores must
    never be written here.
    """
    keys = {k for row in history for k in row} | set(meta)
    if any("test" in k for k in keys):
        raise ValueError("a training history must not contain test scores")
    extra = sorted({k for row in history for k in row} - set(HISTORY_FIELDS))
    meta = {k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in meta.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[*HISTORY_FIELDS, *extra, *meta])
        w.writeheader()
        for row in history:
            w.writerow({**row, **meta})
    return path


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
            raise ValueError(
                f"{path} has columns {header}, expected {FIELDS}; move the old file aside instead of mixing formats"
            )
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


RUN_KEY = ["dataset", "split_sha256", "model", "config", "seed", "phase"]


def load(path: str | Path = "results/registry.csv", latest: bool = True) -> pd.DataFrame:
    """Read the registry; with ``latest``, keep only the newest run of each
    (dataset, split_sha256, model, config, seed, phase).

    ``split_sha256`` is part of the key so that runs on two different splits (e.g. a rebuilt split, or a
    sensitivity split that shares a dataset name) are never treated as re-runs of each other.

    The file itself is append-only: a re-run adds rows and never edits old ones, so readers decide which run
    counts. "Newest" is the largest ``run_at`` (ties: the later line), not the line order, because a union
    merge of two branches may interleave their rows.
    """
    df = pd.read_csv(path, dtype={"git_commit": str, "config": str, "device": str})
    if latest:
        df = (
            df.reset_index(names="_line")
            .sort_values(["run_at", "_line"], kind="stable")
            .drop_duplicates(RUN_KEY, keep="last")
            .sort_values("_line")
            .drop(columns="_line")
            .reset_index(drop=True)
        )
    return df
