"""Build (and persist) a split from a data config."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from tkgat.data.clean import clean
from tkgat.data.leakage import check_split
from tkgat.data.loaders import load
from tkgat.data.split import SplitConfig, TemporalSplit, temporal_split
from tkgat.data.subsample import subsample_users
from tkgat.registry import sha256_file

PARTS = ("train", "val", "test", "seen_before_val", "seen_before_test")


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_split(cfg: dict) -> tuple[TemporalSplit, dict]:
    inter, items = load(cfg["name"], cfg["raw_dir"])
    clean_report = {}
    if cfg.get("clean"):
        inter, items, clean_report = clean(inter, items)
    if cfg.get("user_fraction") is not None:
        # a random user subset (decision 0008): taken before the split, which then runs unchanged on the subset
        inter, subsample_report = subsample_users(inter, float(cfg["user_fraction"]), int(cfg.get("user_seed", 0)))
        clean_report = {**clean_report, "user_subsample": subsample_report}
    s = temporal_split(inter, SplitConfig(**cfg["split"]))
    problems = check_split(s)
    if problems:
        raise RuntimeError("leakage check failed:\n- " + "\n- ".join(problems))
    return s, clean_report


def save_split(s: TemporalSplit, cfg: dict, clean_report: dict) -> Path:
    out = Path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        getattr(s, part).to_csv(out / f"{part}.csv", index=False)
    meta = {
        "dataset": cfg["name"],
        **s.summary(),
        "clean": clean_report,
        "sha256": {p: sha256_file(out / f"{p}.csv") for p in PARTS},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


TEST_PARTS = {"test": ["user", "item", "rating", "ts"], "seen_before_test": ["user", "item", "ts"]}


def load_saved(cfg: dict, with_test: bool = True) -> tuple[TemporalSplit, dict]:
    """Read a saved split. With ``with_test=False`` the test files are not opened and the test parts are empty
    (for runs that only train and select on validation, ``scripts/run.py --val-only``)."""
    out = Path(cfg["processed_dir"])
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    frames = {
        p: pd.DataFrame({c: pd.Series(dtype="int64") for c in TEST_PARTS[p]})
        if p in TEST_PARTS and not with_test
        else pd.read_csv(out / f"{p}.csv")
        for p in PARTS
    }
    s = TemporalSplit(SplitConfig(**meta["config"]), meta["t1"], meta["t2"], **frames)
    s.stats = {k: v for k, v in meta.items() if k not in {"config", "t1", "t2"}}
    return s, meta
