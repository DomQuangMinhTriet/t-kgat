"""Data-only constants and input gates for decision 0021 (still proposed).

The regression test compares these values with the decision text. No training
parameters, model selection or test scores belong in this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from tkgat.data.lastfm_raw import source_sha256, utc_seconds

WINDOW_START = utc_seconds("2014-03-18")
WINDOW_END = utc_seconds("2014-04-01")
Q_VAL = 0.8
Q_TEST = 0.9
KCORE = 10
MAPPING_TRACKS = 1_254_923
SNAPSHOT = "2015-04-16"
MAPPING_COPY = "https://drive.google.com/file/d/1SqoD1_3VgqjrsOo4TJfoR_AeKe_jHl6x/view"
ITEM_LINKS_DRIVE = "gdrive://1SqoD1_3VgqjrsOo4TJfoR_AeKe_jHl6x/lastfm-links-recbole/item_links.tsv"


def check_cut_manifest(meta: dict) -> None:
    expected = {
        "pair_universe": "raw_events", "timestamp_policy": "earliest_in_window",
        "start_inclusive": WINDOW_START, "end_exclusive": WINDOW_END, "min_events_user": 0,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"0021 requires {key}={value!r}; rebuild 01b from raw .inter, not a history cache")
    if not WINDOW_START <= meta["min_ts"] <= meta["max_ts"] < WINDOW_END:
        raise ValueError("Cut timestamps are outside the 0021 window")


def check_build_inputs(pairs, item_links, extraction, links: dict) -> dict:
    """Fail before creating outputs if a legacy source or rule was supplied."""
    cut = json.loads(Path(str(pairs) + ".meta.json").read_text(encoding="utf-8"))
    check_cut_manifest(cut)
    if len(links) != MAPPING_TRACKS:
        raise ValueError(f"0021 requires the new mapping of {MAPPING_TRACKS:,} tracks; got {len(links):,}")
    if extraction.get("source_snapshot") != SNAPSHOT:
        raise ValueError(f"0021 requires Freebase snapshot {SNAPSHOT}")
    if (extraction.get("seed_pairs_sha256") != cut["sha256"]
            or extraction.get("seed_scope") != "protocol-v2 train items"):
        raise ValueError("01c must be rebuilt with --pairs for this 14-day cut and train-only seeds")
    mapping_hash = source_sha256(item_links)
    if extraction.get("item_links_sha256") != mapping_hash:
        raise ValueError("01c extraction uses another mapping or lacks its SHA-256; rebuild with the new --item-links")
    fingerprint = extraction.get("source_fingerprint", {})
    if len(fingerprint.get("sha256", "")) != 64:
        raise ValueError("01c extraction is missing the source dump SHA-256")
    return {
        "decision": "0021-protocol-v2", "status": "proposed",
        "window": {"start_inclusive": WINDOW_START, "end_exclusive": WINDOW_END},
        "timestamp_policy": "earliest", "q_val": Q_VAL, "q_test": Q_TEST,
        "kcore": KCORE, "min_rating": 1, "target_interactions": 0,
        "kg_mode": "two-hop", "kg_entity_min": 1, "kg_relation_min": 1,
        "mapping_tracks": len(links), "item_links_sha256": mapping_hash,
        "dump_sha256": fingerprint["sha256"], "source_snapshot": SNAPSHOT,
    }
