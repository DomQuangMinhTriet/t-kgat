"""Independent source-to-split accounting for the agreed raw Last-FM rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tkgat.data.leakage import check_split
from tkgat.data.split import TemporalSplit, kcore


def _same_rows(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> bool:
    def canonical(frame):
        return frame[columns].sort_values(columns, kind="mergesort").reset_index(drop=True).astype("int64")

    return canonical(left).equals(canonical(right))


def audit_raw_split(interactions: pd.DataFrame, split: TemporalSplit) -> tuple[dict, pd.DataFrame]:
    """Account for every raw pair, and check complete masks against the input.

    Dropped rows carry one exclusive reason, in priority order: train k-core,
    cold user, cold item. Missing Freebase/KG is never an interaction exclusion.
    """
    problems = check_split(split)
    if problems:
        raise ValueError(f"Split audit failed: {problems}")
    if interactions.duplicated(["user", "item"]).any() or not interactions.rating.eq(1).all():
        raise ValueError("Raw split audit requires unique implicit-positive pairs")
    expected_cutoffs = tuple(int(np.quantile(interactions.ts, q)) for q in (split.config.q_val, split.config.q_test))
    if (split.t1, split.t2) != expected_cutoffs:
        raise ValueError("Split cutoffs do not match the raw-pair quantiles")
    columns = ["user", "item", "rating", "ts"]
    expected_train = kcore(interactions[interactions.ts < split.t1], split.config.kcore)
    if not _same_rows(expected_train, split.train, columns):
        raise ValueError("Training rows do not match the complete train-only k-core")
    users, items = set(split.train.user), set(split.train.item)
    keys = ["user", "item", "ts"]
    frame = interactions.merge(
        split.train[keys].assign(in_train=True), on=keys, how="left", validate="one_to_one",
    )
    frame["split"] = np.select([frame.ts < split.t1, frame.ts < split.t2], ["train", "val"], default="test")
    frame["reason"] = np.select(
        [
            frame["split"].eq("train") & frame.in_train.isna(),
            ~frame["split"].eq("train") & ~frame.user.isin(users),
            ~frame["split"].eq("train") & frame.user.isin(users) & ~frame.item.isin(items),
        ],
        ["train_kcore", "cold_user", "cold_item"], default="",
    )
    for part in ("train", "val", "test"):
        expected = frame[frame["split"].eq(part) & frame.reason.eq("")]
        if not _same_rows(expected, getattr(split, part), columns):
            raise ValueError(f"Unexplained missing/changed/extra interactions in {part}")
    # A valid mask must be complete, in addition to containing no future rows.
    for name, cutoff in (("seen_before_val", split.t1), ("seen_before_test", split.t2)):
        expected = interactions.loc[interactions.ts < cutoff, keys]
        if not _same_rows(expected, getattr(split, name), keys):
            raise ValueError(f"Incomplete or changed {name} mask")
    dropped = frame.loc[frame.reason.ne(""), [*columns, "split", "reason"]].copy()
    dropped = dropped.sort_values(["ts", "user", "item"], kind="mergesort").reset_index(drop=True)
    reasons = {
        part: {reason: int((dropped["split"].eq(part) & dropped.reason.eq(reason)).sum())
               for reason in ("train_kcore", "cold_user", "cold_item")}
        for part in ("train", "val", "test")
    }
    kept = {part: len(getattr(split, part)) for part in ("train", "val", "test")}
    total = sum(kept.values()) + len(dropped)
    if total != len(interactions):
        raise ValueError("Input pair accounting does not balance")
    return {
        "input_pairs": len(interactions), "kept": kept, "dropped_by_reason": reasons,
        "dropped_pairs": len(dropped), "accounted_pairs": total, "unexplained_pairs": 0,
        "checks": {
            "global_quantiles": True, "split_boundaries": True, "train_only_kcore": True,
            "known_users_items": True, "complete_past_only_masks": True,
            "no_unexplained_interaction_loss": True,
        },
    }, dropped
