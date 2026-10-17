"""Data-quality fixes applied before splitting.

Each fix is reported so that the data report can quote exact numbers.
"""

from __future__ import annotations

import pandas as pd


def merge_duplicate_titles(inter: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Map items that share an identical title to the smallest id.

    MovieLens-100K lists a handful of films twice under different ids.  If a
    user rated both copies, the later rating becomes a duplicate (user, item)
    pair and is removed by ``split.dedupe_pairs``.
    """
    canon = items.groupby("title")["item"].transform("min")
    remap = dict(zip(items["item"], canon))
    dup_items = items.loc[items["item"] != canon, ["item", "title"]]
    out = inter.copy()
    out["item"] = out["item"].map(remap).astype("int64")
    items_out = items[items["item"] == canon].reset_index(drop=True)
    report = {
        "duplicate_title_items": int(len(dup_items)),
        "duplicate_titles": sorted(dup_items["title"].unique().tolist()),
        "interactions_remapped": int((inter["item"] != out["item"]).sum()),
    }
    return out, items_out, report


def drop_untitled(inter: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Drop items whose title is missing or literally ``unknown``."""
    bad = items["title"].isna() | items["title"].str.strip().str.lower().eq("unknown")
    bad_ids = set(items.loc[bad, "item"])
    report = {
        "untitled_items": sorted(int(x) for x in bad_ids),
        "interactions_dropped": int(inter["item"].isin(bad_ids).sum()),
    }
    return inter[~inter["item"].isin(bad_ids)].reset_index(drop=True), items[~bad].reset_index(drop=True), report


def clean(inter: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    inter, items, r1 = drop_untitled(inter, items)
    inter, items, r2 = merge_duplicate_titles(inter, items)
    return inter, items, {"drop_untitled": r1, "merge_duplicate_titles": r2}
