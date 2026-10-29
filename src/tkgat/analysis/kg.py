"""Descriptive statistics for the temporal-KG notebook (``notebooks/03_kg_temporal.ipynb``).

The inputs are the outputs of ``scripts/06_build_kg.py`` (``kg_meta.json``, ``dropped.csv``, ``entities.csv``,
``kg_t*.tsv``) and the raw Wikidata facts; the notebook reads the files and passes frames or dicts in.
Nothing here re-implements the KG rules: dates are parsed with ``tkgat.kg.temporal``.
"""

from __future__ import annotations

import pandas as pd

from tkgat.kg.temporal import annotate

DATE_COLUMNS = ["point_in_time", "start_time", "tail_publication", "tail_inception", "tail_birth", "film_publication"]
PRECISION_NAMES = {"11": "day", "10": "month", "9": "year", "8": "decade"}
AWARD_RELATIONS = ("award_received", "nominated_for")
REASONS = ("future_statement", "future_tail", "undated")


def relation_date_kinds(meta: dict) -> pd.DataFrame:
    """Per relation: number of facts of the train items and the share with a statement date, only a target
    date, or no date (``kg_meta["date_kind_per_relation"]``), sorted by number of facts."""
    df = pd.DataFrame(meta["date_kind_per_relation"]).set_index("relation")
    return df.sort_values("facts", ascending=False)


def dropped_by_relation(meta: dict) -> pd.DataFrame:
    """Per relation and cut-off: kept facts and facts dropped for each reason (``kg_meta["per_relation"]``).

    Columns are ``<count>_<cutoff>``, e.g. ``kept_t1``, ``future_statement_t2``.
    """
    df = pd.DataFrame(meta["per_relation"])
    wide = df.pivot(index="relation", columns="cutoff", values=["kept", *REASONS])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.astype(int).sort_values("kept_t1", ascending=False)


def variant_triples(metas: dict[str, dict]) -> pd.DataFrame:
    """Triples at t1 and t2 for each KG variant, and whether the variant is time-consistent."""
    rows = {
        v: {"t1": m["n_triples"]["t1"], "t2": m["n_triples"]["t2"], "time_consistent": m["time_consistent"]}
        for v, m in metas.items()
    }
    return pd.DataFrame(rows).T.rename_axis("variant")


def date_precision(facts: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """How precise the Wikidata dates are.

    Returns (counts of ``point_in_time`` by precision name, and per date column the number of filled values
    and the share ending in ``-01-01``, which decision 0006 A reads as year-only when no precision is given).
    """
    f = facts.fillna("")
    pit = f.loc[f["point_in_time"] != "", "point_in_time_precision"].astype(str)
    by_precision = pit.map(lambda p: PRECISION_NAMES.get(p, f"other ({p})")).value_counts()
    rows = {}
    for col in DATE_COLUMNS:
        v = f.loc[f[col] != "", col]
        rows[col] = {"filled": len(v), "share_jan1": float(v.str.endswith("-01-01").mean()) if len(v) else 0.0}
    return by_precision, pd.DataFrame(rows).T


def item_fact_counts(triples: pd.DataFrame, entities: pd.DataFrame) -> pd.Series:
    """Number of KG facts per raw item id (heads of ``triples`` mapped through ``entities``)."""
    items = entities[entities["kind"] == "item"].set_index("entity")["key"].astype(int)
    return triples["h"].map(items).value_counts().rename("facts")


def top_items(train: pd.DataFrame, frac: float) -> set[int]:
    """The ``frac`` most popular train items (by number of train positives)."""
    counts = train["item"].value_counts()
    return set(counts.index[: max(1, int(round(frac * len(counts))))].astype(int))


def popular_concentration(
    fact_counts: pd.Series,
    dropped: pd.DataFrame,
    train: pd.DataFrame,
    frac: float = 0.1,
    relations: tuple[str, ...] = AWARD_RELATIONS,
) -> dict:
    """Do the facts removed by the time filter fall on popular films?

    Compares the share of *all* facts held by the ``frac`` most popular train items with the share of the
    future award/nomination facts (``reason == "future_statement"`` for ``relations``) dropped at one cut-off.
    ``dropped`` must hold one cut-off only.
    """
    if dropped["cutoff"].nunique() > 1:
        raise ValueError("pass the dropped facts of a single cut-off")
    top = top_items(train, frac)
    fut = dropped[dropped["relation"].isin(relations) & (dropped["reason"] == "future_statement")]
    fut_items = fut["item"].astype(int)
    return {
        "top_items": len(top),
        "share_of_all_facts": float(fact_counts[fact_counts.index.isin(top)].sum() / fact_counts.sum()),
        "future_award_facts": len(fut),
        "share_of_future_award_facts": float(fut_items.isin(top).mean()) if len(fut) else 0.0,
        "films_losing_award_facts": int(fut_items.nunique()),
    }


def film_fact_status(film_facts: pd.DataFrame, dropped: pd.DataFrame) -> pd.DataFrame:
    """Status of each fact of one film across the two cut-offs.

    ``film_facts`` are the raw Wikidata rows of the film; ``dropped`` is ``dropped.csv`` of one variant.
    Status: ``kept_t1`` (in the KG from t1), ``added_t2`` (absent at t1, present at t2) or ``dropped_both``.
    Also returns the parsed statement and target dates (decision 0006) and the reason at t1.
    """
    a = annotate(film_facts).drop_duplicates(["relation", "tail"])
    fb = set(a["freebase_id"])
    d = dropped[dropped["freebase_id"].isin(fb)]
    key = ["relation", "tail"]
    at = {c: d[d["cutoff"] == c].set_index(key)["reason"] for c in ("t1", "t2")}
    idx = pd.MultiIndex.from_frame(a[key])
    in1 = idx.isin(at["t1"].index)
    in2 = idx.isin(at["t2"].index)
    status = pd.Series("kept_t1", index=a.index)
    status[in1 & ~in2] = "added_t2"
    status[in1 & in2] = "dropped_both"
    status[~in1 & in2] = "inconsistent"  # never expected: the t1 KG is contained in the t2 KG
    out = a[["relation", "tail", "tail_label", "statement_date", "tail_date"]].copy()
    out["status"] = status.to_numpy()
    out["reason_t1"] = [at["t1"].get((r, t), "") for r, t in zip(out["relation"], out["tail"], strict=True)]
    return out.reset_index(drop=True)


def items_with_filtered_facts(
    static_triples: pd.DataFrame, filtered_triples: pd.DataFrame, entities: pd.DataFrame
) -> set[int]:
    """Raw ids of the items whose number of triples differs between the static KG and a time-filtered KG
    at the same cut-off, i.e. the items that lose at least one fact to the time filter."""
    items = entities[entities["kind"] == "item"].set_index("entity")["key"].astype(int)
    s = static_triples["h"].value_counts()
    f = filtered_triples["h"].value_counts().reindex(s.index, fill_value=0)
    return set(items[s.index[s != f]].astype(int))


def positive_share(positives: pd.DataFrame, items: set[int]) -> float:
    """Share of the positives (rows of ``val.csv`` / ``test.csv``) whose item is in ``items``."""
    return float(positives["item"].isin(items).mean()) if len(positives) else 0.0
