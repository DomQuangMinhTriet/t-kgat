"""Time-consistent knowledge graph from Wikidata facts (decisions 0004 and 0006).

A KG used at cut-off ``t`` keeps a fact only if it was true before the UTC day of ``t``:
its statement date (``point_in_time``, else ``start_time``), if any, and its target's earliest existence
date (publication, inception, birth), if any, must both fall before that day. The film's own release
date is never used to filter (decision 0006 B): every item in the KG was rated before t1.

Variants (decision 0006 D, E):
- ``default`` keeps undated facts (no usable statement date and no usable target date);
- ``drop_undated`` drops them;
- ``static`` keeps every fact of the train items with no time filter (as in the KGAT paper). It is a
  reference for measuring leakage through the KG only and is never time-consistent.
"""

from __future__ import annotations

import calendar
import datetime as dt
import re
from functools import cache

import numpy as np
import pandas as pd

VARIANTS = ("default", "drop_undated", "static")
MOVIELENS_RELATIONS = ("movielens.genre", "movielens.release_decade", "movielens.release_year")
STATIC_NOTE = (
    "static: every fact of the train items, no time filter, kg_t1 = kg_t2 (as in the KGAT paper). "
    "Reference only, to measure the effect of leakage through the KG; it contains future facts "
    "by design and must never be reported as a time-consistent result (decision 0006 E)."
)
TAIL_DATES = ["tail_publication", "tail_inception", "tail_birth"]
VALID_PRECISIONS = {8, 9, 10, 11}  # decade, year, month, day (decision 0006 C)
KNOWN_LIMITATIONS = [
    "Wikidata is a 2026-10-17 snapshot, not the knowledge available in 1997-2003.",
    "Dates without precision (all except point_in_time) that end in -01-01 are read as year-only "
    "(31 December); a fact really dated 1 January is placed too late (decision 0006 A).",
    "point_in_time precision outside 8-11 and unparseable dates (e.g. BC years) count as no date.",
    "Tail entities are Wikidata ids; a tail that is itself a linked film (sequel) is a separate entity "
    "from that film's item, because facts.tsv has no Wikidata id for the head films.",
]


@cache
def end_of_period(value: str, precision: int | None = None) -> dt.date | None:
    """Last day a Wikidata date can refer to, or None when there is no usable date.

    ``precision`` is Wikidata's (11 day, 10 month, 9 year, 8 decade). Without one, ``YYYY-01-01`` is
    read as year-only and any other value as a day.
    """
    if not value:
        return None
    try:
        d = dt.date.fromisoformat(value[:10])
    except ValueError:
        return None
    if precision is None:
        precision = 9 if (d.month, d.day) == (1, 1) else 11
    if precision not in VALID_PRECISIONS:
        return None
    if precision == 11:
        return d
    if precision == 10:
        return d.replace(day=calendar.monthrange(d.year, d.month)[1])
    if precision == 9:
        return dt.date(d.year, 12, 31)
    return dt.date(min(d.year // 10 * 10 + 9, 9999), 12, 31)


def _precision(v: str) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def cutoff_day(ts: int) -> dt.date:
    """UTC calendar day of a Unix timestamp; a fact is kept iff its dates are strictly before it."""
    return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).date()


def annotate(facts: pd.DataFrame) -> pd.DataFrame:
    """Add ``statement_date`` and ``tail_date`` (``datetime.date`` or None) to the raw facts."""
    f = facts.fillna("").copy()
    pit = [
        end_of_period(v, _precision(p) if p != "" else -1) if v else None
        for v, p in zip(f["point_in_time"], f["point_in_time_precision"], strict=True)
    ]
    start = [end_of_period(v) for v in f["start_time"]]
    f["statement_date"] = [a if a is not None else b for a, b in zip(pit, start, strict=True)]
    tails = [[end_of_period(v) for v in f[c]] for c in TAIL_DATES]
    f["tail_date"] = [min((d for d in ds if d is not None), default=None) for ds in zip(*tails, strict=True)]
    return f


def is_undated(f: pd.DataFrame) -> pd.Series:
    """No usable date at all: neither a statement date nor a target existence date (decision 0006 D)."""
    return f["statement_date"].isna() & f["tail_date"].isna()


def drop_reason(f: pd.DataFrame, day: dt.date, variant: str) -> pd.Series:
    """Why each annotated fact is dropped at ``day`` ('' = kept). First matching reason wins."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if variant == "static":
        return pd.Series("", index=f.index)
    st = f["statement_date"].map(lambda d: d is not None and d >= day)
    tail = f["tail_date"].map(lambda d: d is not None and d >= day)
    undated = is_undated(f) if variant == "drop_undated" else pd.Series(False, index=f.index)
    reason = np.select(
        [st.to_numpy(bool), tail.to_numpy(bool), undated.to_numpy(bool)],
        ["future_statement", "future_tail", "undated"],
        default="",
    )
    return pd.Series(reason, index=f.index)


def item_entities(item_ids: np.ndarray, links: pd.DataFrame, item_col: str) -> pd.DataFrame:
    """Train items in encoding order: columns ``h`` (= item index), ``item`` (raw id), ``freebase_id``.

    ``item_ids`` must be ``EncodedSplit.item_ids`` so that entity ``i < n_items`` is item index ``i``.
    """
    lk = links.dropna(subset=[item_col, "freebase_id"])
    fb = pd.Series(lk["freebase_id"].to_numpy(), index=lk[item_col].astype(int).to_numpy())
    items = pd.DataFrame({"h": np.arange(len(item_ids)), "item": np.asarray(item_ids).astype(int)})
    items["freebase_id"] = items["item"].map(fb)
    return items


def _metadata_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def movielens_facts(item_ids: np.ndarray, catalog: pd.DataFrame) -> pd.DataFrame:
    """MovieLens metadata triples for the encoded train items.

    The relations are namespaced so they cannot be confused with Wikidata relations. Genre, release-year and
    release-decade facts are dated at the movie release (exact date when available, otherwise year precision), so
    they pass through the same temporal guards as Wikidata facts. Unknown/no-genre labels are not emitted.
    """
    if catalog["item"].duplicated().any():
        raise ValueError("MovieLens catalog contains duplicate item ids")
    by_item = catalog.set_index("item")
    missing = set(np.asarray(item_ids).astype(int)) - set(by_item.index.astype(int))
    if missing:
        raise ValueError(f"{len(missing)} train items are missing from the MovieLens catalog")

    rows = []
    for h, raw_item in enumerate(np.asarray(item_ids).astype(int)):
        movie = by_item.loc[raw_item]
        release = movie.get("release_date")
        year_value = movie.get("year")
        if pd.notna(release):
            point_in_time = pd.Timestamp(release).strftime("%Y-%m-%d")
            precision = "11"
            year = int(pd.Timestamp(release).year)
        elif pd.notna(year_value):
            year = int(year_value)
            point_in_time = f"{year:04d}-01-01"
            precision = "9"
        else:
            year = None
            point_in_time = precision = ""

        common = {
            "h": h,
            "item": raw_item,
            "freebase_id": "",
            "point_in_time": point_in_time,
            "point_in_time_precision": precision,
            "start_time": "",
            "tail_publication": "",
            "tail_inception": "",
            "tail_birth": "",
            "film_publication": point_in_time,
            "tail_kind": "movielens_metadata",
        }
        genres = [g.strip() for g in str(movie.get("genres", "")).split("|") if g.strip()]
        for genre in genres:
            if genre.lower() in {"unknown", "(no genres listed)"}:
                continue
            rows.append(
                {
                    **common,
                    "relation": "movielens.genre",
                    "tail": f"movielens.genre.{_metadata_key(genre)}",
                    "tail_label": genre,
                }
            )
        if year is not None:
            decade = year // 10 * 10
            rows.extend(
                [
                    {
                        **common,
                        "relation": "movielens.release_year",
                        "tail": f"movielens.year.{year}",
                        "tail_label": str(year),
                    },
                    {
                        **common,
                        "relation": "movielens.release_decade",
                        "tail": f"movielens.decade.{decade}",
                        "tail_label": f"{decade}s",
                    },
                ]
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "h",
                "item",
                "freebase_id",
                "relation",
                "tail",
                "tail_label",
                "point_in_time",
                "point_in_time_precision",
                "start_time",
                "tail_publication",
                "tail_inception",
                "tail_birth",
                "film_publication",
                "tail_kind",
                "statement_date",
                "tail_date",
            ]
        )
    return annotate(pd.DataFrame(rows))


def cutoff_facts(facts: pd.DataFrame, cutoff_ts: int, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split already linked/annotated facts with ``h`` into kept and dropped sets at a cut-off."""
    f = facts.drop_duplicates(["h", "relation", "tail"]).reset_index(drop=True)
    reason = drop_reason(f, cutoff_day(cutoff_ts), variant)
    kept = f[reason == ""].reset_index(drop=True)
    dropped = f[reason != ""].assign(reason=reason[reason != ""]).reset_index(drop=True)
    return kept, dropped


def build_kg(
    facts: pd.DataFrame, items: pd.DataFrame, cutoff_ts: int, variant: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Facts of the train items, split into (kept, dropped) at the cut-off.

    ``facts`` must come from :func:`annotate`; ``items`` from :func:`item_entities`.
    Both frames carry ``h`` (item index), the raw fact columns and, for dropped facts, ``reason``.
    """
    linked = items.dropna(subset=["freebase_id"])[["h", "item", "freebase_id"]]
    f = linked.merge(facts, on="freebase_id", how="inner")
    return cutoff_facts(f, cutoff_ts, variant)


def check_kg(kept: pd.DataFrame, cutoff_ts: int, variant: str, n_items: int) -> list[str]:
    """Independent check of a built KG; returns the problems found (empty list = OK).

    Dates are re-parsed from the raw columns so a bug in the annotated columns cannot hide a leak.
    The ``static`` variant is not time-consistent by design, so its dates are not checked.
    """
    problems: list[str] = []
    if variant != "static":
        day = cutoff_day(cutoff_ts)
        a = annotate(kept.drop(columns=["statement_date", "tail_date"], errors="ignore"))
        future_st = a["statement_date"].map(lambda d: d is not None and d >= day)
        future_tail = a["tail_date"].map(lambda d: d is not None and d >= day)
        if future_st.any():
            problems.append(f"{int(future_st.sum())} facts with a statement date on/after {day}")
        if future_tail.any():
            problems.append(f"{int(future_tail.sum())} facts whose target exists only on/after {day}")
        undated = is_undated(a)
        if variant == "drop_undated" and undated.any():
            problems.append(f"{int(undated.sum())} undated facts in the drop_undated KG")
    h = kept["h"].to_numpy()
    if len(h) and (h.min() < 0 or h.max() >= n_items):
        problems.append("head entity outside the train item range")
    if kept.duplicated(["h", "relation", "tail"]).any():
        problems.append("duplicate triples")
    return problems


def assign_ids(
    n_items: int,
    items: pd.DataFrame,
    kgs: dict[str, pd.DataFrame],
    relations: list[str],
    universe: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """One entity and relation vocabulary shared by the given KGs.

    Entities ``0..n_items-1`` are the train items (encoding order); tail entities follow, sorted by
    Wikidata id. Tails are taken from ``universe`` (columns ``tail``, ``tail_label``) when given, e.g. the
    static KG so that every variant shares the same ids; otherwise from the KGs themselves.
    Returns (entities, relations, {name: integer triples h, r, t}).
    """
    sources = [universe] if universe is not None else list(kgs.values())
    tails = sorted(set().union(*(set(k["tail"]) for k in sources)))
    labels: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for k in sources:
        labels.update(dict(zip(k["tail"], k["tail_label"], strict=True)))
        if "tail_kind" in k:
            kinds.update(
                {tail: kind for tail, kind in zip(k["tail"], k["tail_kind"], strict=True) if pd.notna(kind) and kind}
            )
    missing = set().union(*(set(k["tail"]) for k in kgs.values())) - set(tails)
    if missing:
        raise ValueError(f"{len(missing)} tails of the KGs are not in the universe")
    ent = pd.concat(
        [
            pd.DataFrame(
                {
                    "entity": np.arange(n_items),
                    "kind": "item",
                    "key": items["item"].astype(str),
                    "label": items["freebase_id"].fillna(""),
                }
            ),
            pd.DataFrame(
                {
                    "entity": np.arange(n_items, n_items + len(tails)),
                    "kind": [kinds.get(t, "wikidata") for t in tails],
                    "key": tails,
                    "label": [labels.get(t, "") for t in tails],
                }
            ),
        ],
        ignore_index=True,
    )
    rel = pd.DataFrame({"relation": np.arange(len(relations)), "name": relations})
    tmap = dict(zip(tails, range(n_items, n_items + len(tails)), strict=True))
    rmap = dict(zip(relations, range(len(relations)), strict=True))
    out = {}
    for name, k in kgs.items():
        tri = pd.DataFrame(
            {
                "h": k["h"].astype(np.int64),
                "r": k["relation"].map(rmap).astype(np.int64),
                "t": k["tail"].map(tmap).astype(np.int64),
            }
        )
        out[name] = tri.sort_values(["h", "r", "t"]).reset_index(drop=True)
    return ent, rel, out


def fact_valid_from(facts: pd.DataFrame) -> pd.Series:
    """Unix seconds (UTC midnight) of the first day on which an annotated fact holds under 0004/0006.

    That is the later of its statement date and its target's existence date, both read as ``end_of_period`` (a
    year-only date is 31 Dec). NaN for an undated fact. Used by T-KGAT Stage 2 (decision 0009, item 8) as the
    timestamp of a KG edge.
    """

    def seconds(*days) -> float:
        known = [d for d in days if d is not None and not pd.isna(d)]
        if not known:
            return np.nan
        return dt.datetime.combine(max(known), dt.time(), tzinfo=dt.timezone.utc).timestamp()

    return pd.Series(
        [seconds(a, b) for a, b in zip(facts["statement_date"], facts["tail_date"], strict=True)],
        index=facts.index,
        dtype="float64",
    )


def triple_dates(kept: pd.DataFrame, entities: pd.DataFrame, relations: pd.DataFrame) -> pd.DataFrame:
    """``h, r, t, ts`` of the facts of one KG in the ids of :func:`assign_ids`, in the row order of ``kg_<cut>.tsv``."""
    rmap = dict(zip(relations["name"], relations["relation"], strict=True))
    tails = entities[entities["kind"] != "item"]
    tmap = dict(zip(tails["key"], tails["entity"], strict=True))
    out = pd.DataFrame(
        {
            "h": kept["h"].astype(np.int64).to_numpy(),
            "r": kept["relation"].map(rmap).astype(np.int64).to_numpy(),
            "t": kept["tail"].map(tmap).astype(np.int64).to_numpy(),
            "ts": fact_valid_from(kept).to_numpy(),
        }
    )
    return out.sort_values(["h", "r", "t"]).reset_index(drop=True)


def relation_stats(f: pd.DataFrame) -> pd.DataFrame:
    """Share of facts per relation with a statement date, only a target date, or no date at all."""
    g = f.assign(
        statement_dated=f["statement_date"].notna(),
        tail_only=f["statement_date"].isna() & f["tail_date"].notna(),
        no_date=f["statement_date"].isna() & f["tail_date"].isna(),
    )
    return (
        g.groupby("relation")
        .agg(
            facts=("tail", "size"),
            statement_dated=("statement_dated", "mean"),
            tail_date_only=("tail_only", "mean"),
            no_date=("no_date", "mean"),
        )
        .round(4)
    )
