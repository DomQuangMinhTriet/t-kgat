"""Time-consistent knowledge graph for Last-FM, under the rules of decisions 0004 and 0006.

The protocol is the one of :mod:`tkgat.kg.temporal`; only the sources differ, so the temporal rule itself
(:func:`tkgat.kg.temporal.cutoff_facts` / ``drop_reason``), the entity vocabulary (``assign_ids``) and the
Stage 2 dates (``triple_dates``) are reused unchanged. This module only reads the two Last-FM sources and
re-checks the result:

* the knowledge graph published with KGAT (``Data/last-fm``): ``kg_final.txt`` in the author's remapped
  entity ids, ``entity_list.txt`` (id -> Freebase ``m.xxxx``), ``relation_list.txt``, ``item_list.txt``.
  It carries **no statement dates**, so a fact's only evidence in time is its tail's existence date
  (decision 0006 D: the target's existence date is evidence, because a fact cannot hold before its target
  exists);
* ``data-last-fm/freebase-subset.tsv``, a dump of Freebase of 2015-08-09 with the columns ``subject``,
  ``predicate``, ``object``. The subject has the form of the ``org_id`` of ``entity_list.txt``, so the
  entities match directly, with no entity-linking step.

Existence date of a tail: the earliest of ``music.release.release_date``, ``music.artist.active_start``
and ``people.person.date_of_birth``. The literals are read with :func:`parse_freebase_date`, which applies
decision 0006 through :func:`tkgat.kg.temporal.end_of_period`.

Head items are train items, so, as in decision 0006 B, a head's own release date never drops its facts.

Shape of the KG: the repository's KG is 1-hop item -> attribute (``KGTriples`` requires a head below
``n_items`` and a tail at or above it), so only the triples whose **head** is a train item can be carried.
The triples kept by the item-seeded filter whose head is not a train item are dropped with the reason
``head_not_train_item`` and counted. A tail whose Freebase id is also a train item keeps its own tail
entity id, the same convention as the MovieLens KG (``tkgat.kg.temporal.KNOWN_LIMITATIONS``).
"""

from __future__ import annotations

import datetime as dt
import re
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from tkgat.kg.temporal import VARIANTS, cutoff_day, end_of_period, is_undated

FREEBASE_SNAPSHOT = "2015-08-09"
#: Predicates that date a tail entity; the earliest of them is its existence date.
DATE_PREDICATES = ("music.release.release_date", "music.artist.active_start", "people.person.date_of_birth")
DATE_COLUMNS = ("tail_release_date", "tail_active_start", "tail_date_of_birth")
NAME_PREDICATE = "type.object.name"
HEAD_NOT_ITEM = "head_not_train_item"
KNOWN_LIMITATIONS = [
    f"Freebase is a {FREEBASE_SNAPSHOT} dump, not the knowledge available in 2013-2014.",
    "The published Last-FM KG has no statement dates, so a fact is dated only through its tail's existence "
    "date (decision 0006 D). A fact whose tail existed before the cut-off is kept even if the link itself "
    "was created later.",
    "A date literal ending in -01-01 is read as year-only (31 December); a fact really dated 1 January is "
    "placed too late (decision 0006 A). Freebase writes a year-only date as a gYear literal, which is read "
    "as 31 December as well.",
    "Unparseable date literals (e.g. BC years) count as no date (decision 0006 C).",
    "The KG is 1-hop item -> attribute, so a triple whose head is not a train item cannot be carried; such "
    f"triples are dropped with the reason {HEAD_NOT_ITEM!r}.",
    "A tail whose Freebase id is also a train item is a separate entity from that item, as in the MovieLens "
    "KG: the author's KG has one id space for items and entities, the repository's format has two.",
]

_LITERAL = re.compile(r'^"(?P<value>[^"]*)"(?:\^\^<[^>]*>|@[\w-]+)?$')
_YEAR = re.compile(r"\d{4}")
_YEAR_MONTH = re.compile(r"\d{4}-\d{2}")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def literal_value(literal: str) -> str:
    """The value of a Freebase object literal: ``"2007-09-10"^^<...#date>`` -> ``2007-09-10``.

    A plain (unquoted) object, e.g. another entity id, is returned as it is.
    """
    m = _LITERAL.match(literal.strip())
    return m.group("value") if m else literal.strip()


@cache
def parse_freebase_date(literal: str) -> dt.date | None:
    """Last day a Freebase date literal can refer to, or ``None`` when there is no usable date.

    The three literal forms of the dump are read under decision 0006 through
    :func:`tkgat.kg.temporal.end_of_period`: a year (``"1999"``, gYear) is 31 December of that year, a year
    and month (``"1999-03"``, gYearMonth) the last day of that month, a full date (``"2007-09-10"``, date)
    that day -- except a full date ending in ``-01-01``, which is read as year-only, so 31 December
    (decision 0006 A). A BC year or any other value has no usable date (decision 0006 C).
    """
    value = literal_value(literal)
    if _YEAR.fullmatch(value):
        return end_of_period(f"{value}-01-01", 9)
    if _YEAR_MONTH.fullmatch(value):
        return end_of_period(f"{value}-01", 10)
    if _DAY.fullmatch(value):
        return end_of_period(value)  # no precision: -01-01 is year-only (0006 A)
    return None


def read_freebase_subset(path: str | Path, subjects: set[str]) -> pd.DataFrame:
    """The date literals and the name of every entity of ``subjects`` found in ``freebase-subset.tsv``.

    Returns one row per subject that has at least one of :data:`DATE_PREDICATES` or a name, indexed by the
    Freebase id, with the raw literals in :data:`DATE_COLUMNS` and ``tail_label``. The literals are kept
    raw so that :func:`check_kg` can parse them again independently.
    """
    path = Path(path)
    wanted = {p: c for p, c in zip(DATE_PREDICATES, DATE_COLUMNS, strict=True)}
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", buffering=1 << 24) as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header[:3] != ["subject", "predicate", "object"]:
            raise ValueError(f"{path}: expected the header 'subject predicate object', got {header[:3]}")
        for line_no, line in enumerate(f, 2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no}: expected three tab-separated columns")
            subject, predicate, obj = parts
            if subject not in subjects:
                continue
            column = wanted.get(predicate)
            if column is None and predicate != NAME_PREDICATE:
                continue
            row = rows.setdefault(subject, {})
            if column is None:
                row.setdefault("tail_label", literal_value(obj))
            elif column in row:
                raise ValueError(f"{path}:{line_no}: {subject} has two values for {predicate}")
            else:
                row[column] = obj
    out = pd.DataFrame.from_dict(rows, orient="index")
    for column in (*DATE_COLUMNS, "tail_label"):
        if column not in out:
            out[column] = ""
    return out[[*DATE_COLUMNS, "tail_label"]].fillna("")


def read_author_kg(author_dir: str | Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """The published KGAT Last-FM graph: its triples, the entity id -> Freebase id map, the relation names.

    ``triples`` has the author's integer ids in the columns ``h``, ``r``, ``t``; ``entities`` is indexed by
    the author's entity id; ``relations`` are the relation names in the order of ``relation_list.txt``, so
    relation ``r`` of a triple is ``relations[r]``.
    """
    author_dir = Path(author_dir)
    triples = pd.read_csv(author_dir / "kg_final.txt", sep=" ", names=["h", "r", "t"], dtype="int64", header=None)
    if triples.duplicated().any():
        raise ValueError("the author's kg_final.txt holds duplicate triples")
    entities = pd.read_csv(author_dir / "entity_list.txt", sep=" ", dtype={"org_id": str})
    relations = pd.read_csv(author_dir / "relation_list.txt", sep=" ", dtype={"org_id": str})
    if list(entities.columns) != ["org_id", "remap_id"] or list(relations.columns) != ["org_id", "remap_id"]:
        raise ValueError("entity_list.txt / relation_list.txt must have the columns 'org_id remap_id'")
    if list(relations["remap_id"]) != list(range(len(relations))):
        raise ValueError("relation ids of relation_list.txt are not contiguous")
    names = [org.rsplit("/", 1)[-1] for org in relations["org_id"]]
    return triples, pd.Series(entities["org_id"].to_numpy(), index=entities["remap_id"].to_numpy()), names


def item_entities(item_ids: np.ndarray, items_file: str | Path) -> pd.DataFrame:
    """Train items in encoding order: ``h`` (= item index), ``item`` (the author's item id), ``freebase_id``.

    ``item_ids`` must be ``EncodedSplit.item_ids``, so that entity ``i < n_items`` is item index ``i``. In
    the published data an item's id is also its entity id, which is checked here.
    """
    table = pd.read_csv(Path(items_file), sep=" ", dtype={"freebase_id": str})
    if list(table.columns) != ["org_id", "remap_id", "freebase_id"]:
        raise ValueError("item_list.txt must have the columns 'org_id remap_id freebase_id'")
    fb = pd.Series(table["freebase_id"].to_numpy(), index=table["remap_id"].to_numpy())
    items = pd.DataFrame({"h": np.arange(len(item_ids)), "item": np.asarray(item_ids).astype(np.int64)})
    items["freebase_id"] = items["item"].map(fb)
    if items["freebase_id"].isna().any():
        raise ValueError(f"{int(items['freebase_id'].isna().sum())} train items are not in item_list.txt")
    return items


def seeded_triples(items: pd.DataFrame, triples: pd.DataFrame) -> pd.DataFrame:
    """The author triples with a train item at either end: the item-seeded filter of the Last-FM generator."""
    ids = set(items["item"].to_numpy().tolist())
    return triples[triples["h"].isin(ids) | triples["t"].isin(ids)].reset_index(drop=True)


def lastfm_facts(
    items: pd.DataFrame, triples: pd.DataFrame, entity_ids: pd.Series, relations: list[str], tail_facts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The annotated facts of the train items, and the item-seeded triples the 1-hop format cannot carry.

    ``triples`` is filtered the way the Last-FM generator filters it -- a triple is kept when either end is
    a train item -- and then split: the triples whose head is a train item become facts with ``h`` = that
    item's index and the tail as an attribute entity (keyed by its Freebase id), the rest are returned as
    dropped with the reason :data:`HEAD_NOT_ITEM`. The returned facts carry ``statement_date`` (always
    ``None``: the KG has no statement dates) and ``tail_date``, the earliest existence date of the tail.
    """
    by_item = pd.Series(items["h"].to_numpy(), index=items["item"].to_numpy())
    seeded = seeded_triples(items, triples)
    head_is_item = seeded["h"].isin(by_item.index)

    def frame(part: pd.DataFrame) -> pd.DataFrame:
        """One row per author triple, with the tail as an attribute entity keyed by its Freebase id."""
        f = pd.DataFrame(
            {
                "author_head": part["h"].to_numpy(),
                "author_tail": part["t"].to_numpy(),
                "relation": [relations[r] for r in part["r"]],
                "head_freebase_id": part["h"].map(entity_ids).to_numpy(),
                "tail": part["t"].map(entity_ids).to_numpy(),
            }
        )
        joined = f.join(tail_facts, on="tail")
        for column in (*DATE_COLUMNS, "tail_label"):
            joined[column] = joined[column].fillna("")
        joined["tail_kind"] = "freebase"
        return annotate_lastfm(joined)

    kept = frame(seeded[head_is_item])
    kept.insert(0, "item", seeded.loc[head_is_item, "h"].to_numpy())
    kept.insert(0, "h", seeded.loc[head_is_item, "h"].map(by_item).to_numpy().astype(np.int64))
    dropped = frame(seeded[~head_is_item]).assign(reason=HEAD_NOT_ITEM)
    return kept, dropped


def annotate_lastfm(facts: pd.DataFrame) -> pd.DataFrame:
    """Add ``statement_date`` (always ``None``) and ``tail_date`` to facts holding the raw date literals.

    ``tail_date`` is the earliest date of :data:`DATE_COLUMNS` that :func:`parse_freebase_date` can read,
    and ``None`` when none of them is usable: that fact is *undated* in the sense of decision 0006 D.
    """
    f = facts.copy()
    parsed = [[parse_freebase_date(v) if v else None for v in f[c]] for c in DATE_COLUMNS]
    f["statement_date"] = None  # the published Last-FM KG carries no statement dates
    f["tail_date"] = [min((d for d in ds if d is not None), default=None) for ds in zip(*parsed, strict=True)]
    return f


def check_kg(kept: pd.DataFrame, cutoff_ts: int, variant: str, n_items: int) -> list[str]:
    """Independent check of a built Last-FM KG; returns the problems found (empty list = OK).

    The dates are parsed again from the raw literals, so a bug in the annotated columns cannot hide a leak.
    The ``static`` variant is not time-consistent by design, so its dates are not checked.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    problems: list[str] = []
    if variant != "static":
        day = cutoff_day(cutoff_ts)
        a = annotate_lastfm(kept.drop(columns=["statement_date", "tail_date"], errors="ignore"))
        future_tail = a["tail_date"].map(lambda d: d is not None and d >= day)
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


def relation_stats(facts: pd.DataFrame) -> pd.DataFrame:
    """Share of facts per relation whose tail has an existence date, and share with no date at all."""
    g = facts.assign(tail_dated=facts["tail_date"].notna(), no_date=is_undated(facts))
    return (
        g.groupby("relation")
        .agg(facts=("tail", "size"), tail_dated=("tail_dated", "mean"), no_date=("no_date", "mean"))
        .round(4)
    )


__all__ = [
    "DATE_COLUMNS",
    "DATE_PREDICATES",
    "FREEBASE_SNAPSHOT",
    "HEAD_NOT_ITEM",
    "KNOWN_LIMITATIONS",
    "annotate_lastfm",
    "check_kg",
    "item_entities",
    "lastfm_facts",
    "literal_value",
    "parse_freebase_date",
    "read_author_kg",
    "read_freebase_subset",
    "relation_stats",
    "seeded_triples",
]
