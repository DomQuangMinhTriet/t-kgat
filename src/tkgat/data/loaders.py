"""Loaders for the raw MovieLens files.

Every loader returns two DataFrames with the same schema so that the rest of
the pipeline does not care which MovieLens release it is working on:

* ``interactions``: ``user, item, rating, ts`` (``ts`` = Unix seconds, UTC)
* ``items``: ``item, title, year, release_date, genres``
  (``release_date`` is a ``datetime64`` when the release ships a day-precision
  date, otherwise ``NaT``; ``genres`` is a ``|``-joined string)

Raw ids are kept as they are in the source files. Re-indexing to contiguous
integers happens later, after the split, so that the mapping only ever sees
training data.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ML100K_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _year_from_title(title: str) -> float:
    m = _YEAR_RE.search(title or "")
    return float(m.group(1)) if m else float("nan")


def load_ml100k(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    inter = pd.read_csv(
        raw_dir / "u.data", sep="\t", names=["user", "item", "rating", "ts"],
        dtype={"user": "int64", "item": "int64", "rating": "int64", "ts": "int64"},
    )
    cols = ["item", "title", "release_date", "video_release_date", "imdb_url", *ML100K_GENRES]
    items = pd.read_csv(raw_dir / "u.item", sep="|", names=cols, encoding="latin-1")
    items["release_date"] = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce")
    items["genres"] = items[ML100K_GENRES].apply(
        lambda r: "|".join(g for g in ML100K_GENRES if r[g] == 1), axis=1
    )
    items["year"] = items["title"].map(_year_from_title)
    items.loc[items["year"].isna(), "year"] = items["release_date"].dt.year
    return inter, items[["item", "title", "year", "release_date", "genres", "imdb_url"]]


def load_ml1m(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    inter = pd.read_csv(
        raw_dir / "ratings.dat", sep="::", engine="python",
        names=["user", "item", "rating", "ts"],
    ).astype("int64")
    items = pd.read_csv(
        raw_dir / "movies.dat", sep="::", engine="python",
        names=["item", "title", "genres"], encoding="latin-1",
    )
    items["year"] = items["title"].map(_year_from_title)
    items["release_date"] = pd.NaT
    return inter, items[["item", "title", "year", "release_date", "genres"]]


LOADERS = {"ml-100k": load_ml100k, "ml-1m": load_ml1m}


def load(name: str, raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a dataset by its config name (``ml-100k`` or ``ml-1m``)."""
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; expected one of {sorted(LOADERS)}")
    return LOADERS[name](raw_dir)
