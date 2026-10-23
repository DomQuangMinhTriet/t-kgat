"""Entity linking: MovieLens items -> MovieLens-20M ids -> Freebase / YAGO.

KB4Rec links MovieLens-**20M** movie ids to Freebase and YAGO.  MovieLens-1M
shares the 20M id space (verified in ``notebooks/02_entity_linking.ipynb``),
so ML-1M ids can be looked up directly.  MovieLens-100K uses its own ids, so it
is first matched to ML-1M by normalised title + year.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

_ARTICLES = r"The|A|An|Les|La|Le|L'|Il|Das|Der|Die|El|Los|Las|Det|Den|Une|Un"
_TRAIL_ARTICLE = re.compile(rf"^(.*), ({_ARTICLES})$")
_YEAR = re.compile(r"\s*\((\d{4})\)\s*$")
_AKA = re.compile(r"\s*\(([^()]*)\)\s*$")


def split_title(title: str) -> tuple[str, int | None]:
    m = _YEAR.search(title)
    year = int(m.group(1)) if m else None
    return (_YEAR.sub("", title) if m else title).strip(), year


def norm_title(title: str) -> str:
    """Normalise a MovieLens title (without year) for exact matching."""
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    t = _AKA.sub("", t)  # drop a trailing "(a.k.a. ...)" / foreign title
    t = re.sub(r"\s+", " ", t).strip()
    m = _TRAIL_ARTICLE.match(t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _keyed(items: pd.DataFrame) -> pd.DataFrame:
    parts = items["title"].map(split_title)
    out = items[["item", "title"]].copy()
    out["key"] = [norm_title(t) for t, _ in parts]
    out["key_full"] = [re.sub(r"[^a-z0-9]", "", t.lower()) for t, _ in parts]
    out["key_aka"] = [norm_title(m.group(1)) if (m := _AKA.search(t)) else "" for t, _ in parts]
    out["year"] = [y for _, y in parts]
    return out


def match_ml100k_to_ml1m(items100k: pd.DataFrame, items1m: pd.DataFrame) -> pd.DataFrame:
    """Return one row per ML-100K item with the matched ML-1M id and the rule used.

    Rules, tried in order: (1) normalised title + same year, (2) normalised
    title + year within one year, (3) full title (keeping the a.k.a. part) +
    same year, (4) ML-100K title equals the ML-1M a.k.a. title + same year,
    (5) one normalised title contains the other (>= 6 characters) + same
    year.  A match must be unique, otherwise the item stays unmatched.
    """
    a, b = _keyed(items100k), _keyed(items1m)
    out = a[["item", "title", "year"]].rename(columns={"item": "item_100k"}).copy()
    out["item_ml1m"] = pd.NA
    out["rule"] = pd.NA

    by_key = b.groupby("key")
    by_full = b.groupby("key_full")

    def unique(cands: pd.DataFrame):
        return int(cands["item"].iloc[0]) if len(cands) == 1 else None

    for idx, r in a.iterrows():
        hit, rule = None, None
        if r["key"] in by_key.groups:
            c = by_key.get_group(r["key"])
            hit = unique(c[c["year"] == r["year"]])
            rule = "title+year" if hit is not None else None
            if hit is None and r["year"] is not None:
                hit = unique(c[(c["year"] - r["year"]).abs() <= 1])
                rule = "title+year±1" if hit is not None else None
        if hit is None and r["key_full"] in by_full.groups:
            c = by_full.get_group(r["key_full"])
            hit = unique(c[c["year"] == r["year"]])
            rule = "full_title+year" if hit is not None else None
        if hit is None and r["key"]:
            c = b[(b["key_aka"] == r["key"]) & (b["year"] == r["year"])]
            hit = unique(c)
            rule = "aka_title+year" if hit is not None else None
        if hit is None and len(r["key"]) >= 6:
            same_year = b[b["year"] == r["year"]]
            c = same_year[[(r["key"] in k or (len(k) >= 6 and k in r["key"])) for k in same_year["key"]]]
            hit = unique(c)
            rule = "contains+year" if hit is not None else None
        if hit is not None:
            out.at[idx, "item_ml1m"] = hit
            out.at[idx, "rule"] = rule
    return out


def load_kb4rec(kb4rec_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = Path(kb4rec_dir)
    fb = pd.read_csv(
        d / "ml2fb.txt",
        sep="\t",
        names=["ml20m_id", "freebase_id"],
        dtype={"ml20m_id": int, "freebase_id": str},
    )
    yago = pd.read_csv(
        d / "ml2yago.txt",
        sep="\t",
        names=["ml20m_id", "yago_id"],
        dtype={"ml20m_id": int, "yago_id": str},
        quoting=3,
        encoding="utf-8",
    )
    return fb, yago


def attach_kb_ids(df: pd.DataFrame, id_col: str, fb: pd.DataFrame, yago: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    key = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
    out["freebase_id"] = key.map(dict(zip(fb["ml20m_id"], fb["freebase_id"], strict=True)))
    out["yago_id"] = key.map(dict(zip(yago["ml20m_id"], yago["yago_id"], strict=True)))
    return out
