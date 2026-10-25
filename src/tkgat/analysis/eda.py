"""Descriptive statistics for the EDA notebook (``notebooks/01_eda.ipynb``).

Every function takes interactions with columns ``user, item, rating, ts`` (Unix seconds, UTC) and returns
plain pandas objects, so the notebook only calls them and plots. Nothing here reads files or test scores
other than what the caller passes in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = 3600
DAY = 86400


def _utc(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, unit="s", utc=True)


def overview(inter: pd.DataFrame) -> dict:
    """One row of the overview table: size, density and time span."""
    n_users, n_items = inter["user"].nunique(), inter["item"].nunique()
    t = _utc(inter["ts"])
    return {
        "users": n_users,
        "items": n_items,
        "ratings": len(inter),
        "density": len(inter) / (n_users * n_items),
        "first": t.min().date(),
        "last": t.max().date(),
        "span_days": int((t.max() - t.min()).total_seconds() // DAY),
        "median_ratings_per_user": float(inter.groupby("user").size().median()),
    }


def year_shares(inter: pd.DataFrame) -> pd.Series:
    """Share of ratings per calendar year (UTC)."""
    return _utc(inter["ts"]).dt.year.value_counts(normalize=True).sort_index()


def weekly_counts(inter: pd.DataFrame) -> pd.Series:
    """Number of ratings per week, indexed by the week's start (Monday, UTC, timezone-naive)."""
    t = _utc(inter["ts"]).dt.tz_localize(None)
    weeks = t.dt.to_period("W-SUN").dt.start_time
    return weeks.value_counts().sort_index()


def first_window_share(inter: pd.DataFrame, window_s: int = DAY) -> float:
    """Share of ratings entered within ``window_s`` seconds of the user's first rating."""
    first = inter.groupby("user")["ts"].transform("min")
    return float(((inter["ts"] - first) < window_s).mean())


def first_calendar_day_share(inter: pd.DataFrame) -> float:
    """Share of ratings entered on the same UTC calendar day as the user's first rating."""
    day = inter["ts"] // DAY
    return float((day == day.groupby(inter["user"]).transform("min")).mean())


def sessions(inter: pd.DataFrame, gap_s: int = HOUR) -> pd.DataFrame:
    """Rating sessions: a new session starts after a pause longer than ``gap_s`` seconds.

    Returns one row per session: ``user, start, end, n_ratings, duration_s``.
    """
    d = inter[["user", "ts"]].sort_values(["user", "ts"], kind="stable")
    new = d["user"].ne(d["user"].shift()) | (d["ts"].diff() > gap_s)
    d = d.assign(session=new.cumsum())
    s = d.groupby("session").agg(
        user=("user", "first"), start=("ts", "min"), end=("ts", "max"), n_ratings=("ts", "size")
    )
    return s.assign(duration_s=s["end"] - s["start"]).reset_index(drop=True)


def active_days(inter: pd.DataFrame) -> pd.Series:
    """Number of distinct UTC calendar days on which each user rated."""
    return (inter["ts"] // DAY).groupby(inter["user"]).nunique()


def consecutive_gaps(df: pd.DataFrame, by: str = "user") -> pd.Series:
    """Seconds between consecutive interactions of the same node (``by`` = user or item)."""
    d = df.sort_values([by, "ts"], kind="stable")
    return d.groupby(by)["ts"].diff().dropna()


def node_time_spread(train: pd.DataFrame, by: str = "user", ref: int | None = None) -> pd.DataFrame:
    """Time spread of the train edges of each node; T-KGAT compares Δt among the edges of one node.

    ``by`` = ``"user"`` (a user's items) or ``"item"`` (the users who rated the item). Columns:
    ``n_edges``, ``spread_s`` (max − min ts) and ``std_days``, the sample std of ``(ref − ts)`` in days
    (``ref`` defaults to the latest ts; the std does not depend on it). ``std_days`` is NaN for a node
    with one edge.
    """
    if by not in ("user", "item"):
        raise ValueError("by must be 'user' or 'item'")
    ref = int(train["ts"].max()) if ref is None else int(ref)
    age = (ref - train["ts"]) / DAY
    g = train.assign(age=age).groupby(by)
    return pd.DataFrame(
        {
            "n_edges": g.size(),
            "spread_s": g["ts"].max() - g["ts"].min(),
            "std_days": g["age"].std(ddof=1),
        }
    )


def share_below(values: pd.Series, thresholds: dict[str, float]) -> dict[str, float]:
    """Share of values strictly below each named threshold."""
    v = pd.Series(values).dropna()
    return {name: float((v < t).mean()) for name, t in thresholds.items()}


def rating_distribution(inter: pd.DataFrame) -> pd.Series:
    """Share of ratings per value 1–5."""
    return inter["rating"].value_counts(normalize=True).reindex(range(1, 6), fill_value=0.0)


def film_age_at_rating(inter: pd.DataFrame, items: pd.DataFrame) -> pd.Series:
    """Years between a film's release year (from its title) and the year it was rated; films without a year
    are skipped."""
    year = items.set_index("item")["year"]
    d = inter.assign(film_year=inter["item"].map(year)).dropna(subset=["film_year"])
    return (_utc(d["ts"]).dt.year - d["film_year"]).astype(int).rename("film_age_years")


def split_losses(meta: dict) -> pd.DataFrame:
    """Positives of val and test before filtering, dropped as cold users / items, and kept (from meta.json)."""
    rows = []
    for part in ("val", "test"):
        rows.append(
            {
                "part": part,
                "positives_before_filter": meta[f"{part}_pos_before_filter"],
                "dropped_cold_user": meta[f"{part}_dropped_cold_user_interactions"],
                "dropped_cold_item": meta[f"{part}_dropped_cold_item_interactions"],
                "kept": meta[f"{part}_pos"],
                "users_kept": meta[f"{part}_users"],
            }
        )
    out = pd.DataFrame(rows).set_index("part")
    out["kept_share"] = out["kept"] / out["positives_before_filter"]
    return out


def popularity_curve(train: pd.DataFrame) -> pd.DataFrame:
    """Items sorted by number of train positives: ``item_share`` (rank / n items) and ``cum_share`` of positives."""
    counts = train["item"].value_counts()
    return pd.DataFrame(
        {
            "item_share": np.arange(1, len(counts) + 1) / len(counts),
            "cum_share": counts.cumsum().to_numpy() / counts.sum(),
        },
        index=counts.index,
    )


def top_share(train: pd.DataFrame, frac: float) -> float:
    """Share of train positives that go to the ``frac`` most popular items."""
    counts = train["item"].value_counts()
    k = max(1, int(round(frac * len(counts))))
    return float(counts.iloc[:k].sum() / counts.sum())


def baseline_table(reg: pd.DataFrame, metric: str = "recall@20", phase: str = "test") -> pd.DataFrame:
    """Mean and sample std (ddof=1) over seeds of one metric per dataset, model and config.

    ``reg`` should come from ``tkgat.registry.load()`` (newest run per key).
    """
    r = reg[reg["phase"] == phase]
    return (
        r.groupby(["dataset", "model", "config"])[metric]
        .agg(n_seeds="count", mean="mean", std=lambda s: s.std(ddof=1))
        .reset_index()
    )
