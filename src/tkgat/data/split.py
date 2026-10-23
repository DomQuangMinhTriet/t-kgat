"""Leakage-safe temporal split.

Protocol (see docs/decisions/0002-data-split.md):

1. Deduplicate (user, item) pairs, keeping the earliest interaction.
2. Two global cut-offs ``t1 < t2`` taken as quantiles of *all* timestamps.
   train = ts < t1, val = t1 <= ts < t2, test = ts >= t2.  The same cut-offs
   apply to every user, so no user's future can leak into another user's past.
3. Implicit feedback: an interaction is positive when ``rating >= min_rating``.
4. Iterative k-core filtering is computed on the *train positives only*.
5. Val/test keep only users and items that survive in train.  Cold-start
   users/items are dropped and counted, never silently re-introduced.
6. "Seen" histories used to mask the ranking are built from *every*
   interaction (any rating) strictly before the corresponding cut-off.

Nothing in this module looks at test interactions to make a decision about
train.  ``tests/test_leakage.py`` checks this property on real and synthetic
data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    q_val: float = 0.8
    q_test: float = 0.9
    min_rating: int = 4
    kcore: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.q_val < self.q_test < 1:
            raise ValueError("need 0 < q_val < q_test < 1")


@dataclass
class TemporalSplit:
    config: SplitConfig
    t1: int
    t2: int
    train: pd.DataFrame  # positives, columns user,item,rating,ts (raw ids)
    val: pd.DataFrame
    test: pd.DataFrame
    seen_before_val: pd.DataFrame  # user,item,ts (raw ids), ts < t1, any rating
    seen_before_test: pd.DataFrame  # user,item,ts (raw ids), ts < t2, any rating
    stats: dict = field(default_factory=dict)

    @property
    def users(self) -> np.ndarray:
        return np.sort(self.train["user"].unique())

    @property
    def items(self) -> np.ndarray:
        return np.sort(self.train["item"].unique())

    def summary(self) -> dict:
        return {"config": asdict(self.config), "t1": int(self.t1), "t2": int(self.t2), **self.stats}


def dedupe_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest interaction for every (user, item) pair."""
    return (
        df.sort_values(["ts", "user", "item"], kind="mergesort")
        .drop_duplicates(["user", "item"], keep="first")
        .reset_index(drop=True)
    )


def kcore(train_pos: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iterative k-core on a train-only interaction table."""
    df = train_pos
    if k <= 1:
        return df.reset_index(drop=True)
    while True:
        uc = df["user"].value_counts()
        ic = df["item"].value_counts()
        keep = df["user"].map(uc).ge(k) & df["item"].map(ic).ge(k)
        if keep.all():
            return df.reset_index(drop=True)
        df = df[keep]


def temporal_split(interactions: pd.DataFrame, cfg: SplitConfig) -> TemporalSplit:
    df = dedupe_pairs(interactions[["user", "item", "rating", "ts"]])
    n_dupes = len(interactions) - len(df)

    t1 = int(np.quantile(df["ts"].to_numpy(), cfg.q_val))
    t2 = int(np.quantile(df["ts"].to_numpy(), cfg.q_test))
    if not t1 < t2:
        raise ValueError(f"degenerate cut-offs t1={t1} t2={t2}")

    period_train = df[df["ts"] < t1]
    period_val = df[(df["ts"] >= t1) & (df["ts"] < t2)]
    period_test = df[df["ts"] >= t2]

    pos = lambda x: x[x["rating"] >= cfg.min_rating]  # noqa: E731
    train = kcore(pos(period_train), cfg.kcore)
    users, items = set(train["user"]), set(train["item"])

    def restrict(part: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
        p = pos(part)
        cold_user = ~p["user"].isin(users)
        cold_item = ~p["item"].isin(items)
        kept = p[~cold_user & ~cold_item].reset_index(drop=True)
        return kept, {
            f"{name}_pos_before_filter": int(len(p)),
            f"{name}_dropped_cold_user_interactions": int(cold_user.sum()),
            f"{name}_dropped_cold_item_interactions": int((~cold_user & cold_item).sum()),
            f"{name}_cold_items": int(p.loc[cold_item, "item"].nunique()),
        }

    val, sv = restrict(period_val, "val")
    test, st = restrict(period_test, "test")

    seen_val = df.loc[df["ts"] < t1, ["user", "item", "ts"]].reset_index(drop=True)
    seen_test = df.loc[df["ts"] < t2, ["user", "item", "ts"]].reset_index(drop=True)

    stats = {
        "n_raw": int(len(interactions)),
        "n_duplicate_pairs_removed": int(n_dupes),
        "period_sizes": {
            "train": int(len(period_train)),
            "val": int(len(period_val)),
            "test": int(len(period_test)),
        },
        "train_pos_before_kcore": int(len(pos(period_train))),
        "train_pos": int(len(train)),
        "n_users": int(len(users)),
        "n_items": int(len(items)),
        "val_pos": int(len(val)),
        "test_pos": int(len(test)),
        "val_users": int(val["user"].nunique()),
        "test_users": int(test["user"].nunique()),
        **sv,
        **st,
    }
    return TemporalSplit(cfg, t1, t2, train, val, test, seen_val, seen_test, stats)
