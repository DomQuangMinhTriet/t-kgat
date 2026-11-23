"""Random user subsets and time-window subsets of a dataset, for runs on small machines (decisions 0008, 0016).

A subset keeps *every* rating of a random sample of users, so the calendar span of the data, the way ratings are
entered in sessions and the share of returning users in the late periods are the same as in the full dataset. The
subset is taken before cleaning, the split and the k-core, and then goes through the same pipeline as a full dataset:
quantile cut-offs are those of the subset's own timestamps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def subsample_users(inter: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, dict]:
    """All ratings of ``round(fraction * n_users)`` users drawn uniformly without replacement.

    ``inter`` has columns ``user, item, rating, ts``. The draw depends only on the set of user ids and ``seed``,
    not on the row order. Returns the subset (index reset) and a report that is stored in ``meta.json``.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"user_fraction must be in (0, 1], got {fraction}")
    users = np.sort(inter["user"].unique())
    n_keep = max(1, round(fraction * len(users)))
    picked = np.random.default_rng(seed).choice(users, size=n_keep, replace=False)
    subset = inter[inter["user"].isin(picked)].reset_index(drop=True)
    report = {
        "user_fraction": float(fraction),
        "user_seed": int(seed),
        "users_total": int(len(users)),
        "users_kept": int(n_keep),
        "ratings_total": int(len(inter)),
        "ratings_kept": int(len(subset)),
    }
    return subset, report


def cut_time_window(inter: pd.DataFrame, months: int) -> tuple[pd.DataFrame, dict]:
    """Keep the events of the last ``months`` calendar months of ``inter`` (decision 0016 item 1).

    The window ends at the largest timestamp of ``inter`` and starts ``months`` calendar months before it; an event is
    kept if ``ts >= start``. Only the column ``ts`` is read to decide, so the result depends on nothing else and not on
    the row order. The cut is made on the events before the split and the k-core; t1 and t2 are then computed inside the
    window by the normal pipeline. Returns the window (index reset, original row order kept) and a report for
    ``meta.json``.
    """
    if isinstance(months, bool) or int(months) != months or months < 1:
        raise ValueError(f"window_months must be a positive integer, got {months!r}")
    if inter.empty:
        raise ValueError("cannot cut a time window from an empty table")
    months = int(months)
    ts_max = int(inter["ts"].max())
    start = int((pd.Timestamp(ts_max, unit="s") - pd.DateOffset(months=months)).timestamp())
    window = inter[inter["ts"] >= start].reset_index(drop=True)
    report = {
        "window_months": months,
        "window_start_ts": start,
        "window_end_ts": ts_max,
        "ratings_total": int(len(inter)),
        "ratings_kept": int(len(window)),
        "users_total": int(inter["user"].nunique()),
        "users_kept": int(window["user"].nunique()),
        "items_total": int(inter["item"].nunique()),
        "items_kept": int(window["item"].nunique()),
    }
    return window, report
