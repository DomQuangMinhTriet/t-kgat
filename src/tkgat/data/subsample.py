"""Random user subsets of a dataset, for runs on small machines (decision 0008).

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
