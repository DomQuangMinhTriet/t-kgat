"""Paired significance tests on per-user metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test on matched per-user scores."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    if np.allclose(diff, 0):
        return {"n": int(len(a)), "mean_diff": 0.0, "p_value": 1.0}
    stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"n": int(len(a)), "mean_diff": float(diff.mean()), "p_value": float(stat.pvalue)}
