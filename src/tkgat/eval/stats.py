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


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def paired_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided paired t-test on matched per-user scores (assumes roughly normal differences)."""
    from scipy.stats import ttest_rel

    a, b = _paired(a, b)
    diff = a - b
    if np.allclose(diff, 0):
        return {"n": int(len(a)), "mean_diff": 0.0, "p_value": 1.0}
    res = ttest_rel(a, b)
    return {"n": int(len(a)), "mean_diff": float(diff.mean()), "p_value": float(res.pvalue)}


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap confidence interval for the mean per-user difference a - b."""
    a, b = _paired(a, b)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"n": int(len(diff)), "mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def holm(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values and reject flags (input order)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return {"p_adjusted": adjusted.tolist(), "reject": (adjusted <= alpha).tolist()}
