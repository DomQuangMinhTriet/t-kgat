"""Paired significance tests on per-user metrics.

The paired functions match arrays by position. Load per-user scores with :func:`load_per_user`, which
checks that both files hold the same evaluation users in the same order before returning the arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load_per_user(path_a: str | Path, path_b: str | Path, metric: str = "recall@20") -> tuple[np.ndarray, np.ndarray]:
    """Per-user ``metric`` from two ``results/runs/<run_name>/<tag>_seed<seed>.npz`` files, ready for pairing.

    Raises ``ValueError`` unless both files hold exactly the same ``users`` array (same users, same order),
    e.g. when the files come from different splits or run names.
    """
    with np.load(path_a) as fa, np.load(path_b) as fb:
        users_a, users_b = fa["users"], fb["users"]
        if not np.array_equal(users_a, users_b):
            only_a = len(np.setdiff1d(users_a, users_b))
            only_b = len(np.setdiff1d(users_b, users_a))
            raise ValueError(
                f"per-user files do not hold the same users: {path_a} has {len(users_a)}, {path_b} has "
                f"{len(users_b)} ({only_a} only in the first, {only_b} only in the second"
                f"{', same users in a different order' if only_a == only_b == 0 else ''}); "
                "compare runs of the same split only"
            )
        for path, f in ((path_a, fa), (path_b, fb)):
            if metric not in f:
                raise KeyError(f"{path} has no per-user metric {metric!r}; available: {sorted(f.files)}")
        return np.asarray(fa[metric], dtype=float), np.asarray(fb[metric], dtype=float)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test on matched per-user scores.

    Uses ``zero_method="wilcox"``: users whose difference is exactly 0 (often both models score 0) are
    dropped before ranking, as in Wilcoxon's original test. The result therefore reports both
    ``n_pairs`` (users with a score in both arrays; ``n`` is kept as an alias) and ``n_nonzero`` (the
    differences actually ranked, i.e. the effective sample size of the test). Report ``n_nonzero``.
    """
    a, b = _paired(a, b)
    diff = a - b
    n_pairs, n_nonzero = int(len(a)), int(np.count_nonzero(diff))
    sizes = {"n": n_pairs, "n_pairs": n_pairs, "n_nonzero": n_nonzero}
    if np.allclose(diff, 0):
        return {**sizes, "mean_diff": 0.0, "p_value": 1.0}
    stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {**sizes, "mean_diff": float(diff.mean()), "p_value": float(stat.pvalue)}


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
