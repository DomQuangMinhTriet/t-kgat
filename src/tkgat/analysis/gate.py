"""Rules of the small-set gate (decision 0016 items 5, 6, 8, 9), computed from validation rows only.

``results`` has one row per (model, schedule, grid point, seed), read from the **validation** rows of the registry
(``phase == "val"``) of the small set: columns ``model`` (``bprmf``, ``kgat``, ``kgat_nokg``, ``kgat_pruned``),
``schedule`` (``none`` or ``cosine``), ``point`` (the grid point without the schedule keys, as text), ``lr``, ``seed``
and ``recall@20``. Nothing here reads a test score, and no function picks a value silently: each returns the numbers it
compared and the verdict, so the document that quotes them can be checked line by line.

Rules, fixed here before the KGAT runs of the small set exist:

* schedule: cosine replaces the constant rate if, for every model that has both, the best grid point under cosine has
  a mean validation Recall@20 at least that of the best point under ``none``. A model with one schedule only: pending.
* lr 1e-4: the 1e-4 grid points are dropped if, for every model and schedule that has them, the best 1e-4 point has a
  lower mean than the best 1e-3 point.
* pruned: ``pruned`` replaces ``default`` if its best point has a mean at least that of ``default`` (same schedule)
  with fewer triples (counts passed in).
* gate: KGAT must be above KGAT without KG (best point of each, same schedule), or the advisor is told before the
  official runs.
* epoch budget: the smallest E in {100, 150, 200} at which the median share of the best validation Recall@20 reached
  by epoch E is at least 0.98 for every KGAT-type model run at lr 1e-3 (the schedule is the chosen one).
"""

from __future__ import annotations

import pandas as pd

METRIC = "recall@20"
BUDGETS = (100, 150, 200)
SHARE = 0.98


def best_points(results: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of the metric over seeds per (model, schedule, point), and the best point of each (model, schedule)."""
    g = (
        results.groupby(["model", "schedule", "point", "lr"], sort=True)[METRIC]
        .agg(mean="mean", std="std", seeds="count")
        .reset_index()
    )
    g["best"] = g.groupby(["model", "schedule"])["mean"].transform("max") == g["mean"]
    return g


def _best(points: pd.DataFrame, model: str, schedule: str, lr: float | None = None) -> float | None:
    sel = points[(points["model"] == model) & (points["schedule"] == schedule)]
    if lr is not None:
        sel = sel[sel["lr"] == lr]
    return None if sel.empty else float(sel["mean"].max())


def schedule_verdict(points: pd.DataFrame, models: tuple[str, ...] = ("bprmf", "kgat")) -> dict:
    rows = []
    for m in models:
        none, cos = _best(points, m, "none"), _best(points, m, "cosine")
        rows.append(
            {"model": m, "none": none, "cosine": cos, "cosine_ok": None if None in (none, cos) else cos >= none}
        )
    known = [r["cosine_ok"] for r in rows if r["cosine_ok"] is not None]
    verdict = "pending" if len(known) < len(models) else ("cosine" if all(known) else "none")
    return {"verdict": verdict, "rows": rows}


def lr_verdict(points: pd.DataFrame, low: float = 1e-4, high: float = 1e-3) -> dict:
    rows = []
    for (m, s), _ in points.groupby(["model", "schedule"]):
        lo, hi = _best(points, m, s, low), _best(points, m, s, high)
        if lo is None or hi is None:
            continue
        rows.append({"model": m, "schedule": s, "best_1e-4": lo, "best_1e-3": hi, "drop": lo < hi})
    verdict = "pending" if not rows else ("drop" if all(r["drop"] for r in rows) else "keep")
    return {"verdict": verdict, "rows": rows}


def pruned_verdict(points: pd.DataFrame, triples: dict[str, int]) -> dict:
    """``triples`` has the number of KG triples at t1 of ``default`` and ``pruned``."""
    rows = []
    for s in sorted(points["schedule"].unique()):
        d, p = _best(points, "kgat", s), _best(points, "kgat_pruned", s)
        if d is None or p is None:
            continue
        rows.append(
            {"schedule": s, "default": d, "pruned": p, "pruned_ok": p >= d and triples["pruned"] < triples["default"]}
        )
    verdict = "pending" if not rows else ("pruned" if all(r["pruned_ok"] for r in rows) else "default")
    return {"verdict": verdict, "rows": rows}


def gate_verdict(points: pd.DataFrame) -> dict:
    rows = []
    for s in sorted(points["schedule"].unique()):
        k, n = _best(points, "kgat", s), _best(points, "kgat_nokg", s)
        if k is None or n is None:
            continue
        rows.append({"schedule": s, "kgat": k, "kgat_nokg": n, "kgat_above": k > n})
    verdict = "pending" if not rows else ("pass" if all(r["kgat_above"] for r in rows) else "tell the advisor")
    return {"verdict": verdict, "rows": rows}


def epoch_budget(runs: pd.DataFrame, models: tuple[str, ...], lr: float = 1e-3) -> dict:
    """``runs``: ``13_epoch_budget.collect`` of the small set; ``models``: the model tags that count."""
    sel = runs[runs["model"].isin(models) & (runs["lr"] == lr)]
    if sel.empty:
        return {"verdict": "pending", "rows": []}
    rows = []
    for m, g in sel.groupby("model"):
        rows.append({"model": m, "runs": len(g), **{f"share@{e}": float(g[f"share_at_{e}"].median()) for e in BUDGETS}})
    verdict = next((e for e in BUDGETS if all(r[f"share@{e}"] >= SHARE for r in rows)), "more than 200")
    return {"verdict": verdict, "rows": rows}
