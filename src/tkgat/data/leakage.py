"""Automated leakage checks.

``check_split`` returns a list of human-readable violations; an empty list
means the split passed.  The same function is used by the pytest suite and by
``scripts/02_build_splits.py``, which refuses to write a split that fails.
"""

from __future__ import annotations

import pandas as pd

from tkgat.data.split import TemporalSplit


def _pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(df["user"].to_numpy().tolist(), df["item"].to_numpy().tolist(), strict=True))


def check_split(s: TemporalSplit) -> list[str]:
    v: list[str] = []
    tr, va, te = s.train, s.val, s.test

    # 1. time ordering
    if len(tr) and tr["ts"].max() >= s.t1:
        v.append(f"train has ts >= t1 ({tr['ts'].max()} >= {s.t1})")
    if len(va) and (va["ts"].min() < s.t1 or va["ts"].max() >= s.t2):
        v.append("val has ts outside [t1, t2)")
    if len(te) and te["ts"].min() < s.t2:
        v.append(f"test has ts < t2 ({te['ts'].min()} < {s.t2})")

    # 2. no pair appears in two parts
    p_tr, p_va, p_te = _pairs(tr), _pairs(va), _pairs(te)
    for a, b, name in [(p_tr, p_va, "train/val"), (p_tr, p_te, "train/test"), (p_va, p_te, "val/test")]:
        if a & b:
            v.append(f"{len(a & b)} (user,item) pairs shared by {name}")

    # 3. val/test only contain users and items known at training time
    users, items = set(tr["user"]), set(tr["item"])
    for part, name in [(va, "val"), (te, "test")]:
        if not set(part["user"]) <= users:
            v.append(f"{name} contains users absent from train")
        if not set(part["item"]) <= items:
            v.append(f"{name} contains items absent from train")

    # 4. k-core property holds on train alone
    k = s.config.kcore
    if k > 1 and len(tr):
        if tr["user"].value_counts().min() < k or tr["item"].value_counts().min() < k:
            v.append(f"train violates the {k}-core property")

    # 5. masks only use the past
    if len(s.seen_before_val) and s.seen_before_val["ts"].max() >= s.t1:
        v.append("seen_before_val contains interactions at or after t1")
    if len(s.seen_before_test) and s.seen_before_test["ts"].max() >= s.t2:
        v.append("seen_before_test contains interactions at or after t2")

    # 6. ground truth never overlaps the mask of its own phase
    if p_va & _pairs(s.seen_before_val):
        v.append("val ground truth overlaps seen_before_val")
    if p_te & _pairs(s.seen_before_test):
        v.append("test ground truth overlaps seen_before_test")
    return v
