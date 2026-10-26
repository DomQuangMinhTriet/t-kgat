"""MostPop: ranking by train counts, stable tie-break, and the recent-popularity window."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from tkgat.eval.metrics import evaluate
from tkgat.models.baselines import MostPop

DAY = 86400
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def split_like(items, days, n_items):
    """The three fields MostPop reads from an EncodedSplit; ``days`` are train timestamps in days."""
    items = np.asarray(items, dtype=np.int64)
    train = np.stack([np.zeros(len(items), dtype=np.int64), items], axis=1)
    return SimpleNamespace(train=train, train_ts=np.asarray(days, dtype=np.int64) * DAY, n_items=n_items)


def counts(model):
    """Integer popularity counts: the tie-break adds less than 1e-6 on top of them."""
    return np.round(model.pop).astype(int)


def test_ranks_items_by_train_positive_count():
    data = split_like([2, 2, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5], n_items=4)
    model = MostPop(data)
    assert counts(model).tolist() == [2, 1, 3, 0]
    assert np.argsort(-model.pop).tolist() == [2, 0, 1, 3]
    assert model.score(np.array([5, 6])).shape == (2, 4)  # same scores for every user


def test_tie_break_is_float64_and_orders_equal_counts_by_item_id():
    # 100 items, every one with 100 positives: a float32 score would collapse the 1e-6 tie-break
    # (float32 spacing near 100 is 7.6e-6), so this fails if the scores are ever downcast
    n_items = 100
    items = np.repeat(np.arange(n_items), 100)
    model = MostPop(split_like(items, np.zeros(len(items)), n_items))
    assert model.pop.dtype == np.float64
    assert (np.diff(model.pop) < 0).all()  # strictly decreasing: smaller item id ranks first
    assert len(np.unique(model.pop.astype(np.float32))) == 1  # why float64 matters
    assert np.argsort(-model.pop, kind="stable").tolist() == list(range(n_items))


@pytest.mark.parametrize("device", DEVICES)
def test_ties_rank_by_item_id_through_evaluate(device):
    """Equal counts give the same ranking on every device: item 0 first, then 1, 2, ..."""
    n_items = 100
    items = np.repeat(np.arange(n_items), 100)
    model = MostPop(split_like(items, np.zeros(len(items)), n_items))
    no_mask = sp.csr_matrix((3, n_items), dtype=np.float32)
    truth = {0: np.array([0]), 1: np.array([1]), 2: np.array([2])}  # users 0..2 each hold one item
    scores = lambda users: torch.as_tensor(np.array(model.score(users)), device=device)  # noqa: E731
    res = evaluate(scores, truth, no_mask, ks=(1, 3))
    assert res["per_user"]["recall@1"].tolist() == [1.0, 0.0, 0.0]  # only item 0 is ranked first
    assert res["per_user"]["recall@3"].tolist() == [1.0, 1.0, 1.0]  # items 0, 1, 2 are the top 3


def test_window_days_uses_only_the_last_n_days_of_train():
    # item 0: days 0, 1; item 1: day 5; item 2: days 9, 10; item 3: day 10. The window is anchored at the
    # last train timestamp (day 10), not at the validation cut-off, and includes its first day.
    data = split_like([0, 0, 1, 2, 2, 3], [0, 1, 5, 9, 10, 10], n_items=4)
    assert counts(MostPop(data, window_days=2)).tolist() == [0, 0, 2, 1]  # days >= 8
    assert counts(MostPop(data, window_days=5)).tolist() == [0, 1, 2, 1]  # days >= 5 (boundary included)
    assert counts(MostPop(data, window_days=0)).tolist() == [0, 0, 1, 1]  # only the last day
    assert counts(MostPop(data, window_days=None)).tolist() == [2, 1, 2, 1]  # the whole train period


def test_window_longer_than_train_equals_no_window():
    data = split_like([0, 0, 1, 2, 2, 3], [0, 1, 5, 9, 10, 10], n_items=4)
    full, wide = MostPop(data, window_days=None), MostPop(data, window_days=1000)
    assert np.array_equal(full.pop, wide.pop)
