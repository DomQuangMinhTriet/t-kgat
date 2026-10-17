import pandas as pd

from tkgat.data.split import SplitConfig, dedupe_pairs, kcore, temporal_split


def test_dedupe_keeps_earliest():
    df = pd.DataFrame({"user": [1, 1], "item": [7, 7], "rating": [2, 5], "ts": [20, 10]})
    out = dedupe_pairs(df)
    assert len(out) == 1 and out.loc[0, "ts"] == 10 and out.loc[0, "rating"] == 5


def test_kcore_is_iterative():
    # item 3 has 2 users; removing it drops user 2 below k=2, which then drops item 4
    df = pd.DataFrame({"user": [0, 0, 1, 1, 2, 2], "item": [1, 2, 1, 2, 3, 4], "rating": 5, "ts": 0})
    df = pd.concat([df, pd.DataFrame({"user": [3], "item": [3], "rating": [5], "ts": [0]})])
    out = kcore(df, 2)
    assert set(out["item"]) == {1, 2} and set(out["user"]) == {0, 1}


def test_kcore_never_sees_future(synthetic):
    """An item with too few *train* positives is removed, however popular it becomes later."""
    cfg = SplitConfig(q_val=0.8, q_test=0.9, min_rating=4, kcore=3)
    star = 777
    early = pd.DataFrame({"user": [0, 1], "item": [star, star], "rating": [5, 5], "ts": [1, 2]})
    late = pd.DataFrame({"user": range(2, 40), "item": star, "rating": 5, "ts": 10**9})
    s = temporal_split(pd.concat([synthetic, early, late]), cfg)
    assert star not in set(s.train["item"])  # only 2 train positives < k=3
    assert star not in set(s.test["item"])  # and it is treated as a cold item in test


def test_parts_are_time_ordered(synthetic):
    s = temporal_split(synthetic, SplitConfig(kcore=2))
    assert s.train["ts"].max() < s.t1 <= s.val["ts"].min()
    assert s.val["ts"].max() < s.t2 <= s.test["ts"].min()
    assert (s.train["rating"] >= 4).all() and (s.test["rating"] >= 4).all()
