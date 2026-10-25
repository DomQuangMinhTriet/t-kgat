import numpy as np
import pandas as pd
import pytest

from tkgat.analysis import eda
from tkgat.analysis.eda import DAY, HOUR


def frame(rows):
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])


# user 1: three ratings in the first hour, one 2 days later; user 2: two ratings 25 h apart
INTER = frame(
    [
        (1, 10, 5, 0),
        (1, 11, 4, 600),
        (1, 12, 3, 3000),
        (1, 10, 2, 2 * DAY),
        (2, 10, 4, 10 * HOUR),
        (2, 13, 1, 35 * HOUR),
    ]
)


def test_overview_and_density():
    o = eda.overview(INTER)
    assert (o["users"], o["items"], o["ratings"]) == (2, 4, 6)
    assert o["density"] == pytest.approx(6 / 8)
    assert o["span_days"] == 2 and o["median_ratings_per_user"] == 3.0


def test_first_window_and_calendar_day():
    # within 24 h of the first rating: user 1 -> 3 of 4, user 2 -> 1 of 2 (25 h is outside)
    assert eda.first_window_share(INTER) == pytest.approx(4 / 6)
    assert eda.first_window_share(INTER, window_s=26 * HOUR) == pytest.approx(5 / 6)
    # calendar day: user 2 rates at 10:00 on day 0 and 11:00 on day 1 -> different days
    assert eda.first_calendar_day_share(INTER) == pytest.approx(4 / 6)


def test_window_edge_is_exclusive():
    x = frame([(1, 1, 5, 0), (1, 2, 5, DAY)])
    assert eda.first_window_share(x) == pytest.approx(0.5)


def test_sessions_split_on_pause():
    s = eda.sessions(INTER, gap_s=HOUR)
    assert len(s) == 4  # user 1: [0, 600, 3000] and [2 days]; user 2: two sessions
    first = s.iloc[0]
    assert (first["user"], first["n_ratings"], first["duration_s"]) == (1, 3, 3000)
    assert len(eda.sessions(INTER, gap_s=30 * 60)) == 5  # the 2400 s pause now splits user 1


def test_active_days_and_gaps():
    assert eda.active_days(INTER).to_dict() == {1: 2, 2: 2}
    assert sorted(eda.consecutive_gaps(INTER)) == [600, 2400, 25 * HOUR, 2 * DAY - 3000]


def test_node_time_spread_user_vs_item():
    train = frame(
        [
            (1, 10, 5, 0),
            (1, 11, 5, 60),  # user 1: both edges within a minute
            (2, 10, 5, 30 * DAY),
            (2, 11, 5, 30 * DAY + 60),
        ]
    )
    u = eda.node_time_spread(train, "user")
    i = eda.node_time_spread(train, "item")
    assert u["spread_s"].tolist() == [60, 60]
    assert i["spread_s"].tolist() == [30 * DAY, 30 * DAY]
    assert i.loc[10, "std_days"] == pytest.approx(np.std([30, 0], ddof=1))
    assert u.loc[1, "std_days"] == pytest.approx(np.std([60 / DAY, 0], ddof=1))
    # std does not depend on the reference time
    assert eda.node_time_spread(train, "item", ref=10**9)["std_days"].equals(i["std_days"])
    one = eda.node_time_spread(frame([(1, 10, 5, 0)]), "user")
    assert np.isnan(one.loc[1, "std_days"])
    with pytest.raises(ValueError):
        eda.node_time_spread(train, "rating")


def test_share_below():
    s = pd.Series([0, 30 * 60, HOUR, DAY])
    assert eda.share_below(s, {"<1h": HOUR, "<1d": DAY}) == {"<1h": 0.5, "<1d": 0.75}


def test_rating_distribution_fills_missing_values():
    d = eda.rating_distribution(INTER)
    assert list(d.index) == [1, 2, 3, 4, 5] and d.sum() == pytest.approx(1.0)
    assert d[4] == pytest.approx(2 / 6)


def test_film_age_at_rating_skips_unknown_year():
    items = pd.DataFrame({"item": [10, 11, 12, 13], "year": [1965.0, 1970.0, np.nan, 1969.0]})
    age = eda.film_age_at_rating(INTER, items)
    assert len(age) == 5 and age.max() == 5 and age.min() == 0  # ratings are in 1970


def test_split_losses_from_meta():
    meta = {
        "val_pos_before_filter": 100,
        "val_dropped_cold_user_interactions": 70,
        "val_dropped_cold_item_interactions": 5,
        "val_pos": 25,
        "val_users": 3,
        "test_pos_before_filter": 50,
        "test_dropped_cold_user_interactions": 10,
        "test_dropped_cold_item_interactions": 0,
        "test_pos": 40,
        "test_users": 8,
    }
    t = eda.split_losses(meta)
    assert t.loc["val", "kept_share"] == pytest.approx(0.25) and t.loc["test", "kept"] == 40


def test_popularity_curve_and_top_share():
    train = frame([(u, 1, 5, 0) for u in range(6)] + [(0, 2, 5, 0), (1, 2, 5, 0), (0, 3, 5, 0), (0, 4, 5, 0)])
    c = eda.popularity_curve(train)
    assert c.index[0] == 1 and c["cum_share"].iloc[-1] == pytest.approx(1.0)
    assert c["cum_share"].iloc[0] == pytest.approx(0.6)
    assert eda.top_share(train, 0.25) == pytest.approx(0.6)


def test_baseline_table_uses_sample_std():
    reg = pd.DataFrame(
        {
            "dataset": "d",
            "model": "m",
            "config": "{}",
            "phase": ["test"] * 3 + ["val"],
            "recall@20": [0.1, 0.2, 0.3, 0.9],
        }
    )
    t = eda.baseline_table(reg)
    assert t.loc[0, "n_seeds"] == 3 and t.loc[0, "mean"] == pytest.approx(0.2)
    assert t.loc[0, "std"] == pytest.approx(0.1)  # ddof=1; ddof=0 would give 0.0816
