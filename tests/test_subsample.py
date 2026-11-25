import numpy as np
import pandas as pd
import pytest

from tkgat.data.leakage import check_split
from tkgat.data.loaders import load
from tkgat.data.subsample import cut_time_window, subsample_users
from tkgat.pipeline.build import build_split, load_config

from .conftest import ROOT, data_available


def test_keeps_every_rating_of_the_chosen_users(synthetic):
    sub, rep = subsample_users(synthetic, 0.25, seed=1)
    assert rep["users_total"] == 60 and rep["users_kept"] == 15 and sub["user"].nunique() == 15
    chosen = set(sub["user"])
    full = synthetic[synthetic["user"].isin(chosen)]
    assert len(sub) == len(full) == rep["ratings_kept"] and rep["ratings_total"] == len(synthetic)
    assert (
        sub.sort_values(["user", "item"])
        .reset_index(drop=True)
        .equals(full.sort_values(["user", "item"]).reset_index(drop=True))
    )


def test_is_deterministic_and_independent_of_row_order(synthetic):
    a, _ = subsample_users(synthetic, 0.3, seed=7)
    shuffled = synthetic.sample(frac=1, random_state=0).reset_index(drop=True)
    b, _ = subsample_users(shuffled, 0.3, seed=7)
    assert set(a["user"]) == set(b["user"])
    c, _ = subsample_users(synthetic, 0.3, seed=8)
    assert set(a["user"]) != set(c["user"])  # another seed gives another sample


def test_fraction_bounds_and_full_dataset(synthetic):
    full, rep = subsample_users(synthetic, 1.0, seed=0)
    assert len(full) == len(synthetic) and rep["users_kept"] == 60
    tiny, rep = subsample_users(synthetic, 0.001, seed=0)
    assert rep["users_kept"] == 1 and tiny["user"].nunique() == 1  # never an empty subset
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError, match="user_fraction"):
            subsample_users(synthetic, bad, seed=0)


@pytest.mark.skipif(not data_available("ml-1m"), reason="needs data/raw/ml-1m")
def test_subset_config_goes_through_the_normal_pipeline():
    cfg = load_config(ROOT / "configs" / "data" / "ml1m_u05.yaml")
    assert cfg["run_name"] == "ml-1m-u05" and cfg["name"] == "ml-1m"
    s, report = build_split(cfg)  # raises if check_split fails
    sub = report["user_subsample"]
    assert sub["users_kept"] == round(0.05 * sub["users_total"]) and sub["ratings_kept"] < sub["ratings_total"]
    assert check_split(s) == [] and len(s.val) > 0 and len(s.test) > 0
    inter, _ = load("ml-1m", ROOT / cfg["raw_dir"])
    chosen = set(subsample_users(inter, cfg["user_fraction"], cfg["user_seed"])[0]["user"])
    used = set(s.train["user"]) | set(s.val["user"]) | set(s.test["user"])
    assert used <= chosen  # no user outside the sample reaches any part of the split


# ---- time window (decision 0016 item 1) -------------------------------------------------------------------------


def _calendar_inter():
    """One event per day from 2014-01-01 to 2014-06-30 (UTC) with a rotating user and item."""
    days = pd.date_range("2014-01-01", "2014-06-30", freq="D")
    ts = ((days - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy()
    return pd.DataFrame({"user": np.arange(len(ts)) % 7, "item": np.arange(len(ts)) % 11, "rating": 1, "ts": ts})


def test_window_keeps_the_last_calendar_months():
    inter = _calendar_inter()
    win, rep = cut_time_window(inter, 3)
    assert rep["window_end_ts"] == int(pd.Timestamp("2014-06-30").timestamp())
    assert rep["window_start_ts"] == int(pd.Timestamp("2014-03-30").timestamp())
    assert win["ts"].min() == rep["window_start_ts"] and win["ts"].max() == rep["window_end_ts"]
    assert len(win) == rep["ratings_kept"] == 93 and rep["ratings_total"] == len(inter)
    assert (win["ts"] >= rep["window_start_ts"]).all()
    assert (inter["ts"] < rep["window_start_ts"]).sum() == len(inter) - 93


def test_window_is_independent_of_row_order_and_other_columns():
    inter = _calendar_inter()
    a, ra = cut_time_window(inter, 2)
    shuffled = inter.sample(frac=1, random_state=3).reset_index(drop=True)
    b, rb = cut_time_window(shuffled, 2)
    assert ra == rb
    key = ["ts", "user", "item"]
    assert a.sort_values(key).reset_index(drop=True).equals(b.sort_values(key).reset_index(drop=True))
    scrambled = inter.assign(user=inter["user"][::-1].to_numpy(), item=inter["item"][::-1].to_numpy())
    c, rc = cut_time_window(scrambled, 2)  # the decision reads only ts
    assert rc["window_start_ts"] == ra["window_start_ts"] and sorted(c["ts"]) == sorted(a["ts"])


def test_window_longer_than_the_data_keeps_everything_and_bad_input_fails():
    inter = _calendar_inter()
    win, rep = cut_time_window(inter, 24)
    assert len(win) == len(inter) and rep["users_kept"] == rep["users_total"]
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="window_months"):
            cut_time_window(inter, bad)
    with pytest.raises(ValueError, match="empty"):
        cut_time_window(inter.iloc[0:0], 3)


def test_window_config_goes_through_the_split_and_excludes_user_fraction(tmp_path):
    inter = _calendar_inter()
    rows = pd.concat([inter.assign(user=inter["user"] + 100 * k, item=inter["item"] + 13 * k) for k in range(4)])
    raw = tmp_path / "raw"
    raw.mkdir()
    rows[["user", "item", "ts"]].drop_duplicates(["user", "item"]).to_csv(
        raw / "kgat_author_earliest.tsv", sep="\t", index=False
    )
    cfg = {
        "name": "lastfm",
        "raw_dir": str(raw),
        "window_months": 3,
        "split": {"q_val": 0.8, "q_test": 0.9, "min_rating": 1, "kcore": 1},
    }
    s, report = build_split(cfg)  # raises if check_split fails
    w = report["time_window"]
    assert check_split(s) == [] and w["window_months"] == 3
    used = pd.concat([s.train, s.val, s.test])
    assert used["ts"].min() >= w["window_start_ts"] and s.t1 > w["window_start_ts"] and s.t2 <= w["window_end_ts"]
    with pytest.raises(ValueError, match="alternatives"):
        build_split({**cfg, "user_fraction": 0.5})
