import pytest

from tkgat.data.leakage import check_split
from tkgat.data.loaders import load
from tkgat.data.subsample import subsample_users
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
