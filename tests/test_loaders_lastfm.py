"""The Last-FM loader on a synthetic copy of ``kgat_author_earliest.tsv`` (no real data needed)."""

import numpy as np
import pandas as pd
import pytest

from tkgat.data.loaders import LASTFM_INTERACTIONS, load, load_lastfm
from tkgat.data.split import SplitConfig, temporal_split
from tkgat.pipeline.build import load_config

from .conftest import ROOT

ITEM_COLS = ["item", "title", "year", "release_date", "genres"]


def write_rows(raw_dir, rows) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["user", "item", "ts"]).to_csv(
        raw_dir / LASTFM_INTERACTIONS, sep="\t", index=False, lineterminator="\n"
    )


def synthetic_rows() -> list[tuple[int, int, int]]:
    """30 rows: 6 users x 5 items, one timestamp per pair, a day apart."""
    return [(u, 100 + i, 1_390_000_000 + (u * 5 + i) * 86_400) for u in range(6) for i in range(5)]


@pytest.fixture
def raw_dir(tmp_path):
    write_rows(tmp_path / "lfm1b-tracks", synthetic_rows())
    return tmp_path / "lfm1b-tracks"


def test_schema_matches_the_movielens_loaders(raw_dir):
    inter, items = load_lastfm(raw_dir)
    assert list(inter.columns) == ["user", "item", "rating", "ts"]
    assert list(items.columns) == ITEM_COLS
    assert all(inter[c].dtype == np.int64 for c in inter.columns)


def test_every_interaction_is_a_positive(raw_dir):
    inter, _ = load_lastfm(raw_dir)
    assert len(inter) == 30 and (inter["rating"] == 1).all()
    # min_rating: 1 in the data config keeps all of them
    assert len(temporal_split(inter, SplitConfig(min_rating=1, kcore=2)).train) > 0


def test_timestamps_and_ids_are_read_unchanged(raw_dir):
    inter, _ = load_lastfm(raw_dir)
    expected = pd.DataFrame(synthetic_rows(), columns=["user", "item", "ts"])
    got = inter.sort_values(["user", "item"], ignore_index=True)[["user", "item", "ts"]]
    pd.testing.assert_frame_equal(got, expected.sort_values(["user", "item"], ignore_index=True))


def test_items_are_the_item_ids_with_empty_metadata(raw_dir):
    inter, items = load_lastfm(raw_dir)
    assert list(items["item"]) == sorted(inter["item"].unique())
    assert (items["title"] == "").all() and (items["genres"] == "").all()
    assert items["year"].isna().all() and items["release_date"].isna().all()


def test_registered_under_lastfm(raw_dir):
    inter, items = load("lastfm", raw_dir)
    assert len(inter) == 30 and list(items.columns) == ITEM_COLS


def test_duplicate_pairs_are_refused(tmp_path):
    write_rows(tmp_path / "dup", [(0, 100, 1), (0, 100, 2)])
    with pytest.raises(ValueError, match="duplicate"):
        load_lastfm(tmp_path / "dup")


def test_wrong_columns_are_refused(tmp_path):
    out = tmp_path / "bad"
    out.mkdir()
    (out / LASTFM_INTERACTIONS).write_text("user\titem\trating\n0\t1\t5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        load_lastfm(out)


@pytest.mark.parametrize("name", ["lastfm", "lastfm_u10"])
def test_shipped_configs_use_the_loader_and_the_protocol(name):
    cfg = load_config(ROOT / "configs" / "data" / f"{name}.yaml")
    assert cfg["name"] == "lastfm" and not cfg["clean"]
    assert cfg["split"] == {"q_val": 0.8, "q_test": 0.9, "min_rating": 1, "kcore": 5}
    SplitConfig(**cfg["split"])  # the protocol's own validation
